"""Density-matrix trust state (MASTER_EXECUTION_PLAN §2.2 "State").

Guarantees implemented and unit-tested here:
  * rho_from_params returns a valid density matrix for EVERY theta in R^{d^2}:
    Hermitian, positive semidefinite, trace exactly 1 (Cholesky-style
    parameterization rho = L L^dagger / tr(L L^dagger), L lower-triangular with
    real-positive diagonal via softplus for identifiability).
  * params_init(mixed=True) yields exactly the maximally mixed state rho = I/d,
    the correct vacuous-trust prior (vacuity 1, dissonance 0).
  * vec/unvec use the SAME flattening convention as ltn.models.lindblad
    (vec(rho) = rho.reshape(-1), superoperators built as kron(B.T, A) — the
    pattern validated against qutip at 1.7e-12 in scripts/verify_env.py).

All quantum tensors are complex128; parameters are float64.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CDTYPE = torch.complex128
RDTYPE = torch.float64

_EIG_FLOOR = 1e-12  # eigenvalue clamp inside the entropy log


def _softplus_inv(y: float) -> float:
    """Inverse of softplus: x such that softplus(x) = y (y > 0)."""
    return math.log(math.expm1(y))


def rho_from_params(theta: torch.Tensor) -> torch.Tensor:
    """Map theta in R^{B x d^2} to density matrices [B, d, d] (complex128).

    Layout of theta per row: first d entries are pre-softplus diagonal of L,
    next d(d-1)/2 are Re of the strictly-lower triangle (row-major
    tril_indices order), last d(d-1)/2 are the matching Im parts.
    """
    if theta.dim() != 2:
        raise ValueError(f"theta must be [B, d*d], got shape {tuple(theta.shape)}")
    theta = theta.to(RDTYPE)
    b, dd = theta.shape
    d = math.isqrt(dd)
    if d * d != dd:
        raise ValueError(f"theta second dim {dd} is not a perfect square")
    n_off = d * (d - 1) // 2

    diag = F.softplus(theta[:, :d])  # real-positive diagonal => unique L
    off = torch.complex(theta[:, d:d + n_off], theta[:, d + n_off:])

    L = torch.zeros(b, d, d, dtype=CDTYPE, device=theta.device)
    ar = torch.arange(d, device=theta.device)
    L[:, ar, ar] = diag.to(CDTYPE)
    rows, cols = torch.tril_indices(d, d, offset=-1, device=theta.device)
    L[:, rows, cols] = off

    m = L @ L.conj().transpose(-2, -1)
    tr = torch.diagonal(m, dim1=-2, dim2=-1).sum(-1).real  # > 0 since diag > 0
    return m / tr.to(CDTYPE).view(-1, 1, 1)


def params_init(batch: int, d: int, mixed: bool = True, noise: float = 0.0) -> torch.Tensor:
    """Initial theta [batch, d*d] (float64).

    mixed=True  -> exactly rho = I/d, the vacuous-trust prior.
    mixed=False -> numerically pure |0><0| (softplus cannot reach 0 exactly).
    Optional gaussian `noise` breaks symmetry between nodes.
    """
    theta = torch.zeros(batch, d * d, dtype=RDTYPE)
    if mixed:
        theta[:, :d] = _softplus_inv(1.0)  # L = I -> rho = I/d
    else:
        theta[:, 0] = _softplus_inv(1.0)
        theta[:, 1:d] = -20.0  # softplus(-20) ~ 2e-9 -> purity 1 - O(1e-17)
    if noise > 0:
        theta = theta + noise * torch.randn_like(theta)
    return theta


def vec(rho: torch.Tensor) -> torch.Tensor:
    """Column-stacking vectorization [..., d, d] -> [..., d*d].

    This is the convention for which the superoperator identity
    vec(A rho B) = (B^T (x) A) vec(rho) holds (lindblad.py builds generators as
    kron(B.T, A)). Column-stacking = row-major flatten of the transpose.
    """
    d = rho.shape[-1]
    return rho.transpose(-2, -1).reshape(*rho.shape[:-2], d * d)


def unvec(v: torch.Tensor) -> torch.Tensor:
    """Inverse of the column-stacking vec: [..., d*d] -> [..., d, d]."""
    d = math.isqrt(v.shape[-1])
    return v.reshape(*v.shape[:-1], d, d).transpose(-2, -1)


def von_neumann_entropy(rho: torch.Tensor, normalized: bool = False) -> torch.Tensor:
    """S(rho) = -tr(rho ln rho) via eigvalsh, natural log; /ln(d) if normalized."""
    herm = (rho + rho.conj().transpose(-2, -1)) / 2
    w = torch.linalg.eigvalsh(herm).clamp_min(_EIG_FLOOR)
    s = -(w * w.log()).sum(-1)
    if normalized:
        s = s / math.log(rho.shape[-1])
    return s


def purity(rho: torch.Tensor) -> torch.Tensor:
    """tr(rho^2), real in [1/d, 1]."""
    return torch.einsum("...ij,...ji->...", rho, rho).real


def vacuity(rho: torch.Tensor) -> torch.Tensor:
    """Normalized purity deficit (1 - tr(rho^2)) * d/(d-1) in [0, 1].

    1 at the maximally mixed state (tr rho^2 = 1/d), 0 at any pure state.
    """
    d = rho.shape[-1]
    return (1.0 - purity(rho)) * d / (d - 1)


def dissonance(rho: torch.Tensor) -> torch.Tensor:
    """Normalized l1 off-diagonal coherence in [0, 1].

    C_l1(rho) = sum_{i != j} |rho_ij| is bounded by d-1, with equality only at
    maximally coherent pure states (all entries 1/d, e.g. |+><+|); this is the
    tight max-coherence bound (Baumgratz et al., PRL 113, 140401). We therefore
    normalize by (d-1) — NOT by the off-diagonal count d(d-1), which would cap
    the measure at 1/d — so dissonance(|+><+|) = 1 and dissonance(diagonal) = 0.
    """
    d = rho.shape[-1]
    absr = rho.abs()
    offdiag = absr.sum((-2, -1)) - torch.diagonal(absr, dim1=-2, dim2=-1).sum(-1)
    return offdiag / (d - 1)


class DMState(nn.Module):
    """Learnable bank of n density-matrix trust states (optimize theta only)."""

    def __init__(self, n: int, d: int, mixed: bool = True, noise: float = 0.0) -> None:
        super().__init__()
        self.d = d
        self.theta = nn.Parameter(params_init(n, d, mixed=mixed, noise=noise))

    def forward(self) -> torch.Tensor:
        return rho_from_params(self.theta)

    def rho(self) -> torch.Tensor:
        return self.forward()
