"""Lindblad drift — the signature module (MASTER_EXECUTION_PLAN §2.2).

Over a telemetry gap dt each node's trust state evolves by the exact solution of
    drho/dt = -i[H_v, rho] + sum_k gamma_{v,k} (L_k rho L_k^dag - 1/2 {L_k^dag L_k, rho})
vectorized to a d^2 x d^2 superoperator and exponentiated with batched
torch.matrix_exp. Convention: COLUMN-stacking vec (dmstate.vec) with superoperators
kron(B.T, A), for which vec(A rho B) = (B^T (x) A) vec(rho) holds exactly; this is
cross-checked against qutip.mesolve for general H != 0 at d=4 (< 1e-9) in
tests/test_lindblad.py, not only the degenerate d=2 amplitude-damping case.

Tested guarantees:
  * the channel exp(Lgen*dt) is CPTP for any Hermitian H, any jump ops and any
    gamma >= 0, dt >= 0 (TP error < 1e-8; Choi min eigenvalue > -1e-7);
  * entropy is nondecreasing under pure dephasing (H = 0, diagonal jump op);
  * matches qutip.mesolve at d=2 (amplitude damping) AND d=4 (general H) to < 1e-9;
  * GammaNet has d gamma_k / d cal_age > 0 at initialization (physics prior:
    trust decoheres faster as calibration data ages);
  * dt = 0 is the identity channel.

All quantum tensors complex128; rates/features float64 internally.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltn.models.dmstate import CDTYPE, RDTYPE, unvec, vec

# --- superoperator algebra (verify_env.py pattern, column-stacking kron(B.T, A)) ---


def sprepost(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """rho -> A rho B  ==>  kron(B.T, A) acting on vec(rho)."""
    return torch.kron(B.transpose(-2, -1).contiguous(), A)


def spre(A: torch.Tensor) -> torch.Tensor:
    """rho -> A rho."""
    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return torch.kron(I, A)


def spost(B: torch.Tensor) -> torch.Tensor:
    """rho -> rho B."""
    I = torch.eye(B.shape[-1], dtype=B.dtype, device=B.device)
    return torch.kron(B.transpose(-2, -1).contiguous(), I)


def _bkron(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Batched Kronecker product: [B,d,d] x [B,d,d] -> [B,d^2,d^2]."""
    b, d, _ = X.shape
    return torch.einsum("bij,bkl->bikjl", X, Y).reshape(b, d * d, d * d)


def lindblad_generator(
    H: torch.Tensor,
    Ls: Sequence[torch.Tensor],
    gammas: torch.Tensor,
) -> torch.Tensor:
    """Batched Lindbladian [B, d^2, d^2] from H [B,d,d], shared Ls (each [d,d]),
    per-node rates gammas [B, K] (real, >= 0 for a CPTP channel)."""
    b, d, _ = H.shape
    H = H.to(CDTYPE)
    I = torch.eye(d, dtype=CDTYPE, device=H.device).expand(b, d, d)
    # -i [H, rho]: batched spre(H) - spost(H) via batched kron
    gen = -1j * (_bkron(I, H) - _bkron(H.transpose(-2, -1), I))
    # Dissipators of the SHARED jump ops are batch-independent [K, d^2, d^2];
    # the batch enters only through the rates.
    diss = []
    for Lk in Ls:
        Lk = Lk.to(CDTYPE)
        LdL = Lk.conj().T @ Lk
        diss.append(sprepost(Lk, Lk.conj().T) - 0.5 * spre(LdL) - 0.5 * spost(LdL))
    D = torch.stack(diss)  # [K, d^2, d^2]
    return gen + torch.einsum("bk,kmn->bmn", gammas.to(CDTYPE), D)


def evolve(
    rho: torch.Tensor,
    H: torch.Tensor,
    Ls: Sequence[torch.Tensor],
    gammas: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Exact Lindblad evolution over per-sample gaps dt [B] (float64, >= 0).

    rho' = unvec(matrix_exp(Lgen * dt) @ vec(rho)); the closing hermitization
    and trace renormalization are numerical hygiene ONLY — we assert the drift
    they remove is < 1e-6, so they can never mask a wrong generator.
    """
    gen = lindblad_generator(H, Ls, gammas)
    S = torch.matrix_exp(gen * dt.to(CDTYPE).view(-1, 1, 1))
    rho1 = unvec((S @ vec(rho.to(CDTYPE)).unsqueeze(-1)).squeeze(-1))

    tr = torch.diagonal(rho1, dim1=-2, dim2=-1).sum(-1)
    trace_drift = (tr - 1.0).abs().max()
    herm_drift = (rho1 - rho1.conj().transpose(-2, -1)).abs().max()
    assert trace_drift.item() < 1e-6, f"trace drift {trace_drift.item():.3e} >= 1e-6"
    assert herm_drift.item() < 1e-6, f"hermiticity drift {herm_drift.item():.3e} >= 1e-6"

    rho1 = (rho1 + rho1.conj().transpose(-2, -1)) / 2
    tr = torch.diagonal(rho1, dim1=-2, dim2=-1).sum(-1).real
    return rho1 / tr.to(CDTYPE).view(-1, 1, 1)


class JumpDictionary(nn.Module):
    """Shared learned jump-operator dictionary {L_k}, K=3, complex [K, d, d].

    Initialization (each Frobenius-normalized):
      k=0 lowering ladder  sum_i |i><i+1|          (T1-like energy relaxation)
      k=1 dephasing        diag(0..d-1), traceless (T2-like coherence loss)
      k=2 generalized shift X = sum_i |i+1 mod d><i|.
    The shift X is a single-operator PROXY for depolarizing/mixing: a true
    depolarizing dissipator needs the full d^2-1 Weyl basis, which would bloat
    K; X cyclically mixes all populations and, learned jointly with k=0/1,
    spans the isotropic-mixing direction the dictionary must express.
    """

    def __init__(self, d: int, K: int = 3) -> None:
        super().__init__()
        if K != 3:
            raise ValueError("JumpDictionary init pattern is specified for K=3")
        self.d, self.K = d, K
        init = torch.zeros(K, d, d, dtype=RDTYPE)
        ar = torch.arange(d - 1)
        init[0, ar, ar + 1] = 1.0  # lowering
        deph = torch.arange(d, dtype=RDTYPE)
        init[1] = torch.diag(deph - deph.mean())  # traceless dephasing
        init[2, (torch.arange(d) + 1) % d, torch.arange(d)] = 1.0  # cyclic shift
        for k in range(K):
            init[k] = init[k] / init[k].norm()
        self.ops_real = nn.Parameter(init)
        self.ops_imag = nn.Parameter(torch.zeros(K, d, d, dtype=RDTYPE))

    def ops(self) -> torch.Tensor:
        return torch.complex(self.ops_real, self.ops_imag)

    def forward(self) -> torch.Tensor:
        return self.ops()


class GammaNet(nn.Module):
    """Physics coupling gamma_{v,k} = softplus(g_theta(features)) [B, K].

    Feature layout: [t1_z, t2_z, log1p(cal_age_h), load, platform_emb...],
    cal_age at index `cal_age_idx` (default 2).

    TESTED guarantee — d gamma_k / d cal_age > 0 at initialization: the MLP's
    final layer is zero-initialized (so it contributes neither value nor input
    gradient at init) and a FIXED positive linear skip on the cal_age feature
    feeds every output, giving d gamma_k/d cal_age = softplus'(.) * w > 0.
    The MLP is the learned residual that takes over during training.
    """

    def __init__(
        self,
        in_dim: int,
        K: int = 3,
        hidden: int = 32,
        cal_age_idx: int = 2,
        bias_init: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if not (0 <= cal_age_idx < in_dim):
            raise ValueError("cal_age_idx out of range")
        self.cal_age_idx = cal_age_idx
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=RDTYPE),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=RDTYPE),
            nn.SiLU(),
            nn.Linear(hidden, K, dtype=RDTYPE),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("cal_skip", torch.full((K,), 0.5, dtype=RDTYPE))
        # Per-channel initial rate scale. Trust semantics at init: the UNITAL
        # (mixing) dissipators dominate, so evidence-starved states drift toward
        # maximal mixedness (vacuity), never toward unwarranted confidence; the
        # non-unital T1-like lowering channel (k=0 in JumpDictionary) starts
        # small — softplus(-3) ~ 0.05 vs softplus(-1) ~ 0.31 — and is learned.
        if bias_init is None:
            bias_init = (-3.0,) + (-1.0,) * (K - 1)
        if len(bias_init) != K:
            raise ValueError("bias_init length must equal K")
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=RDTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(RDTYPE)
        pre = self.mlp(x) + self.cal_skip * x[..., self.cal_age_idx : self.cal_age_idx + 1] + self.bias
        return F.softplus(pre)


class HermitianHead(nn.Module):
    """Node embedding -> Hermitian H = A + A^dagger [B, d, d] (complex128).

    Output is scaled small at init so early dynamics are dissipation-dominated.
    """

    def __init__(self, emb_dim: int, d: int, init_scale: float = 1e-2) -> None:
        super().__init__()
        self.d = d
        self.proj = nn.Linear(emb_dim, 2 * d * d, dtype=RDTYPE)
        with torch.no_grad():
            self.proj.weight.mul_(init_scale)
            self.proj.bias.mul_(init_scale)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        b = emb.shape[0]
        out = self.proj(emb.to(RDTYPE)).reshape(b, 2, self.d, self.d)
        A = torch.complex(out[:, 0], out[:, 1])
        return A + A.conj().transpose(-2, -1)
