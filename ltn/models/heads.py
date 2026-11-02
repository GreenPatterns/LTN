"""Readout heads (MASTER_EXECUTION_PLAN §2.2).

POVM heads: E_c = C_c^dag C_c, set-normalized so sum_c E_c = I (a true POVM);
score_c = Re tr(E_c rho) — guaranteed in [0, 1] and summing to 1 across facets.

Forecast head: MLP on (real-vectorized rho, GRU carry, horizon) -> per-observable
Gaussian (mu, sigma). sigma via softplus + floor.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltn.models.dmstate import CDTYPE, RDTYPE


def rho_to_real(rho: torch.Tensor) -> torch.Tensor:
    """[B,d,d] complex Hermitian -> [B, d*d] real features (diag + scaled upper tri re/im)."""
    b, d, _ = rho.shape
    iu = torch.triu_indices(d, d, offset=1, device=rho.device)
    diag = torch.diagonal(rho, dim1=-2, dim2=-1).real
    upper = rho[:, iu[0], iu[1]]
    return torch.cat([diag, upper.real * 2**0.5, upper.imag * 2**0.5], dim=-1).to(RDTYPE)


class POVMHeads(nn.Module):
    """Trust facets as a learned POVM {E_c}, c = 1..n_facets."""

    def __init__(self, d: int, n_facets: int = 2) -> None:
        super().__init__()
        self.d, self.n_facets = d, n_facets
        init = torch.randn(n_facets, 2, d, d, dtype=RDTYPE) / d
        # bias facet 0 toward identity so "overall trust" starts near tr(rho)/n
        init[0, 0] += torch.eye(d, dtype=RDTYPE)
        self.C = nn.Parameter(init)

    def povm(self) -> torch.Tensor:
        C = torch.complex(self.C[:, 0], self.C[:, 1])
        E = C.conj().transpose(-2, -1) @ C  # [n, d, d] PSD
        M = E.sum(0)
        evals, vecs = torch.linalg.eigh(M)
        inv_sqrt = vecs @ torch.diag_embed(evals.clamp_min(1e-10).rsqrt().to(CDTYPE)) @ vecs.conj().transpose(-2, -1)
        return inv_sqrt @ E @ inv_sqrt  # sum = I exactly (up to numerics)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        E = self.povm()
        scores = torch.einsum("nij,bji->bn", E, rho.to(CDTYPE)).real
        return scores.clamp(0.0, 1.0)


class ForecastHead(nn.Module):
    """(rho, carry, horizon) -> per-observable Gaussian (mu, sigma) as z-scores."""

    def __init__(self, d: int, carry_dim: int, n_obs: int, hidden: int = 64) -> None:
        super().__init__()
        in_dim = d * d + carry_dim + 1  # rho features + carry + log-horizon
        self.n_obs = n_obs
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, dtype=RDTYPE),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=RDTYPE),
            nn.SiLU(),
            nn.Linear(hidden, 2 * n_obs, dtype=RDTYPE),
        )

    def forward(
        self, rho: torch.Tensor, carry: torch.Tensor, horizon_steps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.log1p(horizon_steps.to(RDTYPE)).unsqueeze(-1)
        z = torch.cat([rho_to_real(rho), carry.to(RDTYPE), h], dim=-1)
        out = self.net(z)
        mu, raw = out[..., : self.n_obs], out[..., self.n_obs :]
        sigma = F.softplus(raw) + 1e-4
        return mu, sigma


class DivergenceHead(nn.Module):
    """Trust-state -> P(claim-delivery divergence within horizon) for E3.

    Uses the spectral decomposition features (vacuity, dissonance) plus POVM scores —
    the pre-registered story is that dissonance anticipates divergence."""

    def __init__(self, d: int, n_facets: int = 2, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_facets + 2, hidden, dtype=RDTYPE),
            nn.SiLU(),
            nn.Linear(hidden, 1, dtype=RDTYPE),
        )

    def forward(self, povm_scores: torch.Tensor, vacuity: torch.Tensor, dissonance: torch.Tensor) -> torch.Tensor:
        z = torch.cat([povm_scores.to(RDTYPE), vacuity.unsqueeze(-1).to(RDTYPE), dissonance.unsqueeze(-1).to(RDTYPE)], dim=-1)
        return torch.sigmoid(self.net(z)).squeeze(-1)
