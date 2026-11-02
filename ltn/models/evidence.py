"""Evidence channel: telemetry event -> CPTP update (MASTER_EXECUTION_PLAN §2.2).

Features x -> MLP -> r=4 raw complex matrices {A_i}; M = sum_i A_i^dag A_i;
K_i = A_i M^{-1/2}; rho <- sum_i K_i rho K_i^dag.

Guarantee (tested): completeness sum_i K_i^dag K_i = M^{-1/2} M M^{-1/2} = I is
EXACT by construction, up to the eigenvalue clamp (min 1e-8) inside the inverse
square root — max-abs deviation < 1e-5 for random inputs/parameters. The
channel is therefore CPTP and maps valid states to valid states for ANY
network output.

All quantum tensors complex128; the MLP runs in float64.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ltn.models.dmstate import CDTYPE, RDTYPE

_EIG_CLAMP = 1e-8


class EvidenceChannel(nn.Module):
    """x [B, feat_dim] -> Kraus set {K_i} [B, rank, d, d] -> rho update."""

    def __init__(self, feat_dim: int, d: int, rank: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.d, self.rank = d, rank
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden, dtype=RDTYPE),
            nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=RDTYPE),
            nn.SiLU(),
            nn.Linear(hidden, 2 * rank * d * d, dtype=RDTYPE),
        )

    def kraus(self, x: torch.Tensor) -> torch.Tensor:
        """Complete Kraus operators [B, rank, d, d] with sum K^dag K = I."""
        b = x.shape[0]
        raw = self.mlp(x.to(RDTYPE)).reshape(b, self.rank, 2, self.d, self.d)
        A = torch.complex(raw[:, :, 0], raw[:, :, 1])  # [B, r, d, d]
        M = torch.einsum("brji,brjk->bik", A.conj(), A)  # sum_i A_i^dag A_i, PSD
        M = (M + M.conj().transpose(-2, -1)) / 2
        w, V = torch.linalg.eigh(M)
        inv_sqrt = w.clamp_min(_EIG_CLAMP).rsqrt().to(CDTYPE)
        M_inv_sqrt = V @ (inv_sqrt.unsqueeze(-1) * V.conj().transpose(-2, -1))
        return A @ M_inv_sqrt.unsqueeze(1)

    def forward(self, rho: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply the event channel: rho [B,d,d] <- sum_i K_i rho K_i^dag."""
        K = self.kraus(x)
        return torch.einsum("brij,bjk,brlk->bil", K, rho.to(CDTYPE), K.conj())
