"""Kraus propagation over the device graph (MASTER_EXECUTION_PLAN §2.2 "Propagation").

Per round (rounds=2 default), synchronously for every node v:
  message  m_{u->v} = EvidenceChannel(edge features)(rho_u)   (CPTP => valid state)
  attention alpha_{uv} = softmax_{u in In(v)} Re tr(rho_u rho_v) / tau, tau learnable
  update   rho_v <- (1 - beta_v) rho_v + beta_v sum_u alpha_{uv} m_{u->v},
           beta_v = sigmoid(w^T node_feats_v).

Guarantees (tested): every update is a convex mixture of valid density matrices,
so outputs are PSD/trace-1 by construction; attention rows sum to 1; the layer
is permutation-equivariant; a node with NO in-edges keeps its state EXACTLY
(beta never touches it — the vacuous-aggregate edge case is masked out).

Scatter ops use plain index_add_ on a flattened real view [N, 2*d*d]
(torch_scatter is not installed). Quantum tensors complex128.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ltn.models.dmstate import CDTYPE, RDTYPE
from ltn.models.evidence import EvidenceChannel


class KrausProp(nn.Module):
    """Multi-round CPTP message passing over rho [N, d, d]."""

    def __init__(self, d: int, edge_dim: int, node_dim: int, rounds: int = 2,
                 rank: int = 4, hidden: int = 64) -> None:
        super().__init__()
        self.d, self.rounds = d, rounds
        # one evidence channel per round: composed rounds are distinct maps
        self.channels = nn.ModuleList(
            EvidenceChannel(edge_dim, d, rank=rank, hidden=hidden) for _ in range(rounds)
        )
        self.log_tau = nn.Parameter(torch.zeros((), dtype=RDTYPE))  # tau = exp > 0
        self.beta_lin = nn.Linear(node_dim, 1, dtype=RDTYPE)

    def _attention(self, rho: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """alpha [E]: softmax of Re tr(rho_u rho_v)/tau over in-neighbors of v."""
        src, dst = edge_index[0], edge_index[1]
        n = rho.shape[0]
        sim = torch.einsum("eij,eji->e", rho[src], rho[dst]).real  # in [0, 1]
        logits = sim / self.log_tau.exp()
        # logits are bounded (|sim| <= 1), so a plain exp-normalize is stable
        ex = logits.exp()
        denom = torch.zeros(n, dtype=RDTYPE, device=rho.device)
        denom.index_add_(0, dst, ex)
        return ex / denom[dst]

    def forward(
        self,
        rho: torch.Tensor,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """rho [N,d,d] complex, node_feats [N,node_dim], edge_index [2,E] (u->v),
        edge_feats [E,edge_dim] -> updated rho [N,d,d] (and per-round alphas)."""
        rho = rho.to(CDTYPE)
        n, d = rho.shape[0], self.d
        src, dst = edge_index[0], edge_index[1]
        beta = torch.sigmoid(self.beta_lin(node_feats.to(RDTYPE))).squeeze(-1)  # [N]
        has_in = torch.zeros(n, dtype=torch.bool, device=rho.device)
        has_in[dst] = True
        attns: list[torch.Tensor] = []

        for channel in self.channels:
            if edge_index.shape[1] == 0:
                if return_attn:
                    attns.append(torch.zeros(0, dtype=RDTYPE, device=rho.device))
                continue
            alpha = self._attention(rho, edge_index)
            m = channel(rho[src], edge_feats)  # [E, d, d] valid states
            weighted = alpha.to(CDTYPE).view(-1, 1, 1) * m
            # scatter-sum on the flattened real view [N, 2*d*d] via index_add_
            flat = torch.view_as_real(weighted).reshape(-1, 2 * d * d)
            agg = torch.zeros(n, 2 * d * d, dtype=RDTYPE, device=rho.device)
            agg.index_add_(0, dst, flat)
            agg = torch.view_as_complex(agg.reshape(n, d, d, 2))
            mixed = (1.0 - beta).to(CDTYPE).view(-1, 1, 1) * rho \
                + beta.to(CDTYPE).view(-1, 1, 1) * agg
            # nodes without in-edges: aggregate is the zero matrix, not a state;
            # keep their rho untouched instead of a trace-deficient mixture
            rho = torch.where(has_in.view(-1, 1, 1), mixed, rho)
            if return_attn:
                attns.append(alpha)

        return (rho, attns) if return_attn else rho
