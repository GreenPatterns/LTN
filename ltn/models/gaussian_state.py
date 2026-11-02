"""Gaussian-state variant (the -DM ablation and pre-registered G1 pivot).

Same API surface as LTNModel (init_state/step/forecast/divergence) with the density
matrix replaced by a diagonal-covariance Gaussian latent of dimension q = d*d, so
state size matches LTN at d=4 (q=16). Dynamics analog of Lindblad decoherence:
learned OU contraction of the mean + cal_age-coupled variance injection. No CPTP
structure anywhere — that is the point of the ablation.

Uncertainty decomposition analogs (documented for E7 comparability):
  entropy    = mean log sigma^2, squashed to [0,1] against [-8, 4] nats reference
  vacuity    = squashed mean variance (sigmoid of log mean var)
  dissonance = normalized dispersion of per-dim means (|mu| spread — conflict analog)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

RD = torch.float64


@dataclass
class GaussStep:
    mu: torch.Tensor        # [n, q]
    log_var: torch.Tensor   # [n, q]
    carry: torch.Tensor     # [n, C]
    entropy: torch.Tensor   # [n]
    vacuity: torch.Tensor   # [n]
    dissonance: torch.Tensor  # [n]
    # aliases so probe/train code that touches .rho never crashes on the variant
    @property
    def rho(self) -> torch.Tensor:
        return self.mu


def _diag_entropy(log_var: torch.Tensor) -> torch.Tensor:
    raw = log_var.mean(-1)
    return torch.sigmoid((raw + 2.0) / 2.0)


def _vacuity(log_var: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(log_var.exp().mean(-1).log1p() * 2.0 - 1.0)


def _dissonance(mu: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(mu.std(-1) - 1.0)


class GaussianModel(nn.Module):
    def __init__(
        self,
        n_obs: int,
        platform_dim: int,
        d: int = 4,
        carry_dim: int = 32,
        hidden: int = 64,
        K: int = 3,
        n_facets: int = 2,
        evidence_rank: int = 4,
        rounds: int = 2,
        use_graph: bool = True,
        use_lindblad: bool = True,
        physics_gamma: bool = True,
        time_unit_h: float = 8.0,
    ) -> None:
        super().__init__()
        q = d * d
        self.q, self.n_obs, self.platform_dim = q, n_obs, platform_dim
        self.use_graph, self.physics_gamma = use_graph, physics_gamma
        self.time_unit_h = time_unit_h
        self.rounds = rounds
        feat_dim = 2 * n_obs + 2 + platform_dim
        gdim = 4 + platform_dim
        self.drift_gate = nn.Sequential(nn.Linear(gdim, hidden // 2, dtype=RD), nn.SiLU(),
                                        nn.Linear(hidden // 2, 2, dtype=RD))
        self.enc = nn.Sequential(nn.Linear(feat_dim, hidden, dtype=RD), nn.SiLU(),
                                 nn.Linear(hidden, q, dtype=RD))
        self.gate = nn.Sequential(nn.Linear(feat_dim, q, dtype=RD), nn.Sigmoid())
        self.msg = nn.Sequential(nn.Linear(q + 2, q, dtype=RD), nn.SiLU())
        self.gru = nn.GRUCell(feat_dim + q, carry_dim, dtype=RD)
        self.head = nn.Sequential(nn.Linear(2 * q + carry_dim + 1, hidden, dtype=RD), nn.SiLU(),
                                  nn.Linear(hidden, 2 * n_obs, dtype=RD))
        self.div_head = nn.Sequential(nn.Linear(3, 16, dtype=RD), nn.SiLU(), nn.Linear(16, 1, dtype=RD))
        self.carry0 = nn.Parameter(torch.zeros(carry_dim, dtype=RD))

    def init_state(self, n: int, device=None) -> GaussStep:
        mu = torch.zeros(n, self.q, dtype=RD, device=device)
        lv = torch.zeros(n, self.q, dtype=RD, device=device)  # unit variance prior
        carry = self.carry0.expand(n, -1).clone()
        return GaussStep(mu, lv, carry, _diag_entropy(lv), _vacuity(lv), _dissonance(mu))

    def _gfeat(self, x, cal_age_h, dt_steps):
        n = x.shape[0]
        zeros = torch.zeros(n, dtype=RD, device=x.device)
        t1z = x[:, 0].to(RD) if self.n_obs > 0 else zeros
        t2z = x[:, 1].to(RD) if self.n_obs > 1 else zeros
        age = torch.log1p(cal_age_h.to(RD).clamp_min(0))
        if not self.physics_gamma:
            t1z, t2z, age = zeros, zeros, zeros
        load = torch.log1p(dt_steps.to(RD).clamp_min(0))
        plat = x[:, -self.platform_dim:].to(RD) if self.platform_dim else x[:, :0].to(RD)
        return torch.cat([torch.stack([t1z, t2z, age, load], -1).nan_to_num(0.0), plat], -1)

    def _drift(self, mu, log_var, gfeat, dt_steps):
        out = self.drift_gate(gfeat)
        theta = F.softplus(out[:, 0:1]) * 0.3          # contraction rate
        inject = F.softplus(out[:, 1:2]) * 0.3         # variance injection rate
        dt = dt_steps.to(RD).unsqueeze(-1)
        phi = torch.exp(-theta * dt)
        mu2 = mu * phi                                  # decay toward the vacuous prior mean 0
        var2 = log_var.exp() * phi ** 2 + inject * dt   # uncertainty grows without evidence
        return mu2, var2.clamp_min(1e-8).log()

    def step(self, state: GaussStep, x, cal_age, dt_steps, edge_index, edge_attr) -> GaussStep:
        x = x.to(RD)
        mu, lv = self._drift(state.mu, state.log_var, self._gfeat(x, cal_age, dt_steps), dt_steps)
        k = self.gate(x)
        obs = self.enc(x)
        mu = (1 - k) * mu + k * obs
        lv = (lv.exp() * (1 - k) ** 2 + 1e-8).log()
        if self.use_graph and edge_index.numel():
            src, dst = edge_index[0], edge_index[1]
            for _ in range(self.rounds):
                m = self.msg(torch.cat([mu[src], edge_attr.to(RD)], -1))
                agg = torch.zeros_like(mu).index_add_(0, dst, m)
                deg = torch.zeros(mu.shape[0], dtype=RD, device=mu.device).index_add_(
                    0, dst, torch.ones(len(dst), dtype=RD, device=mu.device)).clamp_min(1.0)
                mu = 0.7 * mu + 0.3 * agg / deg.unsqueeze(-1)
        carry = self.gru(torch.cat([x, mu], -1), state.carry)
        return GaussStep(mu, lv, carry, _diag_entropy(lv), _vacuity(lv), _dissonance(mu))

    def forecast(self, state: GaussStep, x_now, cal_age_now, horizon_steps):
        gfeat = self._gfeat(x_now.to(RD), cal_age_now + horizon_steps * self.time_unit_h, horizon_steps)
        mu_h, lv_h = self._drift(state.mu, state.log_var, gfeat, horizon_steps)
        z = torch.cat([mu_h, lv_h, state.carry, torch.log1p(horizon_steps.to(RD)).unsqueeze(-1)], -1)
        out = self.head(z)
        mu, raw = out[..., : self.n_obs], out[..., self.n_obs:]
        return mu, F.softplus(raw) + 1e-4, _diag_entropy(lv_h)

    def divergence(self, state: GaussStep) -> torch.Tensor:
        z = torch.stack([state.entropy, state.vacuity, state.dissonance], -1)
        return torch.sigmoid(self.div_head(z)).squeeze(-1)
