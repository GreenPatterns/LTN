"""LTN: the full model (MASTER_EXECUTION_PLAN §2.2) assembled from the tested modules.

Per device timeline, per snapshot:
  1. DRIFT (between snapshots): rho evolves under the learned Lindblad generator for
     the elapsed gap; rates gamma = GammaNet(physics features from the PREVIOUS
     snapshot — causally stale, exactly like a user extrapolating old calibration).
  2. EVIDENCE: the snapshot's observed features enter through the CPTP evidence
     channel (unobserved entities keep their drifted state — trust decoheres).
  3. PROPAGATION: 2 rounds of KrausProp over the device graph.
  4. CARRY: GRU tracks non-quantum context per entity.

Forecast at origin t for horizon h: a COPY of the state evolves EVIDENCE-FREE for the
horizon gap (cal_age growing linearly) — epistemic decoherence is the forecast's
uncertainty mechanism — then ForecastHead reads out per-observable (mu, sigma) in
z-score space. The E7 event-study reads entropy off exactly this evidence-free path.

Feature layout per entity (built by ltn.exp.dataset from graphbuild+panel):
  x = [z-scored value per target observable (F_obs), observed-mask (F_obs),
       log1p(cal_age_h), log1p(dt_since_prev_h), platform one-hot (P)]
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ltn.models.dmstate import CDTYPE, RDTYPE, dissonance, vacuity, von_neumann_entropy
from ltn.models.evidence import EvidenceChannel
from ltn.models.heads import DivergenceHead, ForecastHead, POVMHeads, rho_to_real
from ltn.models.krausprop import KrausProp
from ltn.models.lindblad import GammaNet, HermitianHead, JumpDictionary, evolve


@dataclass
class LTNStep:
    """Model state after one snapshot (all tensors detached only by the caller)."""

    rho: torch.Tensor      # [n, d, d] complex
    carry: torch.Tensor    # [n, C]
    entropy: torch.Tensor  # [n]
    vacuity: torch.Tensor  # [n]
    dissonance: torch.Tensor  # [n]


class LTNModel(nn.Module):
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
        use_hamiltonian: bool = True,
        time_unit_h: float = 8.0,
    ) -> None:
        super().__init__()
        self.d, self.n_obs, self.platform_dim = d, n_obs, platform_dim
        self.use_graph, self.use_lindblad, self.physics_gamma = use_graph, use_lindblad, physics_gamma
        # use_hamiltonian=False zeroes H in the generator and NOTHING else: the
        # HermitianHead is still constructed, so the parameter count is identical
        # to the full model. This isolates H as a cause of the seed variance, which
        # the v2 model (which changes several things at once) cannot do on its own.
        #
        # It does NOT isolate non-normality. v1's jump dictionary is itself
        # non-normal (lowering ladder / cyclic shift), so zeroing H leaves a MORE
        # non-normal generator, on all 5 E1a seeds: relative Henrici departure goes
        # 0.013-0.161 (full) -> 0.205-0.490 (-H). Only v2 is actually normal
        # (kappa_2(V) = 1.000000 exactly). Measured by scripts/spectra.py.
        self.use_hamiltonian = use_hamiltonian
        self.time_unit_h = time_unit_h
        feat_dim = 2 * n_obs + 2 + platform_dim
        # gamma features: [t1_z, t2_z, cal_age, load(dt), platform...]; when
        # physics_gamma is ablated the physics slots are zeroed (same dim, no info)
        self.gamma_net = GammaNet(in_dim=4 + platform_dim, K=K, hidden=hidden // 2, cal_age_idx=2)
        self.jumps = JumpDictionary(d, K)
        self.hermitian = HermitianHead(carry_dim, d)
        self.evidence = EvidenceChannel(feat_dim, d, rank=evidence_rank)
        self.prop = KrausProp(d, edge_dim=2, node_dim=feat_dim, rounds=rounds)
        self.gru = nn.GRUCell(feat_dim + d * d, carry_dim, dtype=RDTYPE)
        self.forecast_head = ForecastHead(d, carry_dim, n_obs, hidden=hidden)
        self.povm = POVMHeads(d, n_facets)
        self.diverge_head = DivergenceHead(d, n_facets)
        self.carry0 = nn.Parameter(torch.zeros(carry_dim, dtype=RDTYPE))

    # -- state initialization: maximally mixed = vacuous prior --------------------
    def init_state(self, n: int, device=None) -> LTNStep:
        rho = torch.eye(self.d, dtype=CDTYPE, device=device).expand(n, -1, -1).clone() / self.d
        carry = self.carry0.expand(n, -1).clone()
        z = torch.zeros(n, dtype=RDTYPE, device=device)
        return LTNStep(rho=rho, carry=carry, entropy=z.clone(), vacuity=z.clone() + 1.0, dissonance=z.clone())

    def _gamma_features(self, x: torch.Tensor, cal_age_h: torch.Tensor, dt_steps: torch.Tensor) -> torch.Tensor:
        """cal_age_h is RAW hours (log1p applied here — GammaNet's cal_age_idx=2 slot)."""
        n = x.shape[0]
        zeros = torch.zeros(n, dtype=RDTYPE, device=x.device)
        t1z = x[:, 0].to(RDTYPE) if self.n_obs > 0 else zeros
        t2z = x[:, 1].to(RDTYPE) if self.n_obs > 1 else zeros
        age = torch.log1p(cal_age_h.to(RDTYPE).clamp_min(0))
        if not self.physics_gamma:  # ablation: no physics inputs, only load + platform
            t1z, t2z, age = zeros, zeros, zeros
        load = torch.log1p(dt_steps.to(RDTYPE).clamp_min(0))
        plat = x[:, -self.platform_dim:].to(RDTYPE) if self.platform_dim else x[:, :0].to(RDTYPE)
        core = torch.stack([t1z, t2z, age, load], dim=-1)
        return torch.cat([core.nan_to_num(0.0), plat], dim=-1)

    def drift(self, state: LTNStep, x_prev: torch.Tensor, cal_age: torch.Tensor, dt_steps: torch.Tensor) -> torch.Tensor:
        """Evolve rho over dt_steps grid units; returns new rho (no evidence)."""
        if not self.use_lindblad:
            return state.rho  # ablation handled by NodeODE variant instead
        gfeat = self._gamma_features(x_prev, cal_age, dt_steps)
        gammas = self.gamma_net(gfeat)
        H = self._H(state.carry, state.rho)
        return evolve(state.rho, H, list(self.jumps()), gammas, dt_steps.to(RDTYPE))

    def _H(self, carry: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Hamiltonian entering the generator; zero under the use_hamiltonian
        ablation, which makes the generator normal without touching anything else."""
        if not self.use_hamiltonian:
            return torch.zeros_like(rho)
        return self.hermitian(carry)

    def _post_rho(self, rho: torch.Tensor) -> torch.Tensor:
        """Hook applied to rho at the end of every step, BEFORE the carry update
        and the entropy/vacuity/dissonance summaries are computed from it.
        Identity here; LTNv2Model overrides it with the depolarizing purity floor
        so that every consumer of the state sees the same floored rho."""
        return rho

    def step(
        self,
        state: LTNStep,
        x: torch.Tensor,
        cal_age: torch.Tensor,
        dt_steps: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> LTNStep:
        rho = self.drift(state, x, cal_age, dt_steps)
        rho = self.evidence(rho, x.to(RDTYPE))
        if self.use_graph and edge_index.numel():
            rho = self.prop(rho, x.to(RDTYPE), edge_index, edge_attr.to(RDTYPE))
        rho = self._post_rho(rho)
        carry = self.gru(torch.cat([x.to(RDTYPE), rho_to_real(rho)], dim=-1), state.carry)
        return LTNStep(
            rho=rho, carry=carry,
            entropy=von_neumann_entropy(rho), vacuity=vacuity(rho), dissonance=dissonance(rho),
        )

    def forecast(
        self, state: LTNStep, x_now: torch.Tensor, cal_age_now: torch.Tensor, horizon_steps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evidence-free roll-forward + readout. horizon_steps: [n] grid units.

        Returns (mu_z [n, n_obs], sigma_z [n, n_obs], entropy_at_horizon [n])."""
        if self.use_lindblad:
            gfeat = self._gamma_features(x_now, cal_age_now + horizon_steps * self.time_unit_h, horizon_steps)
            gammas = self.gamma_net(gfeat)
            H = self._H(state.carry, state.rho)
            rho_h = evolve(state.rho, H, list(self.jumps()), gammas, horizon_steps.to(RDTYPE))
        else:
            rho_h = state.rho
        mu, sigma = self.forecast_head(rho_h, state.carry, horizon_steps)
        return mu, sigma, von_neumann_entropy(rho_h)

    def divergence(self, state: LTNStep) -> torch.Tensor:
        return self.diverge_head(self.povm(state.rho), state.vacuity, state.dissonance)
