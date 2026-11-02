"""LTN v2, Tier 1: make the density-matrix state load-bearing.

Three repairs over ltn.models.ltn.LTNModel, each targeting a failure the E2/E7
audit measured on v1:

1. Born-rule readout (BornReadout). v1's ForecastHead produced (mu, sigma) from
   an MLP over (rho, carry, log-horizon); the horizon fed sigma directly, so
   predictive uncertainty grew without the Lindblad drift mixing the state and
   the decoherence rates gamma never received gradient (the E7 root cause).
   Here each observable c has a learned Hermitian O_c and
       mu_c    = a_c tr(rho_h O_c) + b_c + resid_c(carry)   (residual mu-only,
                                                             zero-initialized)
       sigma_c = softplus(s_c) sqrt(tr(rho_h O_c^2) - tr(rho_h O_c)^2) + floor,
   the exact Born mean/variance of measuring O_c in rho_h. There is NO carry or
   horizon path into sigma: the only way sigma can move over the horizon is for
   the drift to mix the state, so d sigma / d gamma != 0 by construction
   (unit-tested in tests/test_ltn_v2.py).

   What sigma does over a horizon, stated exactly: Born variance is NOT monotone
   under mixing. It RELAXES toward its vacuous value tr(O_c^2)/d = 1/d (the O_c
   are unit-norm), so as evidence ages sigma converges to the vacuous level from
   whichever side it starts. It therefore GROWS with the horizon exactly when the
   state is more confident than vacuous, which is the regime that matters (the
   state just after evidence arrives) and the one E7 measures. A state that is
   somehow more spread than vacuous along O_c sharpens toward 1/d instead; that
   is the correct semantics, not a bug, since I/d is what total ignorance means.

2. Contractive, UNITAL, dissipative-only propagator. v1 learned a Hamiltonian,
   making the generator non-normal; differentiating matrix_exp of a non-normal
   generator over large dt explodes gradients (v1 full-model seed std 0.026 vs
   <= 0.005 for every ablation). v2 sets H = 0 and passes a saturating effective
   time
       tau(dt) = tau_max (1 - exp(-dt / tau_max)),   tau_max = softplus(theta),
   so the matrix_exp argument is uniformly bounded by tau_max * ||L_gen||.

   v2 also restricts the jump dictionary to HERMITIAN (hence normal, hence
   unital) operators -- see UnitalJumpDictionary. This is not cosmetic. v1's
   dictionary carries a non-unital T1-like lowering ladder whose fixed point is
   the PURE state |0><0|; its rate gamma_0 is small only at initialization and is
   unconstrained afterwards. With gamma_0 learned large, entropy DECREASES along
   the evidence-free horizon and the Born sigma SHRINKS as the forecast reaches
   further out -- the exact inverse of the mechanism this readout exists to
   create, and the E7 entropy signal inverts with it. Unital jumps make the
   guarantee structural rather than an initialization bias: a unital CPTP map is
   doubly stochastic on the spectrum, so Phi(rho) is majorized by rho and
   S(Phi(rho)) >= S(rho) for ANY gammas >= 0 over any gap (Uhlmann majorization --
   NOT Ruskai, whose result is relative-entropy contraction). I/d is then a fixed
   point, so starved of evidence v2 trust decays toward ignorance and can never
   drift toward unwarranted confidence. UNIQUENESS is a separate matter and does
   not follow from Hermiticity: ker(D) is the commutant of {L_k}, so I/d is the
   unique fixed point iff the jumps generate the full matrix algebra. Hermitian
   dephasing jumps are the counterexample (kernel dimension d). It holds
   empirically for every fitted checkpoint -- see scripts/spectra.py.

   SEMANTICS, stated precisely because v1's differ: v2 applies exp(L_gen tau(dt)),
   the exact Lindblad solution over the WARPED time tau(dt), not over the raw gap
   dt. tau is strictly concave, so tau(dt1) + tau(dt2) != tau(dt1 + dt2): the
   channel family is a semigroup in tau, not in physical time, and a device
   observed through one gap of 2 drifts slightly less than one observed at two
   gaps of 1. Past dt >> tau_max the channel saturates, so entropy and sigma are
   flat in horizon beyond ~tau_max (learnable, initialized to 8 grid steps).

3. Observable-aligned state. d defaults to n_obs + 1 (one slack axis); O_c is
   initialized on the |c><c| axis, so diagonal entries carry the per-observable
   predictions and off-diagonals are learned predictive cross-correlations.
   Pre-registered falsifier: with this readout NCRPS must MOVE with d (v1 was
   flat across d in {2,4,8} at 0.309, proving the state was dead weight).

Purity floor: the depolarizing channel rho <- (1 - eps) rho + eps I/d (CPTP) is
applied at the end of every step AND to the horizon-rolled state before readout,
so rho >= (eps/d) I always holds where sigma is computed and no state can be an
eigenstate of a (traceless, unit-norm) O_c.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltn.models.dmstate import (
    CDTYPE,
    RDTYPE,
    dissonance,
    vacuity,
    von_neumann_entropy,
)
from ltn.models.lindblad import evolve
from ltn.models.ltn import LTNModel, LTNStep


class UnitalJumpDictionary(nn.Module):
    """Shared learned jump dictionary constrained to HERMITIAN operators.

    L_k = A_k + A_k^dagger is Hermitian, hence normal (L L^dag = L^2 = L^dag L),
    and a Lindblad dissipator built from a normal jump operator is UNITAL:
        D(I) = L I L^dag - 1/2 {L^dag L, I} = L L^dag - L^dag L = 0,
    so the channel exp(sum_k gamma_k D_k * t) fixes I. Unitality is preserved for
    every gamma >= 0 and every t, and it is what buys the entropy guarantee the
    Born-rule sigma rests on (unital => spectrum majorization => S nondecreasing).
    Crucially the constraint survives LEARNING: any A_k whatsoever yields a
    Hermitian L_k, so no projection, penalty or re-normalization is needed.

    Initialization generalizes the Pauli set -- at d=2 it IS {Z, X, Y}/sqrt(2),
    the depolarizing generator -- each Frobenius-normalized:
      k=0  dephasing        diag(0..d-1) traceless          (T2-like, Z-like)
      k=1  real mixing      (E + E^T)/2                     (X-like)
      k=2  imaginary mixing i(E - E^T)/2                    (Y-like)
    where E = sum_i |i><i+1| is the lowering ladder. E itself is non-normal, but
    the two Hermitian combinations above are not, which is the whole point: v1's
    non-unital lowering channel is deliberately absent (see module docstring).
    The ladder, NOT the cyclic shift, generates k=1/k=2: the cyclic shift is
    symmetric at d=2, which would make the k=2 jump identically zero there.
    """

    def __init__(self, d: int, K: int = 3) -> None:
        super().__init__()
        if K != 3:
            raise ValueError("UnitalJumpDictionary init pattern is specified for K=3")
        if d < 2:
            raise ValueError(f"need d >= 2, got {d}")
        self.d, self.K = d, K
        E = torch.zeros(d, d, dtype=RDTYPE)
        ar = torch.arange(d - 1)
        E[ar, ar + 1] = 1.0  # lowering ladder sum_i |i><i+1|
        deph = torch.arange(d, dtype=RDTYPE)
        re = torch.zeros(K, d, d, dtype=RDTYPE)
        im = torch.zeros(K, d, d, dtype=RDTYPE)
        re[0] = torch.diag(deph - deph.mean())
        re[1] = (E + E.T) / 2
        im[2] = (E - E.T) / 2
        # A_k = (re + i im)/2 so that L_k = A_k + A_k^dag reproduces the target
        for k in range(K):
            n = torch.complex(re[k], im[k]).abs().pow(2).sum().sqrt().clamp_min(1e-12)
            re[k], im[k] = re[k] / n, im[k] / n
        self.A_real = nn.Parameter(re / 2)
        self.A_imag = nn.Parameter(im / 2)

    def ops(self) -> torch.Tensor:
        A = torch.complex(self.A_real, self.A_imag)
        return A + A.conj().transpose(-2, -1)  # Hermitian => normal => unital

    def forward(self) -> torch.Tensor:
        return self.ops()


class BornReadout(nn.Module):
    """(rho, carry) -> per-observable Gaussian (mu, sigma) via Born statistics.

    Each observable is built from ONE real matrix Q_c as
        O_c = sym(Q_c) + i antisym(Q_c)          (Hermitian, exactly d^2 real dof)
    then made TRACELESS and normalized to unit Frobenius norm:
        O_c <- (O_c - tr(O_c)/d I) / ||O_c - tr(O_c)/d I||_F.
    Both steps are load-bearing, not cosmetic:

      * The Q -> sym/antisym construction has no dead directions. Writing
        O = A + A^dag with an unconstrained COMPLEX A (the obvious form) leaves
        antisym(Re A) and sym(Im A) with exactly zero gradient forever, so half
        the parameters are dead weight that Adam still carries.
      * Traceless + unit-norm is what makes the purity floor bind. A shift
        O -> O + lambda I leaves the Born variance unchanged, so trace carries no
        information about sigma (mu absorbs it in `bias`); but an observable that
        DRIFTS toward lambda I has vanishing variance, which would let training
        collapse sigma to the floor for every state and silently restore v1's
        broken "uncertainty ignores the state" behavior. Forcing ||O||_F = 1 with
        tr O = 0 guarantees O has at least two distinct eigenvalues with
        sum(lambda_i^2) = 1, so on any state with rho >= (eps/d) I (the purity
        floor) the Born variance is bounded strictly away from zero.

    sigma sees ONLY the state; the carry residual feeds mu alone and its last
    layer is zero-initialized, so mu = a tr(rho O) + b exactly at init.
    """

    def __init__(self, d: int, carry_dim: int, n_obs: int, hidden: int = 32,
                 sigma_floor: float = 1e-4, init_noise: float = 0.01,
                 var_eps: float = 1e-8, use_carry: bool = True) -> None:
        super().__init__()
        if n_obs > d:
            raise ValueError(f"need d >= n_obs to align observables, got d={d}, n_obs={n_obs}")
        self.d, self.n_obs, self.sigma_floor = d, n_obs, sigma_floor
        # var_eps is INSIDE the sqrt: d/dv sqrt(v + eps) is finite at v = 0,
        # whereas sqrt(clamp_min(v, 0)) has infinite gradient exactly there --
        # which is reachable, since a pure eigenstate of O_c has zero variance.
        self.var_eps = float(var_eps)
        Q = init_noise * torch.randn(n_obs, d, d, dtype=RDTYPE)
        for c in range(n_obs):
            Q[c, c, c] += 1.0  # O_c starts on the |c><c| axis (observable-aligned)
        self.Q = nn.Parameter(Q)
        self.scale = nn.Parameter(torch.ones(n_obs, dtype=RDTYPE))
        self.bias = nn.Parameter(torch.zeros(n_obs, dtype=RDTYPE))
        # softplus(0.5413) ~= 1.0: Born std enters sigma with unit weight at init
        self.s = nn.Parameter(torch.full((n_obs,), 0.5413, dtype=RDTYPE))
        # use_carry=False removes the ONLY carry -> output path in v2 (sigma is a
        # functional of rho alone and LTNv2Model sets self.hermitian = None), so
        # mu = a_c tr(rho_h O_c) + b_c is a pure Born functional of the state.
        # Default True is bit-for-bit identical to the pre-flag model: the module
        # is constructed in the same order, so the RNG stream is unchanged.
        self.use_carry = bool(use_carry)
        if self.use_carry:
            self.resid = nn.Sequential(
                nn.Linear(carry_dim, hidden, dtype=RDTYPE),
                nn.SiLU(),
                nn.Linear(hidden, n_obs, dtype=RDTYPE),
            )
            nn.init.zeros_(self.resid[-1].weight)
            nn.init.zeros_(self.resid[-1].bias)
        else:
            self.resid = None

    def observables(self) -> torch.Tensor:
        """[n_obs, d, d] Hermitian, traceless, unit Frobenius norm."""
        Q = self.Q
        Qt = Q.transpose(-2, -1)
        O = torch.complex((Q + Qt) / 2, (Q - Qt) / 2)  # sym real + i antisym real
        eye = torch.eye(self.d, dtype=CDTYPE, device=Q.device)
        tr = torch.diagonal(O, dim1=-2, dim2=-1).sum(-1)  # real-valued by construction
        O = O - (tr / self.d).view(-1, 1, 1) * eye
        nrm = O.abs().pow(2).sum((-2, -1)).clamp_min(1e-12).sqrt()
        return O / nrm.to(CDTYPE).view(-1, 1, 1)

    def forward(self, rho: torch.Tensor, carry: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        O = self.observables()
        m1 = torch.einsum("cij,bji->bc", O, rho.to(CDTYPE)).real  # tr(rho O_c)
        m2 = torch.einsum("cij,bji->bc", O @ O, rho.to(CDTYPE)).real  # tr(rho O_c^2)
        var = (m2 - m1 * m1).clamp_min(0.0)  # >= 0 exactly for valid rho; clamp is numerics
        mu = self.scale * m1 + self.bias
        if self.use_carry:
            mu = mu + self.resid(carry.to(RDTYPE))
        sigma = F.softplus(self.s) * (var + self.var_eps).sqrt() + self.sigma_floor
        return mu.to(RDTYPE), sigma.to(RDTYPE)


class LTNv2Model(LTNModel):
    """Tier-1 variant: Born readout, unital dissipative-only saturating drift,
    purity floor, observable-aligned d (default n_obs + 1).

    GammaNet, EvidenceChannel, KrausProp, the GRU carry and the POVM/divergence
    heads are inherited unchanged. The HermitianHead is removed entirely and the
    jump dictionary is swapped for the unital one.

    use_carry (default True, bit-for-bit the pre-flag model) gates the readout's
    carry residual. v1 could not support such a flag: there the carry drives the
    Hamiltonian (ltn.py:122-124 self.hermitian(carry)), so a carry-free v1 arm
    would also zero H_v and confound with the -Hamiltonian ablation. v2 sets
    self.hermitian = None, so BornReadout.resid is the sole carry -> output path
    and use_carry=False makes every prediction a pure functional of rho.
    """

    def __init__(self, n_obs: int, platform_dim: int, d: int | None = None,
                 purity_eps: float = 1e-3, tau_max_init: float = 8.0,
                 use_carry: bool = True, **kw) -> None:
        d = int(d) if d else n_obs + 1
        super().__init__(n_obs=n_obs, platform_dim=platform_dim, d=d, **kw)
        self.hermitian = None  # dissipative-only: no learned Hamiltonian
        self.jumps = UnitalJumpDictionary(d, self.jumps.K)
        # v1 biases k=0 low (softplus(-3) ~ 0.05) because k=0 is its non-unital
        # lowering channel; every v2 jump is unital, so all rates start equal.
        with torch.no_grad():
            self.gamma_net.bias.fill_(-1.0)
        self.use_carry = bool(use_carry)
        self.forecast_head = BornReadout(d, self.gru.hidden_size, n_obs,
                                         hidden=max(16, kw.get("hidden", 64) // 2),
                                         use_carry=self.use_carry)
        self.purity_eps = float(purity_eps)
        self.log_tau_max = nn.Parameter(
            torch.tensor(math.log(math.expm1(tau_max_init)), dtype=RDTYPE))

    # -- saturating effective time: linear near 0, bounded by tau_max ----------
    def _tau(self, dt_steps: torch.Tensor) -> torch.Tensor:
        tau_max = F.softplus(self.log_tau_max) + 1e-3
        return tau_max * (1.0 - torch.exp(-dt_steps.to(RDTYPE).clamp_min(0) / tau_max))

    def _post_rho(self, rho: torch.Tensor) -> torch.Tensor:
        """Depolarizing purity floor rho <- (1-eps) rho + eps I/d (CPTP), applied
        inside LTNModel.step so the carry, the entropy/vacuity/dissonance
        summaries and the next drift all see the SAME floored state.

        This is what keeps the Born variance -- hence sigma -- strictly positive:
        the floored state satisfies rho >= (eps/d) I, and BornReadout's
        observables are traceless with unit norm, so no state can be an
        eigenstate of O_c."""
        eps = self.purity_eps
        eye = torch.eye(self.d, dtype=CDTYPE, device=rho.device) / self.d
        return (1.0 - eps) * rho + eps * eye

    def _dissipative_evolve(self, rho: torch.Tensor, gammas: torch.Tensor,
                            dt_steps: torch.Tensor) -> torch.Tensor:
        H0 = torch.zeros_like(rho)
        return evolve(rho, H0, list(self.jumps()), gammas, self._tau(dt_steps))

    def drift(self, state: LTNStep, x_prev: torch.Tensor, cal_age: torch.Tensor,
              dt_steps: torch.Tensor) -> torch.Tensor:
        if not self.use_lindblad:
            return state.rho
        gammas = self.gamma_net(self._gamma_features(x_prev, cal_age, dt_steps))
        return self._dissipative_evolve(state.rho, gammas, dt_steps)

    def forecast(self, state: LTNStep, x_now: torch.Tensor, cal_age_now: torch.Tensor,
                 horizon_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_lindblad:
            gfeat = self._gamma_features(
                x_now, cal_age_now + horizon_steps * self.time_unit_h, horizon_steps)
            gammas = self.gamma_net(gfeat)
            rho_h = self._dissipative_evolve(state.rho, gammas, horizon_steps)
        else:
            rho_h = state.rho
        # the horizon roll leaves the step() path, so it needs the floor applied
        # here too: this is the state sigma and the E7 entropy signal are read off
        rho_h = self._post_rho(rho_h)
        mu, sigma = self.forecast_head(rho_h, state.carry)
        return mu, sigma, von_neumann_entropy(rho_h)
