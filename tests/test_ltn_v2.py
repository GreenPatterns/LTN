"""Tier-1 falsifiers for LTN v2 (cheap CPU tests that kill the design if wrong).

Each test encodes one of the v2 mechanism claims:
  * sigma is Born-only: no carry path, and it is bounded strictly away from the
    floor on any state the model can actually reach (purity floor + traceless
    unit-norm observables);
  * d sigma / d gamma != 0 AND sigma GROWS with the forecast horizon -- the E7
    repair, which must survive trained-scale gammas, not just initialization;
  * the drift is unital, so entropy is monotone nondecreasing for ANY gammas
    (v1's non-unital lowering ladder inverts this once gamma_0 is learned);
  * the saturating tau keeps matrix_exp gradients bounded as the gap grows;
  * the whole thing trains: BPTT chunks, NLL backward, clip, Adam, detach, with
    every v2 parameter receiving a finite nonzero gradient (no dead directions).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from ltn.models.dmstate import CDTYPE, RDTYPE, purity, vacuity, von_neumann_entropy
from ltn.models.lindblad import JumpDictionary, evolve
from ltn.models.ltn_v2 import BornReadout, LTNv2Model, UnitalJumpDictionary

torch.manual_seed(0)

N_OBS, PLAT = 3, 2
FEAT = 2 * N_OBS + 2 + PLAT


def _model(**kw) -> LTNv2Model:
    return LTNv2Model(n_obs=N_OBS, platform_dim=PLAT, carry_dim=8, hidden=16, **kw)


def _no_edges():
    return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2, dtype=RDTYPE)


def _stepped_state(model, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    state = model.init_state(n)
    x = torch.randn(n, FEAT, generator=g, dtype=RDTYPE)
    cal = torch.rand(n, generator=g, dtype=RDTYPE) * 10
    dt = torch.ones(n, dtype=RDTYPE)
    ei, ea = _no_edges()
    return model.step(state, x, cal, dt, ei, ea), x, cal


def _basis_pure(d: int, n: int = 1) -> torch.Tensor:
    rho = torch.zeros(n, d, d, dtype=CDTYPE)
    rho[:, 0, 0] = 1.0
    return rho


def _head_floor(head: BornReadout) -> float:
    """Smallest sigma the head can emit: softplus(s) sqrt(var_eps) + sigma_floor."""
    return float(torch.nn.functional.softplus(head.s.detach()).max()) * math.sqrt(head.var_eps) \
        + head.sigma_floor


# ---------------------------------------------------------------- readout ----


def test_default_d_is_obs_plus_one():
    assert _model().d == N_OBS + 1


def test_rejects_d_below_n_obs():
    with pytest.raises(ValueError):
        LTNv2Model(n_obs=4, platform_dim=PLAT, d=3, carry_dim=8, hidden=16)


@pytest.mark.parametrize("d", [N_OBS, N_OBS + 1, 8])
def test_d_sweep_contract(d):
    """The prereg falsifier sweeps d; every value it sweeps must build and run."""
    m = _model(d=d)
    assert m.d == d
    state, x, cal = _stepped_state(m, n=3, seed=d)
    mu, sigma, _ = m.forecast(state, x, cal, torch.full((3,), 3.0, dtype=RDTYPE))
    assert mu.shape == (3, N_OBS) and torch.isfinite(sigma).all() and (sigma > 0).all()


def test_observables_are_hermitian_traceless_unit_norm():
    """Traceless + unit-norm is the property that makes the purity floor bind:
    it forbids O_c from drifting to a multiple of I, where every Born variance
    would vanish and sigma would collapse to the floor for every state."""
    head = BornReadout(5, carry_dim=8, n_obs=N_OBS)
    with torch.no_grad():  # arbitrary parameters, not just the init
        head.Q.add_(2.0 * torch.randn_like(head.Q))
    O = head.observables()
    assert torch.allclose(O, O.conj().transpose(-2, -1), atol=1e-12)
    tr = torch.diagonal(O, dim1=-2, dim2=-1).sum(-1)
    assert tr.abs().max() < 1e-12
    nrm = O.abs().pow(2).sum((-2, -1)).sqrt()
    assert torch.allclose(nrm, torch.ones_like(nrm), atol=1e-12)


def test_identity_direction_is_degenerate_but_safe():
    """Q exactly on the identity direction is the one degenerate point of the
    parameterization: the traceless projection sends O_c to 0, so sigma sits at
    the floor. The construction does NOT make this unreachable -- it makes it
    unattractive: with the trace gauge removed, shrinking O_c also flattens mu to
    a constant, so the NLL punishes it, unlike v1 where sigma could ignore the
    state for free. What is guaranteed here is only that it stays numerically
    safe (no NaN/inf) if training ever passes near it."""
    head = BornReadout(4, carry_dim=8, n_obs=N_OBS)
    with torch.no_grad():
        head.Q.copy_(torch.eye(4, dtype=RDTYPE).expand(N_OBS, 4, 4) * 1e3)
    O = head.observables()
    assert torch.isfinite(O).all() and O.abs().max() < 1e-6
    rho = torch.eye(4, dtype=CDTYPE).expand(2, 4, 4) / 4
    mu, sigma = head(rho, torch.zeros(2, 8, dtype=RDTYPE))
    assert torch.isfinite(mu).all() and torch.isfinite(sigma).all()
    (grad,) = torch.autograd.grad(sigma.sum() + mu.sum(), [head.Q], allow_unused=True)
    assert grad is not None and torch.isfinite(grad).all(), "degenerate point must not NaN"


def test_born_sigma_ignores_carry_and_mu_is_born_affine_at_init():
    d = N_OBS + 1
    head = BornReadout(d, carry_dim=8, n_obs=N_OBS, init_noise=0.0)
    rho = _basis_pure(d, n=2)
    rho = 0.7 * rho + 0.3 * torch.eye(d, dtype=CDTYPE) / d
    c1, c2 = torch.zeros(2, 8, dtype=RDTYPE), torch.randn(2, 8, dtype=RDTYPE)
    mu1, s1 = head(rho, c1)
    mu2, s2 = head(rho, c2)
    assert torch.equal(s1, s2), "sigma must not depend on the carry"
    O = head.observables()
    m1 = torch.einsum("cij,bji->bc", O, rho).real
    expect = head.scale * m1 + head.bias
    assert torch.allclose(mu1, expect, atol=1e-12)
    assert torch.allclose(mu1, mu2, atol=1e-12)


def test_sigma_at_floor_on_pure_eigenstate_and_grows_under_mixing():
    d = N_OBS + 1
    head = BornReadout(d, carry_dim=8, n_obs=N_OBS, init_noise=0.0)
    carry = torch.zeros(1, 8, dtype=RDTYPE)
    eye = torch.eye(d, dtype=CDTYPE) / d
    pure = _basis_pure(d)  # eigenstate of every O_c at init
    _, s_pure = head(pure, carry)
    assert torch.all(s_pure <= _head_floor(head) + 1e-9)
    _, s_mixed = head(0.6 * pure + 0.4 * eye, carry)
    assert torch.all(s_mixed > s_pure + 1e-4), "mixing must raise Born sigma"


def test_born_variance_nonnegative_without_clamp():
    """Checks the RAW variance, not the clamped one: comparing the clamped value
    against 0 would pass for any input whatsoever, including invalid states."""
    d = N_OBS + 1
    head = BornReadout(d, carry_dim=8, n_obs=N_OBS)
    A = torch.randn(16, d, d, dtype=RDTYPE) + 1j * torch.randn(16, d, d, dtype=RDTYPE)
    rho = A @ A.conj().transpose(-2, -1)
    rho = (rho / torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real.view(-1, 1, 1)).to(CDTYPE)
    O = head.observables()
    m1 = torch.einsum("cij,bji->bc", O, rho).real
    m2 = torch.einsum("cij,bji->bc", O @ O, rho).real
    raw = m2 - m1 * m1  # no clamp: this is the quantity the design claims is >= 0
    assert raw.min() >= -1e-10, f"raw Born variance went negative: {raw.min()}"


def test_purity_floor_bounds_sigma_away_from_floor():
    """The design's sigma lower bound: on any floored state (rho >= eps/d I) a
    traceless unit-norm O_c has strictly positive variance, so sigma cannot sit
    at the floor -- this is what stops v2 from silently regressing to v1's
    state-independent uncertainty."""
    m = _model(purity_eps=1e-3)
    head = m.forecast_head
    worst = m._post_rho(_basis_pure(m.d, n=1))  # floored pure state = worst case
    _, sigma = head(worst, torch.zeros(1, 8, dtype=RDTYPE))
    assert torch.all(sigma > 10 * _head_floor(head)), \
        f"sigma {sigma.tolist()} not bounded away from floor {_head_floor(head):.2e}"


# --------------------------------------------------- gamma gradient (E7 fix) --


def test_sigma_receives_gamma_gradient():
    model = _model()
    state, x, cal = _stepped_state(model)
    hor = torch.full((4,), 3.0, dtype=RDTYPE)
    _, sigma, _ = model.forecast(state, x, cal, hor)
    grads = torch.autograd.grad(
        sigma.sum(), list(model.gamma_net.parameters()), allow_unused=True)
    total = sum(g.abs().sum().item() for g in grads if g is not None)
    assert total > 1e-10, "d sigma / d gamma must be nonzero (E7 repair)"


def test_mu_receives_gamma_gradient_too():
    model = _model()
    state, x, cal = _stepped_state(model)
    mu, _, _ = model.forecast(state, x, cal, torch.full((4,), 3.0, dtype=RDTYPE))
    grads = torch.autograd.grad(mu.sum(), list(model.gamma_net.parameters()), allow_unused=True)
    assert sum(g.abs().sum().item() for g in grads if g is not None) > 1e-12


@pytest.mark.parametrize("gamma", [0.1, 0.3, 2.0])
def test_sigma_grows_with_horizon_from_a_confident_state(gamma):
    """The headline v2 mechanism, stated exactly as it actually holds.

    Born variance is NOT monotone under mixing: it relaxes toward its vacuous
    value tr(O_c^2)/d = 1/d, so a state that is already MORE spread than vacuous
    along O_c gets sharper as it mixes. The claim is therefore scoped to the
    regime that matters -- a state more confident than vacuous, i.e. the state
    right after evidence arrives -- where drifting toward ignorance raises sigma.

    Even there, monotonicity is NOT asserted: with a 3-jump unital channel the
    approach to the fixed point is not a pure depolarizing interpolation, so
    sigma can overshoot the vacuous level slightly and settle back (observed:
    0.514 -> 0.504 at gamma=2). The claims that hold are growth from confident to
    aged, and convergence to the vacuous value."""
    model = _model()
    with torch.no_grad():
        model.gamma_net.bias.fill_(math.log(math.expm1(gamma)))
    rho = model._post_rho(_basis_pure(model.d, n=1))  # maximally confident
    carry = torch.zeros(1, 8, dtype=RDTYPE)
    g = torch.full((1, 3), gamma, dtype=RDTYPE)
    sig = []
    for h in (0.01, 1.0, 3.0, 10.0, 30.0):
        rh = model._post_rho(model._dissipative_evolve(rho, g, torch.tensor([h], dtype=RDTYPE)))
        _, s = model.forecast_head(rh, carry)
        sig.append(float(s.mean().detach()))
    assert sig[-1] > 3 * sig[0], f"sigma must grow materially as evidence ages: {sig}"
    # The fixed point is vacuous (var -> tr(O^2)/d = 1/d for unit-norm O), but it
    # is only REACHED when the channel saturates: total mixing is gamma * tau_max,
    # so a slow gamma stops short of ignorance rather than converging to it.
    sig_vac = float((F.softplus(model.forecast_head.s.detach()) / math.sqrt(model.d)).mean())
    tau_max = float(F.softplus(model.log_tau_max.detach()))
    assert sig[-1] <= sig_vac * 1.15, f"sigma must not overshoot ignorance: {sig[-1]} vs {sig_vac}"
    if gamma * tau_max >= 5.0:
        assert abs(sig[-1] - sig_vac) < 0.25 * sig_vac, \
            f"a saturated channel must relax to the vacuous value {sig_vac:.3f}, got {sig[-1]:.3f}"


# ------------------------------------------------------ unitality (entropy) --


def test_jump_dictionary_is_unital_for_arbitrary_parameters():
    """D(I) = 0 must hold for ANY parameter values, not just at init -- that is
    what makes entropy monotonicity a structural guarantee under training."""
    from ltn.models.dmstate import unvec, vec
    from ltn.models.lindblad import lindblad_generator
    for d in (2, 4, 6):
        jd = UnitalJumpDictionary(d)
        with torch.no_grad():
            jd.A_real.add_(0.5 * torch.randn_like(jd.A_real))
            jd.A_imag.add_(0.5 * torch.randn_like(jd.A_imag))
        L = jd()
        assert torch.allclose(L, L.conj().transpose(-2, -1), atol=1e-12), "jumps must be Hermitian"
        I = torch.eye(d, dtype=CDTYPE).unsqueeze(0)
        gen = lindblad_generator(torch.zeros(1, d, d, dtype=CDTYPE), list(L),
                                 torch.rand(1, 3, dtype=RDTYPE) * 5)
        DI = unvec((gen @ vec(I).unsqueeze(-1)).squeeze(-1))
        assert DI.abs().max() < 1e-10, f"dissipator not unital at d={d}: {DI.abs().max()}"


def test_d2_dictionary_is_the_pauli_set():
    """At d=2 the init IS {Z, X, Y}/sqrt(2), i.e. the depolarizing generator."""
    L = UnitalJumpDictionary(2)().detach()
    sq = math.sqrt(2)
    paulis = {
        "Z": torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE) / sq,
        "X": torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE) / sq,
        "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE) / sq,
    }
    for k, name in enumerate(("Z", "X", "Y")):
        p = paulis[name]
        err = min((L[k] - p).abs().max().item(), (L[k] + p).abs().max().item())
        assert err < 1e-12, f"jump {k} is not +/-{name}/sqrt2 (err {err})"


@pytest.mark.parametrize("gamma0", [0.05, 3.0, 20.0])
def test_entropy_nondecreasing_at_trained_scale_gammas(gamma0):
    """v1's dictionary INVERTS here: its non-unital lowering ladder drives the
    state toward pure |0><0|, so at gamma_0 = 3 entropy (and sigma) DECREASE with
    horizon. v2's unital dictionary must not, at any rate."""
    model = _model()
    rho = model._post_rho(_basis_pure(model.d, n=1))
    g = torch.tensor([[gamma0, 0.05, 0.05]], dtype=RDTYPE)
    ents = []
    for h in (0.01, 0.3, 1.0, 3.0, 10.0, 30.0):
        rh = model._post_rho(model._dissipative_evolve(rho, g, torch.tensor([h], dtype=RDTYPE)))
        ents.append(von_neumann_entropy(rh).item())
    for a, b in zip(ents, ents[1:]):
        assert b >= a - 1e-9, f"entropy decreased with horizon at gamma0={gamma0}: {ents}"


def test_v1_dictionary_would_invert_entropy():
    """Guards the REASON v2 swapped dictionaries: if this ever stops holding, the
    unital constraint is no longer buying anything and can be revisited."""
    d = 4
    jumps = list(JumpDictionary(d)())
    rho = torch.eye(d, dtype=CDTYPE).unsqueeze(0) / d  # maximally mixed: max entropy
    g = torch.tensor([[3.0, 0.05, 0.05]], dtype=RDTYPE)  # non-unital k=0 dominant
    H0 = torch.zeros_like(rho)
    e_short = von_neumann_entropy(evolve(rho, H0, jumps, g, torch.tensor([0.01], dtype=RDTYPE)))
    e_long = von_neumann_entropy(evolve(rho, H0, jumps, g, torch.tensor([8.0], dtype=RDTYPE)))
    assert e_long.item() < e_short.item() - 0.1, \
        "v1's dictionary was expected to lose entropy under a learned lowering rate"


# ------------------------------------------- contraction / bounded gradients --


def _gamma_grad_norm(rho, jumps, H, gammas, dt_scalar, tau=None):
    g = gammas.clone().requires_grad_(True)
    dt = torch.full((rho.shape[0],), float(dt_scalar), dtype=RDTYPE)
    dt_eff = tau(dt) if tau is not None else dt
    out = evolve(rho, H, jumps, g.expand(rho.shape[0], -1), dt_eff)
    (grad,) = torch.autograd.grad(out.real.pow(2).sum() + out.imag.pow(2).sum(), g)
    return grad.norm().item()


def test_saturating_tau_bounds_gradients():
    """Beyond tau_max the effective time saturates, so the matrix_exp argument --
    and therefore the gamma-gradient -- is dt-independent: the channel family
    {exp(L tau(dt))} is uniformly bounded over all gaps."""
    torch.manual_seed(1)
    d = 4
    jumps = list(UnitalJumpDictionary(d)())
    rho = _basis_pure(d, n=3)
    rho = 0.8 * rho + 0.2 * torch.eye(d, dtype=CDTYPE) / d
    gammas = torch.tensor([0.05, 0.3, 0.3], dtype=RDTYPE)
    model = _model()  # provides tau with tau_max ~ 8

    def tau(dt):
        return model._tau(dt).detach()

    H0 = torch.zeros_like(rho)
    v2 = [_gamma_grad_norm(rho, jumps, H0, gammas, s, tau=tau) for s in (10, 100, 1000)]
    assert v2[2] < 2.0 * v2[0] + 1e-9, f"v2 gamma-grad must saturate, got {v2}"
    assert abs(v2[2] - v2[1]) < 1e-4 * max(v2[1], 1e-12), \
        f"past saturation the grad must be dt-independent, got {v2}"


def test_tau_properties():
    model = _model()
    dt = torch.tensor([0.0, 0.5, 1.0, 5.0, 50.0, 5000.0], dtype=RDTYPE)
    t = model._tau(dt)
    tau_max = torch.nn.functional.softplus(model.log_tau_max).item() + 1e-3
    assert t[0].item() == 0.0
    assert torch.all(t[1:] > t[:-1] - 1e-12), "tau must be monotone"
    assert torch.all(t <= tau_max + 1e-9), "tau must be bounded by tau_max"
    assert abs(t[1].item() - 0.5) < 0.05, "tau ~ dt for small gaps"


def test_no_hamiltonian_parameters():
    model = _model()
    assert model.hermitian is None
    assert not any("hermitian" in n for n, _ in model.named_parameters())


# ------------------------------------------------------------- purity floor --


def test_purity_floor_after_step_and_at_forecast():
    model = _model(purity_eps=1e-3)
    state, x, cal = _stepped_state(model, n=6, seed=3)
    lo = 0.99 * model.purity_eps / model.d
    assert torch.all(torch.linalg.eigvalsh(state.rho).min(dim=-1).values >= lo)
    assert torch.all(vacuity(state.rho) > 0)
    tr = torch.diagonal(state.rho, dim1=-2, dim2=-1).sum(-1).real
    assert torch.allclose(tr, torch.ones_like(tr), atol=1e-8)
    # the horizon roll leaves step(), so the floor must be re-applied there
    big = torch.full((6,), 50.0, dtype=RDTYPE)
    for g in (0.05, 20.0):
        with torch.no_grad():
            model.gamma_net.bias.fill_(math.log(math.expm1(g)))
        _, sigma, _ = model.forecast(state, x, cal, big)
        assert torch.all(sigma > _head_floor(model.forecast_head)), \
            f"forecast sigma collapsed to the floor at gamma={g}"


# ------------------------------------------------------ trains end-to-end -----


def test_bptt_training_step_all_params_get_finite_gradients():
    """Runs the real trainer pattern (chunked BPTT -> NLL -> clip -> Adam ->
    detach) and asserts no v2 parameter is dead, NaN, or inf. Single-forward
    autograd checks would miss a dead parameter that only the optimizer reveals."""
    from ltn.exp.train import _detach_state, _nll

    torch.manual_seed(0)
    model = _model()
    n = 4
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ei, ea = _no_edges()
    state = model.init_state(n)
    g = torch.Generator().manual_seed(5)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    seen_grad = {}
    for chunk in range(2):
        loss = None
        for t in range(3):
            x = torch.randn(n, FEAT, generator=g, dtype=RDTYPE)
            cal = torch.rand(n, generator=g, dtype=RDTYPE) * 10
            state = model.step(state, x, cal, torch.ones(n, dtype=RDTYPE), ei, ea)
            mu, sigma, _ = model.forecast(state, x, cal, torch.full((n,), 3.0, dtype=RDTYPE))
            y = torch.randn(n, N_OBS, generator=g, dtype=RDTYPE)
            l = _nll(y, mu, sigma).mean()
            loss = l if loss is None else loss + l
        loss.backward()
        assert torch.isfinite(loss), "loss must stay finite"
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
                seen_grad[name] = seen_grad.get(name, 0.0) + p.grad.abs().sum().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        state = _detach_state(state)

    v2_params = ["log_tau_max", "forecast_head.Q", "forecast_head.scale",
                 "forecast_head.s", "forecast_head.bias",
                 "jumps.A_real", "jumps.A_imag", "gamma_net.bias"]
    for name in v2_params:
        assert seen_grad.get(name, 0.0) > 0, f"{name} received NO gradient (dead parameter)"
    after = model.state_dict()
    for name in v2_params:
        assert not torch.equal(before[name], after[name]), f"{name} did not move under Adam"


def test_observable_parameterization_has_no_dead_directions():
    """O = sym(Q) + i antisym(Q) uses every entry of Q. The alternative form
    O = A + A^dag with complex A leaves half the entries with zero gradient."""
    head = BornReadout(4, carry_dim=8, n_obs=N_OBS)
    rho = torch.randn(2, 4, 4, dtype=RDTYPE)
    rho = (rho @ rho.T if rho.dim() == 2 else rho @ rho.transpose(-2, -1)).to(CDTYPE)
    rho = rho / torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
    mu, sigma = head(rho, torch.randn(2, 8, dtype=RDTYPE))
    (grad,) = torch.autograd.grad((mu.sum() + sigma.sum()), [head.Q])
    assert (grad.abs() > 1e-12).all(), \
        f"{(grad.abs() <= 1e-12).sum().item()} dead entries in Q"


def test_end_to_end_shapes_and_finiteness():
    model = _model()
    n = 5
    state = model.init_state(n)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    ea = torch.randn(4, 2, dtype=RDTYPE)
    g = torch.Generator().manual_seed(7)
    for t in range(3):
        x = torch.randn(n, FEAT, generator=g, dtype=RDTYPE)
        cal = torch.rand(n, generator=g, dtype=RDTYPE) * 20
        state = model.step(state, x, cal, torch.ones(n, dtype=RDTYPE), ei, ea)
    for h in (0.0, 1.0, 9.0):
        mu, sigma, ent = model.forecast(state, x, cal, torch.full((n,), h, dtype=RDTYPE))
        assert mu.shape == (n, N_OBS) and sigma.shape == (n, N_OBS)
        assert torch.isfinite(mu).all() and torch.isfinite(sigma).all()
        assert (sigma > 0).all() and torch.isfinite(ent).all()


# ------------------------------------------------------- use_carry (A4) ----
# The carry ablation. v2 is the only variant where this is a clean experiment:
# v1 routes the carry into the Hamiltonian (ltn.py:122-124), so a carry-free v1
# arm would also zero H_v and confound with the -Hamiltonian ablation. v2 sets
# self.hermitian = None, so BornReadout.resid is the SOLE carry -> output path.

# SHA256 over the sorted state_dict of LTNv2Model(n_obs=4, platform_dim=7,
# carry_dim=31, hidden=126) under torch.manual_seed(0), computed from the
# PRE-FLAG revision of ltn/models/ltn_v2.py (git e451764). Pins the claim that
# use_carry defaults to bit-for-bit the old model: same construction order, so
# the RNG stream and every initialized value are unchanged.
_PREFLAG_SHA256 = "dec0a895f7307e50a06b91b21d76b2e69f3117b9ef893c47933165577b8630e5"
_PREFLAG_NPARAMS = 67852


def _state_dict_sha256(model) -> str:
    import hashlib
    sd = model.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("kw", [{}, {"use_carry": True}])
def test_use_carry_default_is_bit_for_bit_the_preflag_model(kw):
    torch.manual_seed(0)
    m = LTNv2Model(n_obs=4, platform_dim=7, carry_dim=31, hidden=126, **kw)
    assert sum(p.numel() for p in m.parameters()) == _PREFLAG_NPARAMS
    assert _state_dict_sha256(m) == _PREFLAG_SHA256


def test_use_carry_false_removes_every_carry_parameter():
    m = _model(use_carry=False)
    assert m.forecast_head.resid is None
    assert not [n for n, _ in m.named_parameters() if "forecast_head.resid" in n]
    # and the full model still has them
    assert [n for n, _ in _model().named_parameters() if "forecast_head.resid" in n]


def test_carry_free_mu_is_a_pure_functional_of_rho():
    """With use_carry=False, mu must be invariant to the carry and sensitive to
    rho. This is the property the trained Tier-1 arm rests on."""
    m = _model(use_carry=False)
    head = m.forecast_head
    g = torch.Generator().manual_seed(3)
    n, d = 5, m.d
    a = torch.randn(n, d, d, generator=g, dtype=RDTYPE)
    rho = torch.einsum("bij,bkj->bik", a, a).to(CDTYPE)
    rho = rho / torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
    carry = torch.randn(n, 8, generator=g, dtype=RDTYPE)
    mu0, sig0 = head(rho, carry)
    mu1, sig1 = head(rho, torch.randn(n, 8, generator=g, dtype=RDTYPE))
    assert torch.equal(mu0, mu1) and torch.equal(sig0, sig1)  # carry is inert
    mu2, _ = head(rho[torch.tensor([1, 2, 3, 4, 0])], carry)
    assert not torch.allclose(mu0, mu2)                        # rho is not


def test_full_model_mu_moves_with_carry_but_sigma_does_not():
    """The converse, on the DEFAULT model: sigma is Born-only (already covered
    elsewhere) but mu has an open carry path -- which is exactly the bypass the
    A4 ablation quantifies. Trained resid weights are nonzero, so perturb them
    here rather than relying on the zero init."""
    m = _model()
    head = m.forecast_head
    with torch.no_grad():
        head.resid[-1].weight.normal_(0.0, 0.5)
    g = torch.Generator().manual_seed(4)
    n, d = 5, m.d
    a = torch.randn(n, d, d, generator=g, dtype=RDTYPE)
    rho = torch.einsum("bij,bkj->bik", a, a).to(CDTYPE)
    rho = rho / torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
    c0 = torch.randn(n, 8, generator=g, dtype=RDTYPE)
    mu0, sig0 = head(rho, c0)
    mu1, sig1 = head(rho, torch.randn(n, 8, generator=g, dtype=RDTYPE))
    assert not torch.allclose(mu0, mu1)
    assert torch.equal(sig0, sig1)


def test_carry_free_arm_is_parameter_matched_within_10pct():
    """ltn.tex commits ablation variants to +/-10% parameter matching; the
    Tier-1 no-carry arm routes through match_width on `hidden` to honour it."""
    from ltn.models.param_match import count_trainable, match_width

    def build(h, use_carry):
        return LTNv2Model(n_obs=4, platform_dim=7, carry_dim=31, hidden=h,
                          use_carry=use_carry)

    target = count_trainable(build(126, True))
    _, w, n = match_width(target, lambda h: build(h, False), lo=8, hi=4096)
    assert abs(n - target) <= 0.10 * target
    assert w > 126  # widening compensates for the removed residual MLP


def test_carry_free_model_trains_and_the_gru_goes_provably_dead():
    """Structural fact the A4 ablation turns on, asserted rather than assumed:
    in v2 the GRU carry is WRITTEN from (x, rho) every step (ltn.py:147) but is
    READ by nothing except BornReadout.resid, because LTNv2Model sets
    self.hermitian = None. So use_carry=False leaves the GRU with exactly zero
    gradient -- the forecasting model becomes rho-only end to end -- while every
    parameter on the state path still trains."""
    m = _model(use_carry=False)
    state, x, cal = _stepped_state(m)
    mu, sigma, _ = m.forecast(state, x, cal, torch.full((4,), 3.0, dtype=RDTYPE))
    loss = (((torch.zeros_like(mu) - mu) / sigma) ** 2 + 2 * sigma.log()).mean()
    loss.backward()
    live = {n: p for n, p in m.named_parameters()
            if n.startswith(("jumps.", "evidence.", "forecast_head.")) or n == "log_tau_max"}
    for n, p in live.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
        assert float(p.grad.abs().sum()) > 0, n
    for n, p in m.named_parameters():
        if n.startswith("gru.") or n == "carry0":
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, n
    # ... and on the DEFAULT model the same GRU parameters DO receive gradient
    m2 = _model()
    with torch.no_grad():
        m2.forecast_head.resid[-1].weight.normal_(0.0, 0.5)
    st2, x2, cal2 = _stepped_state(m2)
    mu2, s2, _ = m2.forecast(st2, x2, cal2, torch.full((4,), 3.0, dtype=RDTYPE))
    ((mu2 ** 2 / s2 ** 2) + 2 * s2.log()).mean().backward()
    assert float(m2.gru.weight_ih.grad.abs().sum()) > 0
