"""LTN assembly: end-to-end step/forecast validity, gradient flow, and the
epistemic-decoherence signature (entropy grows with evidence-free horizon)."""
import torch

from ltn.models.ltn import LTNModel


def make_model(**kw):
    torch.manual_seed(0)
    defaults = dict(n_obs=4, platform_dim=3, d=4, carry_dim=16, hidden=32)
    defaults.update(kw)
    return LTNModel(**defaults)


def snapshot_inputs(n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    F = 2 * 4 + 2 + 3  # z-vals + mask + cal_age + dt + platform
    x = torch.randn(n, F, generator=g, dtype=torch.float64)
    x[:, 4:8] = (x[:, 4:8] > 0).double()  # mask block in {0,1}
    cal_age = torch.rand(n, generator=g, dtype=torch.float64) * 10
    dt = torch.ones(n, dtype=torch.float64)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]], dtype=torch.long)
    edge_attr = torch.randn(len(edge_index[0]), 2, generator=g, dtype=torch.float64)
    return x, cal_age, dt, edge_index, edge_attr


def valid_state(rho, tol=1e-8):
    tr = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real
    ev = torch.linalg.eigvalsh((rho + rho.conj().transpose(-2, -1)) / 2)
    return (tr - 1).abs().max().item() < tol and ev.min().item() > -1e-7


def test_two_steps_produce_valid_states():
    m = make_model()
    x, cal_age, dt, ei, ea = snapshot_inputs()
    st = m.init_state(5)
    assert valid_state(st.rho) and st.vacuity.max().item() <= 1.0 + 1e-9
    st = m.step(st, x, cal_age, dt, ei, ea)
    assert valid_state(st.rho)
    st = m.step(st, x, cal_age, dt, ei, ea)
    assert valid_state(st.rho)
    assert st.entropy.shape == st.vacuity.shape == (5,)


def test_forecast_shapes_and_uncertainty_grows_with_horizon():
    m = make_model()
    x, cal_age, dt, ei, ea = snapshot_inputs()
    st = m.init_state(5)
    for _ in range(3):
        st = m.step(st, x, cal_age, dt, ei, ea)
    import math

    mu1, s1, ent1 = m.forecast(st, x, cal_age, torch.full((5,), 1.0, dtype=torch.float64))
    mu9, s9, ent9 = m.forecast(st, x, cal_age, torch.full((5,), 9.0, dtype=torch.float64))
    assert mu1.shape == s1.shape == (5, 4) and (s1 > 0).all()
    # Evidence-free drift approaches maximal mixedness at init (unital dissipators
    # dominate; GammaNet bias_init suppresses the non-unital lowering channel).
    # Pointwise monotonicity is NOT a physical guarantee with H != 0 and a small
    # non-unital component — the strict monotone case (pure dephasing) is covered
    # in test_lindblad; here we pin the vacuous fixed point and mean growth.
    assert (ent9 - ent1).min().item() > -0.01
    assert ent9.mean().item() >= ent1.mean().item() - 1e-9
    assert abs(ent9.mean().item() - math.log(4)) < 0.05  # near-maximal mixedness


def test_gradients_reach_all_submodules():
    m = make_model()
    x, cal_age, dt, ei, ea = snapshot_inputs()
    st = m.init_state(5)
    st = m.step(st, x, cal_age, dt, ei, ea)
    mu, sigma, _ = m.forecast(st, x, cal_age, torch.full((5,), 3.0, dtype=torch.float64))
    y = torch.zeros_like(mu)
    loss = (((y - mu) / sigma) ** 2 + 2 * sigma.log()).mean() + m.divergence(st).mean()
    loss.backward()
    reached, missing = [], []
    for name, p in m.named_parameters():
        (reached if p.grad is not None and p.grad.abs().max() > 0 else missing).append(name)
    # every submodule family must receive gradient somewhere
    for fam in ["gamma_net", "jumps", "hermitian", "evidence", "prop", "gru", "forecast_head", "povm", "diverge_head"]:
        assert any(fam in n for n in reached), f"no gradient into {fam}: missing={missing}"


def test_ablation_flags():
    x, cal_age, dt, ei, ea = snapshot_inputs()
    for kw in [dict(use_graph=False), dict(physics_gamma=False)]:
        m = make_model(**kw)
        st = m.step(m.init_state(5), x, cal_age, dt, ei, ea)
        assert valid_state(st.rho)


def test_no_graph_equals_empty_edges():
    m = make_model(use_graph=False)
    x, cal_age, dt, ei, ea = snapshot_inputs()
    st1 = m.step(m.init_state(5), x, cal_age, dt, ei, ea)
    st2 = m.step(m.init_state(5), x, cal_age, dt, torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 2))
    assert torch.allclose(st1.rho, st2.rho)
