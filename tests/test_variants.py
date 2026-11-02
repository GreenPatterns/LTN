"""E2 ablation variants: API compatibility, valid states, gradients, capacity parity."""
import torch

from ltn.models.gaussian_state import GaussianModel
from ltn.models.ltn import LTNModel
from ltn.models.node_ode import NodeODEModel
from ltn.models.param_match import count_trainable
from tests.test_ltn_model import snapshot_inputs, valid_state


def _mk(cls, **kw):
    torch.manual_seed(0)
    return cls(n_obs=4, platform_dim=3, d=4, carry_dim=16, hidden=32, **kw)


def test_node_ode_valid_states_and_grad():
    m = _mk(NodeODEModel)
    x, cal, dt, ei, ea = snapshot_inputs()
    st = m.init_state(5)
    st = m.step(st, x, cal, dt, ei, ea)
    assert valid_state(st.rho, tol=1e-6)
    mu, sigma, ent = m.forecast(st, x, cal, torch.full((5,), 3.0, dtype=torch.float64))
    assert mu.shape == (5, 4) and (sigma > 0).all() and torch.isfinite(ent).all()
    loss = mu.sum() + sigma.sum()
    loss.backward()
    assert any("ode_f" in n and p.grad is not None and p.grad.abs().max() > 0
               for n, p in m.named_parameters())


def test_node_ode_near_identity_at_dt_zero():
    m = _mk(NodeODEModel)
    x, cal, dt, ei, ea = snapshot_inputs()
    st = m.init_state(5)
    st = m.step(st, x, cal, dt, ei, ea)
    rho0 = st.rho.detach().clone()
    rho1 = m.drift(st, x, cal, torch.zeros(5, dtype=torch.float64))
    assert (rho1 - rho0).abs().max().item() < 1e-6


def test_gaussian_valid_and_vacuity_drops_on_evidence():
    m = _mk(GaussianModel)
    x, cal, dt, ei, ea = snapshot_inputs()
    st0 = m.init_state(5)
    st1 = m.step(st0, x, cal, dt, ei, ea)
    assert torch.isfinite(st1.mu).all() and torch.isfinite(st1.log_var).all()
    assert (st1.vacuity < st0.vacuity).all()  # gated evidence shrinks variance
    mu, sigma, ent = m.forecast(st1, x, cal, torch.full((5,), 3.0, dtype=torch.float64))
    assert (sigma > 0).all() and ((ent >= 0) & (ent <= 1)).all()
    (mu.sum() + m.divergence(st1).sum()).backward()


def test_gaussian_uncertainty_grows_without_evidence():
    m = _mk(GaussianModel)
    x, cal, dt, ei, ea = snapshot_inputs()
    st = m.step(m.init_state(5), x, cal, dt, ei, ea)
    _, _, ent1 = m.forecast(st, x, cal, torch.full((5,), 1.0, dtype=torch.float64))
    _, _, ent9 = m.forecast(st, x, cal, torch.full((5,), 9.0, dtype=torch.float64))
    assert ent9.mean().item() >= ent1.mean().item()


def test_capacity_parity_via_forecasters():
    from ltn.exp.variants import GaussianForecaster, NodeODEForecaster

    cfg = {"d": 4, "carry_dim": 16, "hidden": 32}
    target = count_trainable(_mk(LTNModel))
    for fc in (GaussianForecaster(), NodeODEForecaster()):
        n = count_trainable(fc._make_model(4, cfg))
        assert abs(n - target) <= 0.10 * target, f"{fc.name}: {n} vs {target}"
