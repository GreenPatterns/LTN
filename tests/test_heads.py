"""POVM/forecast/divergence heads + param matching."""
import torch

from ltn.models.dmstate import rho_from_params
from ltn.models.heads import DivergenceHead, ForecastHead, POVMHeads, rho_to_real
from ltn.models.param_match import count_trainable, match_width


def rand_rho(b=6, d=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return rho_from_params(torch.randn(b, d * d, generator=g, dtype=torch.float64))


def test_povm_is_resolution_of_identity():
    heads = POVMHeads(d=4, n_facets=3)
    E = heads.povm()
    I = torch.eye(4, dtype=E.dtype)
    assert (E.sum(0) - I).abs().max().item() < 1e-10
    for k in range(3):  # each element PSD
        assert torch.linalg.eigvalsh(E[k]).min().item() > -1e-10


def test_povm_scores_simplex():
    heads = POVMHeads(d=4, n_facets=3)
    s = heads(rand_rho())
    assert (s >= 0).all() and (s <= 1).all()
    assert (s.sum(-1) - 1.0).abs().max().item() < 1e-8


def test_rho_to_real_preserves_information():
    rho = rand_rho(b=2)
    z = rho_to_real(rho)
    assert z.shape == (2, 16)
    assert torch.allclose(z[:, :4].sum(-1), torch.ones(2, dtype=z.dtype), atol=1e-10)


def test_forecast_head_shapes_and_grad():
    fh = ForecastHead(d=4, carry_dim=8, n_obs=5)
    rho = rand_rho(b=6)
    carry = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
    mu, sigma = fh(rho, carry, torch.full((6,), 3.0))
    assert mu.shape == sigma.shape == (6, 5)
    assert (sigma > 0).all()
    mu.sum().backward()
    assert carry.grad is not None and carry.grad.abs().max() > 0


def test_divergence_head_probability():
    dh = DivergenceHead(d=4, n_facets=2)
    p = dh(torch.rand(6, 2, dtype=torch.float64),
           torch.rand(6, dtype=torch.float64), torch.rand(6, dtype=torch.float64))
    assert p.shape == (6,) and (p > 0).all() and (p < 1).all()


def test_match_width_hits_tolerance():
    import torch.nn as nn

    def builder(w):
        return nn.Sequential(nn.Linear(16, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, 4))

    target = count_trainable(builder(64))
    m, w, n = match_width(target, builder)
    assert abs(n - target) <= 0.10 * target
    assert w == 64  # exact target is reachable
