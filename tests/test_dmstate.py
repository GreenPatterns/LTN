"""Unit tests for ltn.models.dmstate — §2.2 State guarantees."""
from __future__ import annotations

import math

import pytest
import torch

from ltn.models.dmstate import (
    DMState,
    dissonance,
    params_init,
    purity,
    rho_from_params,
    unvec,
    vacuity,
    vec,
    von_neumann_entropy,
)

torch.manual_seed(0)


def _assert_valid_state(rho: torch.Tensor, atol: float = 1e-10) -> None:
    assert rho.dtype == torch.complex128
    herm_err = (rho - rho.conj().transpose(-2, -1)).abs().max().item()
    assert herm_err < atol, f"not Hermitian: {herm_err}"
    tr = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)
    assert (tr - 1.0).abs().max().item() < atol
    ev = torch.linalg.eigvalsh(rho)
    assert ev.min().item() > -atol, f"min eig {ev.min().item()}"


@pytest.mark.parametrize("d", [2, 4, 8])
def test_random_theta_gives_valid_state_and_grad(d: int) -> None:
    theta = torch.randn(5, d * d, dtype=torch.float64, requires_grad=True)
    rho = rho_from_params(theta)
    assert rho.shape == (5, d, d)
    _assert_valid_state(rho)
    loss = (rho.real ** 2).sum() + (rho.imag ** 2).sum()
    loss.backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().max().item() > 0


def test_mixed_init_is_vacuous_prior() -> None:
    d = 4
    rho = rho_from_params(params_init(3, d, mixed=True))
    eye = torch.eye(d, dtype=torch.complex128) / d
    assert (rho - eye).abs().max().item() < 1e-12
    assert torch.allclose(vacuity(rho), torch.ones(3, dtype=torch.float64), atol=1e-12)
    assert dissonance(rho).abs().max().item() < 1e-12
    # entropy is maximal: log d (1.0 normalized)
    assert torch.allclose(von_neumann_entropy(rho), torch.full((3,), math.log(d), dtype=torch.float64), atol=1e-9)
    assert torch.allclose(von_neumann_entropy(rho, normalized=True), torch.ones(3, dtype=torch.float64), atol=1e-9)


def test_pure_init_zero_vacuity() -> None:
    d = 4
    rho = rho_from_params(params_init(2, d, mixed=False))
    _assert_valid_state(rho)
    assert vacuity(rho).abs().max().item() < 1e-6
    assert (purity(rho) - 1.0).abs().max().item() < 1e-6
    assert von_neumann_entropy(rho).max().item() < 1e-6


def test_dissonance_range_and_max_coherent_state() -> None:
    d = 4
    # maximally coherent pure state |+><+|: all entries 1/d -> dissonance 1
    plus = torch.full((1, d, d), 1.0 / d, dtype=torch.complex128)
    assert abs(dissonance(plus).item() - 1.0) < 1e-12
    # random states stay in [0, 1]
    rho = rho_from_params(torch.randn(64, d * d, dtype=torch.float64))
    dis = dissonance(rho)
    assert (dis >= -1e-12).all() and (dis <= 1.0 + 1e-12).all()
    vac = vacuity(rho)
    assert (vac >= -1e-12).all() and (vac <= 1.0 + 1e-12).all()


def test_vec_unvec_roundtrip() -> None:
    rho = rho_from_params(torch.randn(3, 16, dtype=torch.float64))
    v = vec(rho)
    assert v.shape == (3, 16)
    assert torch.allclose(unvec(v), rho)
    # convention check: vec is COLUMN-stacking (row-major flatten of the transpose),
    # the convention for which vec(A rho B) = (B^T (x) A) vec(rho) holds in lindblad.py
    assert torch.equal(v, rho.transpose(-2, -1).reshape(3, -1))


def test_dmstate_module_optimizes() -> None:
    state = DMState(4, 4, mixed=True, noise=1e-2)
    rho0 = state.rho().detach()
    _assert_valid_state(state.rho(), atol=1e-9)
    opt = torch.optim.Adam(state.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        loss = vacuity(state.rho()).sum()  # push toward purity
        loss.backward()
        opt.step()
    rho1 = state.rho().detach()
    assert (rho1 - rho0).abs().max().item() > 1e-6
    _assert_valid_state(rho1, atol=1e-9)
