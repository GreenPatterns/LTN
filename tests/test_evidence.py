"""Unit tests for ltn.models.evidence — §2.2 evidence-channel guarantees."""
from __future__ import annotations

import pytest
import torch

from ltn.models.dmstate import rho_from_params
from ltn.models.evidence import EvidenceChannel

torch.manual_seed(0)


@pytest.mark.parametrize("d,rank", [(2, 4), (4, 4), (4, 2), (8, 4)])
def test_completeness_exact_by_construction(d: int, rank: int) -> None:
    torch.manual_seed(d * 10 + rank)
    ch = EvidenceChannel(feat_dim=6, d=d, rank=rank)
    x = torch.randn(16, 6, dtype=torch.float64)
    K = ch.kraus(x)
    assert K.shape == (16, rank, d, d) and K.dtype == torch.complex128
    comp = torch.einsum("brji,brjk->bik", K.conj(), K)  # sum_i K_i^dag K_i
    eye = torch.eye(d, dtype=torch.complex128).expand(16, d, d)
    err = (comp - eye).abs().max().item()
    assert err < 1e-5, f"completeness violated: {err}"


def test_output_is_valid_state() -> None:
    d = 4
    ch = EvidenceChannel(feat_dim=5, d=d)
    rho = rho_from_params(torch.randn(8, d * d, dtype=torch.float64))
    x = torch.randn(8, 5, dtype=torch.float64)
    out = ch(rho, x)
    herm = (out - out.conj().transpose(-2, -1)).abs().max().item()
    assert herm < 1e-10
    tr = torch.diagonal(out, dim1=-2, dim2=-1).sum(-1)
    assert (tr - 1.0).abs().max().item() < 1e-8
    assert torch.linalg.eigvalsh(out).min().item() > -1e-9
    # the channel actually does something (not the identity map)
    assert (out - rho).abs().max().item() > 1e-4


def test_float32_features_accepted() -> None:
    d = 4
    ch = EvidenceChannel(feat_dim=5, d=d)
    rho = rho_from_params(torch.randn(3, d * d, dtype=torch.float64))
    out = ch(rho, torch.randn(3, 5, dtype=torch.float32))
    assert out.dtype == torch.complex128
    tr = torch.diagonal(out, dim1=-2, dim2=-1).sum(-1)
    assert (tr - 1.0).abs().max().item() < 1e-8


def test_gradients_flow_to_features_and_params() -> None:
    d = 4
    ch = EvidenceChannel(feat_dim=5, d=d)
    theta = torch.randn(4, d * d, dtype=torch.float64, requires_grad=True)
    x = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    out = ch(rho_from_params(theta), x)
    loss = (out.real ** 2).sum() + (out.imag ** 2).sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().max().item() > 0
    assert theta.grad is not None and theta.grad.abs().max().item() > 0
    for name, p in ch.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    w0 = ch.mlp[0].weight.grad
    assert w0.abs().max().item() > 0
