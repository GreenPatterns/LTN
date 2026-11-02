"""Unit tests for ltn.models.krausprop — §2.2 propagation guarantees."""
from __future__ import annotations

import torch

from ltn.models.dmstate import rho_from_params
from ltn.models.krausprop import KrausProp

torch.manual_seed(0)
D, EDGE_DIM, NODE_DIM = 4, 3, 5


def _random_graph(n: int, e: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    rho = rho_from_params(torch.randn(n, D * D, dtype=torch.float64, generator=g))
    node_feats = torch.randn(n, NODE_DIM, dtype=torch.float64, generator=g)
    src = torch.randint(0, n, (e,), generator=g)
    dst = (src + 1 + torch.randint(0, n - 1, (e,), generator=g)) % n  # no self-loops
    edge_index = torch.stack([src, dst])
    edge_feats = torch.randn(e, EDGE_DIM, dtype=torch.float64, generator=g)
    return rho, node_feats, edge_index, edge_feats


def test_valid_states_after_two_rounds() -> None:
    prop = KrausProp(D, EDGE_DIM, NODE_DIM, rounds=2)
    rho, nf, ei, ef = _random_graph(n=6, e=14)
    out = prop(rho, nf, ei, ef)
    assert out.shape == rho.shape and out.dtype == torch.complex128
    herm = (out - out.conj().transpose(-2, -1)).abs().max().item()
    assert herm < 1e-10
    tr = torch.diagonal(out, dim1=-2, dim2=-1).sum(-1)
    assert (tr - 1.0).abs().max().item() < 1e-8
    assert torch.linalg.eigvalsh(out).min().item() > -1e-9
    # propagation actually changed states of nodes with in-edges
    assert (out - rho).abs().max().item() > 1e-4


def test_attention_rows_sum_to_one() -> None:
    prop = KrausProp(D, EDGE_DIM, NODE_DIM, rounds=2)
    rho, nf, ei, ef = _random_graph(n=6, e=14, seed=1)
    _, attns = prop(rho, nf, ei, ef, return_attn=True)
    assert len(attns) == 2
    dst = ei[1]
    for alpha in attns:
        sums = torch.zeros(6, dtype=torch.float64)
        sums.index_add_(0, dst, alpha)
        targets = torch.unique(dst)
        assert torch.allclose(sums[targets], torch.ones(len(targets), dtype=torch.float64), atol=1e-12)
        assert (alpha > 0).all()


def test_permutation_consistency() -> None:
    prop = KrausProp(D, EDGE_DIM, NODE_DIM, rounds=2)
    n = 7
    rho, nf, ei, ef = _random_graph(n=n, e=16, seed=2)
    out = prop(rho, nf, ei, ef)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(3))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    # node i in the original graph becomes node inv[i] in the relabeled graph
    out_p = prop(rho[perm], nf[perm], inv[ei], ef)
    assert (out_p - out[perm]).abs().max().item() < 1e-10


def test_node_without_in_edges_unchanged() -> None:
    prop = KrausProp(D, EDGE_DIM, NODE_DIM, rounds=2)
    n = 5
    rho, nf, _, _ = _random_graph(n=n, e=1, seed=4)
    # star into node 0; node 4 has NO in-edges (it only sends)
    edge_index = torch.tensor([[1, 2, 4], [0, 0, 1]])
    ef = torch.randn(3, EDGE_DIM, dtype=torch.float64)
    out = prop(rho, nf, edge_index, ef)
    assert torch.equal(out[4], rho[4].to(torch.complex128))  # exactly untouched
    assert torch.equal(out[2], rho[2].to(torch.complex128))
    assert (out[0] - rho[0]).abs().max().item() > 1e-6  # receiving node updated


def test_empty_graph_is_identity() -> None:
    prop = KrausProp(D, EDGE_DIM, NODE_DIM, rounds=2)
    rho, nf, _, _ = _random_graph(n=4, e=1, seed=5)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    ef = torch.zeros(0, EDGE_DIM, dtype=torch.float64)
    out, attns = prop(rho, nf, edge_index, ef, return_attn=True)
    assert torch.equal(out, rho.to(torch.complex128))
    assert all(a.numel() == 0 for a in attns)


def test_gradients_flow_through_propagation() -> None:
    prop = KrausProp(D, EDGE_DIM, NODE_DIM, rounds=2)
    theta = torch.randn(6, D * D, dtype=torch.float64, requires_grad=True)
    nf = torch.randn(6, NODE_DIM, dtype=torch.float64, requires_grad=True)
    _, _, ei, ef = _random_graph(n=6, e=14, seed=6)
    ef = ef.clone().requires_grad_(True)
    out = prop(rho_from_params(theta), nf, ei, ef)
    loss = (out.real ** 2).sum()
    loss.backward()
    for t, name in [(theta, "theta"), (nf, "node feats"), (ef, "edge feats")]:
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert t.grad.abs().max().item() > 0, f"no gradient into {name}"
    assert prop.log_tau.grad is not None and prop.log_tau.grad.abs().item() > 0
