"""Unit tests for ltn.models.lindblad — §2.2 signature-module guarantees.

Ports the verify_env.py CPTP/Choi/qutip checks into the module API and extends
them with the learned components (JumpDictionary, GammaNet, HermitianHead).
"""
from __future__ import annotations

import pytest
import torch

from ltn.models.dmstate import params_init, rho_from_params, vec, von_neumann_entropy
from ltn.models.lindblad import (
    GammaNet,
    HermitianHead,
    JumpDictionary,
    evolve,
    lindblad_generator,
)

torch.manual_seed(0)
D = 4


def _random_states(b: int, d: int = D) -> torch.Tensor:
    return rho_from_params(torch.randn(b, d * d, dtype=torch.float64))


def _random_setup(b: int, d: int = D):
    Hraw = torch.randn(b, d, d, dtype=torch.complex128)
    H = Hraw + Hraw.conj().transpose(-2, -1)
    Ls = list(JumpDictionary(d).ops().detach())
    gammas = torch.nn.functional.softplus(torch.randn(b, len(Ls), dtype=torch.float64))
    return H, Ls, gammas


def test_channel_is_cptp_for_random_theta_dt() -> None:
    b, d = 3, D
    H, Ls, gammas = _random_setup(b)
    gen = lindblad_generator(H, Ls, gammas)
    assert gen.shape == (b, d * d, d * d)
    dt = torch.tensor([0.1, 0.37, 1.5], dtype=torch.float64)
    S = torch.matrix_exp(gen * dt.to(torch.complex128).view(-1, 1, 1))
    vecI = torch.eye(d, dtype=torch.complex128).reshape(-1)
    for i in range(b):
        # trace preservation: vec(I)^dag S = vec(I)^dag
        tp_err = (S[i].conj().T @ vecI - vecI).abs().max().item()
        assert tp_err < 1e-8, f"sample {i}: TP error {tp_err}"
        # complete positivity: Choi (reshuffle of S) is PSD
        choi = S[i].reshape(d, d, d, d).permute(0, 2, 1, 3).reshape(d * d, d * d)
        choi = (choi + choi.conj().T) / 2
        cp_eig = torch.linalg.eigvalsh(choi).min().item()
        assert cp_eig > -1e-7, f"sample {i}: Choi min eig {cp_eig}"


def test_evolve_valid_states_and_batched_dt() -> None:
    b = 4
    rho = _random_states(b)
    H, Ls, gammas = _random_setup(b)
    dt = torch.tensor([0.0, 0.05, 0.7, 2.0], dtype=torch.float64)
    out = evolve(rho, H, Ls, gammas, dt)
    tr = torch.diagonal(out, dim1=-2, dim2=-1).sum(-1)
    assert (tr - 1.0).abs().max().item() < 1e-10
    assert torch.linalg.eigvalsh(out).min().item() > -1e-9
    # dt = 0 must be the identity channel on that sample only
    assert (out[0] - rho[0]).abs().max().item() < 1e-10
    assert (out[2] - rho[2]).abs().max().item() > 1e-4  # dynamics actually act


def test_entropy_nondecreasing_under_pure_dephasing() -> None:
    # H = 0, only the diagonal (dephasing, k=1) jump op active.
    d = D
    Ls = list(JumpDictionary(d).ops().detach())
    rho0 = _random_states(1, d)
    dts = torch.linspace(0.0, 3.0, 13, dtype=torch.float64)
    b = len(dts)
    rho = rho0.expand(b, d, d).contiguous()
    H = torch.zeros(b, d, d, dtype=torch.complex128)
    gammas = torch.zeros(b, 3, dtype=torch.float64)
    gammas[:, 1] = 0.9
    out = evolve(rho, H, Ls, gammas, dts)
    ent = von_neumann_entropy(out)
    diffs = ent[1:] - ent[:-1]
    assert (diffs > -1e-10).all(), f"entropy not monotone: {ent.tolist()}"
    assert ent[-1].item() > ent[0].item() + 1e-3  # strictly dephases


@pytest.mark.slow
def test_evolve_matches_qutip_amplitude_damping() -> None:
    qt = pytest.importorskip("qutip")
    import numpy as np

    d, gamma, dt = 2, 0.8, 0.5
    sm = torch.tensor([[0, 1], [0, 0]], dtype=torch.complex128)
    rho0 = torch.tensor([[0.3, 0.2 + 0.1j], [0.2 - 0.1j, 0.7]], dtype=torch.complex128)
    ours = evolve(
        rho0.unsqueeze(0),
        torch.zeros(1, d, d, dtype=torch.complex128),
        [sm],
        torch.tensor([[gamma]], dtype=torch.float64),
        torch.tensor([dt], dtype=torch.float64),
    )[0].numpy()
    res = qt.mesolve(
        qt.qzero(2), qt.Qobj(rho0.numpy()), [0, dt],
        c_ops=[np.sqrt(gamma) * qt.destroy(2)],
        options={"atol": 1e-12, "rtol": 1e-12},
    )
    err = np.abs(ours - res.states[-1].full()).max()
    assert err < 1e-6, f"mismatch vs qutip: {err}"


@pytest.mark.slow
def test_evolve_matches_qutip_general_d4() -> None:
    """Regression for the vec-convention bug: general H != 0, d=4, random jumps.
    The amplitude-damping (H=0, d=2) case above is degenerate and hides a
    row-vs-column vec mismatch; this exercises the full generator."""
    qt = pytest.importorskip("qutip")
    import numpy as np
    from ltn.models.dmstate import rho_from_params

    torch.manual_seed(1)
    d, dt = 4, 0.37
    Hraw = torch.randn(1, d, d, dtype=torch.complex128)
    H = Hraw + Hraw.conj().transpose(-2, -1)
    Ls = [torch.randn(d, d, dtype=torch.complex128) for _ in range(3)]
    g = torch.rand(1, 3, dtype=torch.float64)
    rho = rho_from_params(torch.randn(1, d * d, dtype=torch.float64))
    ours = evolve(rho, H, Ls, g, torch.tensor([dt], dtype=torch.float64))[0].numpy()
    ref = qt.mesolve(
        qt.Qobj(H[0].numpy()), qt.Qobj(rho[0].numpy()), [0, dt],
        c_ops=[np.sqrt(float(g[0, k])) * qt.Qobj(Ls[k].numpy()) for k in range(3)],
        options={"atol": 1e-12, "rtol": 1e-12},
    ).states[-1].full()
    err = np.abs(ours - ref).max()
    assert err < 1e-9, f"mismatch vs qutip (d=4, H!=0): {err}"


def test_gradients_flow_from_emb_and_gammanet_inputs() -> None:
    b, d, emb_dim, feat_dim = 3, D, 8, 6
    head = HermitianHead(emb_dim, d)
    gnet = GammaNet(feat_dim, K=3)
    jumps = JumpDictionary(d)
    emb = torch.randn(b, emb_dim, dtype=torch.float64, requires_grad=True)
    x = torch.randn(b, feat_dim, dtype=torch.float64, requires_grad=True)
    rho = _random_states(b)
    dt = torch.full((b,), 0.5, dtype=torch.float64)
    out = evolve(rho, head(emb), list(jumps.ops()), gnet(x), dt)
    loss = von_neumann_entropy(out).sum()
    loss.backward()
    for t, name in [(emb, "emb"), (x, "gamma features")]:
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert t.grad.abs().max().item() > 0, f"no gradient into {name}"
    assert jumps.ops_real.grad is not None and jumps.ops_real.grad.abs().max().item() > 0


def test_gamma_increases_with_cal_age_at_init() -> None:
    torch.manual_seed(1)
    feat_dim, K, idx = 7, 3, 2
    gnet = GammaNet(feat_dim, K=K, cal_age_idx=idx)
    x = torch.randn(16, feat_dim, dtype=torch.float64, requires_grad=True)
    g = gnet(x)
    assert (g > 0).all()
    for k in range(K):
        x.grad = None
        g[:, k].sum().backward(retain_graph=True)
        dg_dcal = x.grad[:, idx]
        assert (dg_dcal > 0).all(), f"d gamma_{k}/d cal_age not positive at init"
        # cal_age is the ONLY input direction alive at init (MLP final layer is zero)
        other = x.grad.clone()
        other[:, idx] = 0.0
        assert other.abs().max().item() < 1e-12


def test_jump_dictionary_init_structure() -> None:
    d = D
    ops = JumpDictionary(d).ops().detach()
    assert ops.shape == (3, d, d) and ops.dtype == torch.complex128
    for k in range(3):
        assert abs(ops[k].norm().item() - 1.0) < 1e-12  # Frobenius-normalized
    # k=0 lowering: support only on the superdiagonal
    low = ops[0]
    mask = torch.zeros(d, d, dtype=torch.bool)
    mask[torch.arange(d - 1), torch.arange(1, d)] = True
    assert low[~mask].abs().max().item() == 0.0 and low[mask].abs().min().item() > 0
    # k=1 dephasing: diagonal and traceless
    deph = ops[1]
    assert (deph - torch.diag(torch.diagonal(deph))).abs().max().item() == 0.0
    assert torch.diagonal(deph).sum().abs().item() < 1e-12
    # k=2 generalized shift: support exactly on |i+1 mod d><i|
    shift = ops[2]
    mask = torch.zeros(d, d, dtype=torch.bool)
    mask[(torch.arange(d) + 1) % d, torch.arange(d)] = True
    assert shift[~mask].abs().max().item() == 0.0 and shift[mask].abs().min().item() > 0
