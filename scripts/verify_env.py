#!/usr/bin/env python
"""
Environment verification for the LTN ("Trust Decoheres") project — conda env `qt`.

Checks that every capability the MASTER_EXECUTION_PLAN depends on actually works:
  - PyTorch + CUDA on the L40S GPUs (complex tensors, matrix_exp, autograd)
  - The Lindblad-drift math core (§2.2): DM-State (Cholesky PSD/trace-1),
    superoperator vectorization + batched matrix_exp, CPTP guarantee
  - PyTorch Geometric message passing
  - Quantum stack: qiskit coupling maps, qiskit-ibm-runtime, qutip Lindblad reference
  - Eval/stats/data stack (CRPS, AUROC, ARIMA, optuna, pandas/pyarrow, wandb)

Run: /home/mucbp/.conda/envs/qt/bin/python scripts/verify_env.py
Exit code 0 iff all REQUIRED checks pass.
"""
import sys, traceback

PASS, FAIL, WARN = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"
results = []  # (name, ok, required, detail)


def check(name, required=True):
    def deco(fn):
        try:
            detail = fn() or ""
            results.append((name, True, required, detail))
        except Exception as e:
            results.append((name, False, required, f"{type(e).__name__}: {e}"))
            if "--verbose" in sys.argv:
                traceback.print_exc()
        return fn
    return deco


# ----------------------------------------------------------------------------
@check("torch import + version")
def _():
    import torch
    return f"torch {torch.__version__}"


@check("CUDA available on GPU")
def _():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    cc = torch.cuda.get_device_capability(0)
    return f"{n} GPU(s): {names}; CUDA={torch.version.cuda}; sm_{cc[0]}{cc[1]}"


@check("GPU tensor matmul (fp32)")
def _():
    import torch
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    c = (a @ b).sum().item()
    assert torch.isfinite(torch.tensor(c)), "non-finite result"
    return f"1024x1024 matmul on {torch.cuda.get_device_name(0)} ok"


@check("complex128 GPU support")
def _():
    import torch
    z = torch.randn(64, 64, dtype=torch.complex128, device="cuda")
    h = z + z.conj().T
    ev = torch.linalg.eigvalsh(h)  # needs cusolver
    return f"complex128 eigvalsh ok (min eig {ev.min().item():.3f})"


@check("DM-State: Cholesky rho is PSD, trace-1, differentiable (§2.2)")
def _():
    import torch
    d = 4
    L = torch.randn(d, d, dtype=torch.complex128, device="cuda", requires_grad=True)
    Ltri = torch.tril(L)
    M = Ltri @ Ltri.conj().T
    rho = M / torch.trace(M).real
    # trace-1
    tr = torch.trace(rho).real.item()
    assert abs(tr - 1.0) < 1e-9, f"trace={tr}"
    # PSD (Hermitian, eigenvalues >= 0)
    ev = torch.linalg.eigvalsh(rho)
    assert ev.min().item() > -1e-9, f"min eig {ev.min().item()}"
    # gradient flows
    loss = (rho.real ** 2).sum()
    loss.backward()
    assert L.grad is not None and torch.isfinite(L.grad).all()
    return f"d={d}: trace={tr:.9f}, min_eig={ev.min().item():.2e}, grad ok"


@check("Lindblad superoperator + batched matrix_exp is CPTP (§2.2 signature module)")
def _():
    import torch
    # d=4 (default). Vectorize drho/dt = -i[H,rho] + sum_k gamma_k D[L_k](rho)
    # to a d^2 x d^2 superoperator, exponentiate over dt, check the channel is
    # trace-preserving and maps a density matrix to a valid density matrix.
    torch.manual_seed(0)
    d = 4
    dev = "cuda"
    cdt = torch.complex128
    I = torch.eye(d, dtype=cdt, device=dev)

    def sprepost(A, B):  # rho -> A rho B  ==>  (B^T ⊗ A) vec(rho)  (column-stacking)
        return torch.kron(B.T.contiguous(), A)

    def spre(A):
        return torch.kron(I, A)

    def spost(B):
        return torch.kron(B.T.contiguous(), I)

    # Random Hermitian H, a few jump operators, positive rates
    Hraw = torch.randn(d, d, dtype=cdt, device=dev)
    H = Hraw + Hraw.conj().T
    Ls = [torch.randn(d, d, dtype=cdt, device=dev) for _ in range(3)]
    gammas = torch.nn.functional.softplus(torch.randn(3, device=dev)).to(torch.float64)

    Lgen = -1j * (spre(H) - spost(H))
    for g, Lk in zip(gammas, Ls):
        LdL = Lk.conj().T @ Lk
        Lgen = Lgen + g * (sprepost(Lk, Lk.conj().T)
                           - 0.5 * spre(LdL) - 0.5 * spost(LdL))
    dt = 0.37
    S = torch.matrix_exp(Lgen * dt)  # d^2 x d^2 channel superoperator

    # Trace preservation: vec(I)^dagger @ S == vec(I)^dagger  (TP condition)
    vecI = I.reshape(-1)  # column-stacked identity
    lhs = S.conj().T @ vecI
    tp_err = (lhs - vecI).abs().max().item()
    assert tp_err < 1e-8, f"trace-preservation error {tp_err}"

    # Complete positivity via Choi matrix PSD (Choi = reshuffle of S)
    Choi = S.reshape(d, d, d, d).permute(0, 2, 1, 3).reshape(d * d, d * d)
    Choi = (Choi + Choi.conj().T) / 2
    cp_eig = torch.linalg.eigvalsh(Choi).min().item()
    assert cp_eig > -1e-7, f"Choi min eig {cp_eig} (not CP)"

    # Apply to a real density matrix, confirm output is a valid state
    Lr = torch.tril(torch.randn(d, d, dtype=cdt, device=dev))
    m = Lr @ Lr.conj().T
    rho = m / torch.trace(m).real
    rho_out = (S @ rho.reshape(-1)).reshape(d, d)
    tr_out = torch.trace(rho_out).real.item()
    out_eig = torch.linalg.eigvalsh((rho_out + rho_out.conj().T) / 2).min().item()
    assert abs(tr_out - 1.0) < 1e-8 and out_eig > -1e-7
    # Batched matrix_exp (device x time), the real training shape
    batch = torch.stack([Lgen * t for t in (0.1, 0.5, 1.0)])
    Sb = torch.matrix_exp(batch)
    assert Sb.shape == (3, d * d, d * d)
    return (f"d={d}: TP_err={tp_err:.1e}, Choi_min_eig={cp_eig:.1e}, "
            f"out_trace={tr_out:.9f}, batched exp {tuple(Sb.shape)} ok")


@check("Lindblad matches qutip reference (amplitude damping)", required=False)
def _():
    import torch, numpy as np
    import qutip as qt
    # Qubit amplitude damping: compare our superoperator exp to qutip mesolve.
    d = 2
    dev = "cpu"
    cdt = torch.complex128
    I = torch.eye(d, dtype=cdt)
    sm = torch.tensor([[0, 1], [0, 0]], dtype=cdt)  # lowering |1><0|... convention
    gamma = 0.8
    dt = 0.5

    def spre(A):  return torch.kron(I, A)
    def spost(B): return torch.kron(B.T.contiguous(), I)
    def sprepost(A, B): return torch.kron(B.T.contiguous(), A)

    LdL = sm.conj().T @ sm
    Lgen = gamma * (sprepost(sm, sm.conj().T) - 0.5 * spre(LdL) - 0.5 * spost(LdL))
    S = torch.matrix_exp(Lgen * dt)
    rho0 = torch.tensor([[0.3, 0.2 + 0.1j], [0.2 - 0.1j, 0.7]], dtype=cdt)
    ours = (S @ rho0.reshape(-1)).reshape(d, d).numpy()

    # qutip reference
    sm_q = qt.destroy(2)
    rho0_q = qt.Qobj(rho0.numpy())
    res = qt.mesolve(qt.qzero(2), rho0_q, [0, dt],
                     c_ops=[np.sqrt(gamma) * sm_q], options={"atol": 1e-12, "rtol": 1e-12})
    ref = res.states[-1].full()
    err = np.abs(ours - ref).max()
    assert err < 1e-6, f"mismatch vs qutip: {err}"
    return f"max|ours - qutip| = {err:.2e}  (physics core validated)"


@check("PyTorch Geometric message passing")
def _():
    import torch
    from torch_geometric.nn import GCNConv
    import torch_geometric
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    x = torch.randn(4, 8)
    conv = GCNConv(8, 16)
    out = conv(x, edge_index)
    assert out.shape == (4, 16)
    return f"PyG {torch_geometric.__version__}: GCNConv 8->16 on 4-node graph ok"


@check("PyG optional C++/CUDA extensions", required=False)
def _():
    import torch_scatter, torch_sparse
    return f"torch_scatter {torch_scatter.__version__}, torch_sparse {torch_sparse.__version__}"


@check("qiskit + coupling map (graph builder, A4)")
def _():
    import qiskit
    from qiskit.transpiler import CouplingMap
    cm = CouplingMap.from_heavy_hex(3)
    n = cm.size()
    return f"qiskit {qiskit.__version__}: heavy-hex coupling map, {n} qubits, {len(cm.get_edges())} edges"


@check("qiskit-ibm-runtime (harvester dependency, A1)")
def _():
    import qiskit_ibm_runtime
    from qiskit_ibm_runtime import QiskitRuntimeService  # noqa: F401
    return f"qiskit-ibm-runtime {qiskit_ibm_runtime.__version__} import ok (no token needed to import)"


@check("qiskit-aer simulator", required=False)
def _():
    import qiskit_aer
    return f"qiskit-aer {qiskit_aer.__version__}"


@check("eval metrics: CRPS (properscoring) + AUROC (sklearn)")
def _():
    import numpy as np, properscoring as ps
    from sklearn.metrics import roc_auc_score
    crps = ps.crps_gaussian(0.5, mu=0.0, sig=1.0)
    auc = roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    return f"CRPS_gaussian={crps:.4f}, AUROC={auc:.3f}"


@check("statsmodels ARIMA baseline")
def _():
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA
    y = np.cumsum(np.random.RandomState(0).randn(50))
    m = ARIMA(y, order=(1, 1, 0)).fit()
    return f"statsmodels ARIMA(1,1,0) fit, aic={m.aic:.1f}"


@check("optuna study")
def _():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study()
    st.optimize(lambda t: (t.suggest_float("x", -1, 1)) ** 2, n_trials=5)
    return f"optuna {optuna.__version__}: 5-trial study, best={st.best_value:.3f}"


@check("pandas + pyarrow parquet round-trip (event lake)")
def _():
    import pandas as pd, pyarrow, tempfile, os
    df = pd.DataFrame({"entity": ["q0"], "observable": ["T1"], "value": [123.4]})
    p = os.path.join(tempfile.gettempdir(), "_qt_verify.parquet")
    df.to_parquet(p)
    back = pd.read_parquet(p)
    os.remove(p)
    assert back.equals(df)
    return f"pandas {pd.__version__} + pyarrow {pyarrow.__version__} parquet ok"


@check("wandb import (logging)", required=False)
def _():
    import wandb
    return f"wandb {wandb.__version__} (offline-capable)"


@check("scipy + h5py + pyyaml + tqdm")
def _():
    import scipy, h5py, yaml, tqdm
    return f"scipy {scipy.__version__}, h5py {h5py.__version__}, pyyaml {yaml.__version__}"


@check("pytest available (CI)")
def _():
    import pytest
    return f"pytest {pytest.__version__}"


@check("stim + pymatching (Google QEC loaders, [DEV-7])")
def _():
    import stim, pymatching
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                               after_clifford_depolarization=0.01)
    dem = c.detector_error_model(decompose_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    shots = c.compile_detector_sampler().sample(64, append_observables=True)
    preds = m.decode_batch(shots[:, : dem.num_detectors])
    assert preds.shape[0] == 64
    return f"stim {stim.__version__} + pymatching {pymatching.__version__}: d=3 decode ok"


# ----------------------------------------------------------------------------
def main():
    width = 66
    print("=" * width)
    print("  LTN env `qt` verification")
    print("=" * width)
    req_fail = 0
    for name, ok, required, detail in results:
        tag = PASS if ok else (FAIL if required else WARN)
        req = "" if required else " (optional)"
        print(f"[{tag}] {name}{req}")
        if detail:
            print(f"        {detail}")
        if not ok and required:
            req_fail += 1
    print("=" * width)
    total = len(results)
    passed = sum(1 for _, ok, *_ in results if ok)
    print(f"  {passed}/{total} checks passed; {req_fail} REQUIRED failure(s)")
    print("=" * width)
    sys.exit(1 if req_fail else 0)


if __name__ == "__main__":
    main()
