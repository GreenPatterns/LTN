# `qt` — conda environment for LTN ("Trust Decoheres")

Reproducible Python environment for the Lindblad Trust Networks program
(`MASTER_EXECUTION_PLAN.md`). Created 2026-07-12.

## What it is

| | |
|---|---|
| Env name | `qt` (conda) |
| Location | `/home/mucbp/.conda/envs/qt` |
| Python | 3.11.15 |
| PyTorch | 2.13.0 + **cu130** (bundled CUDA 13.0.3 runtime) |
| GPUs seen | 3 × NVIDIA L40S (48 GB, sm_89), driver 580.x / CUDA 13.0 |
| Disk | ~6.1 GB |

The PyTorch **cu130** build is an exact match for this host's CUDA 13.0 driver, so
GPU tensor ops, `complex128`, and `torch.matrix_exp` (the Lindblad superoperator core)
all run natively on the L40S.

## Activate & use

```bash
conda activate qt
python scripts/verify_env.py          # 18/19 checks, 0 required failures
```

Or without activating (robust against NFS `$PATH` quirks on this cluster):

```bash
/home/mucbp/.conda/envs/qt/bin/python scripts/verify_env.py
```

A Jupyter kernel **"Python (qt) — LTN"** is registered (`jupyter kernelspec list`)
for the data-quality / inventory notebooks (plan A6).

## Reproduce from scratch

```bash
conda create -y -n qt python=3.11 pip
conda activate qt
pip install -r env/requirements.txt          # curated top-level (validated pins)
# — or — exact frozen graph of all 197 packages:
pip install -r env/requirements-lock.txt
```

`env/environment.yml` is the full `conda env export` (authoritative lock, incl. pip section).

## What's installed (top level)

- **DL/GPU:** torch 2.13.0 (cu130), torch_geometric 2.8.0
- **Quantum:** qiskit 2.5.0, qiskit-ibm-runtime 0.47.0, qiskit-aer 0.17.2, qutip 5.3.0
- **Scientific:** numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, pyarrow 25.0.0, statsmodels 0.14.6,
  scikit-learn 1.9.0, properscoring 0.1 (CRPS), h5py 3.16.0
- **Experiment tooling:** optuna 4.9.0, wandb 0.28.0, pyyaml 6.0.3, matplotlib 3.11.0,
  seaborn 0.13.2, zenodo_get 3.0.3
- **Notebooks/CI:** jupyterlab 4.6.1, ipykernel, pytest 9.1.1, pytest-cov

## Verified capabilities (`scripts/verify_env.py`)

Beyond imports, the verifier exercises the math the plan actually depends on:

- **DM-State (§2.2):** Cholesky-parameterized ρ is PSD, trace-1, and differentiable.
- **Lindblad drift (§2.2 signature module):** vectorized d²×d² superoperator + batched
  `matrix_exp` produces a **CPTP** channel (trace-preservation error ~1e-15, Choi matrix PSD).
- **Physics cross-check:** our Lindblad channel matches `qutip.mesolve` (amplitude damping)
  to **1.7e-12** — the quantum-dynamics core is numerically correct, not just runnable.
- PyG message passing, qiskit coupling maps, CRPS/AUROC/ARIMA/optuna/parquet.

## Known caveat — PyG C/CUDA extensions (optional, not installed)

`torch-scatter`, `torch-sparse`, `pyg-lib` have **no prebuilt wheels for torch 2.13 + cu130**
yet (`https://data.pyg.org/whl/torch-2.13.0+cu130.html` → 403/empty). PyG core message
passing (`GCNConv`, custom `MessagePassing`) works fine without them, so nothing in the LTN
architecture is blocked. Revisit only if a scatter-heavy op needs acceleration — then either
install published wheels or build from source against the CUDA 13 toolkit.

## Multi-GPU note

Three L40S are visible. Pin a device per run with `CUDA_VISIBLE_DEVICES=0` (or 1/2) so
parallel experiments/HPO trials don't contend. ~300–400 GPU-hours of campaign compute
(§2.8) fits comfortably.
