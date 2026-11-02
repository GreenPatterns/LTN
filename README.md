# Does Trust Decohere? Lindblad Trust Networks for quantum-hardware telemetry

Code, frozen benchmark splits, and run records for the paper

> M. Usman and Y. Lee, **"Does Trust Decohere? A Pre-Registered Test for Quantum
> Hardware Reliability."** Department of Computer Science, University of
> Missouri–Kansas City.

Lindblad Trust Networks (LTN) keep a per-device trust state as a density matrix. The
state evolves between observations under a learned Lindblad master equation whose
rates are coupled to measured device physics. New telemetry updates it through
completely positive, trace-preserving (CPTP) evidence channels, and it is propagated
over the device graph by convex CPTP message passing (KrausProp). It is read out by
POVMs and a Gaussian forecast head.

This repository contains what is needed to check the numbers in the paper against the
logged runs, and to re-run the study from public data.

## Contents

| Path | What it holds |
|---|---|
| `ltn/` | The Python package. Data loaders and the unified telemetry schema (`ltn/data`). Models: LTN, LTN-v2 and the ablation variants (`ltn/models`). Baselines, including the classical reliability models (`ltn/baselines`). Experiments E1–E8 (`ltn/exp`). Metrics and statistics (`ltn/eval`). The synthetic fleet (`ltn/sim`). Table and figure generators (`ltn/paper_gen`). The IBM harvester, which was only ever dry-run (`ltn/harvest`). |
| `scripts/` | Data download, campaign drivers, and the stand-alone analyses: state sufficiency, generator spectra, reliability endpoints, computational cost and change-point detectors. |
| `tests/` | Unit tests for the validity of every quantum operation, and for the statistics, loaders and models. |
| `splits/` | Frozen benchmark splits and their SHA-256 seals (`SHA256SUMS`). |
| `runs/` | Per-run records: `results.jsonl`, `config.yaml` and `status.yaml`. Also the model checkpoints (`model.pt`), the Optuna tuning studies (`runs/tuning`), and the outputs of the stand-alone analyses. |
| `paper/tables`, `paper/figures` | Tables and figures as generated from `runs/`. They serve as reference outputs. |
| `docs/prereg.md` | The pre-registration: the frozen text, followed by a post-freeze addendum. |
| `docs/PLAN_DELTA.md` | The deviation register. |
| `notebooks/` | Data-quality notebook for the DAQEC archive. |
| `env/` | Conda environment and pinned requirements. |

## Setup

```bash
conda env create -f env/environment.yml      # creates the environment `qt` (Python 3.11)
conda activate qt
pip install -e .
python scripts/verify_env.py
```

The `Makefile` names an interpreter path on the original machine. Pass your own on
the command line, for example `make PY=$(which python) test`.

## Data

All data are public:

| Source | Used for | Location |
|---|---|---|
| DAQEC, three IBM Eagle devices, 14 days | E1a, E3, E4a, E5 (early stopping), E6b | Zenodo [18045662](https://doi.org/10.5281/zenodo.18045662) |
| Google Quantum AI QEC releases | E5 (zero-shot test) | Zenodo [13273331](https://doi.org/10.5281/zenodo.13273331), [14238907](https://doi.org/10.5281/zenodo.14238907), [14238944](https://doi.org/10.5281/zenodo.14238944), [6804040](https://doi.org/10.5281/zenodo.6804040) |
| Silicon spin qubits, three SiMOS devices | E5 (training) | Zenodo [15653466](https://doi.org/10.5281/zenodo.15653466) |
| Bitcoin-OTC and Bitcoin-Alpha | E8 | [SNAP](https://snap.stanford.edu/data/) |

`scripts/download_data.sh` fetches all of them. First set `ROOT` (the download
directory, normally `data/raw`) and `QT` (your environment's `bin` directory) at the top
of the script. Then build the unified telemetry lake and the synthetic fleet:

```bash
bash scripts/download_data.sh
python -m ltn.data.build_lake                # raw data -> data/uts
make PY=$(which python) sim-data             # Sim-Fleet -> data/uts, data/sim_truth
```

## Pre-registration and frozen splits

- `docs/prereg.md` states the hypotheses, success bars, kill criteria, statistical
  protocol and tuning budgets that were fixed before any experiment was run.
- `splits/` holds the frozen split files. Check them against their seals:

  ```bash
  cd splits && sha256sum -c SHA256SUMS         # 9 files, all OK
  python -m ltn.eval.splits --verify-hashes    # the same check, through the package
  ```

- The splits were frozen in the development repository under the tags `prereg-v1`
  (commit `b69f68d`, 2026-07-12 22:32:56 −05:00) and `prereg-v1.1` (commit `9f4b22c`,
  2026-07-12 22:49:29 −05:00). The earliest run artifact,
  `runs/tuning/optuna_e1a_ewma.db`, was created after the second tag.
- `docs/PLAN_DELTA.md` lists every deviation, separating those decided before the
  results were known from those decided after.

## Checking the paper's numbers (no training)

```bash
make PY=$(which python) ci        # unit tests and seal verification
make PY=$(which python) tables    # paper/tables/T1–T4 from runs/**/results.jsonl (T1 needs the data lake)
make PY=$(which python) figures   # paper/figures/F1–F7 (F1 needs the data lake)
python -m ltn.sim.validate        # Sim-Fleet validation: Table I and Fig. 3 (needs the data lake)
```

The generators write into `paper/tables` and `paper/figures`. If this directory is
kept under git, `git diff paper/` after running them shows any difference from the
reference outputs.

`make repro-fast` additionally compiles the manuscript, which is not part of this
repository. Use the targets above instead.

The per-pair forecast files (`runs/**/predictions.parquet`, 1.4 GB in total) are not
included in this repository because of their size. They are needed only to recompute
the cell-level Wilcoxon tests and the reliability endpoints (Tables VIII–X).
Everything else regenerates from the files here.

## Re-running the experiments

```bash
make PY=$(which python) tune                 # 50 Optuna trials per model and split
make PY=$(which python) e1                   # E1a and E1b, 5 seeds each
make PY=$(which python) e2 e3 e4 e5 e6 e7 e8
make PY=$(which python) repro                # the whole campaign, in order
```

The full campaign takes several days on two GPUs. The reported runs used two NVIDIA
L40S cards, and LTN computes in double precision.

The stand-alone analyses:

```bash
python scripts/carry_ablation.py --exp e1a          # state-sufficiency test (Table XI); also --exp e1b
python scripts/carry_ablation.py --tier 1 --exp e1a # carry-free LTN-v2 retrained from scratch
python scripts/spectra.py --mode all                # generator-spectra diagnostics (Sec. V-A)
python scripts/benchmark_cost.py train              # training cost (Sec. V-B); `latency` needs a GPU
python scripts/e3_changepoint.py                    # change-point detectors on E3 (Sec. V-C)
python scripts/reliability_endpoints.py             # reliability endpoints (Tables VIII–X)
```

## Where each result comes from

| In the paper | Generator | Records |
|---|---|---|
| Table I, Fig. 3 (Sim-Fleet validation) | `python -m ltn.sim.validate` | `paper/figures/simval/sim_vs_real.json` |
| Table II (data inventory) | `ltn/paper_gen/tables.py` (`t1_data`) | data lake |
| Table III, Fig. 4 (`F3_crps.pdf`) | `tables.py` (`t2_main`), `figures.py` (F3) | `runs/E1a`, `runs/E1b` |
| Ten-seed check of the E1a arms (Sec. V-A, V-G) | `python -m ltn.exp.run --split e1a --model <model> --seed <5–9> --tuned --exp E1a-extra`; the −Hamiltonian arm as in `scripts/e2_noH.sh` | `runs/E1a-extra` (seeds 5–9; seeds 0–4 are in `runs/E1a` and `runs/E2-e1a`); `runs/HWCHECK-e1a` (stored seeds re-run on the machine used for seeds 5–9; `docs/PLAN_DELTA.md`, DEV-29) |
| Table IV, Fig. 5 (`F4_ablations.pdf`) | `tables.py` (`t3_ablations`), `figures.py` (F4) | `runs/E2-e1a`, `runs/E2-e1b` |
| Table V (classical reliability models) | `ltn/baselines/reliability.py` through `ltn.exp.run` | `runs/E1a/{gp,kalman,hmm,wiener,gamma}`, `runs/tuning` |
| Table VI (matched pretraining) | `python -m ltn.exp.e4 --part a --base --seeds 5 --models ltn,lstm,tgn,transformer` | `runs/E4a-base` |
| Table VII (E8) | `ltn/exp/e8.py`, `tables.py` (`t4_e8`) | `runs/E8` |
| Tables VIII–X (reliability endpoints) | `scripts/reliability_endpoints.py` | `runs/RELIABILITY`, `paper/tables/T6_reliability.tex` |
| Table XI (state sufficiency), carry-free arm | `scripts/carry_ablation.py` | `runs/CARRY` |
| Fig. 1 | `figures.py` (F1) | data lake |
| Fig. 2 | `figures.py` (F2) | none |
| Fig. 6 (`F5_transfer.pdf`) | `figures.py` (F5) | `runs/E4a`, `runs/E4b`, `runs/E5` |
| Fig. 7 (`F6_regret.pdf`) | `figures.py` (F6) | `runs/E6a` |
| Fig. 8 (`F7_event.pdf`) | `figures.py` (F7) | `runs/E7`, `runs/E7-v2` |
| Generator spectra (Sec. V-A) | `scripts/spectra.py` | `runs/SPECTRA` |
| Computational cost (Sec. V-B) | `scripts/benchmark_cost.py` | `runs/COST` |
| Change-point detectors (Sec. V-C) | `scripts/e3_changepoint.py` | `runs/E3/changepoint_auroc.json` |
| LTN-v2 and its dimension sweep (Sec. V-G) | `ltn.exp.run --model ltn_v2` | `runs/E1a/ltn_v2`, `runs/E1b/ltn_v2`, `runs/DSWEEP-v2-e1a` |

## License

MIT; see `LICENSE`.
