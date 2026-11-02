# Pre-registration — LTN "Trust Decoheres"

**Status: FROZEN at tag `prereg-v1` (2026-07-13), before any experiment run.**
Deviations after the tag must be declared in the paper. The already-known deviations
from the master plan (data-access pivot) are in `PLAN_DELTA.md` [DEV-1..12] and are
incorporated here, not hidden.

Declared at freeze time:
- Pipeline *smoke tests* (2-epoch fits on DAQEC, `tests/test_train_smoke.py`,
  `tests/test_neural_baselines.py`) ran before this tag to verify code paths; no
  hyperparameters, metrics, or model selections were derived from them.
- Split files live in `splits/*.json` (SHA256-sealed); E1a horizons are steps
  {3, 9, 21} on the 8 h DAQEC grid = 1/3/7 days; E1b adds {42} = 14 days.
- Sim-fleet external validity at freeze: KS(real, sim) = 0.22 (T1), 0.29 (T2),
  0.21 (readout), 0.65 (2q gate error — weak, declared); claimed-vs-probe relative
  discrepancy mean 10.4% real vs 15.1% sim. E1b/E2 compare models *within* sim data,
  unaffected; E4b/E6a external validity carries this caveat.

**Amendment v1.1 (tag `prereg-v1.1`, before any campaign run):** E1b moved from an
8-hourly to a DAILY snapshot grid (180 steps; horizons 1/3/7/14 steps = 1/3/7/14 days;
`grid_hours: 24`). Reason: compute feasibility measured on a 2-epoch timing probe
(~2 h/run at 8-hourly × 50-trial budget × 5 seeds is infeasible on 3 GPUs); daily also
matches real IBM calibration cadence. Decided from timing probes only — no model
comparison results existed at amendment time. Tuning protocol for e1b (all models
identically): 50 Optuna trials on a frozen 4-device subset with epochs=20/patience=4;
final 5-seed runs on the full split with the tuned config.

## Hypotheses

- **H1 (dynamics).** Learned Lindblad drift beats equal-capacity (±10% trainable params)
  persistence/EWMA/ARIMA/LSTM/TGN/transformer baselines at probabilistic forecasting
  (CRPS) of device telemetry.
- **H2 (transfer).** (a) Pretraining on other devices/platforms yields few-shot
  adaptation to a held-out device; (b) zero-shot reliability *ranking* on a held-out
  platform (Google) is nontrivial.
- **H3 (trust).** Trust states with vacuity/dissonance decomposition are calibrated,
  anticipate claim-vs-delivery divergence, and convert to lower selection regret than
  calibration-greedy policies.

## Experiments, metrics, success bars

| Exp | Primary metric | Success bar | Kill/demote criterion |
|---|---|---|---|
| E1a (real, DAQEC) | CRPS at horizons 1/3/7 d | best CRPS on ≥2/4 observables, Wilcoxon p<0.05 (Holm) | LTN worse than persistence on all observables → H1 fails on real data |
| E1b (sim) | CRPS at 1/3/7/14 d | best on ≥3/5 observables, p<0.05 | same |
| E2 | ΔCRPS per ablation | each module cut costs performance | Lindblad≈ODE → H1 reframes on physics-γ + E7 interpretability (pre-registered) |
| E3 | AUROC divergence within h∈{1,2,3} snapshots; lead time | AUROC>0.75 on held-out device; lead-time 95% CI excludes 0 | AUROC<0.6 → H3 divergence claim dropped |
| E4a (real) | CRPS vs adaptation days k∈{0,1,3,7} | pretrained k=3 ≤ scratch k=7 | no benefit at any k → H2a fails |
| E4b (sim) | k∈{0,1,7,30} | pretrained k=7 ≤ scratch k=30 (≥4×) | same |
| E5 | Spearman ρ of predicted trust vs decoded logical error/round on Google | ρ>0.4 headline; 0.2–0.4 soft claim | ρ<0.2 → H2 demoted to few-shot only |
| E6a (sim) | cumulative regret | LTN-LCB < calibration-greedy, bootstrap CIs separated; LCB<mean | no separation → H3 operational claim demoted to directional |
| E6b (real, logged pairs) | regret (secondary) | directional, CI reported, no hard bar | n/a (declared secondary) |
| E7 | ECE/Brier/NLL; entropy-growth match during injected gaps | predicted vs empirical growth curves within CI band | mismatch reported honestly; F7 becomes diagnostic |
| E8 | RMSE/AUC on Bitcoin-OTC/Alpha | any improvement over vector-state control | null reported in appendix |

## Statistical protocol (binding)

5 seeds (0–4) for every stochastic model; deterministic baselines run once, paired
against each seed. Paired Wilcoxon signed-rank per device×observable(×horizon) cell;
Holm correction across all comparisons within an experiment; Cliff's δ effect sizes;
95% bootstrap CIs (10k resamples, seed 0) on headline numbers. Temperature scaling fit
on validation only.

## Tuning budgets (binding)

Optuna TPE, **50 trials per model** (persistence exempt: no hyperparameters), objective
= validation CRPS, tuning seed 0, tuned config then rerun on 5 evaluation seeds.
Identical feature sets and split files for all models. Equal-capacity: trainable
parameter counts within ±10% of LTN's (excluding per-node state tensors for all
stateful models symmetrically).

## Frozen splits

Committed under `splits/` with SHA256; E1a temporal 70/10/20 by date (test = final days);
E1b temporal + 4-device holdout; E3 3-fold device rotation; E4 device holdout; E5
platform holdout (Google never loaded during training — enforced by loader flag);
E6 replay windows; E8 temporal edge split. Normalization statistics computed on train
partitions only and stored inside the split files.

---

# POST-FREEZE ADDENDUM (2026-07-18)

**Nothing above this line has been altered.** Everything below is a record of what
happened *after* the freeze, appended so that the pre-registration and the actual
campaign can be compared without reading git history. Deviation IDs refer to
`PLAN_DELTA.md`; evidence rows are in `paper/ledger.md`.

## A. Freeze provenance (re-verified 2026-07-18)

The header above says the freeze is dated 2026-07-13; the git tag is
`prereg-v1` @ **2026-07-12T22:32:56-05:00** (= 2026-07-13T03:32:56Z). Both are correct
— the header is in UTC, the tag in local time. Amendment `prereg-v1.1` @
2026-07-12T22:49:29-05:00.

Re-verified against the filesystem:

| Check | Result |
|---|---|
| `sha256sum -c splits/SHA256SUMS` | **9/9 OK** |
| `python -m ltn.eval.splits --verify-hashes` | `splits: all hashes ok` |
| `splits/` tracked in git | yes, 10 files |
| Earliest run artifact of ANY kind | `runs/tuning/optuna_e1a_ewma.db`, 2026-07-12 23:02:17 — **12 min 48 s after** `prereg-v1.1` |
| Earliest `results.jsonl` | `runs/E1a/lstm/0/results.jsonl`, 2026-07-12 23:50:54 — **1 h 01 min after** `prereg-v1.1` |

So the freeze does precede every artifact, but the true margin is **12 minutes**, not an
hour. Quote the 23:02:17 figure, not the 23:50:54 one: the tighter bound is the honest
one, and anyone who runs `find runs -type f -printf '%T+ %p\n' | sort | head` will
find the tuning database first.

## B. Post-hoc work — EXPLORATORY, NOT PRE-REGISTERED

The v1 campaign closed at tag `results-final` (2026-07-13). Everything below was
designed **after** those results were known. It is reported as exploratory. It does not
inherit the protection of this pre-registration, and none of it retroactively rescues a
pre-registered hypothesis that was killed (H2b/L5, H3-divergence/L6, entropy
signature/L8 all stand as killed).

- **LTN-v2** (DEV-14) — Born-rule readout, unital jump dictionary with $H=0$, saturating
  clock. Motivated by a measured defect in v1: $\sigma$ was a function of log-horizon
  rather than of the state, so $\partial\sigma/\partial\gamma=0$ and the decoherence
  rates never received gradient. **Architecture selected after seeing v1's failures.**
- **$-$Hamiltonian ablation** (DEV-15) — `runs/E2-e1a/ltn_noH`, 5/5 seeds complete.
  A clean single-factor controlled ablation, but the *choice of factor* was made after
  observing the seed-variance blow-up. Attribution claim, not a confirmatory test.
- **E7-v2** (DEV-19) — pre-registered experiment, post-hoc model. The original E7 result
  stands.
- **d-sweep falsifier** (DEV-17) — declared for v2 *before* the sweep ran: NCRPS must
  move with $d$ (v1 was flat at 0.309 across $d\in\{2,4,8\}$, i.e. the state was dead
  weight). This is the one post-freeze item with a genuine advance prediction, so it is
  worth stating as such — but see C, the e1b arm is underpowered.

## C. Budget and completeness departures (declare these)

- **e1b LTN-v2 tuning: 49/50 trials** (DEV-18). `runs/tuning/e1b_ltn_v2.json`
  `n_trials: 49`, best 0.2988. e1a got the full 50 (best 0.2407). Under-budget, so it
  can only understate v2 — but the budget was declared binding above, so it is a
  deviation. Additionally, v2's TPE ran **concurrently** across workers where v1's ran
  sequentially; concurrent TPE is marginally less sample-efficient (workers cannot see
  in-flight results), which again can only understate v2.
- **e1b d-sweep: 6/9 cells** (DEV-17). `runs/DSWEEP-v2-e1b/d{5,6,8}/{0,1}` complete;
  **all three seed-2 cells empty**. Not random attrition — the launcher assigns GPUs by
  `g = (i-1) % 3` over the 9 jobs, so jobs 3/6/9, exactly the three seed-2 cells, went to
  the GPU that had died. **Any e1b d-sweep number is 2 seeds per $d$, not 3, and must be
  reported that way.** The e1a sweep is complete 9/9.

## D. Hardware incident, 2026-07-17 (DEV-16)

The machine dropped from 3 L40S to 2 mid-campaign; `nvidia-smi -L` now lists GPU0 and
GPU1 only. The failure was **silent at the job level**: the shell dispatched work to
`CUDA_VISIBLE_DEVICES=2`, torch then raised

```
RuntimeError: No CUDA GPUs are available
```

at the `.to(dev)` call. Affected workers left **0-byte logs** and **0-line
`results.jsonl` with no `status.yaml`** — a run directory that exists and looks
plausible until you count rows. Casualties: the three d-sweep cells in C, and one
in-flight Optuna trial.

Operational consequence: completeness must be checked by **counting rows**, not by
checking that a path exists. `make gpu-check` now refuses `GPU=2`. Note that
`scripts/campaign_e1.sh`, `scripts/v2_e1b_parallel.sh`, `scripts/v2_gpu2.sh` and
`scripts/rerun_gpu2.sh` still hard-code `CUDA_VISIBLE_DEVICES=2` and will silently drop
work if re-run unmodified — they are kept as the record of what was actually executed.

## E. Retraction (DEV-20)

Proposition 1 is the weakened, diagonal-$\rho$ statement. The stronger theorem, that the
evidence channel "reduces to Bayesian opinion updating" and is "exactly a subjective-logic
opinion $(b,d,u)$", is not claimed; unused source files carrying that wording were deleted
on 2026-07-18.
