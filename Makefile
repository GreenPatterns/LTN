# LTN local CI + campaign driver. GitHub Actions replaced by these targets [DEV-6].
PY := /home/mucbp/.conda/envs/qt/bin/python
TEX := conda run -n tex

# Models in the E1 leaderboard, and the subset that gets an Optuna budget
# (persistence is exempt: it has no hyperparameters).
MODELS      := persistence ewma arima lstm tgn transformer ltn
TUNE_MODELS := ewma arima lstm tgn transformer ltn
SEEDS       := 0 1 2 3 4
TRIALS      := 50

# This box has TWO usable L40S (indices 0 and 1). A third GPU failed on 2026-07-17 and
# CUDA_VISIBLE_DEVICES=2 now dies with "RuntimeError: No CUDA GPUs are available" AFTER the
# scheduler has already handed the job away, which silently drops runs (see PLAN_DELTA
# DEV-16). Targets here never pin a device; `gpu-check` fails loudly if 2 is requested.
GPU ?= 0

.PHONY: env-check gpu-check test test-all ci data sim-data splits splits-verify tune baselines
.PHONY: tables figures paper repro repro-fast clean e1 e1a e1b e2 e3 e4 e5 e6 e7 e8

# Campaign stages consume each other's artifacts; never interleave them.
.NOTPARALLEL:

env-check:
	$(PY) scripts/verify_env.py

test:
	$(PY) -m pytest -m "not slow and not data"

test-all:
	$(PY) -m pytest

ci: test
	$(PY) -m ltn.eval.splits --verify-hashes

data:
	bash scripts/download_data.sh
	$(PY) -m ltn.data.build_lake

sim-data:
	$(PY) -m ltn.sim.fleet --config splits/simfleet_v1_config.json --out data/uts --truth data/sim_truth

# DESTRUCTIVE: rebuilds the split files and re-seals SHA256SUMS. The frozen splits are a
# pre-registration artifact (tags prereg-v1 / prereg-v1.1) and are committed, so a
# reproduction must VERIFY them, not regenerate them -- that is what `splits-verify` and
# `repro` do. Only run this target when deliberately cutting a new pre-registration.
splits:
	$(PY) -m ltn.eval.splits --build --freeze

splits-verify:
	$(PY) -m ltn.eval.splits --verify-hashes
	cd splits && sha256sum -c SHA256SUMS

gpu-check:
	@test "$(GPU)" != "2" || { echo "GPU 2 is DEAD (failed 2026-07-17); use GPU=0 or GPU=1"; exit 1; }
	@$(PY) -c "import torch,sys; n=torch.cuda.device_count(); print(f'visible CUDA devices: {n}'); sys.exit(0 if int('$(GPU)')<n else 1)" \
	  || { echo "GPU=$(GPU) is not visible to torch"; exit 1; }

# Optuna tuning at the pre-registered budget: $(TRIALS) trials per model per split,
# objective = validation CRPS, tuning seed 0. Writes runs/tuning/<split>_<model>.json.
tune: gpu-check
	@for s in e1a e1b; do for m in $(TUNE_MODELS); do \
	  echo "== tune $$s $$m =="; \
	  $(PY) -m ltn.exp.tune --split $$s --model $$m --trials $(TRIALS) --gpu $(GPU) || exit 1; \
	done; done

# 5-seed finals for every model on both E1 splits, using the tuned configs from `make tune`.
# persistence is deterministic and has no tuned config, so it runs once, untuned.
e1a e1b: gpu-check
	@$(PY) -m ltn.exp.run --split $@ --model persistence --seed 0 || exit 1
	@for m in $(filter-out persistence,$(MODELS)); do for s in $(SEEDS); do \
	  echo "== $@ $$m seed $$s =="; \
	  $(PY) -m ltn.exp.run --split $@ --model $$m --seed $$s --tuned || exit 1; \
	done; done

e1: e1a e1b

# Historical alias: the E1 leaderboard IS the baseline comparison.
baselines: e1

# E2 ablations run under exp=E2-<split> so they never touch the E1 leaderboard dirs.
e2:
	bash scripts/campaign_e2.sh $(GPU) e1a
	bash scripts/campaign_e2.sh $(GPU) e1b

e3: gpu-check
	$(PY) -m ltn.exp.e3 --gpu $(GPU)

# E4 has two pre-registered parts: (a) real few-shot, (b) sim cold-start.
e4: gpu-check
	$(PY) -m ltn.exp.e4 --part a --gpu $(GPU)
	$(PY) -m ltn.exp.e4 --part b --gpu $(GPU)

e5: gpu-check
	$(PY) -m ltn.exp.e5 --gpu $(GPU)

e6: gpu-check
	$(PY) -m ltn.exp.e6 --world both --gpu $(GPU)

e7: gpu-check
	$(PY) -m ltn.exp.e7 --gpu $(GPU)

e8:
	$(PY) -m ltn.exp.e8

# Auto-generate table fragments under paper/tables/ (source of record for the numbers
# inlined into the self-contained paper/ltn.tex; run to verify the paper matches runs/).
tables:
	$(PY) -m ltn.paper_gen.tables

figures:
	$(PY) -m ltn.paper_gen.figures

# paper/ltn.tex is a single self-contained submission file: all prose, tables, and the
# bibliography are inlined; only the figures/ images are external. Build needs figures only.
paper: figures
	cd paper && ~/.local/bin/tectonic -X compile ltn.tex --outdir build

# One-command reproduction from public data. Ordered explicitly via recursive $(MAKE)
# rather than as a prerequisite list, because every stage consumes the previous stage's
# artifacts and a parallel `make -j repro` would otherwise interleave them.
#
# COST: this is a multi-day GPU campaign on 2x L40S, not a smoke test -- tuning alone is
# 12 studies x $(TRIALS) trials, and single E1b seeds have taken >2 h. `make repro-fast`
# is the quick check that the pipeline is sound without paying for the campaign.
repro:
	$(MAKE) env-check
	$(MAKE) data
	$(MAKE) sim-data
	$(MAKE) splits-verify
	$(MAKE) test-all
	$(MAKE) tune
	$(MAKE) e1
	$(MAKE) e2
	$(MAKE) e3
	$(MAKE) e4
	$(MAKE) e5
	$(MAKE) e6
	$(MAKE) e7
	$(MAKE) e8
	$(MAKE) tables
	$(MAKE) figures
	$(MAKE) paper

# Cheap end-to-end verification against the committed evidence: checks the environment,
# the split seals, the test suite, and that every table/figure in the paper regenerates
# from the tracked runs/**/results.jsonl. No training.
repro-fast: env-check ci tables figures paper

clean:
	rm -rf paper/build .pytest_cache
