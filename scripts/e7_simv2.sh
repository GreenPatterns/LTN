#!/usr/bin/env bash
# E7 rerun on Sim-Fleet v2, the recalibration-reset refit (docs/PLAN_DELTA.md, DEV-30).
#   bash scripts/e7_simv2.sh <work-dir> [gpu]        (PY=<python> to choose the interpreter)
# ltn.exp.e7 reads the lake from data/uts and the E1b split from splits/e1b.json,
# relative to the working directory. This script therefore builds a separate working
# directory in which source=sim is the v2 fleet and splits/e1b.json is the v2 split;
# DAQEC, every other split, the tuned configs and the published E1a checkpoints are
# linked from this repository unchanged. Nothing in this repository is modified.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
W=$1
GPU=${2:-0}
PY=${PY:-python}
export PYTHONPATH=$REPO
mkdir -p "$W/data/uts" "$W/splits" "$W/runs"
ln -sfn "$REPO/data/uts/source=daqec" "$W/data/uts/source=daqec"
for f in "$REPO"/splits/*.json; do [ "$(basename "$f")" = e1b.json ] || ln -sfn "$f" "$W/splits/"; done
ln -sfn "$REPO/runs/tuning" "$W/runs/tuning"
ln -sfn "$REPO/runs/E1a" "$W/runs/E1a"
cd "$W"

# 1. the v2 fleet at the DAQEC cadence (3 snapshots/day), same seed as v1
$PY -m ltn.sim.fleet --config "$REPO/runs/SIMFLEET-v2/simfleet_v2_config.json" \
    --days 180 --snapshots-per-day 3 --seed 20260712 --out data/uts --truth data/sim_truth
# 2. the E1b split on it (8 h grid)
$PY "$REPO/scripts/simv2_split.py" --lake data/uts --sealed "$REPO/splits/e1b.json" --out splits/e1b.json
# 3. seed-0 checkpoints under the E1b protocol and tuned configs, as E7 used
for m in ltn ltn_v2; do
  CUDA_VISIBLE_DEVICES=$GPU $PY -m ltn.exp.run --split e1b --model "$m" --seed 0 --tuned --exp E1b-simv2
done
# 4. E7 for both models; the E1a parts reuse the published E1a checkpoints
mkdir -p runs/E7-simv2 && cp "$REPO/runs/E7/e3_divergence_events.parquet" runs/E7-simv2/
CUDA_VISIBLE_DEVICES=$GPU $PY -m ltn.exp.e7 --gpu 0 --model ltn --out runs/E7-simv2 \
    --ckpt-e1a runs/E1a/ltn/0/model.pt --ckpt-e1b runs/E1b-simv2/ltn/0/model.pt
CUDA_VISIBLE_DEVICES=$GPU $PY -m ltn.exp.e7 --gpu 0 --model ltn_v2 --out runs/E7-v2-simv2 \
    --ckpt-e1a runs/E1a/ltn_v2/0/model.pt --ckpt-e1b runs/E1b-simv2/ltn_v2/0/model.pt
