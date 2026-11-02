#!/usr/bin/env bash
# E2 ablations under exp=E2 (never touching E1 leaderboard dirs): full LTN reference
# + equal-capacity variants x 5 seeds, LTN's tuned hyperparameters for shared dims.
# Usage: bash scripts/campaign_e2.sh <gpu-id> <split>
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export CUDA_VISIBLE_DEVICES=$1
SPLIT=$2
PARAMS=$($PY -c "import json; p=json.load(open('runs/tuning/${SPLIT}_ltn.json'))['params']; print(' '.join(f'{k}={v}' for k,v in p.items()))")
LOG() { echo "[$(date -u +%H:%M:%S)] $*"; }

run_abl() {  # run_abl <model> <label> [extra cfg...]
  local MODEL=$1 LABEL=$2; shift 2
  for s in 0 1 2 3 4; do
    LOG "E2/$SPLIT $LABEL seed$s"
    $PY -m ltn.exp.run --split $SPLIT --model $MODEL --seed $s \
        --exp "E2-$SPLIT" --as "$LABEL" --cfg $PARAMS "$@" || LOG "FAILED $LABEL $s"
  done
}

run_abl ltn ltn                        # full reference (same conditions as variants)
run_abl ltn_gaussian ltn_gaussian
run_abl ltn_ode ltn_ode
run_abl ltn ltn_nograph use_graph=0
run_abl ltn ltn_nophys physics_gamma=0
LOG "E2-$SPLIT-DONE"
