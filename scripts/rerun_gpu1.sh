#!/usr/bin/env bash
# Re-run lane GPU1 (fixed channel): tune e1b LTN -> E1b finals -> E2-e1b -> E4b
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

LOG "TUNE e1b ltn"; $PY -m ltn.exp.tune --split e1b --model ltn --trials 50 --gpu 0 || LOG "TUNE-FAIL e1b ltn"
for s in 0 1 2 3 4; do LOG "FINAL e1b ltn s$s"; $PY -m ltn.exp.run --split e1b --model ltn --seed $s --tuned || LOG "FAIL e1b ltn $s"; done

PARAMS=$($PY -c "import json;p=json.load(open('runs/tuning/e1b_ltn.json'))['params'];print(' '.join(f'{k}={v}' for k,v in p.items()))")
run_abl(){ local M=$1 L=$2; shift 2; for s in 0 1 2 3 4; do LOG "E2 e1b $L s$s"; $PY -m ltn.exp.run --split e1b --model $M --seed $s --exp E2-e1b --as $L --cfg $PARAMS "$@" || LOG "FAIL $L $s"; done; }
run_abl ltn ltn
run_abl ltn_gaussian ltn_gaussian
run_abl ltn_ode ltn_ode
run_abl ltn ltn_nograph use_graph=0
run_abl ltn ltn_nophys physics_gamma=0

LOG "E4b"; $PY -m ltn.exp.e4 --part b --gpu 0 --seeds 5 || LOG "FAIL E4b"
LOG "GPU1-LANE-DONE"
