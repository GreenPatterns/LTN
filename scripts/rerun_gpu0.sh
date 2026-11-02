#!/usr/bin/env bash
# Re-run lane GPU0 (fixed Lindblad channel): tune e1a LTN -> E1a finals -> E2-e1a -> E3 -> E4a -> E5
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

LOG "TUNE e1a ltn"; $PY -m ltn.exp.tune --split e1a --model ltn --trials 50 --gpu 0 || LOG "TUNE-FAIL e1a ltn"
for s in 0 1 2 3 4; do LOG "FINAL e1a ltn s$s"; $PY -m ltn.exp.run --split e1a --model ltn --seed $s --tuned || LOG "FAIL e1a ltn $s"; done

PARAMS=$($PY -c "import json;p=json.load(open('runs/tuning/e1a_ltn.json'))['params'];print(' '.join(f'{k}={v}' for k,v in p.items()))")
run_abl(){ local M=$1 L=$2; shift 2; for s in 0 1 2 3 4; do LOG "E2 e1a $L s$s"; $PY -m ltn.exp.run --split e1a --model $M --seed $s --exp E2-e1a --as $L --cfg $PARAMS "$@" || LOG "FAIL $L $s"; done; }
run_abl ltn ltn
run_abl ltn_gaussian ltn_gaussian
run_abl ltn_ode ltn_ode
run_abl ltn ltn_nograph use_graph=0
run_abl ltn ltn_nophys physics_gamma=0

LOG "E3"; $PY -m ltn.exp.e3 --gpu 0 --seeds 5 || LOG "FAIL E3"
LOG "E4a"; $PY -m ltn.exp.e4 --part a --gpu 0 --seeds 5 || LOG "FAIL E4a"
LOG "E5"; $PY -m ltn.exp.e5 --gpu 0 --seeds 5 || LOG "FAIL E5"
LOG "GPU0-LANE-DONE"
