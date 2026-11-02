#!/usr/bin/env bash
# LTN-v2 lane GPU0: tune e1a -> 5-seed finals -> d-sweep falsifier on e1a
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

LOG "TUNE e1a ltn_v2"; $PY -m ltn.exp.tune --split e1a --model ltn_v2 --trials 50 --gpu 0 || LOG "TUNE-FAIL e1a ltn_v2"
for s in 0 1 2 3 4; do LOG "FINAL e1a ltn_v2 s$s"; $PY -m ltn.exp.run --split e1a --model ltn_v2 --seed $s --tuned || LOG "FAIL e1a ltn_v2 $s"; done

# d-sweep falsifier (prereg: NCRPS must MOVE with d; v1 was flat at 0.309).
# e1a n_obs=4 => aligned d=5; sweep d in {4, 5, 8} (BornReadout needs d >= n_obs).
PARAMS=$($PY -c "import json;p=json.load(open('runs/tuning/e1a_ltn_v2.json'))['params'];print(' '.join(f'{k}={v}' for k,v in p.items() if k!='d'))")
for d in 4 5 8; do for s in 0 1 2; do
  LOG "DSWEEP-v2 e1a d=$d s$s"; $PY -m ltn.exp.run --split e1a --model ltn_v2 --seed $s --exp DSWEEP-v2-e1a --as d$d --cfg $PARAMS d=$d || LOG "FAIL dsweep d$d $s"
done; done
LOG "V2-GPU0-DONE"
