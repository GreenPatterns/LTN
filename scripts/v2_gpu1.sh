#!/usr/bin/env bash
# LTN-v2 lane GPU1: tune e1b -> 5-seed finals -> d-sweep falsifier on e1b
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

LOG "TUNE e1b ltn_v2"; $PY -m ltn.exp.tune --split e1b --model ltn_v2 --trials 50 --gpu 0 || LOG "TUNE-FAIL e1b ltn_v2"
for s in 0 1 2 3 4; do LOG "FINAL e1b ltn_v2 s$s"; $PY -m ltn.exp.run --split e1b --model ltn_v2 --seed $s --tuned || LOG "FAIL e1b ltn_v2 $s"; done

# d-sweep falsifier: e1b n_obs=5 => aligned d=6; sweep d in {5, 6, 8}.
PARAMS=$($PY -c "import json;p=json.load(open('runs/tuning/e1b_ltn_v2.json'))['params'];print(' '.join(f'{k}={v}' for k,v in p.items() if k!='d'))")
for d in 5 6 8; do for s in 0 1 2; do
  LOG "DSWEEP-v2 e1b d=$d s$s"; $PY -m ltn.exp.run --split e1b --model ltn_v2 --seed $s --exp DSWEEP-v2-e1b --as d$d --cfg $PARAMS d=$d || LOG "FAIL dsweep d$d $s"
done; done
LOG "V2-GPU1-DONE"
