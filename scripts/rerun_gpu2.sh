#!/usr/bin/env bash
# Re-run lane GPU2 (fixed channel): E8 -> d-ablation (M12) -> wait for E1 checkpoints -> E6a,E6b,E7
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

LOG "E8"; $PY -m ltn.exp.e8 --dataset both --seeds 5 || LOG "FAIL E8"

# M12: d in {2,4,8} sensitivity on e1b (waits for e1b tuning from GPU1 lane)
until [ -f runs/tuning/e1b_ltn.json ]; do sleep 30; done
PARAMS=$($PY -c "import json;p=json.load(open('runs/tuning/e1b_ltn.json'))['params'];print(' '.join(f'{k}={v}' for k,v in p.items() if k!='d'))")
for d in 2 4 8; do for s in 0 1 2; do
  LOG "DABL e1b d=$d s$s"; $PY -m ltn.exp.run --split e1b --model ltn --seed $s --exp DABL-e1b --as d$d --cfg $PARAMS d=$d || LOG "FAIL dabl d$d $s"
done; done

# E6/E7 need the fresh E1 checkpoints (seed 0)
until [ -f runs/E1a/ltn/0/model.pt ] && [ -f runs/E1b/ltn/0/model.pt ]; do sleep 60; done
LOG "E6 (sim+logged)"; $PY -m ltn.exp.e6 --world both --gpu 0 --seeds 5 || LOG "FAIL E6"
LOG "E7"; $PY -m ltn.exp.e7 --gpu 0 || LOG "FAIL E7"
LOG "GPU2-LANE-DONE"
