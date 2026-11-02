#!/usr/bin/env bash
# E7-v2: the mechanism test. Does forcing sigma through the state (Born readout) plus
# a unital drift make predictive uncertainty track error under evidence starvation --
# the signature v1 failed, and failed with entropy running backwards?
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export PYTHONUNBUFFERED=1
echo "[$(date -u +%H:%M:%S)] E7-v2 start"
CUDA_VISIBLE_DEVICES=0 $PY -m ltn.exp.e7 --gpu 0 --model ltn_v2 \
  --ckpt-e1a runs/E1a/ltn_v2/0/model.pt --ckpt-e1b runs/E1b/ltn_v2/0/model.pt \
  --out runs/E7-v2
echo "[$(date -u +%H:%M:%S)] E7-v2 done"
