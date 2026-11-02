#!/usr/bin/env bash
# LTN-v2 lane GPU2: wait for v2 E1 checkpoints -> E7-v2 (calibration + gap/recal
# event studies with the Born-rule sigma — the mechanism Tier 1 exists to repair)
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

until [ -f runs/E1a/ltn_v2/0/model.pt ] && [ -f runs/E1b/ltn_v2/0/model.pt ]; do sleep 120; done
LOG "E7-v2"; $PY -m ltn.exp.e7 --gpu 0 --model ltn_v2 \
  --ckpt-e1a runs/E1a/ltn_v2/0/model.pt --ckpt-e1b runs/E1b/ltn_v2/0/model.pt \
  --out runs/E7-v2 || LOG "FAIL E7-v2"
LOG "V2-GPU2-DONE"
