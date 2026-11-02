#!/usr/bin/env bash
# Finish the LTN-v2 e1b lane on the GPUs that actually exist.
#
# GPU2 fell off the bus mid-campaign, so the original lane's `g=$((i % 3))` sharding
# silently pinned a third of its work to a device that no longer answers. This
# reschedules the survivors over the live GPUs, discovered rather than hardcoded, and
# skips anything already on disk.
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
NG=${#GPUS[@]}
LOG "live GPUs: ${GPUS[*]} (n=$NG)"

wait_for(){ while pgrep -f "ltn.exp.run" >/dev/null; do sleep 30; done; }

# 1) d-sweep arms still missing
i=0
for d in 5 6 8; do for s in 0 1 2; do
  f=runs/DSWEEP-v2-e1b/d$d/$s/results.jsonl
  [ -s "$f" ] && continue
  pgrep -f -- "--as d$d" >/dev/null && { LOG "d$d s$s already running"; continue; }
  g=${GPUS[$((i % NG))]}; i=$((i+1))
  LOG "DSWEEP d=$d s=$s -> GPU$g"
  PARAMS=$($PY -c "import json;p=json.load(open('runs/tuning/e1b_ltn_v2.json'))['params'];print(' '.join(f'{k}={v}' for k,v in p.items() if k!='d'))")
  CUDA_VISIBLE_DEVICES=$g $PY -m ltn.exp.run --split e1b --model ltn_v2 --seed $s \
    --exp DSWEEP-v2-e1b --as d$d --cfg $PARAMS d=$d \
    > runs/campaign_logs/v2_e1b_dsweep_d${d}_s$s.log 2>&1 &
  sleep 2
done; done
wait
LOG "DSWEEP done"

# 2) E7-v2: the mechanism test (needs both E1 checkpoints)
if [ ! -f runs/E7-v2/event_study.parquet ]; then
  if [ -f runs/E1a/ltn_v2/0/model.pt ] && [ -f runs/E1b/ltn_v2/0/model.pt ]; then
    LOG "E7-v2"
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY -m ltn.exp.e7 --gpu 0 --model ltn_v2 \
      --ckpt-e1a runs/E1a/ltn_v2/0/model.pt --ckpt-e1b runs/E1b/ltn_v2/0/model.pt \
      --out runs/E7-v2 > runs/campaign_logs/v2_e7.log 2>&1 || LOG "FAIL E7-v2"
  else
    LOG "SKIP E7-v2: missing checkpoints"
  fi
fi
LOG "V2-FINISH-DONE"
