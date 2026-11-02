#!/usr/bin/env bash
# E1 campaign: Optuna tuning (50 trials/model, prereg budget) then 5-seed finals.
# Four lanes run concurrently: GPU0/GPU1/GPU2 + CPU. Logs: runs/campaign_logs/.
# Usage: bash scripts/campaign_e1.sh <lane>   with lane in {gpu0, gpu1, gpu2, cpu}
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
mkdir -p runs/campaign_logs
LOG() { echo "[$(date -u +%H:%M:%S)] $*"; }

tune()  { LOG "TUNE  $1 $2";  $PY -m ltn.exp.tune --split "$1" --model "$2" --trials 50 ${3:+--gpu 0} || LOG "TUNE-FAILED $1 $2"; }
final() { for s in 0 1 2 3 4; do LOG "FINAL $1 $2 seed$s"; $PY -m ltn.exp.run --split "$1" --model "$2" --seed "$s" --tuned || LOG "FINAL-FAILED $1 $2 $s"; done; }
final1() { LOG "FINAL $1 $2 seed0"; $PY -m ltn.exp.run --split "$1" --model "$2" --seed 0 ${3:-} || LOG "FINAL-FAILED $1 $2"; }

case "$1" in
  gpu0)
    export CUDA_VISIBLE_DEVICES=0
    tune e1a ltn gpu; tune e1b ltn gpu
    final e1a ltn; final e1b ltn
    ;;
  gpu1)
    export CUDA_VISIBLE_DEVICES=1
    tune e1a lstm gpu; tune e1b lstm gpu; tune e1a tgn gpu
    final e1a lstm; final e1b lstm; final e1a tgn
    ;;
  gpu2)
    export CUDA_VISIBLE_DEVICES=2
    tune e1a transformer gpu; tune e1b transformer gpu; tune e1b tgn gpu
    final e1a transformer; final e1b transformer; final e1b tgn
    ;;
  cpu)
    tune e1a ewma; tune e1b ewma; tune e1a arima; tune e1b arima
    final1 e1a persistence; final1 e1b persistence
    final1 e1a ewma --tuned; final1 e1b ewma --tuned
    final1 e1a arima --tuned; final1 e1b arima --tuned
    ;;
  *) echo "unknown lane $1"; exit 1;;
esac
LOG "LANE $1 COMPLETE"
