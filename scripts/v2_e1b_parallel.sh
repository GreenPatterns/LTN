#!/usr/bin/env bash
# LTN-v2 e1b lane, parallelized across all 3 GPUs. RESUMABLE: every phase skips
# work already on disk, so it can be relaunched after an interruption.
#
# Budget discipline: the pre-registered tuning budget is 50 trials TOTAL for the
# study, exactly as v1 got. Workers take explicit --worker-trials shares that sum
# to the remainder, and --trials 50 is a hard cap enforced inside tune.py, so no
# relaunch can push the study past 50 COMPLETE trials. Trials left RUNNING by a
# killed worker are zombies: they never finish, so they consume no budget.
# The only deviation from v1's tuning is that TPE runs concurrently rather than
# sequentially, which is marginally LESS sample-efficient (workers cannot see
# each other's in-flight results) -- it can only understate v2, never inflate it.
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

DONE=$($PY -c "
import optuna, warnings; warnings.filterwarnings('ignore')
try:
    s=optuna.load_study(study_name='e1b_ltn_v2', storage='sqlite:///runs/tuning/optuna_e1b_ltn_v2.db')
    print(len([t for t in s.trials if t.state.name=='COMPLETE']))
except Exception: print(0)")
LOG "TUNE e1b ltn_v2: $DONE/50 complete already"
if [ "$DONE" -lt 50 ]; then
  REM=$((50 - DONE))
  # split the remainder across 3 GPUs, ceil-divided so the shares cover it
  A=$(( (REM + 2) / 3 )); B=$(( (REM - A + 1) / 2 )); C=$(( REM - A - B ))
  LOG "  launching remaining $REM trials split $A/$B/$C across GPU0/1/2"
  for spec in "0 $A" "1 $B" "2 $C"; do
    set -- $spec
    [ "$2" -gt 0 ] || continue
    CUDA_VISIBLE_DEVICES=$1 $PY -m ltn.exp.tune --split e1b --model ltn_v2 \
      --trials 50 --worker-trials $2 --gpu 0 >> runs/campaign_logs/v2_e1b_tune_gpu$1.log 2>&1 &
  done
  wait
fi
# a killed worker may leave the JSON unwritten even though the study is full;
# re-run once with a zero-trial share to emit it from the completed study
[ -f runs/tuning/e1b_ltn_v2.json ] || $PY -m ltn.exp.tune --split e1b --model ltn_v2 \
  --trials 50 --worker-trials 0 --gpu 0 >> runs/campaign_logs/v2_e1b_tune_gpu0.log 2>&1
LOG "TUNE done: $($PY -c "import json;d=json.load(open('runs/tuning/e1b_ltn_v2.json'));print(d['n_trials'],'trials, best',round(d['value'],4),d['params'])")"

LOG "FINALS e1b ltn_v2 (5 seeds, parallel)"
for s in 0 1 2 3 4; do
  [ -f runs/E1b/ltn_v2/$s/results.jsonl ] && { LOG "  seed $s already done"; continue; }
  g=$((s % 3))
  CUDA_VISIBLE_DEVICES=$g $PY -m ltn.exp.run --split e1b --model ltn_v2 --seed $s --tuned \
    >> runs/campaign_logs/v2_e1b_final_s$s.log 2>&1 &
done
wait
LOG "FINALS done"

PARAMS=$($PY -c "import json;p=json.load(open('runs/tuning/e1b_ltn_v2.json'))['params'];print(' '.join(f'{k}={v}' for k,v in p.items() if k!='d'))")
LOG "DSWEEP e1b d in {5,6,8} x seeds {0,1,2} (parallel)"
i=0
for d in 5 6 8; do for s in 0 1 2; do
  i=$((i+1))
  [ -f runs/DSWEEP-v2-e1b/d$d/$s/results.jsonl ] && continue
  g=$(( (i-1) % 3 ))
  CUDA_VISIBLE_DEVICES=$g $PY -m ltn.exp.run --split e1b --model ltn_v2 --seed $s \
    --exp DSWEEP-v2-e1b --as d$d --cfg $PARAMS d=$d \
    >> runs/campaign_logs/v2_e1b_dsweep_d${d}_s$s.log 2>&1 &
done; done
wait
LOG "DSWEEP done"

LOG "E7-v2 (the mechanism test: does Born-rule sigma track state entropy)"
CUDA_VISIBLE_DEVICES=0 $PY -m ltn.exp.e7 --gpu 0 --model ltn_v2 \
  --ckpt-e1a runs/E1a/ltn_v2/0/model.pt --ckpt-e1b runs/E1b/ltn_v2/0/model.pt \
  --out runs/E7-v2 >> runs/campaign_logs/v2_e7.log 2>&1 || LOG "FAIL E7-v2"
LOG "V2-E1B-LANE-DONE"
