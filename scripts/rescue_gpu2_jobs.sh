#!/usr/bin/env bash
# GPU2 fell off the bus mid-campaign (nvidia-smi -L now lists 2 L40S, not 3), which
# silently hung every job pinned to CUDA_VISIBLE_DEVICES=2. This re-runs the two
# casualties on the surviving GPUs. Resumable: skips whatever already has results.
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

PARAMS=$($PY -c "import json;p=json.load(open('runs/tuning/e1a_ltn.json'))['params'];print(' '.join(f'{k}={v}' for k,v in p.items()))")

if [ ! -s runs/E2-e1a/ltn_noH/2/results.jsonl ]; then
  LOG "rerun E2-e1a ltn_noH seed 2 on GPU0"
  CUDA_VISIBLE_DEVICES=0 $PY -m ltn.exp.run --split e1a --model ltn --seed 2 \
    --exp E2-e1a --as ltn_noH --cfg $PARAMS use_hamiltonian=0 \
    > runs/campaign_logs/e2_noH_s2.log 2>&1 &
fi
if [ ! -s runs/E1b/ltn_v2/2/results.jsonl ]; then
  LOG "rerun E1b ltn_v2 seed 2 on GPU1"
  CUDA_VISIBLE_DEVICES=1 $PY -m ltn.exp.run --split e1b --model ltn_v2 --seed 2 --tuned \
    > runs/campaign_logs/v2_e1b_final_s2.log 2>&1 &
fi
wait
LOG "RESCUE DONE"
$PY - <<'PY'
import glob, numpy as np, pandas as pd
for lab in ('ltn','ltn_noH'):
    v=[]
    for f in sorted(glob.glob(f'runs/E2-e1a/{lab}/*/results.jsonl')):
        d=pd.read_json(f,lines=True)
        if 'ncrps' in d.columns and len(d): v.append(d[d.ncrps.notna()].ncrps.mean())
    v=np.array(v)
    if len(v): print(f'{lab:9s} n={len(v)} mean={v.mean():.4f} med={np.median(v):.4f} sd={v.std(ddof=1):.4f} max={v.max():.4f}')
PY
