#!/usr/bin/env bash
# E2 ablation "-Hamiltonian": v1 with H forced to zero in the generator and NOTHING
# else changed (same tuned config, same parameter count -- the HermitianHead is still
# built, just inert). This is the controlled test of the variance diagnosis: the v2
# model removes H but also swaps the readout, the jump dictionary, the clock and d,
# so it cannot attribute the variance drop to H on its own. This can.
set -uo pipefail
cd /home/mucbp/PROJECTS/QuTen
PY=/home/mucbp/.conda/envs/qt/bin/python
export PYTHONUNBUFFERED=1
LOG(){ echo "[$(date -u +%H:%M:%S)] $*"; }

PARAMS=$($PY -c "import json;p=json.load(open('runs/tuning/e1a_ltn.json'))['params'];print(' '.join(f'{k}={v}' for k,v in p.items()))")
LOG "E2-e1a ltn_noH, 5 seeds, tuned v1 params: $PARAMS"
for s in 0 1 2 3 4; do
  [ -f runs/E2-e1a/ltn_noH/$s/results.jsonl ] && { LOG "  seed $s done"; continue; }
  g=$((s % 3))
  CUDA_VISIBLE_DEVICES=$g $PY -m ltn.exp.run --split e1a --model ltn --seed $s \
    --exp E2-e1a --as ltn_noH --cfg $PARAMS use_hamiltonian=0 \
    >> runs/campaign_logs/e2_noH_s$s.log 2>&1 &
done
wait
LOG "E2-noH DONE"
$PY - <<'PY'
import glob, numpy as np, pandas as pd
def sd(lab):
    v=[]
    for f in sorted(glob.glob(f'runs/E2-e1a/{lab}/*/results.jsonl')):
        d=pd.read_json(f,lines=True); d=d[d.ncrps.notna()]
        if len(d): v.append(d.ncrps.mean())
    return np.array(v)
for lab in ('ltn','ltn_noH'):
    v=sd(lab)
    if len(v): print(f'{lab:10s} mean={v.mean():.4f} median={np.median(v):.4f} sd={v.std(ddof=1):.4f} max={v.max():.4f} seeds={[round(x,4) for x in v]}')
PY
