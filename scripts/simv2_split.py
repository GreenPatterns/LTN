"""The E1b split rebuilt on a Sim-Fleet v2 lake, for the E7 rerun (see scripts/e7_simv2.sh).

    python scripts/simv2_split.py --lake <work>/data/uts --out <work>/splits/e1b.json

Same experiment definition as the sealed splits/e1b.json (observables, holdout
devices, train/validation boundaries), except that v2 is sampled three times a day:
grid_hours = 8, so the 1/3/7/14-day horizons become 3/9/21/42 steps. Normalization
statistics come from v2's own training region, computed exactly as
ltn.eval.splits.build_all computes them. The output is not a sealed split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ltn.data.uts import read_uts
from ltn.eval.splits import E1B_OBSERVABLES, SIM_HOLDOUT_DEVICES, _norm_stats, _temporal_bounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, required=True, help="lake holding the v2 fleet as source=sim")
    ap.add_argument("--sealed", type=Path, default=Path("splits/e1b.json"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sealed = json.loads(args.sealed.read_text())
    sim = read_uts(args.lake, sources=["sim"])
    sim_in = sim[~sim["device"].isin(SIM_HOLDOUT_DEVICES)]
    tr_end, va_end = _temporal_bounds(sim_in, E1B_OBSERVABLES)
    if (tr_end, va_end) != (sealed["train_end"], sealed["val_end"]):
        raise SystemExit(f"boundaries {tr_end}, {va_end} differ from the sealed E1b split")
    split = {"experiment": "E1b", "sources": ["sim"], "observables": E1B_OBSERVABLES,
             "horizons_steps": [3, 9, 21, 42], "grid_hours": 8.0,
             "holdout_devices": SIM_HOLDOUT_DEVICES, "train_end": tr_end, "val_end": va_end,
             "norm_stats": _norm_stats(sim_in, E1B_OBSERVABLES, tr_end),
             "note": "Sim-Fleet v2 (recalibration-reset refit), 3 snapshots/day; not a sealed split"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=1) + "\n")
    print(f"wrote {args.out} (train_end {tr_end}, val_end {va_end})")


if __name__ == "__main__":
    main()
