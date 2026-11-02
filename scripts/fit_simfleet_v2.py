"""Refit the synthetic fleet with the recalibration-reset process (Sim-Fleet v2).

    python scripts/fit_simfleet_v2.py --out runs/SIMFLEET-v2/simfleet_v2_config.json

Starts from the registered v1 fit (splits/simfleet_v1_config.json) and changes only
what the recalibration-reset process needs:

* ``recal_groups = 6``: DAQEC carries six distinct calibration timestamps per
  snapshot, and the pooled interval between them is the 1.31 h of Table I;
* ``reset_hour = 23``: a daily full recalibration sweep, which resets a
  within-day log-T1 decline (the one-day period in DAQEC's T1 autocorrelation);
* degradation episodes off: the 14-day DAQEC record shows no multi-day T1
  decline (its below-median runs last 5.5 h on average and reach 0.35 at most),
  and over 180 days the episodes dominate the fleet's lag-1 autocorrelation.

It then refits v1's ten knobs plus the reset decay, the T1 and Tphi medians and the
two-qubit error floor with ``ltn.sim.fit.fit_fleet_config`` at its default
evaluation settings (2 devices x 60 days, 3 snapshots/day, fixed seed), exactly as
v1 was fitted. Validate the result with

    python -m ltn.sim.validate --sim-config runs/SIMFLEET-v2/simfleet_v2_config.json \
        --sim-label simfleet-v2 --out runs/SIMFLEET-v2 --tex runs/SIMFLEET-v2
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ltn.data.uts import read_uts
from ltn.sim.fit import DEFAULT_KNOBS, fit_fleet_config
from ltn.sim.fleet import FleetConfig

V2_KNOBS = tuple(DEFAULT_KNOBS) + ("reset_decay_per_h", "t1_median_us", "tphi_median_us", "err2q_a2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, default=Path("data/uts"))
    ap.add_argument("--v1", type=Path, default=Path("splits/simfleet_v1_config.json"))
    ap.add_argument("--out", type=Path, default=Path("runs/SIMFLEET-v2/simfleet_v2_config.json"))
    args = ap.parse_args()
    real = read_uts(args.lake, sources=["daqec"])
    v1 = FleetConfig.from_json(args.v1.read_text())
    base = replace(v1, recal_groups=6, reset_hour=23.0, reset_decay_per_h=0.02,
                   degrade_rate_per_day=0.0)
    fitted = fit_fleet_config(real, base, V2_KNOBS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fitted.to_json(args.out)
    print(f"wrote {args.out}")
    for k in V2_KNOBS:
        print(f"  {k:20s} v1 {getattr(v1, k):.6g} -> v2 {getattr(fitted, k):.6g}")


if __name__ == "__main__":
    main()
