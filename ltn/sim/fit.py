"""Moment-matching calibration of the synthetic fleet against real UTS data [DEV-2].

``target_stats(uts_df)`` reduces a UTS event table to a named vector of scalar
statistics (per-observable moments/quantiles, lag-1/lag-3 autocorrelation,
claimed-vs-measured discrepancy, recalibration cadence from cal_ts deltas,
day-over-day persistence MAE). ``fit_fleet_config`` tunes ~8-12 scalar
FleetConfig knobs with :func:`scipy.optimize.least_squares` so that a small
fixed-seed simulation (default 2 devices x 60 days) reproduces the real
statistics. The evaluation seed is FIXED, so the objective is a deterministic
function of the knobs. validate.py reuses ``target_stats`` for its report.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .fleet import FleetConfig, FleetSim

#: observables whose marginal moments are matched
FIT_OBSERVABLES = ("t1_us", "t2_us", "readout_error", "gate_error_2q")

#: default knobs tuned by fit_fleet_config (all strictly positive -> log-space)
DEFAULT_KNOBS = (
    "ou_theta_t1", "ou_sigma_t1",
    "tls_on_rate", "tls_off_rate",
    "recal_median_h",
    "claimed_noise", "probe_noise",
    "ou_sigma_tphi", "readout_sigma", "err_noise_rel",
)


def _autocorr(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    sa, sb = a.std(), b.std()
    if sa <= 0 or sb <= 0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _nanmean(vals: Sequence[float]) -> float:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _measured_channel(vis: pd.DataFrame) -> pd.DataFrame:
    """Rows of the preferred 'fresh measurement' channel: probe > delivered > claimed."""
    for ch in ("probe", "delivered", "claimed"):
        sub = vis[vis["channel"] == ch]
        if len(sub):
            return sub
    return vis


def target_stats(uts_df: pd.DataFrame) -> dict[str, float]:
    """Named scalar statistics of a UTS event table (truth channel excluded).

    Missing pieces (e.g. no claimed/probe pairing) yield NaN entries;
    ``fit_fleet_config`` only matches keys finite in BOTH real and sim.
    """
    out: dict[str, float] = {}
    vis = uts_df[uts_df["channel"] != "truth"]

    for obs in FIT_OBSERVABLES:
        v = vis.loc[vis["observable"] == obs, "value"].to_numpy(dtype=np.float64)
        if v.size:
            out[f"{obs}.mean"] = float(v.mean())
            out[f"{obs}.std"] = float(v.std())
            out[f"{obs}.q10"] = float(np.quantile(v, 0.10))
            out[f"{obs}.q50"] = float(np.quantile(v, 0.50))
            out[f"{obs}.q90"] = float(np.quantile(v, 0.90))
        else:
            for k in ("mean", "std", "q10", "q50", "q90"):
                out[f"{obs}.{k}"] = float("nan")

    # temporal structure of T1 on the freshest available channel
    meas = _measured_channel(vis)
    t1 = meas[meas["observable"] == "t1_us"]
    ac1, ac3, persist = [], [], []
    for _, g in t1.groupby(["device", "entity_id"], observed=True):
        g = g.sort_values("ts")
        x = g["value"].to_numpy(dtype=np.float64)
        if len(x) >= 12:
            ac1.append(_autocorr(x, 1))
            ac3.append(_autocorr(x, 3))
        daily = g.groupby(g["ts"].dt.floor("D"))["value"].mean().to_numpy(dtype=np.float64)
        if len(daily) >= 3 and daily.mean() > 0:
            persist.append(float(np.mean(np.abs(np.diff(daily))) / daily.mean()))
    out["t1.acf1"] = _nanmean(ac1)
    out["t1.acf3"] = _nanmean(ac3)
    out["t1.persist_mae"] = _nanmean(persist)

    # claimed-vs-measured relative discrepancy |claimed - probe| / claimed
    claimed = vis[vis["channel"] == "claimed"]
    other = pd.DataFrame()
    for ch in ("probe", "delivered"):
        other = vis[vis["channel"] == ch]
        if len(other):
            break
    if len(claimed) and len(other):
        keys = ["device", "entity_type", "entity_id", "observable", "ts"]
        m = claimed[keys + ["value"]].merge(
            other[keys + ["value"]], on=keys, suffixes=("_c", "_m"))
        if len(m):
            d = np.abs(m["value_c"] - m["value_m"]) / (np.abs(m["value_c"]) + 1e-12)
            out["disc.mean"] = float(d.mean())
            out["disc.p95"] = float(np.quantile(d, 0.95))
    out.setdefault("disc.mean", float("nan"))
    out.setdefault("disc.p95", float("nan"))

    # recalibration cadence from cal_ts deltas (hours)
    intervals: list[float] = []
    with_cal = vis.dropna(subset=["cal_ts"])
    for _, g in with_cal.groupby("device", observed=True):
        ct = np.sort(g["cal_ts"].astype("int64").unique())  # ns since epoch
        if len(ct) >= 2:
            diffs = np.diff(ct).astype(np.float64) / 3.6e12  # ns -> hours
            intervals.extend(diffs.tolist())
    out["recal.mean_h"] = _nanmean(intervals)
    return out


def stat_residual(
    sim_stats: dict[str, float], real_stats: dict[str, float]
) -> tuple[np.ndarray, list[str]]:
    """Normalized residual over keys finite in both stat dicts."""
    keys = [k for k in real_stats
            if np.isfinite(real_stats[k]) and np.isfinite(sim_stats.get(k, float("nan")))]
    res = np.array(
        [(sim_stats[k] - real_stats[k]) / (abs(real_stats[k]) + 1e-9) for k in keys],
        dtype=np.float64)
    return res, keys


def fit_fleet_config(
    real_uts: pd.DataFrame,
    base: FleetConfig,
    knobs: Sequence[str] = DEFAULT_KNOBS,
    *,
    eval_days: int = 60,
    eval_devices: int = 2,
    eval_topology: dict | None = None,
    eval_seed: int = 20260601,
    snapshots_per_day: int = 3,
    max_nfev: int = 16,
    diff_step: float = 0.25,
) -> FleetConfig:
    """Moment-match FleetConfig knobs to real telemetry.

    Each candidate is evaluated by simulating ``eval_devices`` small devices
    for ``eval_days`` days with the FIXED ``eval_seed`` and comparing the
    normalized ``target_stats`` vectors. Knobs are optimized in log-space
    (all are strictly positive rates/scales).
    """
    for k in knobs:
        if not hasattr(base, k):
            raise ValueError(f"unknown FleetConfig knob {k!r}")
        if getattr(base, k) <= 0:
            raise ValueError(f"knob {k!r} must be > 0 for log-space fitting")

    topo = eval_topology or {"kind": "linear", "n": 8}
    real = target_stats(real_uts)

    def make_cfg(x: np.ndarray) -> FleetConfig:
        upd = {k: float(np.exp(xi)) for k, xi in zip(knobs, x)}
        upd["topologies"] = [dict(topo) for _ in range(eval_devices)]
        return replace(base, **upd)

    def residual(x: np.ndarray) -> np.ndarray:
        cfg = make_cfg(x)
        events, _ = FleetSim(cfg, eval_seed).run(eval_days, snapshots_per_day)
        res, _ = stat_residual(target_stats(events), real)
        if res.size == 0:
            raise ValueError("no overlapping finite statistics between real and sim")
        return res

    x0 = np.log([getattr(base, k) for k in knobs])
    sol = least_squares(
        residual, x0,
        bounds=(x0 - 3.0, x0 + 3.0),
        diff_step=diff_step, max_nfev=max_nfev, method="trf")
    fitted = {k: float(np.exp(xi)) for k, xi in zip(knobs, sol.x)}
    return replace(base, **fitted)
