"""Sim-vs-real diagnostics for the synthetic fleet [DEV-2] — paper appendix artifact.

``sim_vs_real_report(real_uts, sim_uts, out_dir)`` compares model-visible
telemetry distributions between a real UTS lake and a synthetic one:

* two-sample KS statistic + p-value per observable,
* lag-1..10 autocorrelation overlays of the T1 series,
* claimed-vs-measured relative-discrepancy histograms,
* the full ``ltn.sim.fit.target_stats`` vectors for both datasets,
* an HONEST per-statistic validation table (real, sim, relative error, and
  whether the statistic was actually a term in the fit objective),
* extra validation axes that the moment-fit never targeted: cross-observable
  correlation, distribution tails (p95/p99), longer-lag autocorrelation,
  drift-episode duration/magnitude, and the missingness pattern.

Figures
-------
Two SINGLE-PAGE PDFs are written (they used to be one two-page file, whose
second page was silently dropped by ``\\includegraphics`` without ``page=``):

* ``sim_vs_real_marginals.pdf`` — four ECDF marginals with KS annotations,
* ``sim_vs_real_dynamics.pdf``  — T1 autocorrelation overlay + discrepancy
  histograms + drift-episode and missingness summaries.

Tables
------
With ``tex_dir=`` set, writes ``T5_simval.tex`` (fitted-target table) and
``T5b_simval_axes.tex`` (extra axes) as standalone ``tabular`` fragments.

Also writes ``sim_vs_real.json`` in ``out_dir`` and returns the report as a
dict (NaN -> None, so the JSON is strict). Deterministic: no randomness.

CLI
---
``python -m ltn.sim.validate`` regenerates every artifact from the registered
``splits/simfleet_v1_config.json`` (see ``main`` for the exact defaults).
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from .fit import (  # noqa: E402
    FIT_OBSERVABLES,
    _autocorr,
    _measured_channel,
    stat_residual,
    target_stats,
)

# validated 2-slot categorical palette (dataviz reference palette, light mode):
# blue = real, yellow = sim; legend + direct labels supply the contrast relief.
C_REAL = "#2a78d6"
C_SIM = "#eda100"
GRID = dict(color="#d9d8d4", linewidth=0.6)
INK = "#0b0b0b"
INK2 = "#52514e"

AUTOCORR_LAGS = tuple(range(1, 11))
#: lags reported in the LaTeX/JSON table (1..7)
TABLE_LAGS = tuple(range(1, 8))
#: fractional drop below a series' own median that counts as a drift episode
DRIFT_THRESHOLD = 0.10
#: series shorter than this are skipped by the temporal diagnostics
MIN_SERIES_LEN = 12

SERIES_KEYS = ["device", "entity_type", "entity_id"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _visible(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["channel"] != "truth"]


def _values(df: pd.DataFrame, obs: str) -> np.ndarray:
    vis = _visible(df)
    return vis.loc[vis["observable"] == obs, "value"].to_numpy(dtype=np.float64)


def _nanmean_list(vals) -> float:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _rel_err(real, sim) -> float:
    """Signed relative error (sim - real) / |real|; NaN unless both are finite.

    Accepts None (``_clean`` maps NaN -> None for the JSON artifact).
    """
    if real is None or sim is None:
        return float("nan")
    real, sim = float(real), float(sim)
    if not (np.isfinite(real) and np.isfinite(sim)) or real == 0.0:
        return float("nan")  # relative error is undefined against a zero baseline
    return float((sim - real) / abs(real))


def _preferred_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the freshest channel chosen INDEPENDENTLY PER OBSERVABLE.

    ``fit._measured_channel`` picks one channel globally, which silently drops
    observables that only exist on a staler channel (in DAQEC, ``gate_error_2q``
    is ``claimed``-only while T1/T2/readout have ``probe`` rows). For the
    cross-observable axes we want every observable represented, so the channel
    preference probe > delivered > claimed is applied per observable.
    """
    vis = _visible(df)
    parts = []
    for obs in FIT_OBSERVABLES:
        sub = vis[vis["observable"] == obs]
        if not len(sub):
            continue
        for ch in ("probe", "delivered", "claimed"):
            c = sub[sub["channel"] == ch]
            if len(c):
                parts.append(c)
                break
    if not parts:
        return vis.iloc[:0]
    return pd.concat(parts, ignore_index=True)


def _series_iter(df: pd.DataFrame, obs: str, *, deinterleave: bool = True):
    """Yield ``(key, values, ts_ns)`` for each per-entity time series of ``obs``.

    DAQEC records several rows at the SAME ``ts`` for one ``entity_id`` (the
    device-level entity carries one row per code distance 3/5/7). Sorting such
    a group by ``ts`` and treating it as one series -- which
    ``fit.target_stats`` does -- concatenates three distinct measurement
    streams and injects a spurious period-3 oscillation into the ACF. With
    ``deinterleave`` (the default) each group is split into ts-unique
    sub-series by rank within timestamp, ordered by the ``context`` string so
    the assignment is deterministic.
    """
    rows = _preferred_rows(df)
    rows = rows[rows["observable"] == obs]
    if not len(rows):
        return
    rows = rows.assign(_ctx=rows["context"].astype("string").fillna(""))
    for key, g in rows.groupby(SERIES_KEYS, observed=True):
        g = g.sort_values(["ts", "_ctx"], kind="mergesort")
        vals = g["value"].to_numpy(dtype=np.float64)
        ts = g["ts"].astype("int64").to_numpy()
        if deinterleave:
            rank = g.groupby("ts", observed=True).cumcount().to_numpy()
        else:
            rank = np.zeros(len(g), dtype=int)
        for r in range(int(rank.max()) + 1 if len(rank) else 0):
            sel = np.flatnonzero(rank == r)
            if len(sel) < MIN_SERIES_LEN:
                continue
            sel = sel[np.argsort(ts[sel], kind="mergesort")]
            yield (tuple(key) + (r,)), vals[sel], ts[sel]


def _clean(obj):
    """NaN/inf -> None recursively so the JSON artifact is strict."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


# --------------------------------------------------------------------------
# existing diagnostics (unchanged numerics)
# --------------------------------------------------------------------------
def _autocorr_curve(df: pd.DataFrame) -> list[float]:
    """Mean lag-k autocorrelation (k=1..10) of per-entity T1 series.

    Kept BIT-IDENTICAL to the frozen artifact: groups by (device, entity_id)
    exactly as ``fit.target_stats`` does, interleaving included. The
    de-interleaved counterpart is reported separately by
    ``_autocorr_curve_deinterleaved``.
    """
    vis = _visible(df)
    t1 = _measured_channel(vis)
    t1 = t1[t1["observable"] == "t1_us"]
    curves = []
    for _, g in t1.groupby(["device", "entity_id"], observed=True):
        x = g.sort_values("ts")["value"].to_numpy(dtype=np.float64)
        if len(x) >= max(AUTOCORR_LAGS) + 2:
            curves.append([_autocorr(x, k) for k in AUTOCORR_LAGS])
    if not curves:
        return [float("nan")] * len(AUTOCORR_LAGS)
    return [float(v) for v in np.nanmean(np.asarray(curves), axis=0)]


def _autocorr_curve_deinterleaved(df: pd.DataFrame) -> list[float]:
    """Mean lag-k autocorrelation over ts-unique (de-interleaved) T1 series."""
    curves = []
    for _, x, _ts in _series_iter(df, "t1_us"):
        if len(x) >= max(AUTOCORR_LAGS) + 2:
            curves.append([_autocorr(x, k) for k in AUTOCORR_LAGS])
    if not curves:
        return [float("nan")] * len(AUTOCORR_LAGS)
    return [float(v) for v in np.nanmean(np.asarray(curves), axis=0)]


def _discrepancies(df: pd.DataFrame) -> np.ndarray:
    """|claimed - measured| / |claimed| over matched rows."""
    vis = _visible(df)
    claimed = vis[vis["channel"] == "claimed"]
    other = pd.DataFrame()
    for ch in ("probe", "delivered"):
        other = vis[vis["channel"] == ch]
        if len(other):
            break
    if not len(claimed) or not len(other):
        return np.empty(0, dtype=np.float64)
    keys = ["device", "entity_type", "entity_id", "observable", "ts"]
    m = claimed[keys + ["value"]].merge(other[keys + ["value"]], on=keys, suffixes=("_c", "_m"))
    return (np.abs(m["value_c"] - m["value_m"]) / (np.abs(m["value_c"]) + 1e-12)).to_numpy()


# --------------------------------------------------------------------------
# new validation axes
# --------------------------------------------------------------------------
def tail_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """p95 / p99 and the shape-free tail index p99/p50 per observable."""
    out: dict[str, dict[str, float]] = {}
    for obs in FIT_OBSERVABLES:
        v = _values(df, obs)
        if v.size:
            p50 = float(np.quantile(v, 0.50))
            p95 = float(np.quantile(v, 0.95))
            p99 = float(np.quantile(v, 0.99))
            out[obs] = {"p95": p95, "p99": p99,
                        "p99_over_p50": p99 / p50 if p50 else float("nan"),
                        "n": int(v.size)}
        else:
            out[obs] = {"p95": float("nan"), "p99": float("nan"),
                        "p99_over_p50": float("nan"), "n": 0}
    return out


def cross_observable_corr(df: pd.DataFrame) -> dict[str, dict]:
    """Within-device Pearson correlation between device-snapshot observable means.

    Observables live on different entity types (qubits vs couplers in the sim,
    subgraphs vs device in DAQEC), so they are first reduced to a per
    ``(device, ts)`` mean. Correlating within each device and then averaging
    avoids a between-device Simpson effect.
    """
    long = _preferred_rows(df)
    if not len(long):
        return {}
    agg = (long.groupby(["device", "ts", "observable"], observed=True)["value"]
           .mean().unstack("observable"))
    out: dict[str, dict] = {}
    for a, b in itertools.combinations(FIT_OBSERVABLES, 2):
        vals: list[float] = []
        if a in agg.columns and b in agg.columns:
            for _dev, g in agg.groupby(level="device", observed=True):
                x = g[a].to_numpy(dtype=np.float64)
                y = g[b].to_numpy(dtype=np.float64)
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() >= MIN_SERIES_LEN and x[m].std() > 0 and y[m].std() > 0:
                    vals.append(float(np.corrcoef(x[m], y[m])[0, 1]))
        out[f"{a}~{b}"] = {"pearson": _nanmean_list(vals), "n_devices": len(vals)}
    return out


def drift_episodes(df: pd.DataFrame, threshold: float = DRIFT_THRESHOLD) -> dict:
    """Duration/magnitude of T1 degradation episodes.

    An episode is a maximal run of consecutive snapshots in which T1 sits at
    least ``threshold`` below that series' own median. Durations are reported
    in HOURS (not snapshots) because the real and synthetic lakes need not
    share a sampling cadence.
    """
    durations: list[float] = []
    mags: list[float] = []
    n_series = n_samples = n_in = 0
    total_hours = 0.0
    for _key, x, ts in _series_iter(df, "t1_us"):
        med = float(np.median(x))
        if not np.isfinite(med) or med <= 0:
            continue
        dev = x / med - 1.0
        below = dev <= -threshold
        n_series += 1
        n_samples += len(x)
        n_in += int(below.sum())
        dt_h = float(np.median(np.diff(ts))) / 3.6e12 if len(ts) > 1 else float("nan")
        total_hours += float(ts[-1] - ts[0]) / 3.6e12
        i = 0
        while i < len(below):
            if below[i]:
                j = i
                while j + 1 < len(below) and below[j + 1]:
                    j += 1
                durations.append(float(ts[j] - ts[i]) / 3.6e12 + dt_h)
                mags.append(float(np.mean(np.abs(dev[i:j + 1]))))
                i = j + 1
            else:
                i += 1
    d = np.asarray(durations, dtype=np.float64)
    m = np.asarray(mags, dtype=np.float64)
    return {
        "threshold": float(threshold),
        "n_series": int(n_series),
        "n_episodes": int(d.size),
        "episodes_per_1000h": float(1000.0 * d.size / total_hours) if total_hours > 0 else float("nan"),
        "frac_samples_in_episode": float(n_in / n_samples) if n_samples else float("nan"),
        "mean_duration_h": float(d.mean()) if d.size else float("nan"),
        "median_duration_h": float(np.median(d)) if d.size else float("nan"),
        "p90_duration_h": float(np.quantile(d, 0.90)) if d.size else float("nan"),
        "mean_magnitude": float(m.mean()) if m.size else float("nan"),
        "max_magnitude": float(m.max()) if m.size else float("nan"),
        "total_observed_hours": float(total_hours),
        "durations_h": [float(v) for v in d],
    }


def missingness(df: pd.DataFrame) -> dict:
    """Missing-cell rate and burst structure of the observation grid.

    The reference grid is the LAKE-WIDE set of snapshot timestamps, clipped to
    each device's own observed span. Using the device's own timestamps instead
    would hide device-wide outages entirely (every entity drops out together,
    so the timestamp simply never appears), which is exactly the dominant
    dropout mode of the synthetic fleet. ``burstiness`` is the observed mean
    missing-run length divided by the mean run length expected under a
    memoryless (MCAR) process at the same rate, 1/(1-p); 1.0 means MCAR,
    > 1 means gaps arrive in bursts.
    """
    vis = _visible(df)
    if not len(vis):
        return {}
    full_grid = np.sort(pd.Series(vis["ts"].unique()).astype("int64").to_numpy())
    spans = {dev: (int(g["ts"].astype("int64").min()), int(g["ts"].astype("int64").max()))
             for dev, g in vis.groupby("device", observed=True)}
    grids = {dev: full_grid[(full_grid >= lo) & (full_grid <= hi)]
             for dev, (lo, hi) in spans.items()}
    total_cells = 0
    total_missing = 0
    runs: list[int] = []
    for (dev, _obs, _ch), g in vis.groupby(["device", "observable", "channel"], observed=True):
        grid = grids[dev]
        if not len(grid):
            continue
        ent_codes, ents = pd.factorize(g["entity_id"])
        row_ts = g["ts"].astype("int64").to_numpy()
        pos = np.searchsorted(grid, row_ts)
        ok = (pos < len(grid)) & (grid[np.clip(pos, 0, len(grid) - 1)] == row_ts)
        present = np.zeros((len(ents), len(grid)), dtype=bool)
        present[ent_codes[ok], pos[ok]] = True
        total_cells += present.size
        total_missing += int((~present).sum())
        for row in ~present:
            if not row.any():
                continue
            padded = np.concatenate(([False], row, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            runs.extend((edges[1::2] - edges[0::2]).tolist())
    r = np.asarray(runs, dtype=np.float64)
    p = total_missing / total_cells if total_cells else float("nan")
    expected = 1.0 / (1.0 - p) if np.isfinite(p) and p < 1.0 else float("nan")
    mean_run = float(r.mean()) if r.size else float("nan")
    return {
        "missing_frac": float(p),
        "n_cells": int(total_cells),
        "n_missing": int(total_missing),
        "n_gaps": int(r.size),
        "mean_gap_len": mean_run,
        "median_gap_len": float(np.median(r)) if r.size else float("nan"),
        "p95_gap_len": float(np.quantile(r, 0.95)) if r.size else float("nan"),
        "max_gap_len": float(r.max()) if r.size else float("nan"),
        "mcar_expected_gap_len": float(expected),
        "burstiness": float(mean_run / expected) if r.size and np.isfinite(expected) else float("nan"),
        "frac_gaps_len_ge2": float((r >= 2).mean()) if r.size else float("nan"),
    }


def recal_diagnostic(df: pd.DataFrame) -> dict:
    """Why ``recal.mean_h`` is not comparable across the two lakes.

    ``fit.target_stats`` derives ``recal.mean_h`` from diffs of the pooled set
    of DISTINCT ``cal_ts`` values per device. If a lake carries several distinct
    ``cal_ts`` per snapshot (DAQEC has one per code distance), that pooled diff
    measures cal_ts DENSITY, i.e. roughly cadence / multiplicity -- not the
    recalibration interval. This records both factors so the mismatch can be
    attributed.
    """
    vis = _visible(df)
    cad: list[float] = []
    mult: list[float] = []
    for _dev, g in vis.groupby("device", observed=True):
        ts = np.sort(pd.Series(g["ts"].unique()).astype("int64").to_numpy())
        if len(ts) >= 2:
            # MEAN, not median: recal.mean_h is itself a mean of cal_ts diffs,
            # and the DAQEC cadence is irregular (08:00/13:00/18:00 -> 5,5,14 h).
            cad.append(float(np.mean(np.diff(ts))) / 3.6e12)
        per = g.dropna(subset=["cal_ts"]).groupby("ts", observed=True)["cal_ts"].nunique()
        if len(per):
            mult.append(float(per.median()))
    cadence = _nanmean_list(cad)
    multiplicity = _nanmean_list(mult)
    return {
        "snapshot_cadence_h": cadence,
        "cal_ts_per_snapshot": multiplicity,
        "implied_recal_mean_h": (cadence / multiplicity
                                 if np.isfinite(cadence) and multiplicity else float("nan")),
    }


# --------------------------------------------------------------------------
# honest validation table
# --------------------------------------------------------------------------
def validation_table(real_stats: dict, sim_stats: dict) -> list[dict]:
    """Every target statistic: real, sim, relative error, in-fit-objective flag.

    The objective flag is derived from ``fit.stat_residual``, which is what the
    optimizer actually minimized: a key is a term iff it is finite in BOTH
    dicts. Nothing is filtered out for looking bad.
    """
    _res, obj_keys = stat_residual(sim_stats, real_stats)
    obj = set(obj_keys)
    rows = []
    for k in sorted(set(real_stats) | set(sim_stats)):
        rv = float(real_stats.get(k, float("nan")))
        sv = float(sim_stats.get(k, float("nan")))
        rows.append({
            "stat": k,
            "real": rv,
            "sim": sv,
            "rel_err": _rel_err(rv, sv),
            "in_fit_objective": k in obj,
            "sign_flip": bool(np.isfinite(rv) and np.isfinite(sv) and rv * sv < 0),
        })
    return rows


# --------------------------------------------------------------------------
# LaTeX emission
# --------------------------------------------------------------------------
def _tex_num(v: float, sig: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "--"
    return f"{v:.{sig}g}"


def _tex_pct(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "--"
    return f"{100.0 * v:+.1f}\\%"


def _tex_esc(s: str) -> str:
    return s.replace("_", "\\_")


def _write_t5(rows: list[dict], ks: dict, path: Path) -> None:
    """Fitted-target validation table (T5_simval.tex)."""
    out = [
        "% AUTO-GENERATED by ltn/sim/validate.py -- do not edit by hand.",
        "% Honest sim-vs-real validation of every fitted target statistic.",
        "% 'Obj.' = the statistic was a term in the least-squares fit objective",
        "% (fit.stat_residual keeps every key finite in both real and sim).",
        "\\begin{tabular}{lrrrc}",
        "\\toprule",
        "Statistic & Real (DAQEC) & Sim (simfleet-v1) & Rel.\\ err. & Obj. \\\\",
        "\\midrule",
    ]
    for r in rows:
        flag = "\\checkmark" if r["in_fit_objective"] else "--"
        name = _tex_esc(r["stat"])
        if r["sign_flip"]:
            name += "$^{\\ddagger}$"
        out.append(f"{name} & {_tex_num(r['real'])} & {_tex_num(r['sim'])} & "
                   f"{_tex_pct(r['rel_err'])} & {flag} \\\\")
    out.append("\\midrule")
    for obs in FIT_OBSERVABLES:
        k = ks.get(obs, {})
        st, pv = k.get("statistic"), k.get("pvalue")
        st_s = _tex_num(st) if st is not None else "--"
        if pv is None or not np.isfinite(float(pv)):
            pv_s = "p=\\text{--}"
        elif float(pv) < 1e-16:
            pv_s = "p<10^{-16}"
        else:
            pv_s = f"p={_tex_num(pv, 2)}"
        out.append(f"KS {_tex_esc(obs)} & \\multicolumn{{2}}{{c}}{{$D={st_s}$, ${pv_s}$}} "
                   f"& -- & -- \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(out) + "\n")


def _write_t5b(report: dict, path: Path) -> None:
    """Extra validation axes never targeted by the fit (T5b_simval_axes.tex)."""
    ax = report["extra_axes"]
    out = [
        "% AUTO-GENERATED by ltn/sim/validate.py -- do not edit by hand.",
        "% Validation axes OUTSIDE the moment-fit objective: none of these were",
        "% fitted, so they are an out-of-sample test of the generative model.",
        "% Correlation-valued rows (ACF, cross-observable) report an ABSOLUTE",
        "% difference; a percentage against a near-zero correlation is meaningless.",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Axis & Quantity & Real & Sim & Discrepancy \\\\",
        "\\midrule",
    ]

    def row(axis: str, quant: str, rv: float, sv: float, *, absolute: bool = False) -> None:
        if absolute:
            d = (float("nan") if rv is None or sv is None
                 else float(sv) - float(rv))
            disc = "--" if not np.isfinite(d) else f"$\\Delta={d:+.3f}$"
        else:
            disc = _tex_pct(_rel_err(rv, sv))
        out.append(f"{axis} & {quant} & {_tex_num(rv)} & {_tex_num(sv)} & {disc} \\\\")

    first = True
    for obs in FIT_OBSERVABLES:
        rr = ax["tails"]["real"].get(obs, {})
        ss = ax["tails"]["sim"].get(obs, {})
        for q in ("p95", "p99"):
            row("Tails" if first else "", f"{_tex_esc(obs)} {q}", rr.get(q), ss.get(q))
            first = False
    out.append("\\midrule")

    first = True
    for k in sorted(ax["cross_observable_corr"]["real"]):
        rv = ax["cross_observable_corr"]["real"][k]["pearson"]
        sv = ax["cross_observable_corr"]["sim"].get(k, {}).get("pearson")
        row("Cross-obs.\\ corr." if first else "", _tex_esc(k.replace("~", " vs ")),
            rv, sv, absolute=True)
        first = False
    out.append("\\midrule")

    first = True
    for lag in TABLE_LAGS:
        i = AUTOCORR_LAGS.index(lag)
        row("T1 ACF" if first else "", f"lag {lag}",
            report["autocorr"]["real"][i], report["autocorr"]["sim"][i], absolute=True)
        first = False
    out.append("\\midrule")

    de = ax["drift_episodes"]
    first = True
    for q, lbl in (("episodes_per_1000h", "episodes / 1000\\,h"),
                   ("mean_duration_h", "mean duration (h)"),
                   ("median_duration_h", "median duration (h)"),
                   ("mean_magnitude", "mean magnitude"),
                   ("frac_samples_in_episode", "frac.\\ time in episode")):
        row("Drift episodes" if first else "", lbl, de["real"].get(q), de["sim"].get(q))
        first = False
    out.append("\\midrule")

    ms = ax["missingness"]
    first = True
    for q, lbl in (("missing_frac", "missing fraction"),
                   ("mean_gap_len", "mean gap (snapshots)"),
                   ("p95_gap_len", "p95 gap (snapshots)"),
                   ("burstiness", "burstiness vs MCAR")):
        row("Missingness" if first else "", lbl, ms["real"].get(q), ms["sim"].get(q))
        first = False
    out.append("\\midrule")

    rc = ax["recal_diagnostic"]
    first = True
    for q, lbl in (("snapshot_cadence_h", "snapshot cadence (h)"),
                   ("cal_ts_per_snapshot", "distinct cal\\_ts per snapshot"),
                   ("implied_recal_mean_h", "implied recal.mean\\_h")):
        row("cal\\_ts audit" if first else "", lbl, rc["real"].get(q), rc["sim"].get(q))
        first = False
    out += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(out) + "\n")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.set_ylabel(ylabel, fontsize=8, color=INK2)
    ax.tick_params(labelsize=7, colors=INK2)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def _ecdf(ax: plt.Axes, v: np.ndarray, color: str, label: str) -> None:
    if v.size == 0:
        return
    x = np.sort(v)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, linewidth=2.0, label=label)


def _fig_marginals(real_uts, sim_uts, ks, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
    fig.suptitle("Sim-Fleet (synthetic) vs real telemetry — marginals",
                 fontsize=11, color=INK, x=0.02, ha="left")
    for ax, obs in zip(axes.ravel(), FIT_OBSERVABLES):
        _ecdf(ax, _values(real_uts, obs), C_REAL, "real")
        _ecdf(ax, _values(sim_uts, obs), C_SIM, "sim")
        k = ks[obs]
        sub = (f"KS={k['statistic']:.3f}  p={k['pvalue']:.2g}"
               if math.isfinite(k["statistic"]) else "no data")
        _style(ax, f"{obs}   ({sub})", obs, "ECDF")
        if k["n_real"] and k["n_sim"]:
            ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)  # format inferred from the suffix (.pdf for the artifacts)
    plt.close(fig)


def _fig_dynamics(report, ac_real, ac_sim, d_real, d_sim, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0))
    fig.suptitle("Sim-Fleet vs real telemetry — dynamics, drift and missingness",
                 fontsize=11, color=INK, x=0.02, ha="left")
    ax1, ax2, ax3, ax4 = axes.ravel()

    lags = np.asarray(AUTOCORR_LAGS)
    for curve, color, label in ((ac_real, C_REAL, "real"), (ac_sim, C_SIM, "sim")):
        arr = np.asarray(curve, dtype=np.float64)
        if np.isfinite(arr).any():
            ax1.plot(lags, arr, color=color, linewidth=2.0, marker="o",
                     markersize=4, label=label)
    _style(ax1, "T1 autocorrelation (sign disagrees at lag 1)", "lag (snapshots)", "autocorr")
    ax1.axhline(0.0, color="#c3c2b7", linewidth=0.8)
    ax1.legend(fontsize=7, frameon=False)

    both = (np.concatenate([d for d in (d_real, d_sim) if d.size])
            if (d_real.size or d_sim.size) else np.empty(0))
    if both.size:
        hi = float(np.quantile(both, 0.99)) or 1e-3
        bins = np.linspace(0.0, hi, 30)
        for d, color, label in ((d_real, C_REAL, "real"), (d_sim, C_SIM, "sim")):
            if d.size:
                ax2.hist(np.clip(d, bins[0], bins[-1]), bins=bins, density=True,
                         histtype="step", linewidth=2.0, color=color, label=label)
        ax2.legend(fontsize=7, frameon=False)
    _style(ax2, "|claimed - measured| / claimed", "relative discrepancy", "density")

    de = report["extra_axes"]["drift_episodes"]
    for side, color, label in (("real", C_REAL, "real"), ("sim", C_SIM, "sim")):
        d = np.asarray(de[side].get("durations_h", []), dtype=np.float64)
        d = d[np.isfinite(d)]
        if d.size:
            ax3.hist(d, bins=np.linspace(0, max(1.0, float(np.quantile(d, 0.98))), 25),
                     density=True, histtype="step", linewidth=2.0, color=color, label=label)
    if ax3.has_data():
        ax3.legend(fontsize=7, frameon=False)
    _style(ax3, f"T1 drift-episode duration (>{int(100 * DRIFT_THRESHOLD)}% below median)",
           "duration (h)", "density")

    ms = report["extra_axes"]["missingness"]
    labels = ["missing\nfraction", "mean gap\n(snapshots)", "burstiness\nvs MCAR"]
    keys = ["missing_frac", "mean_gap_len", "burstiness"]
    xpos = np.arange(len(keys))
    for off, side, color, label in ((-0.18, "real", C_REAL, "real"), (0.18, "sim", C_SIM, "sim")):
        # None appears when the report dict has been through _clean (NaN -> None)
        vals = [float("nan") if ms[side].get(k) is None else float(ms[side][k])
                for k in keys]
        ax4.bar(xpos + off, vals, width=0.34, color=color, label=label)
        for xi, v in zip(xpos + off, vals):
            if np.isfinite(v):
                ax4.text(xi, v, f"{v:.3g}", ha="center", va="bottom",
                         fontsize=6.5, color=INK2)
    ax4.set_xticks(xpos)
    ax4.set_xticklabels(labels, fontsize=7)
    ax4.legend(fontsize=7, frameon=False)
    _style(ax4, "Missingness pattern", "", "value")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)  # format inferred from the suffix (.pdf for the artifacts)
    plt.close(fig)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def sim_vs_real_report(
    real_uts: pd.DataFrame,
    sim_uts: pd.DataFrame,
    out_dir: str | Path,
    tex_dir: str | Path | None = None,
) -> dict:
    """Build the sim-vs-real diagnostic report; write PDFs + JSON into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marg_path = out_dir / "sim_vs_real_marginals.pdf"
    dyn_path = out_dir / "sim_vs_real_dynamics.pdf"
    json_path = out_dir / "sim_vs_real.json"

    ks: dict[str, dict] = {}
    for obs in FIT_OBSERVABLES:
        vr, vs = _values(real_uts, obs), _values(sim_uts, obs)
        entry: dict = {"n_real": int(vr.size), "n_sim": int(vs.size),
                       "statistic": float("nan"), "pvalue": float("nan")}
        if vr.size and vs.size:
            res = sps.ks_2samp(vr, vs)
            entry["statistic"] = float(res.statistic)
            entry["pvalue"] = float(res.pvalue)
        ks[obs] = entry

    ac_real, ac_sim = _autocorr_curve(real_uts), _autocorr_curve(sim_uts)
    acd_real, acd_sim = (_autocorr_curve_deinterleaved(real_uts),
                         _autocorr_curve_deinterleaved(sim_uts))
    d_real, d_sim = _discrepancies(real_uts), _discrepancies(sim_uts)
    real_stats, sim_stats = target_stats(real_uts), target_stats(sim_uts)

    report = {
        "ks": ks,
        "autocorr": {"lags": list(AUTOCORR_LAGS), "real": ac_real, "sim": ac_sim},
        "autocorr_deinterleaved": {"lags": list(AUTOCORR_LAGS),
                                   "real": acd_real, "sim": acd_sim},
        "discrepancy": {
            "real_mean": float(np.mean(d_real)) if d_real.size else float("nan"),
            "real_p95": float(np.quantile(d_real, 0.95)) if d_real.size else float("nan"),
            "sim_mean": float(np.mean(d_sim)) if d_sim.size else float("nan"),
            "sim_p95": float(np.quantile(d_sim, 0.95)) if d_sim.size else float("nan"),
        },
        "target_stats": {"real": real_stats, "sim": sim_stats},
        "validation_table": validation_table(real_stats, sim_stats),
        "extra_axes": {
            "tails": {"real": tail_stats(real_uts), "sim": tail_stats(sim_uts)},
            "cross_observable_corr": {"real": cross_observable_corr(real_uts),
                                      "sim": cross_observable_corr(sim_uts)},
            "drift_episodes": {"real": drift_episodes(real_uts),
                               "sim": drift_episodes(sim_uts)},
            "missingness": {"real": missingness(real_uts), "sim": missingness(sim_uts)},
            "recal_diagnostic": {"real": recal_diagnostic(real_uts),
                                 "sim": recal_diagnostic(sim_uts)},
        },
        "pdfs": {"marginals": str(marg_path), "dynamics": str(dyn_path)},
        "json": str(json_path),
    }

    _fig_marginals(real_uts, sim_uts, ks, marg_path)
    _fig_dynamics(report, ac_real, ac_sim, d_real, d_sim, dyn_path)

    # the per-episode duration vectors exist only to draw the histogram above
    # (~10^5 floats for the sim); keep them out of the committed JSON artifact
    for side in report["extra_axes"]["drift_episodes"].values():
        side.pop("durations_h", None)

    report_clean = _clean(report)
    json_path.write_text(json.dumps(report_clean, indent=2, sort_keys=True) + "\n")

    if tex_dir is not None:
        tex_dir = Path(tex_dir)
        tex_dir.mkdir(parents=True, exist_ok=True)
        _write_t5(report["validation_table"], ks, tex_dir / "T5_simval.tex")
        _write_t5b(report_clean, tex_dir / "T5b_simval_axes.tex")
    return report_clean


def main(argv: list[str] | None = None) -> int:
    """Regenerate the sim-vs-real artifacts from the registered fleet config."""
    import argparse

    from ltn.data.uts import read_uts

    from .fleet import FleetConfig, FleetSim

    p = argparse.ArgumentParser(prog="python -m ltn.sim.validate")
    p.add_argument("--lake", type=Path, default=Path("data/uts"))
    p.add_argument("--real-source", default="daqec")
    p.add_argument("--sim-config", type=Path, default=Path("splits/simfleet_v1_config.json"),
                   help="regenerate the sim lake in-memory from this FleetConfig JSON")
    p.add_argument("--sim-source", default=None,
                   help="instead read the sim lake from --lake under this source name")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--snapshots-per-day", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--out", type=Path, default=Path("paper/figures/simval"))
    p.add_argument("--tex", type=Path, default=Path("paper/tables"))
    args = p.parse_args(argv)

    real = read_uts(args.lake, sources=[args.real_source])
    if args.sim_source:
        sim = read_uts(args.lake, sources=[args.sim_source])
        src = f"lake source={args.sim_source}"
    else:
        cfg = FleetConfig.from_json(args.sim_config.read_text())
        sim, _ = FleetSim(cfg, args.seed).run(args.days, args.snapshots_per_day)
        src = (f"{args.sim_config} seed={args.seed} days={args.days} "
               f"snapshots_per_day={args.snapshots_per_day}")
    print(f"real: {len(real):,} rows ({args.real_source})   sim: {len(sim):,} rows ({src})")
    rep = sim_vs_real_report(real, sim, args.out, tex_dir=args.tex)
    print("wrote", rep["pdfs"]["marginals"], rep["pdfs"]["dynamics"], rep["json"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
