"""E3 (H3): claim-vs-delivered divergence prediction on DAQEC — 100% real test data.

  python -m ltn.exp.e3 [--folds all] [--gpu 0] [--pretrain-ckpt PATH]

Design (frozen in splits/e3.json):
  Label: at snapshot t, entity e diverges if the relative claim discrepancy
  |probe - claimed| / claimed of ANY monitored observable, z-scored against the
  TRAIN-fold distribution for that (device, observable), exceeds z_threshold=2
  at any of the next h in {1,2,3} snapshots.
  Folds: 3-fold device rotation (train 2 IBM devices, test the 3rd).

Predictors compared (all evaluated on the held-out device only):
  ltn_dissonance   raw dissonance of the LTN trust state — zero extra training;
                   the pre-registered mechanism claim (dissonance anticipates divergence)
  ltn_probe        logistic probe on (entropy, vacuity, dissonance, p_diverge),
                   fit on train-device states only
  cal_age          staleness-only logistic control (does the model beat the trivial
                   "old calibration diverges" heuristic?)
  last_disc        most recent observed relative discrepancy (carry-forward control)

Lead time: mean dissonance trajectory in the 6 snapshots before each divergence
event vs matched non-event windows; the reported lead time is the earliest offset
whose bootstrap CI excludes the non-event mean. Written to runs/E7-compatible
event rows for F7's dissonance panel.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ltn.data.series import make_panel, to_series
from ltn.data.uts import read_uts
from ltn.eval import metrics as M
from ltn.eval.runlog import RunLogger
from ltn.eval.splits import load_split
from ltn.exp.probe import extract_states
from ltn.exp.run import model_cfg, parse_norm_stats
from ltn.exp.train import LTNForecaster

E3_OBS = [("t1_us", "probe"), ("t2_us", "probe"), ("readout_error", "probe")]


def discrepancy_frame(uts: pd.DataFrame) -> pd.DataFrame:
    """Per (device, entity, ts, observable): relative |probe-claimed|/claimed."""
    piv = uts[uts["observable"].isin([o for o, _ in E3_OBS])]
    g = piv.pivot_table(index=["device", "entity_id", "ts", "observable"],
                        columns="channel", values="value", aggfunc="mean", observed=True).reset_index()
    g = g.dropna(subset=["claimed", "probe"])
    g["rel_disc"] = (g["probe"] - g["claimed"]).abs() / g["claimed"].abs().clip(lower=1e-12)
    return g


def make_labels(disc: pd.DataFrame, train_devices: list[str], z_thr: float,
                horizons: list[int]) -> pd.DataFrame:
    """z-score rel_disc against the TRAIN-device (device, observable) distribution
    is wrong for the held-out device — the spec z-references per device x observable
    on the TRAIN FOLD; for the test device we use pooled train-fold observable stats
    (no test statistics leak into the threshold)."""
    stats = (disc[disc["device"].isin(train_devices)]
             .groupby("observable", observed=True)["rel_disc"].agg(["mean", "std"]))
    d = disc.merge(stats, on="observable")
    d["z"] = (d["rel_disc"] - d["mean"]) / d["std"].clip(lower=1e-12)
    d["exceed"] = d["z"] > z_thr
    # entity-level: diverges at t if any monitored observable exceeds
    ent = (d.groupby(["device", "entity_id", "ts"], observed=True)["exceed"].any()
           .reset_index().sort_values(["device", "entity_id", "ts"]))
    # label(t) = any exceed within the NEXT h snapshots
    hmax = max(horizons)
    out = []
    for (dev, e), g in ent.groupby(["device", "entity_id"], observed=True):
        ex = g["exceed"].to_numpy()
        fut = np.zeros(len(ex), dtype=bool)
        for h in range(1, hmax + 1):
            fut[:-h] |= ex[h:]
        gg = g.copy()
        gg["label"] = fut
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def lag1_autocorr(x: np.ndarray) -> float:
    """Standard (biased, Bartlett-consistent) lag-1 ACF estimator

        r1 = sum_t (x_t - xbar)(x_{t+1} - xbar) / sum_t (x_t - xbar)^2

    Returns NaN for series shorter than 3 points or with zero variance
    (a constant divergence indicator has no defined autocorrelation).
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan")
    xc = x - x.mean()
    den = float((xc * xc).sum())
    if den <= 0.0:
        return float("nan")
    return float((xc[:-1] * xc[1:]).sum() / den)


def _acf_series(df: pd.DataFrame, keys: list[str], value: str) -> list[dict]:
    rows = []
    for k, g in df.groupby(keys, observed=True):
        g = g.sort_values("ts")
        v = g[value].to_numpy(dtype=float)
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec["n"] = int(len(v))
        rec["r1"] = lag1_autocorr(v)
        rec["mean"] = float(np.nanmean(v)) if len(v) else float("nan")
        rows.append(rec)
    return rows


def _acf_summary(rows: list[dict]) -> dict:
    r = np.array([x["r1"] for x in rows], dtype=float)
    ns = np.array([x["n"] for x in rows], dtype=float)
    fin = np.isfinite(r)
    n_med = float(np.median(ns[fin])) if fin.any() else float("nan")
    band = 1.96 / np.sqrt(n_med) if n_med and np.isfinite(n_med) else float("nan")
    return {
        "n_series": int(len(rows)),
        "n_series_defined": int(fin.sum()),
        "n_series_degenerate": int((~fin).sum()),
        "series_length_median": n_med,
        "white_noise_band_95": float(band),
        "max_abs_r1": float(np.max(np.abs(r[fin]))) if fin.any() else float("nan"),
        "median_abs_r1": float(np.median(np.abs(r[fin]))) if fin.any() else float("nan"),
        "mean_r1": float(np.mean(r[fin])) if fin.any() else float("nan"),
        "n_outside_band": int((np.abs(r[fin]) > band).sum()) if fin.any() else 0,
        "series": rows,
    }


def divergence_autocorrelation(uts: pd.DataFrame, split: dict) -> dict:
    """Per-entity lag-1 autocorrelation of the E3 divergence signal.

    Uses exactly the E3 entity/label machinery (``discrepancy_frame`` +
    ``make_labels`` with the split's ``z_threshold``), so the series scored here
    are the same ones the E3 classifiers are evaluated on.

    Three readings are returned:
      ``binary_exceed``  -- the divergence-EVENT indicator 1[z > z_thr] at t,
                            per (device, entity); z-referenced on the fold in
                            which that device is HELD OUT (the evaluated fold).
      ``binary_label``   -- the forward-window E3 target (any exceed in the next
                            h in label_horizons); autocorrelated by construction
                            because overlapping windows share snapshots.
      ``continuous_rel_disc`` -- the raw relative discrepancy per
                            (device, entity, observable); fold-independent.
    ``binary_exceed_all_folds`` repeats reading (a) for every fold assignment.
    """
    disc = discrepancy_frame(uts)
    z_thr = float(split["divergence_def"]["z_threshold"])
    horizons = list(split["label_horizons_snapshots"])

    per_fold = {}
    ev_rows, lb_rows = [], []
    for fold in split["folds"]:
        train_devs, test_dev = list(fold["train"]), fold["test"][0]
        lab = make_labels(disc, train_devs, z_thr, horizons)
        lab["exceed"] = lab["exceed"].astype(float)
        lab["label"] = lab["label"].astype(float)
        per_fold[test_dev] = {
            "train_devices": train_devs,
            "exceed": _acf_summary(_acf_series(lab, ["device", "entity_id"], "exceed")),
        }
        held = lab[lab["device"] == test_dev]
        ev_rows += _acf_series(held, ["device", "entity_id"], "exceed")
        lb_rows += _acf_series(held, ["device", "entity_id"], "label")

    return {
        "z_threshold": z_thr,
        "label_horizons_snapshots": horizons,
        "estimator": "r1 = sum (x_t - xbar)(x_{t+1} - xbar) / sum (x_t - xbar)^2",
        "binary_exceed": _acf_summary(ev_rows),
        "binary_label": _acf_summary(lb_rows),
        "continuous_rel_disc": _acf_summary(
            _acf_series(disc, ["device", "entity_id", "observable"], "rel_disc")),
        "binary_exceed_all_folds": per_fold,
    }


def claim_age_stats(master_path: str = "data/raw/daqec/master.parquet") -> dict:
    """Three distinct definitions of 'claim age' on the DAQEC master table.

    per_run              age of the calibration attached to each individual run
                         (ts - calibration_timestamp), one value per run row.
    freshest_per_snapshot  per snapshot ts, ts - max(calibration_timestamp);
                         this is the definition ltn/paper_gen/figures.py uses
                         (groupby("ts")["cal_ts"].max()).
    oldest_per_snapshot  per snapshot ts, ts - min(calibration_timestamp).
    mean_per_snapshot    per snapshot ts, mean over runs of ts - calibration_timestamp;
                         this matches the ``cal_age_h`` feature E3 feeds to the
                         staleness-only control in ``run_fold``.
    """
    m = pd.read_parquet(master_path, columns=["backend", "timestamp_utc", "calibration_timestamp"])
    m = m.dropna(subset=["timestamp_utc", "calibration_timestamp"])
    m["age_h"] = (m["timestamp_utc"] - m["calibration_timestamp"]).dt.total_seconds() / 3600.0

    def _desc(s: pd.Series) -> dict:
        s = s.dropna()
        return {"n": int(len(s)), "median_h": float(s.median()),
                "q25_h": float(s.quantile(0.25)), "q75_h": float(s.quantile(0.75)),
                "iqr_h": float(s.quantile(0.75) - s.quantile(0.25)),
                "min_h": float(s.min()), "max_h": float(s.max())}

    out: dict = {"source": master_path, "definitions": {}}
    per_run = m
    fresh = (m.groupby(["backend", "timestamp_utc"], observed=True)["calibration_timestamp"]
             .max().reset_index())
    fresh["age_h"] = (fresh["timestamp_utc"] - fresh["calibration_timestamp"]).dt.total_seconds() / 3600.0
    old = (m.groupby(["backend", "timestamp_utc"], observed=True)["calibration_timestamp"]
           .min().reset_index())
    old["age_h"] = (old["timestamp_utc"] - old["calibration_timestamp"]).dt.total_seconds() / 3600.0
    mean = (m.groupby(["backend", "timestamp_utc"], observed=True)["age_h"]
            .mean().reset_index())
    for name, df in [("per_run", per_run), ("freshest_per_snapshot", fresh),
                     ("oldest_per_snapshot", old), ("mean_per_snapshot", mean)]:
        out["definitions"][name] = {
            "pooled": _desc(df["age_h"]),
            "per_device": {dev: _desc(g["age_h"])
                           for dev, g in df.groupby("backend", observed=True)},
        }
    return out


def run_fold(fold: dict, split: dict, e1a_split: dict, uts: pd.DataFrame,
             gpu: int | None, pretrain_ckpt: str | None, seed: int) -> dict:
    train_devs, test_dev = fold["train"], fold["test"][0]
    # --- fit LTN on train devices (E1a objective) -------------------------------
    observables = [tuple(x) for x in e1a_split["observables"]]
    series = to_series(uts, observables=[o for o, _ in observables],
                       channels=list({c for _, c in observables}))
    keep = [(o, c) in observables for o, c in zip(series["observable"], series["channel"])]
    panel = make_panel(series[keep], uts)
    tr_end = pd.Timestamp(e1a_split["train_end"])
    va_end = pd.Timestamp(e1a_split["val_end"])
    tr = np.asarray(panel.ts < tr_end)
    va = np.asarray((panel.ts >= tr_end) & (panel.ts < va_end))
    cfg = model_cfg(e1a_split, seed=seed, extra={})
    cfg.update(uts=uts, devices=train_devs + [test_dev], epochs=40, patience=6)
    if gpu is not None:
        cfg["device"] = f"cuda:{gpu}"
    if pretrain_ckpt:
        cfg["init_from"] = pretrain_ckpt
    # IMPORTANT: only train-device targets contribute to the loss; the test device's
    # sequence is rolled for state extraction but masked out of training.
    m = LTNForecaster()
    full_seq_devices = cfg["devices"]
    cfg["devices"] = train_devs
    m.fit(panel, tr, va, cfg)
    # rebuild sequences to include the test device for state extraction
    from ltn.exp.dataset import build_device_sequence
    for d in full_seq_devices:
        if d not in m.seqs:
            m.seqs[d] = build_device_sequence(uts, panel, d, observables, cfg["norm_stats"],
                                              time_unit_h=cfg["time_unit_h"])
    states = extract_states(m)

    # --- labels ------------------------------------------------------------------
    disc = discrepancy_frame(uts)
    labels = make_labels(disc, train_devs, split["divergence_def"]["z_threshold"],
                         split["label_horizons_snapshots"])
    sl = states.merge(labels, on=["device", "entity_id", "ts"], how="inner")
    # staleness + last-disc controls
    tmp = uts.dropna(subset=["cal_ts"]).copy()
    tmp["cal_age_h"] = (tmp["ts"] - tmp["cal_ts"]).dt.total_seconds() / 3600
    cal = tmp.groupby(["device", "entity_id", "ts"], observed=True)["cal_age_h"].mean().reset_index()
    sl = sl.merge(cal, on=["device", "entity_id", "ts"], how="left").fillna({"cal_age_h": 0.0})
    last_disc = (disc.groupby(["device", "entity_id", "ts"], observed=True)["rel_disc"].mean()
                 .groupby(level=[0, 1]).shift(1).rename("last_disc").reset_index())
    sl = sl.merge(last_disc, on=["device", "entity_id", "ts"], how="left").fillna({"last_disc": 0.0})

    tr_sl = sl[sl["device"].isin(train_devs)]
    te_sl = sl[sl["device"] == test_dev]
    if te_sl["label"].nunique() < 2:
        return {"fold": test_dev, "error": "degenerate test labels", "n": len(te_sl)}

    out = {"fold": test_dev, "n_test": len(te_sl), "base_rate": float(te_sl["label"].mean())}
    # provenance for the "divergence events are not strongly autocorrelated" claim:
    # lag-1 ACF of the held-out device's event indicator and raw discrepancy.
    held_lab = labels[labels["device"] == test_dev].assign(
        exceed=lambda d: d["exceed"].astype(float))
    out["acf1_exceed"] = _acf_summary(_acf_series(held_lab, ["device", "entity_id"], "exceed"))
    out["acf1_rel_disc"] = _acf_summary(
        _acf_series(disc[disc["device"] == test_dev], ["device", "entity_id", "observable"], "rel_disc"))
    scores = {"ltn_dissonance": te_sl["dissonance"].to_numpy()}
    feats = ["entropy", "vacuity", "dissonance", "p_diverge"]
    lr = LogisticRegression(max_iter=1000).fit(tr_sl[feats], tr_sl["label"])
    scores["ltn_probe"] = lr.predict_proba(te_sl[feats])[:, 1]
    lr_age = LogisticRegression(max_iter=1000).fit(tr_sl[["cal_age_h"]], tr_sl["label"])
    scores["cal_age"] = lr_age.predict_proba(te_sl[["cal_age_h"]])[:, 1]
    scores["last_disc"] = te_sl["last_disc"].to_numpy()
    y = te_sl["label"].to_numpy()
    for name, s in scores.items():
        out[f"auroc_{name}"] = float(M.auroc(y, s))
        out[f"auprc_{name}"] = float(M.auprc(y, s))

    # --- lead-time event study (dissonance around divergence onsets) --------------
    ev_rows = []
    te_states = states[states["device"] == test_dev]
    onsets = te_sl[te_sl["label"] & ~te_sl.groupby("entity_id")["label"].shift(1).fillna(False).astype(bool)]
    for _, r in onsets.iterrows():
        s_ent = te_states[te_states["entity_id"] == r["entity_id"]].sort_values("ts").reset_index(drop=True)
        pos = s_ent.index[s_ent["ts"] == r["ts"]]
        if not len(pos):
            continue
        p = int(pos[0])
        for k in range(-6, 1):
            if 0 <= p + k < len(s_ent):
                ev_rows.append({"t_rel": k, "dissonance": s_ent.loc[p + k, "dissonance"],
                                "entropy_pred": s_ent.loc[p + k, "entropy"],
                                "err_emp": np.nan, "event_type": "divergence", "fold": test_dev})
    out["event_rows"] = ev_rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--pretrain-ckpt", default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--autocorr-only", action="store_true",
                    help="compute the divergence autocorrelation + claim-age provenance "
                         "artifacts only (no model training) and exit")
    args = ap.parse_args()
    split = load_split("e3")
    uts = read_uts("data/uts", sources=["daqec"])

    import pathlib
    pathlib.Path("runs/E3").mkdir(parents=True, exist_ok=True)
    acf = divergence_autocorrelation(uts, split)
    pathlib.Path("runs/E3/divergence_autocorr.json").write_text(json.dumps(acf, indent=1))
    print("[E3] wrote runs/E3/divergence_autocorr.json")
    try:
        age = claim_age_stats()
        pathlib.Path("runs/E3/claim_age.json").write_text(json.dumps(age, indent=1))
        print("[E3] wrote runs/E3/claim_age.json")
    except FileNotFoundError as e:
        print(f"[E3] claim-age skipped: {e}")
    if args.autocorr_only:
        return

    e1a = load_split("e1a")
    all_events = []
    for seed in range(args.seeds):
        for fold in split["folds"]:
            res = run_fold(fold, split, e1a, uts, args.gpu, args.pretrain_ckpt, seed)
            ev = res.pop("event_rows", [])
            all_events += ev
            # one run dir per fold (RunLogger truncates on re-entry — same name would
            # overwrite the previous fold's rows)
            with RunLogger("E3", f"ltn-{res.get('fold')}", seed, {"fold": res.get("fold")}, root="runs") as log:
                log.log(**res)
            print(json.dumps(res))
    if all_events:
        import pathlib
        pathlib.Path("runs/E7").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_events).to_parquet("runs/E7/e3_divergence_events.parquet", index=False)


if __name__ == "__main__":
    main()
