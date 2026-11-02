#!/usr/bin/env python
"""Compute IEEE-TR-style reliability endpoints from already-stored forecasts (lane A6).

Everything here is a closed-form transform of columns that already exist in
runs/**/predictions.parquet (mu, sigma, y) or of the `reward` column of the E6
replay parquets. No model is retrained, refit or re-run.

Endpoints
  1. SLA / threshold-violation probability, scored with Brier / ECE / Brier skill.
  2. Discrete-time reliability R(h) and hazard over the stored horizon grid.
  3. Expected wasted shots N_shots*(1 - p(S)) and availability, on the E6 replays.

Outputs JSON + CSV under runs/RELIABILITY/ and a standalone LaTeX fragment at
paper/tables/T6_reliability.tex.

Usage:  python scripts/reliability_endpoints.py [--n-shots 4096]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ltn.eval import reliability  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "RELIABILITY"

MODELS = ["ltn", "ltn_v2", "arima", "ewma", "lstm", "persistence", "tgn", "transformer"]
PRETTY = {
    "ltn": "LTN", "ltn_v2": "LTN-v2", "arima": "ARIMA", "ewma": "EWMA",
    "lstm": "LSTM", "persistence": "Persistence", "tgn": "TGN",
    "transformer": "Transformer",
}
BASE_RATE = 0.10                     # headline SLA severity
SENSITIVITY_RATES = [0.05, 0.10, 0.20]
DEFAULT_N_SHOTS = 4096


# --------------------------------------------------------------------- io

def seed_files(exp: str, model: str) -> list[str]:
    return sorted(glob.glob(str(REPO / "runs" / exp / model / "*" / "predictions.parquet")))


def load(path: str, channel: str | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if channel is not None and "channel" in df.columns:
        df = df[df["channel"] == channel].reset_index(drop=True)
    return df


# --------------------------------------------------------------------- E1

def score_experiment(exp: str, channel: str | None, base_rate: float,
                     want_curve: bool) -> dict:
    """SLA scores (+ survival curve) for every model/seed of `exp`."""
    ref_files = seed_files(exp, "ltn")
    if not ref_files:
        raise FileNotFoundError(f"no ltn predictions under runs/{exp}")
    ref = load(ref_files[0], channel)
    thr = reliability.sla_thresholds(ref, base_rate=base_rate)

    per_seed, curves = [], []
    for model in MODELS:
        for f in seed_files(exp, model):
            seed = int(Path(f).parent.name)
            df = load(f, channel)
            # y must be identical across models for the comparison to be fair
            j = reliability.attach_violations(df, thr)
            row = {"model": model, "seed": seed} | reliability.sla_scores(j)
            for h, g in j.groupby("horizon"):
                s = reliability.sla_scores(g)
                row[f"brier_h{h}"] = s["brier"]
                row[f"ece_h{h}"] = s["ece"]
                row[f"brier_skill_h{h}"] = s["brier_skill"]
            per_seed.append(row)
            if want_curve:
                c = reliability.reliability_curve(j)
                c.insert(0, "model", model)
                c.insert(1, "seed", seed)
                curves.append(c)

    per_seed = pd.DataFrame(per_seed)
    agg_cols = [c for c in per_seed.columns if c not in ("model", "seed", "n")]
    agg = per_seed.groupby("model")[agg_cols].agg(["mean", "std", "count"])
    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return {"thresholds": thr, "per_seed": per_seed, "agg": agg, "curve": curve}


def brier_separation(exp: str, channel: str | None, base_rate: float,
                     reference: str = "ltn", n_boot: int = 2000) -> pd.DataFrame:
    """Paired device-clustered bootstrap of the Brier difference vs `reference`.

    Per-row loss for a model is the squared error of its violation probability,
    averaged over that model's seeds, so every model contributes one loss vector
    on an identical row set. Clustering on device keeps the interval honest about
    the within-device correlation of telemetry.
    """
    ref_files = seed_files(exp, "ltn")
    thr = reliability.sla_thresholds(load(ref_files[0], channel), base_rate=base_rate)

    losses, order_key, devices, viol = {}, None, None, None
    for model in MODELS:
        acc, n = None, 0
        for f in seed_files(exp, model):
            j = reliability.attach_violations(load(f, channel), thr)
            j = j.sort_values(["device", "observable", "s_idx", "t_idx", "horizon"],
                              kind="mergesort").reset_index(drop=True)
            if order_key is None:
                order_key = j[["device", "observable", "s_idx", "t_idx", "horizon"]]
                devices = j["device"].to_numpy()
                viol = j["viol"].to_numpy(dtype=np.float64)
            se = (j["p_viol"].to_numpy(dtype=np.float64) - j["viol"].to_numpy(dtype=np.float64)) ** 2
            acc = se if acc is None else acc + se
            n += 1
        if n:
            losses[model] = acc / n

    rows = []
    for model, loss in losses.items():
        if model == reference:
            continue
        r = reliability.paired_bootstrap_diff(losses[reference], loss, devices,
                                              n_boot=n_boot, seed=0)
        rows.append({"reference": reference, "model": model,
                     "brier_ref": float(losses[reference].mean()),
                     "brier_model": float(loss.mean())} | r)
    return pd.DataFrame(rows)


def sanity_check_y_identical(exp: str, channel: str | None) -> dict:
    """The comparison is only fair if every model scored the same outcomes."""
    key = ["t_idx", "s_idx", "horizon", "device", "observable"]
    ref = load(seed_files(exp, "ltn")[0], channel)
    worst = 0.0
    for model in MODELS:
        fs = seed_files(exp, model)
        if not fs:
            continue
        d = load(fs[0], channel)
        m = ref.merge(d, on=key, suffixes=("_r", "_m"))
        if len(m) != len(ref):
            return {"ok": False, "reason": f"{model}: row mismatch {len(m)} vs {len(ref)}"}
        worst = max(worst, float(np.abs(m["y_r"] - m["y_m"]).max()))
    return {"ok": worst == 0.0, "max_abs_y_diff": worst}


# --------------------------------------------------------------------- E6

def e6_wasted(n_shots: int) -> dict:
    """Expected wasted shots per policy for the E6a synthetic and E6b logged replays."""
    out = {"n_shots": n_shots, "E6a": [], "E6b": []}

    for f in sorted(glob.glob(str(REPO / "runs" / "E6a" / "replay_*.parquet"))):
        stem = Path(f).stem                       # replay_<policy>_w<width>_s<seed>
        body = stem[len("replay_"):]
        policy, width, seed = body.rsplit("_w", 1)[0], body.rsplit("_w", 1)[1].split("_s")[0], body.split("_s")[-1]
        d = pd.read_parquet(f)
        out["E6a"].append({"policy": policy, "width": int(width), "seed": int(seed)}
                          | reliability.wasted_shots(d["reward"], n_shots, d["best_reward"]))

    for f in sorted(glob.glob(str(REPO / "runs" / "E6b" / "replay_*.parquet"))):
        body = Path(f).stem[len("replay_"):]
        policy, seed = body.rsplit("_s", 1)
        d = pd.read_parquet(f)
        out["E6b"].append({"policy": policy, "seed": int(seed)}
                          | reliability.wasted_shots(d["reward"], n_shots, d["best_reward"]))
    return out


# --------------------------------------------------------------------- latex

def _fmt(x, nd=4):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_latex(e1b_agg: pd.DataFrame, e1b_curve: pd.DataFrame, e1a_agg: pd.DataFrame,
                e6: dict, base_rate: float, path: Path) -> None:
    L = []
    A = L.append
    A("% Auto-generated by scripts/reliability_endpoints.py -- do not edit by hand.")
    A("% Lane A6: reliability endpoints derived in closed form from stored (mu, sigma).")
    A("")
    # --- panel (a): SLA violation forecasting, E1b
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\caption{Reliability endpoints derived in closed form from the stored Gaussian")
    A(r"forecasts. (a) SLA-violation probability $P(Q_{t+h}$ out of spec$)$ on the")
    A(rf"synthetic fleet (E1b), SLA set per device--observable at a "
      rf"{base_rate * 100:.0f}\% empirical")
    A(r"violation base rate; mean $\pm$ s.d. over seeds. Brier skill is against the")
    A(r"climatological base rate (higher is better; $0$ = no skill). $\Delta R(14)$ is the")
    A(r"chained-survival error at the longest horizon, $R_{\mathrm{pred}}-R_{\mathrm{emp}}$")
    A(r"(negative = pessimistic).}")
    A(r"\label{tab:reliability}")
    A(r"\begin{tabular}{lcccc}")
    A(r"\toprule")
    A(r"Model & Brier $\downarrow$ & ECE $\downarrow$ & Brier skill $\uparrow$ & "
      r"$\Delta R(14)$ \\")
    A(r"\midrule")
    gap = {}
    if len(e1b_curve):
        cm = e1b_curve.groupby(["model", "horizon"]).mean(numeric_only=True)
        h_last = sorted(e1b_curve["horizon"].unique())[-1]
        for m in e1b_curve["model"].unique():
            row = cm.loc[(m, h_last)]
            gap[m] = float(row["R_pred"] - row["R_emp"])
    order = e1b_agg[("brier", "mean")].sort_values().index
    for m in order:
        r = e1b_agg.loc[m]
        n = int(r[("brier", "count")])
        sd = lambda k: "" if n < 2 or not np.isfinite(r[(k, "std")]) else rf"$\pm${r[(k,'std')]:.4f}"
        g = f"{gap[m]:+.3f}" if m in gap else "--"
        A(f"{PRETTY.get(m, m)} & {_fmt(r[('brier','mean')])}{sd('brier')} & "
          f"{_fmt(r[('ece','mean')])}{sd('ece')} & {_fmt(r[('brier_skill','mean')],3)} & "
          f"{g} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")
    # --- panel (b): survival / hazard
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(r"\caption{Discrete-time reliability $R(h)=P(\text{no SLA violation through }h)$ and")
    A(r"hazard on the E1b horizon grid, fixed cohort of origins observed at every horizon.")
    A(r"$R_{\mathrm{emp}}$ is a property of the data and the SLA, identical for all models;")
    A(r"$R_{\mathrm{pred}}$ is the LTN forecast chained under conditional independence.}")
    A(r"\label{tab:reliability-survival}")
    A(r"\begin{tabular}{lcccc}")
    A(r"\toprule")
    A(r"$h$ & $R_{\mathrm{pred}}$ & $R_{\mathrm{emp}}$ & haz.\ pred & haz.\ emp \\")
    A(r"\midrule")
    ltn_curve = e1b_curve[e1b_curve.model == "ltn"].groupby("horizon").mean(numeric_only=True)
    for h, r in ltn_curve.iterrows():
        A(f"{int(h)} & {_fmt(r['R_pred'],3)} & {_fmt(r['R_emp'],3)} & "
          f"{_fmt(r['hazard_pred'],3)} & {_fmt(r['hazard_emp'],3)} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    A("")
    # --- panel (c): wasted shots
    A(r"\begin{table}[t]")
    A(r"\centering")
    A(rf"\caption{{Expected wasted shots per job of $N={e6['n_shots']}$ shots, "
      r"$N(1-p(S))$, on the E6a synthetic replay ($w{=}12$). Informed policies are")
    A(r"\emph{single-seed}; random is averaged over 5 seeds. The excess column is the")
    A(r"shots an operator loses relative to the oracle selection.}")
    A(r"\label{tab:reliability-shots}")
    A(r"\begin{tabular}{lccc}")
    A(r"\toprule")
    A(r"Policy & Availability & Wasted shots & Excess vs.\ oracle \\")
    A(r"\midrule")
    e6a = pd.DataFrame(e6["E6a"])
    w12 = e6a[e6a.width == 12].groupby("policy").mean(numeric_only=True)
    for p in ["oracle", "ltn_lcb", "ltn_mean", "greedy_claimed", "random"]:
        if p not in w12.index:
            continue
        r = w12.loc[p]
        name = p.replace("_", r"\_")
        A(f"{name} & {_fmt(r['availability'],4)} & "
          f"{r['wasted_shots_mean']:.1f} & {r['excess_wasted_shots_vs_oracle']:.1f} \\\\")
    A(r"\bottomrule")
    A(r"\end{tabular}")
    A(r"\end{table}")
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shots", type=int, default=DEFAULT_N_SHOTS)
    ap.add_argument("--base-rate", type=float, default=BASE_RATE)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"base_rate": args.base_rate, "n_shots": args.n_shots}

    # ---- E1b (primary) -----------------------------------------------------
    print("[E1b] sanity-checking that all models scored identical outcomes ...")
    manifest["E1b_y_identical"] = sanity_check_y_identical("E1b", None)
    print("      ", manifest["E1b_y_identical"])

    print("[E1b] SLA violation scoring + survival ...")
    e1b = score_experiment("E1b", None, args.base_rate, want_curve=True)
    e1b["per_seed"].to_csv(OUT / "e1b_sla_per_seed.csv", index=False)
    e1b["agg"].to_csv(OUT / "e1b_sla_agg.csv")
    e1b["curve"].to_csv(OUT / "e1b_survival_per_seed.csv", index=False)
    e1b["thresholds"].to_csv(OUT / "e1b_thresholds.csv", index=False)

    print("[E1b] paired device-clustered bootstrap of Brier vs LTN ...")
    sep = brier_separation("E1b", None, args.base_rate, reference="ltn")
    sep.to_csv(OUT / "e1b_brier_separation.csv", index=False)
    print(sep.round(6).to_string(index=False))
    manifest["E1b_brier_separation_vs_ltn"] = sep.to_dict(orient="records")

    # ---- sensitivity to SLA severity --------------------------------------
    sens = []
    for br in SENSITIVITY_RATES:
        if br == args.base_rate:
            r = e1b["per_seed"]
        else:
            r = score_experiment("E1b", None, br, want_curve=False)["per_seed"]
        t = r.groupby("model")[["brier", "ece", "brier_skill", "base_rate"]].mean()
        t["sla_base_rate"] = br
        sens.append(t.reset_index())
    sens = pd.concat(sens, ignore_index=True)
    sens.to_csv(OUT / "e1b_sla_sensitivity.csv", index=False)

    # ---- E1a (small-sample real-data check) --------------------------------
    print("[E1a] SLA violation scoring (small sample, probe channel) ...")
    e1a = score_experiment("E1a", "probe", args.base_rate, want_curve=False)
    e1a["per_seed"].to_csv(OUT / "e1a_sla_per_seed.csv", index=False)
    e1a["agg"].to_csv(OUT / "e1a_sla_agg.csv")

    # E1a cannot support a survival curve -- record why, do not fabricate one.
    ref_e1a = load(seed_files("E1a", "ltn")[0], "probe")
    hpo = ref_e1a.groupby(["device", "observable", "s_idx", "t_idx"]).horizon.nunique()
    manifest["E1a_survival"] = {
        "feasible": bool((hpo == ref_e1a.horizon.nunique()).any()),
        "n_horizons_in_grid": int(ref_e1a.horizon.nunique()),
        "max_horizons_per_origin": int(hpo.max()),
        "n_origins": int(len(hpo)),
        "n_rows": int(len(ref_e1a)),
        "note": ("No E1a forecast origin is observed at every horizon, so no fixed-cohort "
                 "survival curve exists for the real-data lane."),
    }
    print("      ", manifest["E1a_survival"])

    # ---- E6 wasted shots ---------------------------------------------------
    print("[E6] expected wasted shots ...")
    e6 = e6_wasted(args.n_shots)
    pd.DataFrame(e6["E6a"]).to_csv(OUT / "e6a_wasted_shots.csv", index=False)
    pd.DataFrame(e6["E6b"]).to_csv(OUT / "e6b_wasted_shots.csv", index=False)

    e6b = pd.DataFrame(e6["E6b"])
    rnd = e6b[e6b.policy == "random"]
    manifest["E6b_degeneracy"] = {
        "random_availability_mean": float(rnd.availability.mean()),
        "random_excess_wasted_frac_vs_oracle_mean": float(rnd.excess_wasted_frac_vs_oracle.mean()),
        "random_excess_wasted_shots_vs_oracle_mean": float(rnd.excess_wasted_shots_vs_oracle.mean()),
        "oracle_availability": float(e6b[e6b.policy == "oracle"].availability.iloc[0]),
        "n_random_seeds": int(len(rnd)),
        "note": ("Logged-arm replay is degenerate: random selection loses a fraction of a "
                 "percent of shots against the oracle, so no policy can separate."),
    }
    print("      ", manifest["E6b_degeneracy"])

    # ---- outputs -----------------------------------------------------------
    manifest["E1b_sla_agg"] = json.loads(
        e1b["agg"][[("brier", "mean"), ("brier", "std"), ("ece", "mean"), ("ece", "std"),
                    ("brier_skill", "mean"), ("brier_skill", "std"), ("brier", "count")]]
        .set_axis(["brier_mean", "brier_std", "ece_mean", "ece_std",
                   "brier_skill_mean", "brier_skill_std", "n_seeds"], axis=1)
        .to_json(orient="index"))
    manifest["E1a_sla_agg"] = json.loads(
        e1a["agg"][[("brier", "mean"), ("ece", "mean"), ("brier_skill", "mean"),
                    ("brier", "count")]]
        .set_axis(["brier_mean", "ece_mean", "brier_skill_mean", "n_seeds"], axis=1)
        .to_json(orient="index"))
    _c = e1b["curve"].groupby(["model", "horizon"]).mean(numeric_only=True).reset_index()
    _c["survival_gap"] = _c["R_pred"] - _c["R_emp"]
    manifest["E1b_survival_all_models"] = _c[
        ["model", "horizon", "R_pred", "R_emp", "survival_gap",
         "hazard_pred", "hazard_emp"]].to_dict(orient="records")
    manifest["E1b_survival_ltn"] = json.loads(
        e1b["curve"][e1b["curve"].model == "ltn"].groupby("horizon")
        .mean(numeric_only=True).to_json(orient="index"))
    manifest["E6a_w12"] = json.loads(
        pd.DataFrame(e6["E6a"]).query("width == 12").groupby("policy")
        .mean(numeric_only=True).to_json(orient="index"))

    (OUT / "reliability.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tex = REPO / "paper" / "tables" / "T6_reliability.tex"
    write_latex(e1b["agg"], e1b["curve"], e1a["agg"], e6, args.base_rate, tex)
    print(f"\nwrote {OUT}/reliability.json and {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
