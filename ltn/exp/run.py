"""Unified E1-family experiment runner.

  python -m ltn.exp.run --split e1a --model persistence --seed 0 [--cfg k=v ...]

Per run writes runs/<EXP>/<model>/<seed>/:
  config.yaml, results.jsonl (one line per device x observable x horizon cell),
  predictions.parquet (per-pair mu/sigma/y — the pairing substrate for Wilcoxon).

Metrics per cell: CRPS (raw units), NCRPS (CRPS / train std of the observable —
scale-free, the tuning + headline aggregate), MAE, Gaussian NLL, 90% coverage.
Deterministic baselines (persistence) run at seed 0 only; stats pair them against
every seed of stochastic models (declared in prereg).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ltn.data.series import forecast_pairs, make_panel, split_panel_by_date, to_series
from ltn.data.uts import read_uts
from ltn.eval import metrics as M
from ltn.eval.runlog import RunLogger
from ltn.eval.splits import load_split

MODELS = {}


def _register():
    from ltn.baselines.arima import ARIMAForecaster
    from ltn.baselines.ewma import EWMA
    from ltn.baselines.neural import LSTMBaseline, TGNBaseline, TransformerBaseline
    from ltn.baselines.persistence import Persistence
    from ltn.baselines.reliability import RELIABILITY_MODELS
    from ltn.exp.train import LTNForecaster
    from ltn.exp.variants import VARIANT_MODELS
    MODELS.update({
        "persistence": Persistence, "ewma": EWMA, "arima": ARIMAForecaster,
        "lstm": LSTMBaseline, "tgn": TGNBaseline, "transformer": TransformerBaseline,
        "ltn": LTNForecaster,
    })
    # Classical reliability/degradation baselines (kalman, gp, wiener, gamma, hmm):
    # the comparison with the reliability-modelling literature. Same Forecaster
    # protocol, so they run through the identical evaluation path.
    MODELS.update(RELIABILITY_MODELS)
    MODELS.update(VARIANT_MODELS)


def parse_norm_stats(split: dict) -> dict:
    return {tuple(k.split("|")): tuple(v) for k, v in split["norm_stats"].items()}


def build_data(split: dict, lake="data/uts"):
    """-> (uts, panel, masks (train, val, test)) on the split's frozen boundaries."""
    uts = read_uts(lake, sources=split["sources"])
    holdout = set(split.get("holdout_devices", []))
    if holdout:
        uts = uts[~uts["device"].isin(holdout)].reset_index(drop=True)
    observables = [tuple(x) for x in split["observables"]]
    series = to_series(uts, observables=[o for o, _ in observables],
                       channels=list({c for _, c in observables}))
    keep = [(o, c) in observables for o, c in zip(series["observable"], series["channel"])]
    panel = make_panel(series[keep], uts)
    tr_end = pd.Timestamp(split["train_end"])
    va_end = pd.Timestamp(split["val_end"])
    tr = (panel.ts < tr_end).to_numpy() if hasattr(panel.ts < tr_end, "to_numpy") else np.asarray(panel.ts < tr_end)
    va = np.asarray((panel.ts >= tr_end) & (panel.ts < va_end))
    te = np.asarray(panel.ts >= va_end)
    return uts, panel, (tr, va, te)


def model_cfg(split: dict, seed: int, extra: dict) -> dict:
    cfg = {
        "observables": [tuple(x) for x in split["observables"]],
        "norm_stats": parse_norm_stats(split),
        "horizons": split["horizons_steps"],
        "time_unit_h": split.get("grid_hours", 8.0),
        "seed": seed,
    }
    cfg.update(extra)
    return cfg


def evaluate(pairs: pd.DataFrame, panel, mu: np.ndarray, sigma: np.ndarray,
             norm_stats: dict) -> pd.DataFrame:
    """Per-cell metric frame; cell = device x observable x horizon."""
    df = pairs.copy()
    df["mu"], df["sigma"] = mu, sigma
    keys = panel.keys
    df["device"] = keys["device"].to_numpy()[df["s_idx"]]
    df["observable"] = keys["observable"].to_numpy()[df["s_idx"]]
    df["channel"] = keys["channel"].to_numpy()[df["s_idx"]]
    df["platform"] = panel.platform.to_numpy()[df["s_idx"]]
    rows = []
    for (dev, obs, ch, h), g in df.groupby(["device", "observable", "channel", "horizon"], observed=True):
        y, m, s = g["y"].to_numpy(), g["mu"].to_numpy(), g["sigma"].to_numpy()
        p = g["platform"].iloc[0]
        train_std = norm_stats.get((p, obs, ch), (0.0, 1.0))[1] or 1.0
        rows.append({
            "device": dev, "observable": obs, "channel": ch, "horizon": int(h), "n": len(g),
            "crps": M.crps_gaussian(y, m, s), "ncrps": M.crps_gaussian(y, m, s) / train_std,
            "mae": M.mae(y, m), "nll": M.gaussian_nll(y, m, s),
            "coverage90": M.coverage(y, m, s, 0.9),
        })
    return pd.DataFrame(rows), df


def run_one(split_name: str, model_name: str, seed: int, extra_cfg: dict,
            lake="data/uts", root="runs", exp_override: str | None = None,
            label: str | None = None) -> pd.DataFrame:
    """exp_override/label route runs (e.g. E2 ablations) to their own run dirs so
    they can never clobber the E1 leaderboard runs of the same base model."""
    _register()
    split = load_split(split_name)
    exp = exp_override or split["experiment"]
    uts, panel, (tr, va, te) = build_data(split, lake)
    cfg = model_cfg(split, seed, extra_cfg)
    cfg["uts"] = uts
    m = MODELS[model_name]()
    loggable = {
        k: v for k, v in cfg.items()
        if k not in ("uts", "norm_stats", "observables")
    }
    loggable["observables"] = ["/".join(x) for x in cfg["observables"]]
    loggable["norm_stats_keys"] = sorted("|".join(k) for k in cfg["norm_stats"])
    with RunLogger(exp, label or model_name, seed, loggable, root=root) as log:
        m.fit(panel, tr, va, cfg)
        pairs = forecast_pairs(panel, split["horizons_steps"], te)
        if cfg.get("devices"):  # device-restricted runs (tuning probes, E4 adaptation)
            dev_of = panel.keys["device"].to_numpy()
            pairs = pairs[np.isin(dev_of[pairs["s_idx"].to_numpy()], cfg["devices"])].reset_index(drop=True)
        mu, sigma = m.predict(pairs, panel)
        cells, preds = evaluate(pairs, panel, mu, sigma, parse_norm_stats(split))
        for _, r in cells.iterrows():
            log.log(**r.to_dict())
        if hasattr(m, "n_params"):
            log.log_params(int(m.n_params()))
        preds.drop(columns=["platform"]).to_parquet(log.dir / "predictions.parquet", index=False)
        if hasattr(m, "save"):
            m.save(log.dir / "model.pt")
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", nargs="*", default=[], help="k=v extra model config (int/float parsed)")
    ap.add_argument("--tuned", action="store_true", help="load best params from runs/tuning/<split>_<model>.json")
    ap.add_argument("--exp", default=None, help="override experiment name for run-dir routing (e.g. E2)")
    ap.add_argument("--as", dest="label", default=None, help="model label for run-dir routing")
    args = ap.parse_args()
    extra = {}
    if args.tuned:
        import json
        from pathlib import Path
        best = json.loads(Path(f"runs/tuning/{args.split}_{args.model}.json").read_text())
        extra.update(best["params"])
    for kv in args.cfg:
        k, v = kv.split("=", 1)
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                pass
        extra[k] = v
    cells = run_one(args.split, args.model, args.seed, extra,
                    exp_override=args.exp, label=args.label)
    print(cells.groupby(["observable", "horizon"])["ncrps"].mean().round(4).to_string())
    print(f"\nmean NCRPS: {cells['ncrps'].mean():.4f}")


if __name__ == "__main__":
    main()
