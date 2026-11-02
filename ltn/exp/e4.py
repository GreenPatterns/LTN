"""E4 (H2a): cold-start few-shot adaptation to a held-out device.

  python -m ltn.exp.e4 --part a|b [--gpu 0] [--seeds 5]

E4a (real): pretrain = masked-telemetry on sim + forecast fine-tune on
ibm_brisbane+ibm_kyoto; adapt on ibm_osaka with the k in {0,1,3,7} days immediately
preceding osaka's eval region (final 20% of the grid). Scratch control trains from
random init on the same k days. k=0 pretrained = zero-shot; k=0 scratch = untrained
(reported for completeness).

E4b (sim): pretrain on the 12 train devices; adapt on each of the 4 frozen holdout
devices with k in {0,1,7,30} days; aggregate cells across devices.

Cells: (device, observable, horizon, k, arm) with CRPS/NCRPS — the F5 curve substrate.

BASELINE CONTROL (--base): the same pretrain -> fine-tune protocol is run for the
sequence baselines (lstm/transformer/tgn), so the data-efficiency claim can be
attributed to the density-matrix state rather than to pretraining per se. Identical
pretraining data, masked-telemetry objective, epoch/patience budget and adaptation
protocol for every family; by default every family also runs at the same E4 default
hyper-parameters, so the LTN arm reproduces the published E4 numbers exactly.
--tuned is the robustness variant in which EVERY family instead takes its own E1
Optuna architecture (the same 50-trial budget each model received). Results land
under runs/E4{a,b}-base/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ltn.data.series import forecast_pairs, make_panel, to_series
from ltn.data.uts import read_uts
from ltn.eval.runlog import RunLogger
from ltn.eval.splits import load_split
from ltn.exp.run import evaluate, model_cfg, parse_norm_stats
from ltn.exp.train import LTNForecaster

CKPT_DIR = Path("runs/E4/ckpt")

# arch/optim keys carried over from a model's E1 Optuna study (every family got the
# same 50-trial budget). epochs, patience and the adaptation lr are fixed by the E4
# protocol for every family, so `bptt` and stage budgets are deliberately excluded.
_TUNED_KEYS = ("width", "n_layers", "n_heads", "window",       # baselines
               "carry_dim", "hidden", "evidence_rank", "rounds", "d",  # ltn
               "lr")


def _model_cls(name: str):
    from ltn.exp.run import MODELS, _register
    if not MODELS:
        _register()
    return MODELS[name]


def _tuned(part: str, model: str, enabled: bool) -> dict:
    """E1-tuned params (same Optuna budget every model received). Off by default so
    the published E4 LTN protocol -- which ran at E4 defaults -- is reproduced
    bit-for-bit; the --base control sweep turns it on for EVERY family at once."""
    if not enabled:
        return {}
    split_name = "e1a" if part == "a" else "e1b"
    p = Path(f"runs/tuning/{split_name}_{model}.json")
    if not p.exists():
        return {}
    params = json.loads(p.read_text()).get("params", {})
    return {k: v for k, v in params.items() if k in _TUNED_KEYS}


def _ckpt_path(part: str, seed: int, model: str, tuned: bool = False) -> Path:
    # untuned 'ltn' keeps the historical filename so published checkpoints stay valid
    if model == "ltn" and not tuned:
        return CKPT_DIR / f"pretrained_{part}_{seed}.pt"
    suffix = "_tuned" if tuned else ""
    return CKPT_DIR / f"pretrained_{model}{suffix}_{part}_{seed}.pt"


def _panel_for(split, uts):
    observables = [tuple(x) for x in split["observables"]]
    series = to_series(uts, observables=[o for o, _ in observables],
                       channels=list({c for _, c in observables}))
    keep = [(o, c) in observables for o, c in zip(series["observable"], series["channel"])]
    return make_panel(series[keep], uts), observables


def pretrain(part: str, gpu: int | None, seed: int, model: str = "ltn",
             tuned: bool = False) -> Path:
    """Stage 1: masked pretraining (+ forecast fine-tune for E4a) -> checkpoint.

    `model` selects the architecture family; the data, objective and budget are
    identical for every family (that is the point of the baseline control).
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out = _ckpt_path(part, seed, model, tuned)
    if out.exists():
        return out
    cls = _model_cls(model)
    arch = _tuned(part, model, tuned)
    e1a, e1b = load_split("e1a"), load_split("e1b")
    dev = f"cuda:{gpu}" if gpu is not None else None
    if part == "a":
        # masked pretraining on sim, in the E1a observable/feature space
        sim = read_uts("data/uts", sources=["sim"])
        panel_s, _ = _panel_for(e1a | {"observables": e1a["observables"]}, sim)
        T = len(panel_s.ts)
        tr = np.arange(T) < int(T * 0.9)
        va = ~tr
        cfg = model_cfg(e1a, seed, {"pretrain": True, "epochs": 15, "patience": 3,
                                    "bptt": 32, **arch})
        cfg["norm_stats"] = {  # sim-platform stats, computed on the pretrain region only
            k: v for k, v in _sim_norm_stats(sim, panel_s, tr).items()
        }
        cfg["uts"] = sim
        if dev:
            cfg["device"] = dev
        m = cls()
        m.fit(panel_s, tr, va, cfg)
        # forecast fine-tune on the two real pretrain devices
        real = read_uts("data/uts", sources=["daqec"])
        panel_r, _ = _panel_for(e1a, real)
        tr_end, va_end = pd.Timestamp(e1a["train_end"]), pd.Timestamp(e1a["val_end"])
        trr = np.asarray(panel_r.ts < tr_end)
        var = np.asarray((panel_r.ts >= tr_end) & (panel_r.ts < va_end))
        stage1 = CKPT_DIR / (f"_simstage_{seed}.pt" if (model == "ltn" and not tuned)
                             else f"_simstage_{model}{'_tuned' if tuned else ''}_{seed}.pt")
        m.save(stage1)
        cfg2 = model_cfg(e1a, seed, {"epochs": 40, "patience": 6,
                                     "init_from": str(stage1),
                                     "devices": ["ibm_brisbane", "ibm_kyoto"], **arch})
        cfg2["uts"] = real
        if dev:
            cfg2["device"] = dev
        m2 = cls()
        m2.fit(panel_r, trr, var, cfg2)
        m2.save(out)
    else:
        sim = read_uts("data/uts", sources=["sim"])
        holdout = set(e1b["holdout_devices"])
        sim_in = sim[~sim["device"].isin(holdout)].reset_index(drop=True)
        panel_s, _ = _panel_for(e1b, sim_in)
        tr_end, va_end = pd.Timestamp(e1b["train_end"]), pd.Timestamp(e1b["val_end"])
        tr = np.asarray(panel_s.ts < tr_end)
        va = np.asarray((panel_s.ts >= tr_end) & (panel_s.ts < va_end))
        cfg = model_cfg(e1b, seed, {"epochs": 30, "patience": 5, **arch})
        cfg["uts"] = sim_in
        if dev:
            cfg["device"] = dev
        m = cls()
        m.fit(panel_s, tr, va, cfg)
        m.save(out)
    return out


def _sim_norm_stats(sim_uts, panel, tr_mask):
    stats = {}
    vals = panel.values[tr_mask]
    for j, k in panel.keys.iterrows():
        v = vals[:, j]
        v = v[~np.isnan(v)]
        key = ("sim_sc", k.observable, k.channel)
        stats.setdefault(key, []).append((np.nanmean(v), np.nanstd(v)))
    return {k: (float(np.mean([a for a, _ in vs])), float(np.mean([b for _, b in vs])) or 1.0)
            for k, vs in stats.items()}


def adapt_and_eval(part: str, k_days: int, arm: str, ckpt: Path | None,
                   gpu: int | None, seed: int, model: str = "ltn",
                   tuned: bool = False) -> pd.DataFrame:
    split = load_split("e1a" if part == "a" else "e1b")
    arch = _tuned(part, model, tuned)
    e4 = load_split("e4")[f"e4{part}"]
    grid_h = split.get("grid_hours", 8.0)
    steps_per_day = int(round(24.0 / grid_h))
    sources = ["daqec"] if part == "a" else ["sim"]
    uts = read_uts("data/uts", sources=sources)
    targets = [e4["holdout_device"]] if part == "a" else e4["holdout_devices"]
    panel, observables = _panel_for(split, uts)
    T = len(panel.ts)
    eval_start = int(T * 0.8)
    te = np.arange(T) >= eval_start
    adapt_lo = max(0, eval_start - k_days * steps_per_day)
    ad = (np.arange(T) >= adapt_lo) & (np.arange(T) < eval_start)
    va = ad.copy()  # tiny adaptation sets: early-stop on the adaptation window itself

    # adaptation protocol is fixed for every family: the tuned lr is deliberately
    # overridden by the E4 fine-tune lr so the arms differ only in initialization
    extra = {k: v for k, v in arch.items() if k != "lr"}
    cfg = model_cfg(split, seed, {"devices": targets, "epochs": 0 if k_days == 0 else 25,
                                  "patience": 4, "lr": 3e-4, **extra})
    cfg["uts"] = uts
    if gpu is not None:
        cfg["device"] = f"cuda:{gpu}"
    if arm == "pretrained":
        cfg["init_from"] = str(ckpt)
    elif k_days == 0:
        cfg["epochs"] = 0  # scratch k=0 = untrained baseline, reported as such
    m = _model_cls(model)()
    m.fit(panel, ad, va, cfg)
    pairs = forecast_pairs(panel, split["horizons_steps"], te)
    dev_of = panel.keys["device"].to_numpy()
    pairs = pairs[np.isin(dev_of[pairs["s_idx"].to_numpy()], targets)].reset_index(drop=True)
    mu, sigma = m.predict(pairs, panel)
    cells, _ = evaluate(pairs, panel, mu, sigma, parse_norm_stats(split))
    cells["k_days"], cells["arm"], cells["base"] = k_days, arm, model
    cells["n_params"] = int(m.n_params()) if hasattr(m, "n_params") else -1
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed-list", default=None,
                    help="comma-separated seeds (overrides --seeds), e.g. 0,2,4")
    ap.add_argument("--models", default="ltn",
                    help="comma-separated model families, e.g. ltn,lstm,transformer,tgn")
    ap.add_argument("--base", action="store_true",
                    help="write to runs/E4{a,b}-base/ (the baseline-control sweep)")
    ap.add_argument("--tuned", action="store_true",
                    help="give EVERY family its E1 Optuna arch (robustness variant); "
                         "off by default so all families share the E4 default protocol "
                         "and the LTN arm reproduces the published E4 numbers exactly")
    ap.add_argument("--ks", default=None, help="comma-separated adapt days (default: split)")
    args = ap.parse_args()
    tuned = args.tuned
    ks = ([int(x) for x in args.ks.split(",")] if args.ks
          else load_split("e4")[f"e4{args.part}"]["adapt_days"])
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = ([int(s) for s in args.seed_list.split(",")] if args.seed_list
             else list(range(args.seeds)))
    exp = f"E4{args.part}-base" if args.base else f"E4{args.part}"
    for model in models:
        for seed in seeds:
            ckpt = pretrain(args.part, args.gpu, seed, model, tuned)
            for k in ks:
                for arm in ("pretrained", "scratch"):
                    cells = adapt_and_eval(args.part, k, arm, ckpt, args.gpu, seed,
                                           model, tuned)
                    if tuned:  # keep tuned/untuned variants separable in analysis
                        cells["base"] = f"{model}-tuned"
                    label = f"{arm}-k{k}" if (model == "ltn" and not args.base) \
                        else f"{model}{'-tuned' if tuned else ''}-{arm}-k{k}"
                    with RunLogger(exp, label, seed,
                                   {"k_days": k, "arm": arm, "base": model,
                                    "tuned": tuned}, root="runs") as log:
                        for _, r in cells.iterrows():
                            log.log(**r.to_dict())
                    print(f"[e4{args.part}:{model}] seed{seed} k={k} {arm}: mean NCRPS "
                          f"{cells['ncrps'].mean():.4f} ({len(cells)} cells)", flush=True)


if __name__ == "__main__":
    main()
