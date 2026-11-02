"""Optuna HPO — 50-trial budget per model (prereg-binding; persistence exempt).

  python -m ltn.exp.tune --split e1a --model lstm [--trials 50] [--gpu 0]

Objective: mean per-cell NCRPS on the split's VALIDATION region. Tuning protocol
(identical for every model, declared in the paper): on large splits (e1b) tuning
runs on the frozen 4-device subset TUNE_DEVICES with a shortened schedule; final
5-seed runs use the tuned config on the full split. Best config -> runs/tuning/
<split>_<model>.json; the study db is runs/tuning/optuna.db (resumable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna

from ltn.data.series import forecast_pairs
from ltn.eval.splits import load_split
from ltn.exp.run import MODELS, _register, build_data, evaluate, model_cfg, parse_norm_stats

TUNE_DEVICES = {"e1b": ["sim_dev_00", "sim_dev_04", "sim_dev_08", "sim_dev_10"]}
TUNE_SCHEDULE = {"e1b": {"epochs": 20, "patience": 4}}


def search_space(model: str, trial: optuna.Trial) -> dict:
    if model == "ewma":
        return {"alpha": trial.suggest_float("alpha", 0.05, 1.0),
                "var_scale": trial.suggest_float("var_scale", 0.25, 1.0)}
    if model == "arima":
        return {"p": trial.suggest_int("p", 0, 3), "d": trial.suggest_int("d", 0, 1),
                "q": trial.suggest_int("q", 0, 2)}
    # Classical reliability/degradation baselines carry their own search spaces so
    # they get the same 50-trial budget as every other baseline, so the classical
    # models are not handicapped relative to the others.
    from ltn.baselines.reliability import SEARCH_SPACE as REL_SPACE
    if model in REL_SPACE:
        return REL_SPACE[model](trial)
    common = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "bptt": trial.suggest_categorical("bptt", [16, 32, 64]),
    }
    if model in ("lstm", "tgn", "transformer"):
        common["width"] = trial.suggest_int("width", 16, 192, log=True)
        if model == "transformer":
            common["n_layers"] = trial.suggest_int("n_layers", 1, 3)
        return common
    if model in ("ltn", "ltn_v2"):
        common["carry_dim"] = trial.suggest_int("carry_dim", 8, 64, log=True)
        common["hidden"] = trial.suggest_int("hidden", 32, 128, log=True)
        if model == "ltn":
            # KNOWN NO-OPS, kept only so this branch still reproduces the tuning
            # that produced the paper's LTN numbers: LTNForecaster._make_model
            # never forwards evidence_rank/rounds to LTNModel, so both stay at
            # their defaults (4, 2) whatever Optuna picks, and the values stored
            # in runs/tuning/*.json for them are inert. ltn_v2 deliberately omits
            # them so that both models tune the same FOUR effective dimensions
            # under the same 50-trial budget and share the same defaults --
            # otherwise a v2-vs-v1 win would be confounded by a wider search.
            common["evidence_rank"] = trial.suggest_categorical("evidence_rank", [2, 4])
            common["rounds"] = trial.suggest_categorical("rounds", [1, 2])
        return common
    raise ValueError(model)


def objective_factory(split_name: str, model_name: str, gpu: int | None):
    from ltn.data.series import subset_panel

    split = load_split(split_name)
    uts, panel, (tr, va, te) = build_data(split)
    tune_devs = TUNE_DEVICES.get(split_name)
    sched = TUNE_SCHEDULE.get(split_name, {})
    norm = parse_norm_stats(split)
    if tune_devs is not None:  # subset applies at the DATA level, identically for all models
        panel = subset_panel(panel, tune_devs)
    val_pairs = forecast_pairs(panel, split["horizons_steps"], va)

    def objective(trial: optuna.Trial) -> float:
        cfg = model_cfg(split, seed=0, extra=search_space(model_name, trial))
        cfg["uts"] = uts
        cfg.update(sched)
        if tune_devs is not None:
            cfg["devices"] = tune_devs
        if gpu is not None:
            cfg["device"] = f"cuda:{gpu}"
        m = MODELS[model_name]()
        m.fit(panel, tr, va, cfg)
        mu, sigma = m.predict(val_pairs, panel)
        cells, _ = evaluate(val_pairs, panel, mu, sigma, norm)
        return float(cells["ncrps"].mean())

    return objective


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trials", type=int, default=50,
                    help="TOTAL trial budget for this study (pre-registered; never exceeded)")
    ap.add_argument("--worker-trials", type=int, default=None,
                    help="trials for THIS worker when several share the study across GPUs. "
                         "The shares must be chosen to sum to the --trials budget: each worker "
                         "reads n_done once at startup, so concurrent workers cannot rely on "
                         "the cap alone to divide the remainder between them.")
    ap.add_argument("--gpu", type=int, default=None)
    args = ap.parse_args()
    _register()
    out = Path("runs/tuning")
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{args.split}_{args.model}",
        storage=f"sqlite:///{out}/optuna_{args.split}_{args.model}.db",  # one db per study: no cross-lane SQLite lock contention
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
        direction="minimize",
    )
    n_done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, args.trials - n_done)
    if args.worker_trials is not None:
        remaining = min(args.worker_trials, remaining)  # budget stays the hard cap
    if remaining:
        study.optimize(objective_factory(args.split, args.model, args.gpu),
                       n_trials=remaining, gc_after_trial=True)
    # n_trials counts COMPLETE trials only -- the number the paper's budget claim
    # refers to. len(study.trials) would also count FAILed trials and the RUNNING
    # records a killed worker leaves behind, overstating the budget actually spent.
    best = {"value": study.best_value, "params": study.best_params,
            "n_trials": len([t for t in study.trials if t.state.name == "COMPLETE"]),
            "n_trial_records": len(study.trials)}
    (out / f"{args.split}_{args.model}.json").write_text(json.dumps(best, indent=1))
    print(json.dumps(best))


if __name__ == "__main__":
    main()
