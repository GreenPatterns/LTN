"""E5 (H2b): zero-shot reliability ranking on a held-out platform (Google).

  python -m ltn.exp.e5 [--gpu 0] [--seeds 5] [--pretrain-ckpt PATH]

Protocol (frozen in splits/e5.json):
  Train an LTN forecaster on DAQEC (3 IBM devices) + silicon (episodic), never
  touching google_qec. Zero-shot inputs on Google are STRICTLY pre-outcome: per-qubit
  DEM error intensities (the fitted-noise calibration analog) + topology. Ranking
  target: pymatching-decoded logical_error_per_round per subgraph (code placement x
  distance x basis). Metric: Spearman rho, permutation null (10k, seed 0).

Declared mappings (also in the paper):
  * google per-qubit dem_error_prob (channel=claimed) enters the model through the
    epc_1q feature slot; per-subgraph/device means through gate_error_2q/derived —
    the semantically closest error observables in the training vocabulary.
  * Input z-scoring on the test platform uses google INPUT statistics only
    (dem_error_prob) — outcome columns are never touched before scoring.
  * Trust score per subgraph = predicted error (epc_1q slot) at horizon 1, negated
    for ranking; control 'raw_dem' ranks by mean input intensity directly (does the
    learned state add anything over the raw inputs?).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from ltn.data.series import make_panel, to_series
from ltn.data.uts import read_uts
from ltn.eval.runlog import RunLogger
from ltn.eval.splits import load_split
from ltn.exp.dataset import build_device_sequence
from ltn.exp.run import parse_norm_stats
from ltn.exp.train import LTNForecaster

E5_OBS = [("t1_us", "probe"), ("t2_us", "probe"), ("readout_error", "probe"),
          ("gate_error_2q", "claimed"), ("gate_error_2q", "derived"), ("epc_1q", "derived")]


def _norm_stats_from(uts: pd.DataFrame, observables) -> dict:
    out = {}
    want = {tuple(o) for o in observables}
    for (p, o, c), g in uts.groupby(["platform", "observable", "channel"], observed=True):
        if (o, c) in want:
            std = float(g["value"].std())
            out[(p, o, c)] = (float(g["value"].mean()), std if np.isfinite(std) and std > 0 else 1.0)
    return out


def google_as_model_inputs(guts: pd.DataFrame) -> pd.DataFrame:
    """Remap google inputs into the training observable vocabulary (declared)."""
    g = guts[guts["observable"] == "dem_error_prob"].copy()
    qubit = g["entity_type"] == "qubit"
    g.loc[qubit, "observable"] = "epc_1q"
    g.loc[qubit, "channel"] = "derived"
    g.loc[~qubit, "observable"] = "gate_error_2q"
    g.loc[~qubit, "channel"] = "derived"
    return g


def train_source_model(gpu, seed, pretrain_ckpt=None) -> tuple[LTNForecaster, dict]:
    uts = read_uts("data/uts", sources=["daqec", "silicon"])
    series = to_series(uts, observables=[o for o, _ in E5_OBS],
                       channels=list({c for _, c in E5_OBS}))
    keep = [(o, c) in E5_OBS for o, c in zip(series["observable"], series["channel"])]
    panel = make_panel(series[keep], uts)
    T = len(panel.ts)
    tr = np.arange(T) < int(T * 0.85)
    va = ~tr
    norm = _norm_stats_from(uts[uts["ts"] < panel.ts[int(T * 0.85) - 1]], E5_OBS)
    cfg = {
        "uts": uts, "observables": E5_OBS, "norm_stats": norm,
        "horizons": [1, 3], "time_unit_h": 8.0, "seed": seed,
        "epochs": 40, "patience": 6,
    }
    if pretrain_ckpt:
        cfg["init_from"] = pretrain_ckpt
    if gpu is not None:
        cfg["device"] = f"cuda:{gpu}"
    m = LTNForecaster()
    m.fit(panel, tr, va, cfg)
    return m, norm


@torch.no_grad()
def score_google(m: LTNForecaster, seed: int) -> pd.DataFrame:
    guts = read_uts("data/uts", sources=["google_qec"])
    inputs = google_as_model_inputs(guts)
    # google-input-only normalization (declared): stats from the remapped inputs
    gnorm = _norm_stats_from(inputs, E5_OBS)
    norm = dict(m._cfg_cache["norm_stats"])
    norm.update(gnorm)

    # target LER per (device, subgraph, run-ts); each pseudo-time snapshot IS one
    # run, so the run's qubits are exactly the qubit entities observed at that ts
    target = (guts[(guts["observable"] == "logical_error_per_round") & (guts["entity_type"] == "subgraph")]
              [["device", "entity_id", "ts", "value"]].rename(columns={"value": "ler"}))

    series = to_series(inputs, observables=[o for o, _ in E5_OBS], channels=list({c for _, c in E5_OBS}))
    panel = make_panel(series, inputs)
    model = m.model
    model.eval()
    dev = m._torch_device
    n_obs = len(E5_OBS)
    rows = []
    for d in sorted(inputs["device"].unique()):
        seq = build_device_sequence(inputs, panel, d, E5_OBS, norm, time_unit_h=8.0)
        x, cal = seq.x.to(dev), seq.cal_age_h.to(dev)
        ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
        st = model.init_state(len(seq.entity_ids), device=dev)
        is_qubit = torch.tensor([e.startswith("q") for e in seq.entity_ids])
        f_epc = seq.observables.index("epc_1q/derived") if "epc_1q/derived" in seq.observables else None
        if f_epc is None:
            continue
        for t in range(len(seq.ts)):
            st = model.step(st, x[t], cal[t], seq.dt_steps[t].expand(x.shape[1]).to(dev), ei, ea)
            mu, sigma, _ = model.forecast(
                st, x[t], cal[t], torch.ones(len(seq.entity_ids), dtype=torch.float64, device=dev))
            observed = (seq.x[t, :, n_obs + f_epc] > 0.5) & is_qubit  # this run's qubits
            if not observed.any():
                continue
            pred = float(mu[observed.to(dev), f_epc].mean())
            raw = float(seq.x[t, observed, f_epc].mean())
            rows.append({"device": d, "ts": seq.ts[t], "ltn_score": -pred, "raw_dem_score": -raw})
    sc = pd.DataFrame(rows)
    merged = sc.merge(target, on=["device", "ts"], how="inner")
    return (merged.groupby(["device", "entity_id"], observed=True)
            [["ltn_score", "raw_dem_score", "ler"]].mean().reset_index())


def spearman_with_null(scored: pd.DataFrame, col: str, n_perm: int = 10000, seed: int = 0) -> dict:
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    per_dev, pooled_s, pooled_r = {}, [], []
    for d, g in scored.groupby("device", observed=True):
        if len(g) < 4:
            continue
        rho = float(spearmanr(g[col], -g["ler"]).statistic)  # score high = LER low
        per_dev[d] = {"rho": rho, "n": len(g)}
        pooled_s.append(g[col].rank(pct=True).to_numpy())
        pooled_r.append((-g["ler"]).rank(pct=True).to_numpy())
    s, r = np.concatenate(pooled_s), np.concatenate(pooled_r)
    rho_pool = float(spearmanr(s, r).statistic)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = spearmanr(s, rng.permutation(r)).statistic
    p = float((np.abs(null) >= abs(rho_pool)).mean())
    return {"rho_pooled": rho_pool, "perm_p": p, "per_device": per_dev}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pretrain-ckpt", default=None)
    args = ap.parse_args()
    for seed in range(args.seeds):
        m, _ = train_source_model(args.gpu, seed, args.pretrain_ckpt)
        scored = score_google(m, seed)
        res_ltn = spearman_with_null(scored, "ltn_score")
        res_raw = spearman_with_null(scored, "raw_dem_score")
        with RunLogger("E5", "ltn", seed, {"pretrain": bool(args.pretrain_ckpt)}) as log:
            log.log(metric="spearman_pooled", ltn=res_ltn["rho_pooled"], ltn_p=res_ltn["perm_p"],
                    raw_dem=res_raw["rho_pooled"], raw_p=res_raw["perm_p"],
                    n_subgraphs=int(len(scored)))
            for d, v in res_ltn["per_device"].items():
                log.log(metric="spearman_device", device=d, rho=v["rho"], n=v["n"])
        scored.to_parquet(f"runs/E5/ltn/{seed}/scored.parquet", index=False)
        print(f"[e5] seed{seed}: ltn rho={res_ltn['rho_pooled']:.3f} (p={res_ltn['perm_p']:.4f}) "
              f"raw rho={res_raw['rho_pooled']:.3f} n={len(scored)}")


if __name__ == "__main__":
    main()
