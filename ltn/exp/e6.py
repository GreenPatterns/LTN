"""E6 (H3): trust-aware backend/subgraph selection — regret replay.

  python -m ltn.exp.e6 --world sim --checkpoint runs/E1b/ltn/0/model.pt [--gpu 0]

E6a (primary): sim replay on the 12 train devices, TEST-region days only (the frozen
e6.json window [91,180] includes model-train days; we restrict to the post-validation
region and declare it — the model must never be scored where it trained).
Truth channel drives outcomes; policies see claimed values or LTN forecasts (h=1).
E6b (secondary [DEV-5]): DAQEC logged-arms replay, reward = 1 - delivered LER.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ltn.data.series import make_panel, to_series
from ltn.data.uts import read_uts
from ltn.eval.regret_sim import (POLICIES, Arm, connected_candidates,
                                 make_policy_ltn_lcb, replay, success_prob)
from ltn.eval.runlog import RunLogger
from ltn.eval.splits import load_split
from ltn.exp.run import model_cfg, parse_norm_stats
from ltn.exp.train import LTNForecaster

ERR_OBS = ["readout_error", "epc_1q", "gate_error_2q"]


def device_topology(uts_dev: pd.DataFrame) -> list[tuple[int, int]]:
    edges = []
    for e in uts_dev.loc[uts_dev["entity_type"] == "coupler", "entity_id"].unique():
        try:
            a, b = e[1:].split("-")
            edges.append((int(a), int(b)))
        except ValueError:
            continue
    return sorted(set(edges))


def _values_at(df: pd.DataFrame, ts, obs: str) -> dict:
    """Latest value per entity at or before ts for one observable."""
    d = df[(df["observable"] == obs) & (df["ts"] <= ts)]
    if d.empty:
        return {}
    d = d.sort_values("ts").groupby("entity_id", observed=True).last()
    return d["value"].to_dict()


def _qubit_map(vals: dict) -> dict[int, float]:
    out = {}
    for e, v in vals.items():
        if e.startswith("q") and "_" not in e:
            try:
                out[int(e[1:])] = float(v)
            except ValueError:
                pass
    return out


def _coupler_map(vals: dict) -> dict[tuple[int, int], float]:
    out = {}
    for e, v in vals.items():
        if e.startswith("c") and "-" in e:
            try:
                a, b = e[1:].split("-")
                out[(int(a), int(b))] = float(v)
            except ValueError:
                pass
    return out


@torch.no_grad()
def ltn_forecasts(forecaster: LTNForecaster, horizon: float = 1.0) -> dict:
    """{device: {ts: {obs: {entity_key: (mu, sigma)}}}} in RAW units."""
    model = forecaster.model
    model.eval()
    dev = forecaster._torch_device
    out: dict = {}
    for d, seq in forecaster.seqs.items():
        x, cal = seq.x.to(dev), seq.cal_age_h.to(dev)
        ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
        st = model.init_state(len(seq.entity_ids), device=dev)
        per_ts = {}
        for t in range(len(seq.ts)):
            st = model.step(st, x[t], cal[t], seq.dt_steps[t].expand(x.shape[1]).to(dev), ei, ea)
            mu, sigma, _ = model.forecast(
                st, x[t], cal[t], torch.full((len(seq.entity_ids),), horizon, dtype=torch.float64, device=dev))
            mu, sigma = mu.cpu().numpy(), sigma.cpu().numpy()
            by_obs: dict = {}
            for f, lbl in enumerate(seq.observables):
                obs = lbl.split("/")[0]
                if obs not in ERR_OBS:
                    continue
                m0, s0 = seq.norm.get(lbl, (0.0, 1.0))
                ent_map = {}
                for i, e in enumerate(seq.entity_ids):
                    ent_map[e] = (float(mu[i, f] * s0 + m0), float(abs(sigma[i, f] * s0)))
                by_obs[obs] = ent_map
            per_ts[seq.ts[t]] = by_obs
        out[d] = per_ts
    return out


def _pred_feature_maps(by_obs: dict) -> dict:
    def qmap(obs):
        return {int(e[1:]): v for e, v in by_obs.get(obs, {}).items()
                if e.startswith("q") and "_" not in e and e[1:].isdigit()}

    g2 = {}
    for e, v in by_obs.get("gate_error_2q", {}).items():
        if e.startswith("c") and "-" in e:
            try:
                a, b = e[1:].split("-")
                g2[(int(a), int(b))] = v
            except ValueError:
                pass
    return {"readout_error": qmap("readout_error"), "epc_1q": qmap("epc_1q"), "gate_error_2q": g2}


def run_sim_world(ckpt: str, gpu: int | None, seeds: int) -> None:
    e6 = load_split("e6")["e6a"]
    e1b = load_split("e1b")
    uts = read_uts("data/uts", sources=["sim"])
    uts = uts[~uts["device"].isin(e1b["holdout_devices"])].reset_index(drop=True)
    truth = read_uts("data/sim_truth", include_truth=True)
    truth = truth[truth["channel"] == "truth"]

    # rebuild the trained forecaster (epochs=0 + init_from)
    observables = [tuple(x) for x in e1b["observables"]]
    series = to_series(uts, observables=[o for o, _ in observables],
                       channels=list({c for _, c in observables}))
    keep = [(o, c) in observables for o, c in zip(series["observable"], series["channel"])]
    panel = make_panel(series[keep], uts)
    T = len(panel.ts)
    cfg = model_cfg(e1b, 0, {"epochs": 0, "init_from": ckpt})
    cfg["uts"] = uts
    if gpu is not None:
        cfg["device"] = f"cuda:{gpu}"
    m = LTNForecaster()
    dummy = np.zeros(T, dtype=bool)
    m.fit(panel, dummy, dummy, cfg)
    preds = ltn_forecasts(m)

    va_end = pd.Timestamp(e1b["val_end"])
    replay_ts = [t for t in panel.ts if t >= va_end]
    devices = sorted(uts["device"].unique())
    widths = e6["jobs_widths"]
    kappa = float(e6.get("lcb_kappa", 1.0))

    truth_by_dev = {d: g for d, g in truth.groupby("device", observed=True)}
    claimed_by_dev = {d: g[g["channel"] == "claimed"] for d, g in uts.groupby("device", observed=True)}
    topo = {d: device_topology(uts[uts["device"] == d]) for d in devices}
    cands = {(d, w): connected_candidates(topo[d], w, k=20, seed=0)
             for d in devices for w in widths if topo[d]}

    Path("runs/E6a").mkdir(parents=True, exist_ok=True)
    policies = dict(POLICIES)
    policies["ltn_lcb"] = make_policy_ltn_lcb(kappa)

    for w in widths:
        arms_by_t: dict = {}
        for ts in replay_ts:
            arms = []
            for d in devices:
                if (d, w) not in cands:
                    continue
                tr_ro = _qubit_map(_values_at(truth_by_dev[d], ts, "readout_error"))
                tr_e1 = _qubit_map(_values_at(truth_by_dev[d], ts, "epc_1q"))
                tr_g2 = _coupler_map(_values_at(truth_by_dev[d], ts, "gate_error_2q"))
                cl_ro = _qubit_map(_values_at(claimed_by_dev[d], ts, "readout_error"))
                cl_e1 = _qubit_map(_values_at(claimed_by_dev[d], ts, "epc_1q"))
                cl_g2 = _coupler_map(_values_at(claimed_by_dev[d], ts, "gate_error_2q"))
                p_ts = preds[d].get(ts, {})
                pred_maps = _pred_feature_maps(p_ts) if p_ts else None
                for S in cands[(d, w)]:
                    feats = {
                        "claimed": {"readout_error": {q: (v, 0.0) for q, v in cl_ro.items()},
                                    "epc_1q": {q: (v, 0.0) for q, v in cl_e1.items()},
                                    "gate_error_2q": {e: (v, 0.0) for e, v in cl_g2.items()}},
                    }
                    if pred_maps:
                        feats["pred"] = pred_maps
                    arms.append(Arm(
                        device=d, subset=S, features=feats,
                        truth_reward=success_prob(S, topo[d], tr_ro, tr_e1, tr_g2),
                    ))
            arms_by_t[ts] = [a for a in arms if "pred" in a.features]
        for pname, pol in policies.items():
            n_seeds = seeds if pname == "random" else 1
            for s in range(n_seeds):
                df = replay(arms_by_t, pol, seed=s)
                df.to_parquet(f"runs/E6a/replay_{pname}_w{w}_s{s}.parquet", index=False)
                with RunLogger("E6a", f"{pname}-w{w}", s, {"width": w, "kappa": kappa,
                                                           "window": "test-region (declared)"}) as log:
                    log.log(cum_regret=float(df["regret"].sum()),
                            mean_regret=float(df["regret"].mean()),
                            mean_reward=float(df["reward"].mean()), n_rounds=len(df))
                print(f"[e6a] {pname} w={w} s={s}: cum_regret={df['regret'].sum():.4f}")


def run_logged_world(seeds: int, ckpt: str = "runs/E1a/ltn/0/model.pt",
                     gpu: int | None = None) -> None:
    e1a = load_split("e1a")
    uts = read_uts("data/uts", sources=["daqec"])
    va_end = pd.Timestamp(e1a["val_end"])
    ler = uts[(uts["observable"] == "logical_error_rate") & (uts["channel"] == "delivered")
              & (uts["entity_type"] == "subgraph") & (uts["ts"] >= va_end)].copy()
    ler["distance"] = ler["context"].map(lambda c: json.loads(c).get("distance"))

    # LTN arm scores from the E1a checkpoint: predicted composite error per subgraph
    pred_score = {}
    if Path(ckpt).exists():
        observables = [tuple(x) for x in e1a["observables"]]
        series = to_series(uts, observables=[o for o, _ in observables],
                           channels=list({c for _, c in observables}))
        keep = [(o, c) in observables for o, c in zip(series["observable"], series["channel"])]
        panel = make_panel(series[keep], uts)
        cfg = model_cfg(e1a, 0, {"epochs": 0, "init_from": ckpt})
        cfg["uts"] = uts
        if gpu is not None:
            cfg["device"] = f"cuda:{gpu}"
        m = LTNForecaster()
        T = len(panel.ts)
        dummy = np.zeros(T, dtype=bool)
        m.fit(panel, dummy, dummy, cfg)
        fc = ltn_forecasts(m)  # raw-unit (mu, sigma) for error observables at h=1
        for d, per_ts in fc.items():
            for ts, by_obs in per_ts.items():
                for ent in set(by_obs.get("readout_error", {})) | set(by_obs.get("gate_error_2q", {})):
                    if not ent.startswith("sub:"):
                        continue
                    mu = sig = 0.0
                    for obs in ("readout_error", "gate_error_2q"):
                        if ent in by_obs.get(obs, {}):
                            mu += by_obs[obs][ent][0]
                            sig += by_obs[obs][ent][1]
                    pred_score[(d, ent, ts)] = (mu, sig)

    claimed = uts[(uts["channel"] == "claimed") & (uts["entity_type"] == "subgraph")
                  & (uts["observable"].isin(["readout_error", "gate_error_2q"]))]
    claimed_sum = claimed.groupby(["device", "entity_id", "ts"], observed=True)["value"].sum()

    kappa = 1.0
    arms_by_t: dict = {}
    for (ts, dist), g in ler.groupby(["ts", "distance"], observed=True):
        arms = []
        for _, r in g.iterrows():
            key = (r["device"], r["entity_id"], ts)
            feats = {"pred_err": pred_score.get(key), "claimed_err": claimed_sum.get(key)}
            arms.append(Arm(device=r["device"], subset=frozenset([r["entity_id"]]),
                            features=feats, truth_reward=1.0 - float(r["value"])))
        if len(arms) >= 2:
            arms_by_t[(ts, dist)] = arms

    def pol_greedy(arms, rng):
        vals = [a.features["claimed_err"] if a.features["claimed_err"] is not None else np.inf
                for a in arms]
        return int(np.argmin(vals))

    def pol_ltn(kap):
        def p(arms, rng):
            vals = []
            for a in arms:
                pe = a.features["pred_err"]
                vals.append(pe[0] + kap * pe[1] if pe is not None else np.inf)
            return int(np.argmin(vals))
        return p

    policies = {"random": POLICIES["random"], "oracle": POLICIES["oracle"],
                "greedy_claimed": pol_greedy, "ltn_mean": pol_ltn(0.0),
                "ltn_lcb": pol_ltn(kappa)}
    Path("runs/E6b").mkdir(parents=True, exist_ok=True)
    for pname, pol in policies.items():
        if pname.startswith("ltn") and not pred_score:
            print(f"[e6b] skip {pname}: no checkpoint")
            continue
        for s in range(seeds if pname == "random" else 1):
            df = replay(arms_by_t, pol, seed=s)
            df.to_parquet(f"runs/E6b/replay_{pname}_s{s}.parquet", index=False)
            with RunLogger("E6b", pname, s, {"note": "secondary, logged-arms partial feedback [DEV-5]",
                                             "kappa": kappa}) as log:
                log.log(cum_regret=float(df["regret"].sum()), mean_regret=float(df["regret"].mean()),
                        mean_reward=float(df["reward"].mean()), n_rounds=len(df))
            print(f"[e6b] {pname} s={s}: cum_regret={df['regret'].sum():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=["sim", "logged", "both"], default="both")
    ap.add_argument("--checkpoint", default="runs/E1b/ltn/0/model.pt")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    if args.world in ("sim", "both"):
        run_sim_world(args.checkpoint, args.gpu, args.seeds)
    if args.world in ("logged", "both"):
        run_logged_world(args.seeds, ckpt="runs/E1a/ltn/0/model.pt", gpu=args.gpu)


if __name__ == "__main__":
    main()
