"""E7: uncertainty quality — calibration + the two event studies behind F7.

  python -m ltn.exp.e7 [--gpu 0] [--ckpt-e1a runs/E1a/ltn/0/model.pt] [--ckpt-e1b runs/E1b/ltn/0/model.pt]

Part 1  Calibration: PIT coverage at levels {50,80,90,95}% and Gaussian NLL on the
        frozen test pairs of E1a and E1b, before/after temperature scaling fit on the
        VALIDATION pairs only.
Part 2  Gap injection (sim): hide L in {1..5} consecutive snapshots of held-out test
        entities; record predicted state entropy S(rho_t) through the gap vs the
        empirical |z-error| of the model's h=0 reconstruction at each gap position
        (truth known on sim). The pre-registered claim: the two growth curves match.
Part 3  Recalibration event study: dissonance/entropy trajectories in the 6 snapshots
        around real DAQEC recalibrations (cal_ts jumps) and sim recals.

Outputs runs/E7/event_study.parquet with rows
  (t_rel, entropy_pred, err_emp, dissonance, event_type in {gap, recal_real, recal_sim})
— exactly the schema figures.py's F7 consumes — plus calibration cells via RunLogger.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ltn.data.series import forecast_pairs, make_panel, to_series
from ltn.data.uts import read_uts
from ltn.eval.calibrate import TemperatureScaler
from ltn.eval.metrics import coverage, gaussian_nll
from ltn.eval.runlog import RunLogger
from ltn.eval.splits import load_split
from ltn.exp.probe import extract_states
from ltn.exp.run import build_data, model_cfg, parse_norm_stats
from ltn.exp.train import LTNForecaster

OUT = Path("runs/E7")
MODEL_NAME = "ltn"  # set by main(); "ltn_v2" evaluates the Tier-1 variant


def _load_forecaster(split_name: str, ckpt: str, gpu) -> tuple[LTNForecaster, object, tuple]:
    from ltn.exp.run import MODELS, _register
    _register()
    split = load_split(split_name)
    uts, panel, (tr, va, te) = build_data(split)
    cfg = model_cfg(split, 0, {"epochs": 0, "init_from": ckpt})
    cfg["uts"] = uts
    if gpu is not None:
        cfg["device"] = f"cuda:{gpu}"
    m = MODELS[MODEL_NAME]()
    m.fit(panel, tr, va, cfg)
    return m, panel, (tr, va, te, split)


def part1_calibration(split_name: str, ckpt: str, gpu) -> None:
    m, panel, (tr, va, te, split) = _load_forecaster(split_name, ckpt, gpu)
    va_pairs = forecast_pairs(panel, split["horizons_steps"], va)
    te_pairs = forecast_pairs(panel, split["horizons_steps"], te)
    mu_v, sig_v = m.predict(va_pairs, panel)
    mu_t, sig_t = m.predict(te_pairs, panel)
    scaler = TemperatureScaler().fit(va_pairs["y"].to_numpy(), mu_v, sig_v)
    sig_t_cal = scaler.transform(sig_t)
    y = te_pairs["y"].to_numpy()
    with RunLogger(OUT.name, f"calibration-{split_name}", 0, {"ckpt": ckpt, "temperature": float(scaler.t_)}) as log:
        for lv in (0.5, 0.8, 0.9, 0.95):
            log.log(cell="coverage", level=lv,
                    raw=float(coverage(y, mu_t, sig_t, lv)),
                    temp_scaled=float(coverage(y, mu_t, sig_t_cal, lv)))
        log.log(cell="nll", raw=float(gaussian_nll(y, mu_t, sig_t)),
                temp_scaled=float(gaussian_nll(y, mu_t, sig_t_cal)))
    print(f"[e7:cal:{split_name}] T={scaler.t_:.3f} cov90 raw="
          f"{coverage(y, mu_t, sig_t, 0.9):.3f} scaled={coverage(y, mu_t, sig_t_cal, 0.9):.3f}")


@torch.no_grad()
def part2_gap_injection(ckpt: str, gpu, n_events: int = 60, max_len: int = 5, seed: int = 0) -> list[dict]:
    m, panel, (tr, va, te, split) = _load_forecaster("e1b", ckpt, gpu)
    model = m.model
    model.eval()
    dev = m._torch_device
    rng = np.random.default_rng(seed)
    n_obs_count = len(split["observables"])
    rows = []
    devices = sorted(m.seqs)
    te_idx = np.nonzero(te)[0]
    ts_to_row = {t: i for i, t in enumerate(panel.ts)}
    for _ in range(n_events):
        d = devices[int(rng.integers(len(devices)))]
        seq = m.seqs[d]
        rows_d = np.array([ts_to_row[t] for t in seq.ts])
        local_test = np.nonzero(np.isin(rows_d, te_idx))[0]
        if len(local_test) < max_len + 8:
            continue
        g0 = int(rng.choice(local_test[3:-max_len - 1]))
        L = int(rng.integers(1, max_len + 1))
        x = seq.x.clone()
        # inject the gap: wipe values + observed-mask for all entities in [g0, g0+L)
        x[g0:g0 + L, :, :n_obs_count] = 0.0
        x[g0:g0 + L, :, n_obs_count:2 * n_obs_count] = 0.0
        xd, cal = x.to(dev), seq.cal_age_h.to(dev)
        ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
        st = model.init_state(len(seq.entity_ids), device=dev)
        for t in range(min(g0 + L + 2, len(seq.ts))):
            st = model.step(st, xd[t], cal[t], seq.dt_steps[t].expand(x.shape[1]).to(dev), ei, ea)
            k = t - g0  # position relative to gap start
            if -3 <= k <= L + 1:
                mu, sigma, _ = model.forecast(
                    st, xd[t], cal[t], torch.zeros(x.shape[1], dtype=torch.float64, device=dev))
                truth = seq.y[t]  # z-space truth (probe values, hidden from the model in the gap)
                obs = ~torch.isnan(truth)
                err = float((truth[obs] - mu.cpu()[obs]).abs().mean()) if obs.any() else np.nan
                rows.append({"t_rel": k, "gap_len": L,
                             "entropy_pred": float(st.entropy.mean()),
                             "dissonance": float(st.dissonance.mean()),
                             # predictive sigma at the h=0 readout — declared POST-HOC
                             # diagnostic (prereg specified entropy; see paper §5)
                             "sigma_pred": float(sigma.cpu()[obs].mean()) if obs.any() else np.nan,
                             "err_emp": err, "event_type": "gap"})
    return rows


def part3_recal_events(split_name: str, ckpt: str, gpu, event_type: str) -> list[dict]:
    m, panel, _ = _load_forecaster(split_name, ckpt, gpu)
    states = extract_states(m)
    uts = read_uts("data/uts", sources=load_split(split_name)["sources"])
    cal = uts.dropna(subset=["cal_ts"]).groupby(["device", "ts"], observed=True)["cal_ts"].max().reset_index()
    rows = []
    for d, g in cal.groupby("device", observed=True):
        g = g.sort_values("ts").reset_index(drop=True)
        recals = g.index[g["cal_ts"].diff() > pd.Timedelta(0)]
        sd = states[states["device"] == d]
        ent_mean = sd.groupby("ts")[["entropy", "dissonance"]].mean()
        for ridx in recals:
            ts0 = g.loc[ridx, "ts"]
            pos = ent_mean.index.get_indexer([ts0])
            if pos[0] < 0:
                continue
            p = int(pos[0])
            for k in range(-6, 3):
                if 0 <= p + k < len(ent_mean):
                    r = ent_mean.iloc[p + k]
                    rows.append({"t_rel": k, "entropy_pred": float(r["entropy"]),
                                 "dissonance": float(r["dissonance"]),
                                 "err_emp": np.nan, "event_type": event_type})
    return rows


def main() -> None:
    global OUT, MODEL_NAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--ckpt-e1a", default="runs/E1a/ltn/0/model.pt")
    ap.add_argument("--ckpt-e1b", default="runs/E1b/ltn/0/model.pt")
    ap.add_argument("--model", default="ltn", help="forecaster registry name the ckpts belong to")
    ap.add_argument("--out", default="runs/E7", help="output dir (use runs/E7-v2 for ltn_v2)")
    args = ap.parse_args()
    MODEL_NAME = args.model
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    part1_calibration("e1a", args.ckpt_e1a, args.gpu)
    part1_calibration("e1b", args.ckpt_e1b, args.gpu)
    rows = part2_gap_injection(args.ckpt_e1b, args.gpu)
    rows += part3_recal_events("e1a", args.ckpt_e1a, args.gpu, "recal_real")
    rows += part3_recal_events("e1b", args.ckpt_e1b, args.gpu, "recal_sim")
    ev = pd.DataFrame(rows)
    extra = OUT / "e3_divergence_events.parquet"
    if extra.exists():
        ev = pd.concat([ev, pd.read_parquet(extra).assign(event_type="divergence")], ignore_index=True)
    ev.to_parquet(OUT / "event_study.parquet", index=False)
    print(f"[e7] event rows: {len(ev)} -> {OUT/'event_study.parquet'}")


if __name__ == "__main__":
    main()
