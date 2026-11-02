"""Frozen experiment splits (pre-registration machinery, master plan §2.4/B1).

`python -m ltn.eval.splits --build`   builds splits/*.json + SHA256SUMS from the lake
`python -m ltn.eval.splits --verify-hashes`   fails CI if any frozen file changed

Every split file carries: the data selection (sources/devices/observables/channels),
temporal boundaries (ISO timestamps), holdout definitions, and the TRAIN-ONLY
normalization statistics used by every model (the leakage guard lives here, not in
model code). Committed to git; hashes are the tamper seal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ltn.data.series import make_panel, split_panel_by_date, to_series
from ltn.data.uts import read_uts

SPLIT_DIR = Path("splits")
LAKE = Path("data/uts")

E1A_OBSERVABLES = [["t1_us", "probe"], ["t2_us", "probe"], ["readout_error", "probe"], ["gate_error_2q", "claimed"]]
E1B_OBSERVABLES = [["t1_us", "probe"], ["t2_us", "probe"], ["readout_error", "probe"], ["epc_1q", "probe"], ["gate_error_2q", "probe"]]
E1A_HORIZONS_STEPS = [3, 9, 21]        # 1/3/7 days on the 8h DAQEC grid [DEV-3]
E1B_HORIZONS_STEPS = [1, 3, 7, 14]     # 1/3/7/14 days on the DAILY sim grid (prereg-v1.1)
SIM_HOLDOUT_DEVICES = ["sim_dev_12", "sim_dev_13", "sim_dev_14", "sim_dev_15"]


def _norm_stats(uts: pd.DataFrame, observables, train_end: str) -> dict:
    """{(platform|observable|channel) -> [mean, std]} from events strictly before train_end."""
    tr = uts[uts["ts"] < pd.Timestamp(train_end)]
    out = {}
    for (p, o, c), g in tr.groupby(["platform", "observable", "channel"], observed=True):
        if [o, c] in observables or o == "gate_error_2q":
            std = float(g["value"].std())
            out[f"{p}|{o}|{c}"] = [float(g["value"].mean()), std if np.isfinite(std) and std > 0 else 1.0]
    return out


def _temporal_bounds(uts: pd.DataFrame, observables, train_frac=0.7, val_frac=0.1) -> tuple[str, str]:
    series = to_series(uts, observables=[o for o, _ in observables], channels=list({c for _, c in observables}))
    keep = [(o, c) in [tuple(x) for x in observables] for o, c in zip(series["observable"], series["channel"])]
    panel = make_panel(series[keep], uts)
    tr, va, te = split_panel_by_date(panel, train_frac, val_frac)
    train_end = panel.ts[np.nonzero(tr)[0][-1] + 1].isoformat()
    val_end = panel.ts[np.nonzero(va)[0][-1] + 1].isoformat() if va.any() else train_end
    return train_end, val_end


def build_all(lake_root: Path = LAKE, out: Path = SPLIT_DIR) -> dict[str, Path]:
    out.mkdir(exist_ok=True)
    written = {}

    # ---- E1a: DAQEC temporal --------------------------------------------------
    daqec = read_uts(lake_root, sources=["daqec"])
    tr_end, va_end = _temporal_bounds(daqec, E1A_OBSERVABLES)
    e1a = {
        "experiment": "E1a",
        "sources": ["daqec"],
        "observables": E1A_OBSERVABLES,
        "horizons_steps": E1A_HORIZONS_STEPS,
        "grid_hours": 8.0,
        "train_end": tr_end,
        "val_end": va_end,
        "norm_stats": _norm_stats(daqec, E1A_OBSERVABLES, tr_end),
    }
    written["e1a"] = out / "e1a.json"

    # ---- E1b: sim temporal + device holdout ------------------------------------
    sim = read_uts(lake_root, sources=["sim"])
    e1b = {"experiment": "E1b", "sources": ["sim"], "observables": E1B_OBSERVABLES,
           "horizons_steps": E1B_HORIZONS_STEPS, "grid_hours": 24.0,
           "holdout_devices": SIM_HOLDOUT_DEVICES}
    if len(sim):
        sim_in = sim[~sim["device"].isin(SIM_HOLDOUT_DEVICES)]
        tr_end_b, va_end_b = _temporal_bounds(sim_in, E1B_OBSERVABLES)
        e1b.update(train_end=tr_end_b, val_end=va_end_b,
                   norm_stats=_norm_stats(sim_in, E1B_OBSERVABLES, tr_end_b))
    else:
        e1b.update(train_end=None, val_end=None, norm_stats={},
                   note="sim lake not yet generated at build time; rebuild after sim-data")
    written["e1b"] = out / "e1b.json"

    # ---- E3: divergence, 3-fold device rotation --------------------------------
    e3 = {
        "experiment": "E3",
        "sources": ["daqec"],
        "folds": [{"train": ["ibm_brisbane", "ibm_kyoto"], "test": ["ibm_osaka"]},
                  {"train": ["ibm_brisbane", "ibm_osaka"], "test": ["ibm_kyoto"]},
                  {"train": ["ibm_kyoto", "ibm_osaka"], "test": ["ibm_brisbane"]}],
        "divergence_def": {"stat": "abs(probe-claimed)/claimed", "z_threshold": 2.0,
                           "z_ref": "train fold, per device x observable"},
        "label_horizons_snapshots": [1, 2, 3],
        "observables": [["t1_us", "probe"], ["t2_us", "probe"], ["readout_error", "probe"]],
        "pretrain_sources_allowed": ["sim", "silicon"],
    }
    written["e3"] = out / "e3.json"

    # ---- E4: cold start ---------------------------------------------------------
    e4 = {
        "experiment": "E4",
        "e4a": {"holdout_device": "ibm_osaka", "adapt_days": [0, 1, 3, 7],
                "pretrain": ["ibm_brisbane", "ibm_kyoto", "silicon", "sim"],
                "eval": "final 20% of ibm_osaka grid"},
        "e4b": {"holdout_devices": SIM_HOLDOUT_DEVICES, "adapt_days": [0, 1, 7, 30],
                "pretrain": "sim non-holdout", "eval": "final 20% of holdout device grids"},
    }
    written["e4"] = out / "e4.json"

    # ---- E5: platform holdout ----------------------------------------------------
    e5 = {
        "experiment": "E5",
        "train_sources": ["daqec", "silicon"],
        "pretrain_sources_allowed": ["sim"],
        "test_source": "google_qec",
        "forbidden_in_training": ["google_qec"],
        "ranking_target": ["logical_error_per_round", "delivered"],
        "inputs": [["dem_error_prob", "claimed"]],
        "rank_over": "subgraph entities per google device",
        "metric": "spearman",
        "null": "permutation, 10000 resamples, seed 0",
    }
    written["e5"] = out / "e5.json"

    # ---- E6: regret replay ---------------------------------------------------------
    e6 = {
        "experiment": "E6",
        "e6a": {"world": "sim", "replay_days": [91, 180], "jobs_widths": [5, 12, 27],
                "policies": ["random", "greedy_claimed", "ltn_mean", "ltn_lcb", "oracle"],
                "lcb_kappa": 1.0, "seed": 0},
        "e6b": {"world": "daqec_logged", "region": "test (after e1a val_end)",
                "choice_set": "3 backends x logged subgraphs per session x distance",
                "outcome": ["logical_error_rate", "delivered"], "declared": "secondary [DEV-5]"},
    }
    written["e6"] = out / "e6.json"

    # ---- E8: bitcoin --------------------------------------------------------------
    e8 = {
        "experiment": "E8",
        "datasets": {
            "bitcoin_otc": "data/raw/snap/soc-sign-bitcoinotc.csv.gz",
            "bitcoin_alpha": "data/raw/snap/soc-sign-bitcoinalpha.csv.gz",
        },
        "split": {"temporal_edge_fracs": [0.7, 0.1, 0.2]},
        "task": "next-window edge weight prediction (scaled to [0,1])",
        "control": "vector+GRU state, same pipeline, param-matched",
    }
    written["e8"] = out / "e8.json"

    payloads = {"e1a": e1a, "e1b": e1b, "e3": e3, "e4": e4, "e5": e5, "e6": e6, "e8": e8}
    for k, path in written.items():
        path.write_text(json.dumps(payloads[k], indent=1, sort_keys=True) + "\n")
    _write_hashes(out)
    return written


def _write_hashes(out: Path) -> None:
    lines = []
    for f in sorted(out.glob("*.json")):
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append(f"{h}  {f.name}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def verify_hashes(out: Path = SPLIT_DIR) -> bool:
    sums = (out / "SHA256SUMS").read_text().strip().splitlines()
    ok = True
    for line in sums:
        h, name = line.split(None, 1)
        f = out / name.strip()
        actual = hashlib.sha256(f.read_bytes()).hexdigest()
        if actual != h:
            print(f"HASH MISMATCH: {name}")
            ok = False
    print("splits: all hashes ok" if ok else "splits: FROZEN FILES CHANGED")
    return ok


def load_split(name: str, out: Path = SPLIT_DIR) -> dict:
    return json.loads((out / f"{name}.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify-hashes", action="store_true")
    args = ap.parse_args()
    if args.build:
        w = build_all()
        print(f"wrote {len(w)} split files + SHA256SUMS")
    if args.verify_hashes:
        raise SystemExit(0 if verify_hashes() else 1)


if __name__ == "__main__":
    main()
