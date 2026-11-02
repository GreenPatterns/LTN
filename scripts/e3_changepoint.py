"""Classical change-point / drift detectors on E3's divergence-anticipation task.

Change-point and drift-detection baselines for E3. Unlike the degradation models in ``ltn.baselines.reliability``, these are
not forecasters -- they emit an anomaly score, not a predictive (mu, sigma) -- so
they cannot enter the NCRPS leaderboard. They belong on E3, whose endpoint is AUROC
for anticipating a claim-versus-probe divergence event, and where the paper reports a
pre-registered null (LTN dissonance 0.497, LTN state probe 0.511, staleness control
0.459, best-anywhere last-discrepancy carry-forward 0.58, against a 0.75 bar).

This is a genuine test of the paper's conclusion, not a formality. The paper argues
that divergence is untestable at DAQEC's cadence because claims are refreshed faster
than the drift timescale. If a classical detector also fails, that conclusion is
strengthened by a method built for exactly this job. If one succeeds, that is a
finding against LTN and must be reported as such.

  python scripts/e3_changepoint.py

Uses the SAME fold rotation, labels and z-threshold as ltn/exp/e3.py, so the numbers
sit next to the paper's without a protocol caveat.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ltn.baselines.reliability import binseg_scores, cusum_scores, page_hinkley_scores
from ltn.data.uts import read_uts
from ltn.exp.e3 import discrepancy_frame, make_labels

OUT = Path("runs/E3")
Z_THR = 2.0
HORIZONS = [1, 2, 3]

#: (name, scorer, causal). Non-causal detectors are reported separately as an
#: optimistic bound -- never as a peer of the causal forecasters.
DETECTORS = [
    ("CUSUM", lambda x: cusum_scores(x, k=0.5), True),
    ("Page--Hinkley", lambda x: page_hinkley_scores(x, delta=0.005), True),
    ("Binary segmentation", lambda x: binseg_scores(x, n_bkps=3), False),
]


def _entity_scores(disc: pd.DataFrame, scorer) -> pd.DataFrame:
    """Score each (device, entity) series, aggregating observables by max."""
    rows = []
    for (dev, ent, obs), g in disc.groupby(["device", "entity_id", "observable"],
                                           observed=True):
        g = g.sort_values("ts")
        s = scorer(g["rel_disc"].to_numpy())
        rows.append(pd.DataFrame({"device": dev, "entity_id": ent, "ts": g["ts"].to_numpy(),
                                  "observable": obs, "score": s}))
    if not rows:
        return pd.DataFrame(columns=["device", "entity_id", "ts", "score"])
    allr = pd.concat(rows)
    return (allr.groupby(["device", "entity_id", "ts"], observed=True)["score"]
            .max().reset_index())


def main() -> None:
    uts = read_uts("data/uts")
    disc = discrepancy_frame(uts[uts["source"] == "daqec"])
    devices = sorted(disc["device"].unique())
    print(f"devices: {devices}")

    results: dict[str, dict] = {}
    for name, scorer, causal in DETECTORS:
        per_fold = {}
        for test_dev in devices:  # same 3-fold device rotation as E3
            train = [d for d in devices if d != test_dev]
            lab = make_labels(disc, train, Z_THR, HORIZONS)
            sc = _entity_scores(disc[disc["device"] == test_dev], scorer)
            m = lab[lab["device"] == test_dev].merge(
                sc, on=["device", "entity_id", "ts"], how="inner")
            y = m["label"].to_numpy().astype(int)
            if y.min() == y.max():          # degenerate fold: AUROC undefined
                per_fold[test_dev] = None
                continue
            per_fold[test_dev] = float(roc_auc_score(y, m["score"].to_numpy()))
        vals = [v for v in per_fold.values() if v is not None]
        results[name] = {"per_fold": per_fold, "mean_auroc": float(np.mean(vals)) if vals else None,
                         "causal": causal, "n_folds": len(vals)}
        mean = results[name]["mean_auroc"]
        tag = "" if causal else "  [NON-CAUSAL, optimistic bound]"
        print(f"{name:22s} mean AUROC {mean:.3f}   per-fold "
              f"{ {k: (round(v,3) if v is not None else None) for k,v in per_fold.items()} }{tag}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "changepoint_auroc.json").write_text(json.dumps(results, indent=1))
    print(f"[e3-cp] wrote {OUT / 'changepoint_auroc.json'}")
    print("\nPaper's E3 reference points: LTN dissonance 0.497, LTN state probe 0.511, "
          "staleness control 0.459, last-discrepancy carry-forward 0.58, bar 0.75.")


if __name__ == "__main__":
    main()
