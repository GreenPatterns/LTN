"""Baseline checkpointing: save -> load_weights round-trip and init_from warm start.

These guard the E4 baseline control (lane A5-e4base): the pretrain -> fine-tune
protocol only means anything for the sequence baselines if a checkpoint's weights
AND its build dims survive the round trip. Synthetic panel, CPU, no raw data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ltn.baselines.neural import LSTMBaseline, TGNBaseline, TransformerBaseline
from ltn.data.series import Panel, forecast_pairs

OBS = [("t1", "probe"), ("readout_error", "probe")]
LABELS = ["t1/probe", "readout_error/probe"]


def _toy(n_t: int = 48, seed: int = 0):
    """Two devices x two entities on a shared 8h grid, with UTS rows the sequence
    builder needs (device/platform/entity_id/ts/observable/channel/value/cal_ts)."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n_t, freq="8h")
    devices, ents = ["dev_a", "dev_b"], ["q0", "q1"]
    keys, cols = [], []
    rows = []
    for d in devices:
        for e in ents:
            for o, c in OBS:
                v = rng.normal(0.0, 1.0, n_t).cumsum() * 0.05 + 1.0
                keys.append({"device": d, "entity_id": e, "observable": o, "channel": c})
                cols.append(v)
                rows += [{"device": d, "platform": "sim_sc", "entity_id": e, "ts": t,
                          "observable": o, "channel": c, "value": float(x),
                          "cal_ts": t - pd.Timedelta("4h"), "context": "{}"}
                         for t, x in zip(ts, v)]
    panel = Panel(ts=ts, keys=pd.DataFrame(keys), values=np.stack(cols, axis=1),
                  platform=pd.Series(["sim_sc"] * len(keys)))
    uts = pd.DataFrame(rows)
    norm = {("sim_sc", o, c): (1.0, 0.5) for o, c in OBS}
    return uts, panel, norm


def _cfg(uts, norm, **over):
    cfg = {"uts": uts, "observables": OBS, "norm_stats": norm, "horizons": [1, 3],
           "epochs": 2, "patience": 2, "bptt": 16, "width": 12, "seed": 0,
           "device": "cpu", "time_unit_h": 8.0}
    cfg.update(over)
    return cfg


@pytest.mark.parametrize("cls", [LSTMBaseline, TGNBaseline, TransformerBaseline])
def test_save_load_weights_roundtrip(cls, tmp_path):
    uts, panel, norm = _toy()
    tr = np.arange(len(panel.ts)) < 32
    va = ~tr
    m = cls()
    m.fit(panel, tr, va, _cfg(uts, norm))
    ckpt = tmp_path / "m.pt"
    m.save(ckpt)

    pairs = forecast_pairs(panel, [1, 3], va).head(20)
    mu0, sig0 = m.predict(pairs, panel)

    # a fresh, differently-seeded model must diverge, then match exactly after load
    m2 = cls()
    m2.fit(panel, tr, va, _cfg(uts, norm, epochs=1, seed=7))
    mu_pre, _ = m2.predict(pairs, panel)
    assert not np.allclose(mu_pre, mu0)

    m2.load_weights(ckpt)
    mu1, sig1 = m2.predict(pairs, panel)
    assert np.allclose(mu0, mu1) and np.allclose(sig0, sig1)


@pytest.mark.parametrize("cls", [LSTMBaseline, TGNBaseline, TransformerBaseline])
def test_init_from_adopts_checkpoint_dims(cls, tmp_path):
    """A warm start with no explicit width must rebuild the checkpoint's shapes
    and load strictly -- otherwise state_dicts silently mismatch across E1a/E1b."""
    uts, panel, norm = _toy()
    tr = np.arange(len(panel.ts)) < 32
    va = ~tr
    m = cls()
    m.fit(panel, tr, va, _cfg(uts, norm, width=20))
    ckpt = tmp_path / "m.pt"
    m.save(ckpt)
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)["cfg"]

    child = cls()
    cfg = _cfg(uts, norm, epochs=0, init_from=str(ckpt))
    cfg.pop("width")  # deliberately unspecified: must come from the checkpoint
    child.fit(panel, tr, va, cfg)
    assert child.model.width == saved["width"]
    # epochs=0 + init_from == zero-shot evaluation of the loaded checkpoint
    pairs = forecast_pairs(panel, [1, 3], va).head(20)
    assert np.allclose(m.predict(pairs, panel)[0], child.predict(pairs, panel)[0])


def test_zero_shot_warm_start_beats_untrained():
    """epochs=0 + init_from keeps the pretrained weights (the E4 k=0 pretrained arm);
    epochs=0 without it is the untrained control."""
    uts, panel, norm = _toy()
    tr = np.arange(len(panel.ts)) < 32
    va = ~tr
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        ck = Path(td) / "m.pt"
        m = LSTMBaseline()
        m.fit(panel, tr, va, _cfg(uts, norm, epochs=6))
        m.save(ck)
        warm = LSTMBaseline()
        warm.fit(panel, tr, va, _cfg(uts, norm, epochs=0, init_from=str(ck)))
        cold = LSTMBaseline()
        cold.fit(panel, tr, va, _cfg(uts, norm, epochs=0))
    pairs = forecast_pairs(panel, [1, 3], va)
    from ltn.eval.metrics import crps_gaussian
    y = pairs["y"].to_numpy()
    c_warm = crps_gaussian(y, *warm.predict(pairs, panel))
    c_cold = crps_gaussian(y, *cold.predict(pairs, panel))
    assert c_warm < c_cold


def test_masked_pretrain_objective_runs():
    """pretrain=True routes through the masked-telemetry branch (same objective as
    ltn.exp.train) and still produces finite predictions."""
    uts, panel, norm = _toy()
    tr = np.arange(len(panel.ts)) < 32
    va = ~tr
    m = LSTMBaseline()
    m.fit(panel, tr, va, _cfg(uts, norm, pretrain=True, epochs=2))
    pairs = forecast_pairs(panel, [1, 3], va).head(20)
    mu, sigma = m.predict(pairs, panel)
    assert np.isfinite(mu).all() and (sigma > 0).all()
