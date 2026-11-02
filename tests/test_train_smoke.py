"""End-to-end E1a pipeline smoke: DAQEC -> panel -> LTN fit (2 epochs) -> pair predictions."""
import numpy as np
import pytest
import torch

from conftest import requires_raw

E1A_OBSERVABLES = [
    ("t1_us", "probe"), ("t2_us", "probe"), ("readout_error", "probe"), ("gate_error_2q", "claimed"),
]


def inline_norm_stats(series, panel, train_mask):
    """Train-only normalization; mirrors what the split builder stores."""
    stats = {}
    for j, k in panel.keys.iterrows():
        v = panel.values[train_mask, j]
        v = v[~np.isnan(v)]
        key = ("ibm_sc", k.observable, k.channel)
        if key not in stats:
            stats[key] = []
        stats[key].append((v.mean(), v.std()))
    return {k: (float(np.mean([m for m, _ in vs])), float(np.mean([s for _, s in vs])) or 1.0)
            for k, vs in stats.items()}


@pytest.mark.data
@pytest.mark.slow
@requires_raw("daqec/master.parquet")
def test_ltn_e1a_pipeline_smoke(raw_root):
    from ltn.data.loaders.daqec import load_daqec
    from ltn.data.series import forecast_pairs, make_panel, split_panel_by_date, to_series
    from ltn.exp.train import LTNForecaster

    uts = load_daqec(raw_root / "daqec")
    series = to_series(uts, observables=[o for o, _ in E1A_OBSERVABLES],
                       channels=["probe", "claimed"])
    # keep exactly the E1A series (drop e.g. t1/claimed which pairs with probe)
    series = series[[
        (o, c) in E1A_OBSERVABLES for o, c in zip(series["observable"], series["channel"])
    ]]
    panel = make_panel(series, uts)
    tr, va, te = split_panel_by_date(panel)
    stats = inline_norm_stats(series, panel, tr)

    m = LTNForecaster()
    m.fit(panel, tr, va, {
        "uts": uts, "observables": E1A_OBSERVABLES, "norm_stats": stats,
        "horizons": [1, 3], "epochs": 2, "patience": 2, "bptt": 16,
        "carry_dim": 8, "hidden": 16, "seed": 0,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    })
    pairs = forecast_pairs(panel, [1, 3], te).head(50)
    mu, sigma = m.predict(pairs, panel)
    assert np.isfinite(mu).all() and (sigma > 0).all()
    y = pairs["y"].to_numpy()
    # sanity: raw-unit predictions live on the right scale (not z-space or garbage)
    assert np.abs(mu - y).mean() < 10 * np.abs(np.diff(y)).mean() + 1e3
    assert m.n_params() > 0
