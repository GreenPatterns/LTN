"""Neural baselines smoke: fit 2 epochs on DAQEC, finite pair predictions, no crash."""
import numpy as np
import pytest
import torch

from conftest import requires_raw
from tests.test_train_smoke import E1A_OBSERVABLES, inline_norm_stats


@pytest.mark.data
@pytest.mark.slow
@requires_raw("daqec/master.parquet")
@pytest.mark.parametrize("name", ["lstm", "tgn", "transformer"])
def test_neural_baseline_smoke(raw_root, name):
    from ltn.baselines.neural import LSTMBaseline, TGNBaseline, TransformerBaseline
    from ltn.data.loaders.daqec import load_daqec
    from ltn.data.series import forecast_pairs, make_panel, split_panel_by_date, to_series

    uts = load_daqec(raw_root / "daqec")
    series = to_series(uts, observables=[o for o, _ in E1A_OBSERVABLES], channels=["probe", "claimed"])
    series = series[[(o, c) in E1A_OBSERVABLES for o, c in zip(series["observable"], series["channel"])]]
    panel = make_panel(series, uts)
    tr, va, te = split_panel_by_date(panel)
    stats = inline_norm_stats(series, panel, tr)

    cls = {"lstm": LSTMBaseline, "tgn": TGNBaseline, "transformer": TransformerBaseline}[name]
    m = cls()
    m.fit(panel, tr, va, {
        "uts": uts, "observables": E1A_OBSERVABLES, "norm_stats": stats,
        "horizons": [1, 3], "epochs": 2, "patience": 2, "bptt": 16, "width": 16,
        "seed": 0, "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    })
    pairs = forecast_pairs(panel, [1, 3], te).head(30)
    mu, sigma = m.predict(pairs, panel)
    assert np.isfinite(mu).all() and (sigma > 0).all()
    assert m.n_params() > 0
