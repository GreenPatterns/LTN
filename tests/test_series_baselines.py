"""Series/panel layer + classical baselines: correctness and the no-lookahead guarantee."""
import numpy as np
import pandas as pd
import pytest

from ltn.baselines.arima import ARIMAForecaster
from ltn.baselines.ewma import EWMA
from ltn.baselines.persistence import Persistence
from ltn.data import uts
from ltn.data.series import forecast_pairs, make_panel, split_panel_by_date, to_series
from ltn.eval.metrics import crps_gaussian


def synth_uts(T=120, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=T, freq="8h", tz="UTC")
    frames = []
    for dev in ["sim_dev_00", "sim_dev_01"]:
        ar = np.empty(T)
        ar[0] = 100.0
        for t in range(1, T):  # strongly mean-reverting AR(1): persistence is clearly suboptimal
            ar[t] = 100 + 0.3 * (ar[t - 1] - 100) + rng.normal(0, 2.0)
        for i, t in enumerate(ts):
            frames.append(
                uts.make_events(
                    source="sim", platform="sim_sc", device=dev,
                    entity_type="device", entity_id="device", ts=t,
                    rows=[("t1_us", "probe", ar[i], None)],
                )
            )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    events = synth_uts()
    return make_panel(to_series(events), events)


def test_panel_shape_and_keys(panel):
    assert panel.values.shape == (120, 2)
    assert set(panel.keys["device"]) == {"sim_dev_00", "sim_dev_01"}
    assert (panel.platform == "sim_sc").all()


def test_duplicate_observations_averaged():
    events = synth_uts()
    dup = events.copy()
    dup["value"] = dup["value"] + 2.0
    s = to_series(pd.concat([events, dup], ignore_index=True))
    s0 = to_series(events)
    assert np.allclose(s["value"].to_numpy(), s0["value"].to_numpy() + 1.0)


def test_forecast_pairs_target_in_mask(panel):
    tr, va, te = split_panel_by_date(panel)
    pairs = forecast_pairs(panel, [1, 3], te)
    t_target = pairs["t_idx"] + pairs["horizon"]
    assert te[t_target.to_numpy()].all()
    # origins may lie in val (test targets forecast from recent past) but never in test-only future
    assert (pairs["t_idx"] < len(panel.ts) - 1).all()


@pytest.mark.parametrize("cls", [Persistence, EWMA, ARIMAForecaster])
def test_no_lookahead(panel, cls):
    tr, va, te = split_panel_by_date(panel)
    pairs = forecast_pairs(panel, [1, 3], te).head(20)
    import dataclasses

    m = cls()
    m.fit(panel, tr, va, {"horizons": [1, 3]})
    mu1, s1 = m.predict(pairs, panel)
    # per pair: corrupt EVERYTHING strictly after that pair's own origin
    for i, (_, r) in enumerate(pairs.iterrows()):
        v = panel.values.copy()
        v[int(r.t_idx) + 1 :, :] += 1e3
        tampered = dataclasses.replace(panel, values=v)
        one = pairs.iloc[[i]]
        mu2, s2 = m.predict(one, tampered)
        np.testing.assert_allclose(mu1[i], mu2[0], err_msg=f"lookahead in mu, pair {i}")
        np.testing.assert_allclose(s1[i], s2[0], err_msg=f"lookahead in sigma, pair {i}")


def test_persistence_mu_is_last_value(panel):
    tr, va, te = split_panel_by_date(panel)
    pairs = forecast_pairs(panel, [1], te)
    m = Persistence()
    m.fit(panel, tr, va, {"horizons": [1]})
    mu, sig = m.predict(pairs, panel)
    np.testing.assert_allclose(mu, pairs["x_last"].to_numpy())
    assert (sig > 0).all()


def test_arima_beats_persistence_on_ar1(panel):
    tr, va, te = split_panel_by_date(panel)
    pairs = forecast_pairs(panel, [1], te)
    p, a = Persistence(), ARIMAForecaster()
    p.fit(panel, tr, va, {"horizons": [1]})
    a.fit(panel, tr, va, {"p": 1, "d": 0, "q": 0})
    y = pairs["y"].to_numpy()
    crps_p = crps_gaussian(y, *p.predict(pairs, panel))
    crps_a = crps_gaussian(y, *a.predict(pairs, panel))
    assert crps_a < crps_p  # AR(1) structure is learnable


def test_ewma_alpha_one_tracks_last(panel):
    tr, va, te = split_panel_by_date(panel)
    pairs = forecast_pairs(panel, [1], te).head(10)
    m = EWMA()
    m.fit(panel, tr, va, {"alpha": 1.0})
    mu, _ = m.predict(pairs, panel)
    np.testing.assert_allclose(mu, pairs["x_last"].to_numpy())
