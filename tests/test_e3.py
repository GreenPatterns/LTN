"""E3 provenance tests: lag-1 autocorrelation of the divergence signal and the
three (four) claim-age definitions. These back numbers printed in the paper, so
they are pinned here rather than left to a one-off script."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ltn.exp.e3 import (_acf_series, _acf_summary, claim_age_stats,
                        discrepancy_frame, divergence_autocorrelation, lag1_autocorr)

DAQEC = Path("data/uts/source=daqec")
MASTER = Path("data/raw/daqec/master.parquet")


def test_lag1_autocorr_matches_manual_formula():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    xc = x - x.mean()
    want = float((xc[:-1] * xc[1:]).sum() / (xc * xc).sum())
    assert lag1_autocorr(x) == pytest.approx(want)


def test_lag1_autocorr_perfect_alternation_is_near_minus_one():
    x = np.array([0.0, 1.0] * 20)
    # r1 of a period-2 square wave -> -(n-1)/n
    assert lag1_autocorr(x) == pytest.approx(-39 / 40)


def test_lag1_autocorr_degenerate_cases_are_nan():
    assert np.isnan(lag1_autocorr(np.ones(50)))       # zero variance
    assert np.isnan(lag1_autocorr(np.array([1.0, 2.0])))  # too short


def test_acf_summary_white_noise_band():
    rows = [{"n": 42, "r1": 0.1, "mean": 0.0}, {"n": 42, "r1": -0.4, "mean": 0.0},
            {"n": 42, "r1": float("nan"), "mean": 0.0}]
    s = _acf_summary(rows)
    assert s["n_series"] == 3 and s["n_series_defined"] == 2
    assert s["n_series_degenerate"] == 1
    assert s["white_noise_band_95"] == pytest.approx(1.96 / np.sqrt(42))
    assert s["max_abs_r1"] == pytest.approx(0.4)
    assert s["n_outside_band"] == 1


def test_acf_series_sorts_by_ts():
    df = pd.DataFrame({
        "device": ["d"] * 5,
        "ts": pd.to_datetime(["2025-01-05", "2025-01-01", "2025-01-03",
                              "2025-01-02", "2025-01-04"], utc=True),
        "v": [5.0, 1.0, 3.0, 2.0, 4.0],
    })
    rows = _acf_series(df, ["device"], "v")
    assert len(rows) == 1 and rows[0]["n"] == 5
    assert rows[0]["r1"] == pytest.approx(lag1_autocorr(np.arange(1.0, 6.0)))


@pytest.mark.skipif(not DAQEC.exists(), reason="daqec UTS lake not present")
def test_divergence_autocorrelation_on_real_daqec():
    from ltn.data.uts import read_uts
    from ltn.eval.splits import load_split
    uts = read_uts("data/uts", sources=["daqec"])
    acf = divergence_autocorrelation(uts, load_split("e3"))

    ex, rd = acf["binary_exceed"], acf["continuous_rel_disc"]
    # 3 devices x 4 entities; the device-level entity never exceeds z=2 -> degenerate
    assert ex["n_series"] == 12 and ex["n_series_defined"] == 9
    assert rd["n_series"] == 36 and rd["n_series_defined"] == 36
    assert ex["series_length_median"] == 42
    assert ex["white_noise_band_95"] == pytest.approx(1.96 / np.sqrt(42))

    # regression pins: the audited values. NOT <= 0.14 as the paper claimed.
    assert ex["max_abs_r1"] == pytest.approx(0.280, abs=5e-3)
    assert rd["max_abs_r1"] == pytest.approx(0.299, abs=5e-3)
    assert ex["max_abs_r1"] > 0.14 and rd["max_abs_r1"] > 0.14
    # ...but both stay inside the white-noise band, which is the defensible claim
    assert ex["n_outside_band"] == 0 and rd["n_outside_band"] == 0

    # the forward-window E3 target is autocorrelated by construction (overlapping
    # windows share snapshots) -- it is NOT the series the paper should quote
    assert acf["binary_label"]["max_abs_r1"] > 0.6
    assert acf["binary_label"]["n_outside_band"] == 9


@pytest.mark.skipif(not DAQEC.exists(), reason="daqec UTS lake not present")
def test_discrepancy_frame_shape():
    from ltn.data.uts import read_uts
    d = discrepancy_frame(read_uts("data/uts", sources=["daqec"]))
    assert set(d["observable"]) == {"t1_us", "t2_us", "readout_error"}
    assert (d["rel_disc"] >= 0).all()
    assert d.groupby(["device", "entity_id", "observable"], observed=True).size().eq(42).all()


@pytest.mark.skipif(not MASTER.exists(), reason="daqec master.parquet not present")
def test_claim_age_definitions_are_distinct_and_ordered():
    s = claim_age_stats(str(MASTER))["definitions"]
    assert set(s) == {"per_run", "freshest_per_snapshot", "oldest_per_snapshot",
                      "mean_per_snapshot"}
    fresh = s["freshest_per_snapshot"]["pooled"]["median_h"]
    mean = s["mean_per_snapshot"]["pooled"]["median_h"]
    old = s["oldest_per_snapshot"]["pooled"]["median_h"]
    assert fresh < mean < old, "the three definitions must not be interchangeable"
    # regression pins on the real table
    assert s["per_run"]["pooled"]["n"] == 756
    assert s["per_run"]["pooled"]["median_h"] == pytest.approx(6.54, abs=0.02)
    assert fresh == pytest.approx(1.96, abs=0.02)
    assert old == pytest.approx(10.82, abs=0.02)
    assert mean == pytest.approx(6.42, abs=0.02)
    # no definition puts the MEDIAN in the paper's "3--6 h" window
    for name, d in s.items():
        assert not (3.0 <= d["pooled"]["median_h"] <= 6.0), name
