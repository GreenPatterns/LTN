"""Tests for ltn.eval.reliability (lane A6: TR-appropriate reliability endpoints).

Reference values are hand-computed or cross-checked against scipy directly, in the
same spirit as tests/test_eval.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sstats

from ltn.eval import metrics, reliability

RNG = np.random.default_rng(11)


# ------------------------------------------------------------ closed forms

def test_violation_probability_lower_matches_scipy():
    mu = np.array([100.0, 120.0, 80.0])
    sigma = np.array([10.0, 5.0, 20.0])
    thr = 90.0
    got = reliability.violation_probability(mu, sigma, thr, "lower")
    want = sstats.norm.cdf((thr - mu) / sigma)
    assert got == pytest.approx(want, rel=1e-12)


def test_violation_probability_upper_is_complement_of_lower():
    mu, sigma, thr = 0.01, 0.002, 0.013
    up = reliability.violation_probability(mu, sigma, thr, "upper")
    lo = reliability.violation_probability(mu, sigma, thr, "lower")
    assert float(up) + float(lo) == pytest.approx(1.0, rel=1e-12)


def test_violation_probability_at_threshold_is_half():
    for d in ("upper", "lower"):
        assert float(reliability.violation_probability(5.0, 1.3, 5.0, d)) == pytest.approx(0.5)


def test_violation_probability_monotone_in_margin():
    # 'upper': larger mu -> more likely out of spec.
    p = reliability.violation_probability(np.array([0.5, 1.0, 1.5]), 1.0, 1.0, "upper")
    assert p[0] < p[1] < p[2]


def test_bad_direction_raises():
    with pytest.raises(ValueError):
        reliability.violation_probability(0.0, 1.0, 0.0, "sideways")
    with pytest.raises(ValueError):
        reliability.violation_label(np.zeros(3), 0.0, "sideways")


def test_violation_label_is_strict_inequality():
    y = np.array([0.9, 1.0, 1.1])
    assert reliability.violation_label(y, 1.0, "upper").tolist() == [0, 0, 1]
    assert reliability.violation_label(y, 1.0, "lower").tolist() == [1, 0, 0]


def test_direction_convention_matches_physics():
    assert reliability.direction_for("t1_us") == "lower"      # short T1 is bad
    assert reliability.direction_for("t2_us") == "lower"
    assert reliability.direction_for("readout_error") == "upper"  # high error is bad
    assert reliability.direction_for("gate_error_2q") == "upper"
    assert reliability.direction_for("epc_1q") == "upper"
    assert reliability.direction_for("unseen_thing") == "upper"   # documented fallback


# ------------------------------------------------------------ thresholds

def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y_err = rng.normal(0.01, 0.002, n)
    y_t1 = rng.normal(150.0, 20.0, n)
    return pd.concat([
        pd.DataFrame({"device": "d0", "observable": "readout_error", "y": y_err,
                      "mu": y_err, "sigma": 0.002,
                      "horizon": np.tile([1, 3], n // 2),
                      "s_idx": np.repeat(np.arange(n // 2), 2),
                      "t_idx": 0}),
        pd.DataFrame({"device": "d0", "observable": "t1_us", "y": y_t1,
                      "mu": y_t1, "sigma": 20.0,
                      "horizon": np.tile([1, 3], n // 2),
                      "s_idx": np.repeat(np.arange(n // 2), 2),
                      "t_idx": 0}),
    ], ignore_index=True)


def test_sla_thresholds_hit_the_nominal_base_rate():
    df = _toy()
    thr = reliability.sla_thresholds(df, base_rate=0.10)
    joined = reliability.attach_violations(df, thr)
    for obs, g in joined.groupby("observable"):
        # empirical quantile => realised base rate is the nominal one up to ties
        assert g["viol"].mean() == pytest.approx(0.10, abs=0.01), obs


def test_sla_thresholds_pick_the_correct_tail():
    thr = reliability.sla_thresholds(_toy(), base_rate=0.10)
    row_err = thr[thr.observable == "readout_error"].iloc[0]
    row_t1 = thr[thr.observable == "t1_us"].iloc[0]
    assert row_err["direction"] == "upper" and row_err["threshold"] > 0.01
    assert row_t1["direction"] == "lower" and row_t1["threshold"] < 150.0


def test_sla_thresholds_are_model_independent():
    """Same y, different mu/sigma => identical thresholds. This is the property
    that makes the cross-model comparison fair."""
    df = _toy()
    other = df.copy()
    other["mu"] = other["mu"] * 3.0 + 1.0
    other["sigma"] = other["sigma"] * 7.0
    a = reliability.sla_thresholds(df).sort_values(["device", "observable"])
    b = reliability.sla_thresholds(other).sort_values(["device", "observable"])
    assert a["threshold"].to_numpy() == pytest.approx(b["threshold"].to_numpy())


def test_base_rate_out_of_range_raises():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            reliability.sla_thresholds(_toy(), base_rate=bad)


def test_attach_violations_rejects_incomplete_threshold_table():
    df = _toy()
    thr = reliability.sla_thresholds(df)
    with pytest.raises(ValueError):
        reliability.attach_violations(df, thr[thr.observable == "t1_us"])


# ------------------------------------------------------------ scoring

def test_sla_scores_agree_with_metrics_module():
    df = reliability.attach_violations(_toy(), reliability.sla_thresholds(_toy()))
    s = reliability.sla_scores(df)
    assert s["brier"] == pytest.approx(metrics.brier(df["viol"], df["p_viol"]))
    assert s["ece"] == pytest.approx(metrics.ece(df["viol"], df["p_viol"]))
    assert s["n"] == len(df)


def test_brier_skill_is_zero_for_climatology_forecast():
    v = (RNG.random(1000) < 0.3).astype(int)
    base = v.mean()
    df = pd.DataFrame({"viol": v, "p_viol": np.full(1000, base)})
    s = reliability.sla_scores(df)
    assert s["brier_skill"] == pytest.approx(0.0, abs=1e-12)


def test_brier_skill_is_one_for_a_perfect_forecast():
    v = (RNG.random(500) < 0.4).astype(int)
    df = pd.DataFrame({"viol": v, "p_viol": v.astype(float)})
    assert reliability.sla_scores(df)["brier_skill"] == pytest.approx(1.0)


def test_brier_skill_negative_for_an_anti_forecast():
    v = (RNG.random(500) < 0.4).astype(int)
    df = pd.DataFrame({"viol": v, "p_viol": 1.0 - v.astype(float)})
    assert reliability.sla_scores(df)["brier_skill"] < 0.0


# ------------------------------------------------------------ survival

def test_reliability_curve_hand_computed():
    """Two origins, horizons {1,2}; survival and hazard by hand."""
    df = pd.DataFrame({
        "device": ["d", "d", "d", "d"],
        "observable": ["o"] * 4,
        "s_idx": [0, 0, 1, 1],
        "t_idx": [0, 0, 0, 0],
        "horizon": [1, 2, 1, 2],
        "p_viol": [0.0, 0.5, 0.5, 0.5],
        "viol": [0, 1, 0, 0],
    })
    c = reliability.reliability_curve(df).set_index("horizon")
    # predicted: origin A 1.0 -> 0.5 ; origin B 0.5 -> 0.25 ; means 0.75, 0.375
    assert c.loc[1, "R_pred"] == pytest.approx(0.75)
    assert c.loc[2, "R_pred"] == pytest.approx(0.375)
    assert c.loc[1, "hazard_pred"] == pytest.approx(0.25)
    assert c.loc[2, "hazard_pred"] == pytest.approx(1.0 - 0.375 / 0.75)
    # empirical: both survive h=1 -> 1.0 ; A dies at h=2 -> 0.5
    assert c.loc[1, "R_emp"] == pytest.approx(1.0)
    assert c.loc[2, "R_emp"] == pytest.approx(0.5)
    assert c.loc[2, "hazard_emp"] == pytest.approx(0.5)
    assert int(c.loc[1, "n_origins"]) == 2


def test_reliability_curve_is_monotone_nonincreasing():
    df = reliability.attach_violations(_toy(seed=3), reliability.sla_thresholds(_toy(seed=3)))
    c = reliability.reliability_curve(df)
    assert (np.diff(c["R_pred"].to_numpy()) <= 1e-12).all()
    assert (np.diff(c["R_emp"].to_numpy()) <= 1e-12).all()


def test_reliability_curve_uses_only_complete_cohort():
    """An origin missing a horizon must be dropped, not silently treated as surviving."""
    base = pd.DataFrame({
        "device": "d", "observable": "o", "t_idx": 0,
        "s_idx": [0, 0, 1], "horizon": [1, 2, 1],
        "p_viol": [0.1, 0.1, 0.9], "viol": [0, 0, 1],
    })
    c = reliability.reliability_curve(base)
    assert int(c["n_origins"].iloc[0]) == 1          # origin s_idx=1 dropped
    assert c.set_index("horizon").loc[1, "R_pred"] == pytest.approx(0.9)


def test_reliability_curve_raises_when_no_complete_origin():
    base = pd.DataFrame({
        "device": "d", "observable": "o", "t_idx": 0,
        "s_idx": [0, 1], "horizon": [1, 2],
        "p_viol": [0.1, 0.1], "viol": [0, 0],
    })
    with pytest.raises(ValueError):
        reliability.reliability_curve(base)


# ------------------------------------------------------------ wasted shots

def test_wasted_shots_hand_computed():
    r = np.array([0.75, 0.5])
    w = reliability.wasted_shots(r, n_shots=1000)
    assert w["availability"] == pytest.approx(0.625)
    assert w["wasted_shots_mean"] == pytest.approx(375.0)   # 1000*(0.25+0.5)/2
    assert w["wasted_shots_worst_round"] == pytest.approx(500.0)
    assert w["n_rounds"] == 2


def test_wasted_shots_excess_vs_oracle():
    r = np.array([0.90, 0.80])
    b = np.array([0.95, 0.95])
    w = reliability.wasted_shots(r, n_shots=4096, best_rewards=b)
    assert w["excess_wasted_frac_vs_oracle"] == pytest.approx(0.10)
    assert w["excess_wasted_shots_vs_oracle"] == pytest.approx(409.6)


def test_wasted_shots_zero_for_perfect_policy():
    w = reliability.wasted_shots(np.ones(5), n_shots=8192, best_rewards=np.ones(5))
    assert w["wasted_shots_mean"] == pytest.approx(0.0)
    assert w["excess_wasted_shots_vs_oracle"] == pytest.approx(0.0)


# ------------------------------------------------------------ bootstrap

def test_paired_bootstrap_identical_losses_do_not_separate():
    x = RNG.random(600)
    cl = np.repeat(np.arange(12), 50)
    r = reliability.paired_bootstrap_diff(x, x, cl, n_boot=500)
    assert r["diff"] == pytest.approx(0.0)
    assert r["ci_lo"] == pytest.approx(0.0) and r["ci_hi"] == pytest.approx(0.0)
    assert r["separates"] is False
    assert r["n_clusters"] == 12


def test_paired_bootstrap_detects_a_large_consistent_gap():
    cl = np.repeat(np.arange(12), 50)
    a = RNG.random(600)
    b = a + 1.0                       # b is uniformly worse by 1.0 on every row
    r = reliability.paired_bootstrap_diff(a, b, cl, n_boot=500)
    assert r["diff"] == pytest.approx(-1.0)
    assert r["separates"] is True and r["ci_hi"] < 0.0


def test_paired_bootstrap_does_not_separate_pure_noise():
    cl = np.repeat(np.arange(12), 50)
    a, b = RNG.normal(size=600), RNG.normal(size=600)
    r = reliability.paired_bootstrap_diff(a, b, cl, n_boot=500)
    assert r["ci_lo"] < 0.0 < r["ci_hi"]
    assert r["separates"] is False


def test_paired_bootstrap_ci_brackets_the_point_estimate():
    cl = np.repeat(np.arange(10), 40)
    a = RNG.normal(size=400)
    b = a + RNG.normal(0.3, 1.0, 400)
    r = reliability.paired_bootstrap_diff(a, b, cl, n_boot=800)
    assert r["ci_lo"] <= r["diff"] <= r["ci_hi"]


def test_paired_bootstrap_cluster_ci_is_wider_than_iid():
    """Correlated clusters => cluster bootstrap must not be over-confident."""
    n_cl, per = 12, 60
    cl = np.repeat(np.arange(n_cl), per)
    # a per-cluster offset makes the paired difference correlated within device
    offs = RNG.normal(0.0, 1.0, n_cl).repeat(per)
    a = RNG.normal(size=n_cl * per)
    b = a - offs
    clustered = reliability.paired_bootstrap_diff(a, b, cl, n_boot=1000, seed=1)
    iid = reliability.paired_bootstrap_diff(a, b, np.arange(n_cl * per), n_boot=1000, seed=1)
    assert (clustered["ci_hi"] - clustered["ci_lo"]) > (iid["ci_hi"] - iid["ci_lo"])


def test_paired_bootstrap_shape_and_cluster_guards():
    with pytest.raises(ValueError):
        reliability.paired_bootstrap_diff(np.zeros(5), np.zeros(4), np.zeros(5))
    with pytest.raises(ValueError):
        reliability.paired_bootstrap_diff(np.zeros(5), np.zeros(5), np.zeros(5))
