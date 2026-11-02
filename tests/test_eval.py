"""Tests for the evaluation harness (ltn.eval.{metrics,stats,calibrate,runlog}).

Reference values are hand-computed or cross-checked against
properscoring/scipy/sklearn directly, per master plan §2.4 B2.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from scipy import stats as sstats
from sklearn.metrics import brier_score_loss, roc_auc_score

from ltn.eval import calibrate, metrics, runlog, stats

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- metrics

def test_mae_rmse_hand_computed():
    y = np.array([0.0, 1.0, 2.0])
    yhat = np.array([1.0, 1.0, 0.0])
    assert metrics.mae(y, yhat) == pytest.approx(1.0)  # (1 + 0 + 2)/3
    assert metrics.rmse(y, yhat) == pytest.approx(np.sqrt(5.0 / 3.0))


def test_crps_gaussian_closed_form():
    # CRPS(N(mu, s^2), y=mu) = s * (2*phi(0) - 1/sqrt(pi))
    s = 1.7
    expected = s * (2.0 / np.sqrt(2 * np.pi) - 1.0 / np.sqrt(np.pi))
    got = metrics.crps_gaussian(np.full(5, 3.0), np.full(5, 3.0), np.full(5, s))
    assert got == pytest.approx(expected, rel=1e-10)


def test_gaussian_nll_matches_scipy_logpdf():
    y = RNG.normal(size=50)
    mu = RNG.normal(size=50)
    sigma = np.abs(RNG.normal(size=50)) + 0.5
    ref = -np.mean(sstats.norm.logpdf(y, loc=mu, scale=sigma))
    assert metrics.gaussian_nll(y, mu, sigma) == pytest.approx(ref, rel=1e-12)


def test_ece_hand_computed():
    # bin [0, .5): p = .1, .1, acc = .5, conf = .1 -> |.4| * weight .5
    # bin [.5, 1]: p = .9, .9, acc = 1., conf = .9 -> |.1| * weight .5
    y = np.array([0, 1, 1, 1])
    p = np.array([0.1, 0.1, 0.9, 0.9])
    assert metrics.ece(y, p, n_bins=2) == pytest.approx(0.25)


def test_ece_perfectly_calibrated_extremes():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])  # p = 1.0 must land in the top bin
    assert metrics.ece(y, p, n_bins=15) == pytest.approx(0.0)


def test_brier_matches_sklearn():
    y = np.array([0, 1, 1, 0, 1])
    p = np.array([0.1, 0.8, 0.6, 0.4, 0.9])
    assert metrics.brier(y, p) == pytest.approx(brier_score_loss(y, p), rel=1e-12)


def test_auroc_auprc():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.auroc(y, s) == pytest.approx(1.0)
    assert metrics.auprc(y, s) == pytest.approx(1.0)
    s_mixed = np.array([0.1, 0.8, 0.2, 0.9])
    assert metrics.auroc(y, s_mixed) == pytest.approx(roc_auc_score(y, s_mixed))


def test_spearman_monotone_invariance():
    a = RNG.normal(size=40)
    assert metrics.spearman(a, np.exp(a)) == pytest.approx(1.0)
    assert metrics.spearman(a, -(a**3)) == pytest.approx(-1.0)


def test_coverage_matches_nominal_level():
    n = 200_000
    mu = np.zeros(n)
    sigma = np.full(n, 2.0)
    y = RNG.normal(0.0, 2.0, size=n)
    assert metrics.coverage(y, mu, sigma, level=0.9) == pytest.approx(0.9, abs=5e-3)
    assert metrics.coverage(y, mu, sigma, level=0.5) == pytest.approx(0.5, abs=5e-3)


def test_regret():
    assert metrics.regret(3.0, 5.0) == pytest.approx(2.0)
    np.testing.assert_allclose(
        metrics.regret(np.array([1.0, 5.0]), np.array([4.0, 5.0])), [3.0, 0.0]
    )


# ---------------------------------------------------------------- soft ECE

def test_soft_ece_close_to_hard_ece_smooth_case():
    rng = np.random.default_rng(0)
    n = 5000
    p = rng.uniform(0.02, 0.98, size=n)
    # miscalibrated outcomes: true prob is a sharpened version of p
    q = 1.0 / (1.0 + np.exp(-2.5 * np.log(p / (1 - p))))
    y = (rng.uniform(size=n) < q).astype(float)
    hard = metrics.ece(y, p, n_bins=15)
    soft = metrics.soft_ece(
        torch.tensor(p, dtype=torch.float64), torch.tensor(y, dtype=torch.float64),
        n_bins=15, temperature=200.0,
    ).item()
    assert hard > 0.05  # the case actually exercises miscalibration
    assert soft == pytest.approx(hard, abs=0.02)


def test_soft_ece_differentiable_nonzero_grad():
    p = torch.tensor([0.1, 0.3, 0.6, 0.8, 0.55, 0.45], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=torch.float64)
    loss = metrics.soft_ece(p, y, n_bins=5, temperature=50.0)
    loss.backward()
    assert p.grad is not None
    assert torch.isfinite(p.grad).all()
    assert p.grad.abs().sum().item() > 0.0


def test_soft_ece_memberships_partition():
    # zero miscalibration -> soft ECE ~ 0 even with soft bin boundaries
    p = torch.linspace(0.05, 0.95, 200, dtype=torch.float64)
    assert metrics.soft_ece(p, p.clone(), n_bins=15, temperature=50.0).item() == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------- stats

def test_cliffs_delta_reference_cases():
    assert stats.cliffs_delta([2.0, 3.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert stats.cliffs_delta([0.0, 1.0], [2.0, 3.0]) == pytest.approx(-1.0)
    assert stats.cliffs_delta([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)
    # hand count: pairs (1,1)=tie, (1,2)=lt, (3,1)=gt, (3,2)=gt -> (2-1)/4
    assert stats.cliffs_delta([1.0, 3.0], [1.0, 2.0]) == pytest.approx(0.25)


def test_paired_wilcoxon_holm_identical_cell_survives():
    rng = np.random.default_rng(1)
    b = rng.normal(size=30)
    cells = {
        "clearly_different": (b + 0.5 + 0.01 * rng.normal(size=30), b),
        "identical": (b.copy(), b.copy()),  # all-zero diffs: must not crash
    }
    df = stats.paired_wilcoxon_holm(cells)
    by_cell = df.set_index("cell")
    assert by_cell.loc["clearly_different", "holm_p"] < 0.05
    assert by_cell.loc["identical", "wilcoxon_p"] == pytest.approx(1.0)
    assert by_cell.loc["identical", "holm_p"] >= 0.05  # survives correction
    assert by_cell.loc["identical", "significant"] == 0.0
    assert by_cell.loc["clearly_different", "cliffs_delta"] > 0.0
    # Holm never decreases a p-value
    body = df[df["cell"] != stats.SUMMARY_ROW]
    assert (body["holm_p"] >= body["wilcoxon_p"] - 1e-15).all()
    # summary row present and aggregates correctly
    assert stats.SUMMARY_ROW in df["cell"].values
    assert by_cell.loc[stats.SUMMARY_ROW, "n"] == 60
    assert by_cell.loc[stats.SUMMARY_ROW, "significant"] == pytest.approx(0.5)


def test_paired_wilcoxon_matches_scipy_raw_p():
    rng = np.random.default_rng(2)
    a = rng.normal(size=25)
    b = a + rng.normal(scale=0.3, size=25)
    df = stats.paired_wilcoxon_holm({"only": (a, b)}).set_index("cell")
    ref = sstats.wilcoxon(a, b, zero_method="pratt").pvalue
    assert df.loc["only", "wilcoxon_p"] == pytest.approx(ref, rel=1e-12)
    # single comparison: Holm is identity
    assert df.loc["only", "holm_p"] == pytest.approx(ref, rel=1e-12)


def test_bootstrap_ci_contains_true_mean_and_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(loc=2.0, scale=1.0, size=400)
    lo, hi = stats.bootstrap_ci(np.mean, x, n_boot=3000, seed=0)
    assert lo < 2.0 < hi
    assert hi - lo < 0.5  # sane width for n=400
    lo2, hi2 = stats.bootstrap_ci(np.mean, x, n_boot=3000, seed=0)
    assert (lo, hi) == (lo2, hi2)
    lo3, _ = stats.bootstrap_ci(np.mean, x, n_boot=3000, seed=1)
    assert lo3 != lo


def test_bootstrap_ci_joint_resampling():
    # paired stat: mean difference; joint resampling must preserve pairing exactly
    a = np.arange(100, dtype=np.float64)
    b = a + 1.0
    lo, hi = stats.bootstrap_ci(lambda x, y: float(np.mean(y - x)), a, b, n_boot=500, seed=0)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)


# ---------------------------------------------------------------- calibrate

def test_temperature_scaler_recovers_inflation_factor():
    rng = np.random.default_rng(4)
    n = 4000
    sigma_true = np.abs(rng.normal(size=n)) + 1.0
    mu = rng.normal(size=n)
    y = mu + sigma_true * rng.normal(size=n)
    understated = sigma_true / 2.0
    ts = calibrate.TemperatureScaler().fit(y, mu, understated)
    assert ts.t_ == pytest.approx(2.0, rel=0.1)
    np.testing.assert_allclose(ts.transform(understated), ts.t_ * understated)
    # calibrated NLL is no worse than uncalibrated
    assert metrics.gaussian_nll(y, mu, ts.transform(understated)) <= metrics.gaussian_nll(
        y, mu, understated
    )


def test_temperature_scaler_identity_when_calibrated():
    rng = np.random.default_rng(5)
    n = 4000
    mu = rng.normal(size=n)
    sigma = np.full(n, 1.5)
    y = mu + sigma * rng.normal(size=n)
    ts = calibrate.TemperatureScaler().fit(y, mu, sigma)
    assert ts.t_ == pytest.approx(1.0, rel=0.05)


def test_platt_binary_improves_ece():
    rng = np.random.default_rng(6)
    n = 6000
    q = rng.uniform(0.05, 0.95, size=n)  # true probabilities
    y = (rng.uniform(size=n) < q).astype(int)
    # reported probabilities are under-confident (logits shrunk 3x)
    p = 1.0 / (1.0 + np.exp(-np.log(q / (1 - q)) / 3.0))
    pb = calibrate.PlattBinary().fit(y, p)
    assert pb.a_ == pytest.approx(3.0, rel=0.15)  # recovers the shrink factor
    assert metrics.ece(y, pb.transform(p)) < metrics.ece(y, p)


def test_calibrators_raise_before_fit():
    with pytest.raises(RuntimeError):
        calibrate.TemperatureScaler().transform(np.ones(3))
    with pytest.raises(RuntimeError):
        calibrate.PlattBinary().transform(np.full(3, 0.5))


# ---------------------------------------------------------------- runlog

def test_runlogger_roundtrip(tmp_path):
    cfg = {"lr": 1e-3, "d": 4}
    with runlog.RunLogger("E1", "ltn", 0, cfg, root=tmp_path) as rl:
        rl.log_params(12345)
        rl.log(device="ibm_fake", observable="t1_us", crps=0.12)
        rl.log(device="ibm_fake", observable="t2_us", crps=0.34)
    run_dir = tmp_path / "E1" / "ltn" / "0"
    header = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert header["exp"] == "E1" and header["model"] == "ltn" and header["seed"] == 0
    assert header["config"] == cfg
    assert "git_sha" in header and "start_time" in header
    assert header["n_params"] == 12345
    status = yaml.safe_load((run_dir / "status.yaml").read_text())
    assert status["status"] == "ok"
    assert status["duration_s"] >= 0.0

    df = runlog.read_results(tmp_path)
    assert len(df) == 2
    assert set(["exp", "model", "seed", "crps"]) <= set(df.columns)
    assert (df["exp"] == "E1").all() and (df["seed"] == 0).all()
    assert sorted(df["observable"]) == ["t1_us", "t2_us"]


def test_runlogger_failure_writes_traceback_and_propagates(tmp_path):
    with pytest.raises(ValueError, match="boom"):
        with runlog.RunLogger("E1", "ltn", 1, {}, root=tmp_path) as rl:
            rl.log(crps=0.5)
            raise ValueError("boom")
    status = yaml.safe_load((tmp_path / "E1" / "ltn" / "1" / "status.yaml").read_text())
    assert status["status"] == "failed"
    assert "ValueError: boom" in status["traceback"]
    # partial results are still readable
    assert len(runlog.read_results(tmp_path)) == 1


def test_read_results_filters_and_rerun_replaces(tmp_path):
    for model in ("ltn", "persistence"):
        for seed in (0, 1):
            with runlog.RunLogger("E1", model, seed, {}, root=tmp_path) as rl:
                rl.log(mae=float(seed))
    with runlog.RunLogger("E3", "ltn", 0, {}, root=tmp_path) as rl:
        rl.log(auroc=0.8)

    assert len(runlog.read_results(tmp_path)) == 5
    assert len(runlog.read_results(tmp_path, exp="E1")) == 4
    df = runlog.read_results(tmp_path, exp="E1", model="ltn")
    assert len(df) == 2 and (df["model"] == "ltn").all()
    assert runlog.read_results(tmp_path, exp="E9").empty

    # rerunning a (exp, model, seed) replaces its rows instead of double-counting
    with runlog.RunLogger("E3", "ltn", 0, {}, root=tmp_path) as rl:
        rl.log(auroc=0.9)
    df3 = runlog.read_results(tmp_path, exp="E3")
    assert len(df3) == 1 and df3["auroc"].iloc[0] == pytest.approx(0.9)
