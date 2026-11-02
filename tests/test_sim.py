"""Tests for the synthetic device-fleet telemetry generator [DEV-2] (ltn.sim)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ltn.data import uts
from ltn.sim import processes as proc
from ltn.sim.fit import fit_fleet_config, target_stats
from ltn.sim.fleet import FleetConfig, FleetSim, default_fleet_mix
from ltn.sim.topology import make_topology
from ltn.sim.validate import sim_vs_real_report


def small_cfg(**kw) -> FleetConfig:
    return FleetConfig(
        topologies=[{"kind": "linear", "n": 5}, {"kind": "ring", "n": 4}], **kw)


@pytest.fixture(scope="module")
def small_run():
    cfg = small_cfg()
    return cfg, *FleetSim(cfg, 20260712).run(20, 3)


# ------------------------------------------------------------------- topology

def test_topologies():
    n, edges = make_topology("heavy_hex", distance=3)
    assert n > 0 and len(edges) > 0
    assert all(0 <= u < v < n for u, v in edges)
    assert len(set(edges)) == len(edges)

    n, edges = make_topology("grid", rows=2, cols=3)
    assert (n, len(edges)) == (6, 7)

    n, edges = make_topology("linear", n=5)
    assert (n, edges) == (5, [(0, 1), (1, 2), (2, 3), (3, 4)])

    n, edges = make_topology("ring", n=6)
    assert n == 6 and len(edges) == 6

    with pytest.raises(ValueError, match="unknown topology"):
        make_topology("torus", n=4)


def test_default_fleet_mix():
    mix = default_fleet_mix()
    assert len(mix) == 16
    kinds = [m["kind"] for m in mix]
    assert kinds.count("heavy_hex") == 13 and kinds.count("grid") == 2 and kinds.count("linear") == 1
    # 10x d=5 and 3x d=7
    dists = [m.get("distance") for m in mix if m["kind"] == "heavy_hex"]
    assert dists.count(5) == 10 and dists.count(7) == 3


# ------------------------------------------------------------------ processes

def test_ou_exact_discretization_moments():
    theta, sigma, dt = 2.0, 0.5, 0.1
    rng = np.random.default_rng(0)
    x = proc.ou_init(np.zeros(4000), theta, sigma, rng)
    for _ in range(50):
        x = proc.ou_step(x, 0.0, theta, sigma, dt, rng)
    stat_std = proc.ou_stationary_std(theta, sigma)
    assert abs(float(x.std()) - stat_std) < 0.05 * stat_std
    # innovation variance matches the exact AR(1) formula
    phi = proc.ou_phi(theta, dt)
    assert np.isclose(proc.ou_eps_std(theta, sigma, dt) ** 2,
                      sigma**2 * (1 - phi**2) / (2 * theta))


def test_recal_times_median():
    rng = np.random.default_rng(1)
    times = proc.sample_recal_times(24.0 * 2000, 8.0, 0.35, rng)
    ivals = np.diff(np.concatenate([[0.0], times]))
    assert 6.5 < np.median(ivals) < 9.5  # LogNormal median ~ 8 h
    assert (times < 24.0 * 2000).all()


def test_offline_days_and_degradation():
    rng = np.random.default_rng(2)
    mask = proc.sample_offline_days(365, 0.05, 1 / 60, 0.4, rng)
    assert mask.dtype == bool and mask.shape == (365,) and 0 < mask.sum() < 365
    eps = proc.sample_degradation_episodes(365, 1 / 30, 10.0, 0.02, rng)
    assert eps.shape[1] == 3 and (eps[:, 2] <= 0).all()
    assert proc.degradation_offset(eps, 0.0) == 0.0
    assert proc.degradation_offset(eps, 365.0) <= 0.0


# ----------------------------------------------------------------- guarantees

def test_t2_le_2t1_everywhere(small_run):
    _, events, truth = small_run
    allrows = pd.concat([events, truth], ignore_index=True)
    piv = allrows[allrows["observable"].isin(["t1_us", "t2_us"])].pivot_table(
        index=["device", "entity_id", "channel", "ts"],
        columns="observable", values="value")
    piv = piv.dropna()
    assert len(piv) > 0
    assert (piv["t2_us"] <= 2.0 * piv["t1_us"] + 1e-9).all()
    assert (allrows["value"] > 0).all()


def test_bitwise_determinism_and_seed_sensitivity(small_run):
    cfg, events, truth = small_run
    ev2, tr2 = FleetSim(cfg, 20260712).run(20, 3)
    pd.testing.assert_frame_equal(events, ev2, check_exact=True)
    pd.testing.assert_frame_equal(truth, tr2, check_exact=True)
    ev3, _ = FleetSim(cfg, 12345).run(20, 3)
    assert not np.array_equal(events["value"].to_numpy(), ev3["value"].to_numpy())


def test_validate_uts_clean(small_run):
    _, events, truth = small_run
    uts.validate_uts(events)
    uts.validate_uts(truth)
    assert set(events["channel"].unique()) == {"claimed", "probe"}
    assert set(truth["channel"].unique()) == {"truth"}
    assert set(events["device"].unique()) == {"sim_dev_00", "sim_dev_01"}
    assert (events["source"] == "sim").all() and (events["platform"] == "sim_sc").all()
    assert "tphi_us" in set(truth["observable"]) and "tphi_us" not in set(events["observable"])


def test_claimed_refreshes_exactly_at_recals(small_run):
    _, events, _ = small_run
    q = events[(events["device"] == "sim_dev_00")
               & (events["entity_id"] == "q0")
               & (events["observable"] == "t1_us")
               & (events["channel"] == "claimed")].sort_values("ts")
    assert (q["ts"] >= q["cal_ts"]).all()
    assert q["cal_ts"].is_monotonic_increasing
    groups = q.groupby("cal_ts", sort=True)["value"]
    assert groups.ngroups >= 3  # ~3 recals/day for 20 days
    # claimed is frozen between recals ...
    assert (groups.nunique() == 1).all()
    # ... and refreshes to a new snapshot at every recal
    vals = groups.first().to_numpy()
    assert (np.diff(vals) != 0.0).all()
    # staleness (ts - cal_ts) resets after each recal
    stale_h = (q["ts"] - q["cal_ts"]).dt.total_seconds().to_numpy() / 3600.0
    first_ts = q.groupby("cal_ts", sort=True)["ts"].min()
    first_stale = ((first_ts - first_ts.index).dt.total_seconds() / 3600.0).to_numpy()
    assert (first_stale <= 8.0).all()  # a snapshot lands every 8 h at spd=3
    assert stale_h.max() > first_stale.min()


def test_probe_tracks_truth_and_claimed_goes_stale(small_run):
    cfg, events, truth = small_run
    keys = ["device", "entity_id", "observable", "ts"]
    t1t = truth[(truth["observable"] == "t1_us")][keys + ["value"]]
    for ch, rel in (("probe", cfg.probe_noise), ("claimed", cfg.claimed_noise)):
        obs = events[(events["observable"] == "t1_us") & (events["channel"] == ch)]
        m = obs[keys + ["value"]].merge(t1t, on=keys, suffixes=("_o", "_t"))
        relerr = np.abs(m["value_o"] - m["value_t"]) / m["value_t"]
        if ch == "probe":  # fresh every snapshot: ~ rel * |N(0,1)|
            assert relerr.mean() < 6 * rel
        else:  # stale between recals: strictly worse than probe noise on average
            assert relerr.mean() > cfg.probe_noise * np.sqrt(2 / np.pi) * 0.5


def test_long_series_lag1_autocorr_matches_theory():
    cfg = FleetConfig(
        topologies=[{"kind": "linear", "n": 8}],
        tls_on_rate=0.0, degrade_rate_per_day=0.0,
        recal_median_h=1e9,  # no recalibration re-pinning
        offline_p_mcar=0.0, offline_burst_rate=0.0)
    _, truth = FleetSim(cfg, 42).run(250, 3)
    dt = 1.0 / 3.0
    expected = np.exp(-cfg.ou_theta_t1 * dt)
    t1 = truth[truth["observable"] == "t1_us"]
    acs = []
    for _, g in t1.groupby("entity_id"):
        x = np.log(g.sort_values("ts")["value"].to_numpy())
        acs.append(np.corrcoef(x[:-1], x[1:])[0, 1])
    assert abs(float(np.mean(acs)) - expected) < 0.15


def test_offline_gaps_hit_visible_channels_only():
    cfg = small_cfg(offline_p_mcar=0.5, offline_burst_rate=0.0)
    events, truth = FleetSim(cfg, 3).run(20, 3)
    for dev in ("sim_dev_00", "sim_dev_01"):
        n_vis = events.loc[events["device"] == dev, "ts"].nunique()
        n_truth = truth.loc[truth["device"] == dev, "ts"].nunique()
        assert n_truth == 20 * 3
        assert n_vis < n_truth
        assert n_vis % 3 == 0  # whole offline days drop all 3 snapshots


def test_inject_gaps_masks_exactly_the_requested_windows(small_run):
    _, events, _ = small_run
    start = events["ts"].min() + pd.Timedelta(days=3)
    end = start + pd.Timedelta(days=2)
    spec = [{"device": "sim_dev_00", "start": start, "end": end},
            {"device": "sim_dev_01", "start": start, "end": end,
             "observables": ["t1_us"], "channels": ["probe"]}]
    masked = proc.inject_gaps(events, spec)
    m0 = (events["device"] == "sim_dev_00") & (events["ts"] >= start) & (events["ts"] < end)
    m1 = ((events["device"] == "sim_dev_01") & (events["ts"] >= start) & (events["ts"] < end)
          & (events["observable"] == "t1_us") & (events["channel"] == "probe"))
    expected = events.loc[~(m0 | m1)].reset_index(drop=True)
    assert m0.sum() > 0 and m1.sum() > 0
    pd.testing.assert_frame_equal(masked, expected)


# --------------------------------------------------------------- config & CLI

def test_config_json_roundtrip(tmp_path):
    cfg = small_cfg(recal_median_h=6.5, ou_sigma_t1=0.21)
    text = cfg.to_json(tmp_path / "cfg.json")
    back = FleetConfig.from_json(text)
    assert back == cfg
    assert FleetConfig.from_json((tmp_path / "cfg.json").read_text()) == cfg
    with pytest.raises(ValueError, match="unknown FleetConfig"):
        FleetConfig.from_json(json.dumps({"nope": 1}))


def test_cli_writes_separate_uts_and_truth_roots(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    small_cfg().to_json(cfg_path)
    out, truth_root = tmp_path / "lake", tmp_path / "truth"
    cmd = [sys.executable, "-m", "ltn.sim.fleet",
           "--config", str(cfg_path), "--days", "3", "--seed", "7",
           "--out", str(out), "--truth", str(truth_root)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    vis = uts.read_uts(out)
    assert len(vis) > 0 and "truth" not in set(vis["channel"])
    # truth lake: invisible by default, present when explicitly requested
    assert len(uts.read_uts(truth_root)) == 0
    tr = uts.read_uts(truth_root, include_truth=True)
    assert len(tr) > 0 and set(tr["channel"]) == {"truth"}
    # --write-default-config
    dflt = tmp_path / "default.json"
    r2 = subprocess.run([sys.executable, "-m", "ltn.sim.fleet",
                         "--write-default-config", str(dflt)],
                        capture_output=True, text=True)
    assert r2.returncode == 0 and len(FleetConfig.from_json(dflt.read_text()).topologies) == 16


# ------------------------------------------------------------- fit & validate

def test_target_stats_shape(small_run):
    _, events, _ = small_run
    stats = target_stats(events)
    for obs in ("t1_us", "t2_us", "readout_error", "gate_error_2q"):
        assert np.isfinite(stats[f"{obs}.mean"]) and stats[f"{obs}.std"] >= 0
    assert -1.0 <= stats["t1.acf1"] <= 1.0
    assert stats["disc.mean"] > 0
    assert 2.0 < stats["recal.mean_h"] < 30.0  # median 8 h cadence


def test_fit_smoke_recovers_recal_and_ou_sigma_direction():
    # ~3 s: known config -> perturbed start -> least_squares moment matching
    topo = {"kind": "linear", "n": 8}
    cfg_true = FleetConfig(topologies=[dict(topo), dict(topo)])
    real, _ = FleetSim(cfg_true, 99).run(60, 3)
    start = replace(cfg_true, recal_median_h=20.0, ou_sigma_t1=0.06)
    fitted = fit_fleet_config(
        real, start, knobs=("recal_median_h", "ou_sigma_t1"),
        eval_days=60, eval_topology=topo, max_nfev=16)
    # recal interval pulled toward the true 8 h (loose tolerance)
    assert abs(fitted.recal_median_h - cfg_true.recal_median_h) < \
        abs(start.recal_median_h - cfg_true.recal_median_h)
    # ou_sigma moved in the right direction (was 3x too small)
    assert fitted.ou_sigma_t1 > start.ou_sigma_t1


def test_sim_vs_real_report(tmp_path, small_run):
    cfg, events, _ = small_run
    other, _ = FleetSim(small_cfg(ou_sigma_t1=0.30), 5).run(15, 3)
    report = sim_vs_real_report(other, events, tmp_path, tex_dir=tmp_path)

    # Two SINGLE-page PDFs, not one two-page file: \includegraphics without a
    # page= option renders page 1 only, which silently dropped the dynamics
    # panel from the paper for as long as this was a single 2-page artifact.
    for name in ("sim_vs_real_marginals.pdf", "sim_vs_real_dynamics.pdf"):
        raw = (tmp_path / name).read_bytes()
        counts = [int(m) for m in re.findall(rb"/Count\s+(\d+)", raw)]
        assert counts == [1], f"{name} must be exactly one page, got /Count {counts}"
    assert not (tmp_path / "sim_vs_real.pdf").exists()

    loaded = json.loads((tmp_path / "sim_vs_real.json").read_text())
    for obs in ("t1_us", "t2_us", "readout_error", "gate_error_2q"):
        assert 0.0 <= report["ks"][obs]["statistic"] <= 1.0
        assert loaded["ks"][obs]["n_real"] > 0
    assert len(report["autocorr"]["lags"]) == 10
    assert report["discrepancy"]["sim_mean"] > 0

    # every target statistic is listed, with its fit-objective membership
    stats = {r["stat"] for r in report["validation_table"]}
    assert {"t1.acf1", "t1.persist_mae", "recal.mean_h"} <= stats
    assert all(set(r) >= {"real", "sim", "rel_err", "in_fit_objective"}
               for r in report["validation_table"])
    assert any(r["in_fit_objective"] for r in report["validation_table"])

    ax = report["extra_axes"]
    assert ax["tails"]["real"]["t1_us"]["p99"] >= ax["tails"]["real"]["t1_us"]["p95"]
    assert ax["drift_episodes"]["sim"]["n_series"] > 0
    assert 0.0 <= ax["missingness"]["sim"]["missing_frac"] <= 1.0
    assert ax["recal_diagnostic"]["sim"]["cal_ts_per_snapshot"] >= 1.0
    for name in ("T5_simval.tex", "T5b_simval_axes.tex"):
        assert "\\begin{tabular}" in (tmp_path / name).read_text()
