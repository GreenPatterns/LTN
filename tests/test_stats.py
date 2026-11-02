"""Cluster bootstrap + cell-level Wilcoxon (lane A3-clusterboot).

The paper's CIs were an iid bootstrap over nested rows (many rolling forecast
origins per device x observable series). These tests pin the two replacements:
`cluster_bootstrap_ci` (resamples cluster labels) and `cell_paired_wilcoxon`
(one test over cell means), and assert the legacy `bootstrap_ci` is untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from ltn.eval.stats import (
    bootstrap_ci,
    cell_paired_wilcoxon,
    cliffs_delta,
    cluster_bootstrap_ci,
    paired_wilcoxon_holm,
)

MEAN = lambda v: float(np.mean(v))  # noqa: E731


def _clustered(k=12, m=200, icc_shift=1.0, seed=0):
    """k clusters x m rows; all within-cluster variance comes from a shared offset."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, icc_shift, size=k)
    vals = np.concatenate([offsets[j] + rng.normal(0.0, 0.05, size=m) for j in range(k)])
    groups = np.repeat(np.arange(k), m)
    return vals, groups


# ----------------------------------------------------------- determinism / contract
def test_cluster_bootstrap_is_deterministic_under_seed():
    v, g = _clustered()
    a = cluster_bootstrap_ci(MEAN, v, groups=g, n_boot=500, seed=7)
    b = cluster_bootstrap_ci(MEAN, v, groups=g, n_boot=500, seed=7)
    assert a == b
    assert cluster_bootstrap_ci(MEAN, v, groups=g, n_boot=500, seed=8) != a


def test_cluster_bootstrap_brackets_the_point_estimate():
    v, g = _clustered()
    lo, hi = cluster_bootstrap_ci(MEAN, v, groups=g, n_boot=1000, seed=0)
    assert lo < float(np.mean(v)) < hi


def test_cluster_bootstrap_widens_ci_on_clustered_data():
    """The whole point: with strong within-cluster correlation the iid CI is too narrow."""
    v, g = _clustered(k=12, m=200, icc_shift=1.0)
    i_lo, i_hi = bootstrap_ci(MEAN, v, n_boot=1000, seed=0)
    c_lo, c_hi = cluster_bootstrap_ci(MEAN, v, groups=g, n_boot=1000, seed=0)
    assert (c_hi - c_lo) > 5.0 * (i_hi - i_lo)


def test_cluster_bootstrap_matches_iid_when_every_row_is_its_own_cluster():
    """Singleton clusters reduce the cluster bootstrap to the iid bootstrap."""
    rng = np.random.default_rng(3)
    v = rng.normal(size=300)
    g = np.arange(300)
    i = bootstrap_ci(MEAN, v, n_boot=4000, seed=0)
    c = cluster_bootstrap_ci(MEAN, v, groups=g, n_boot=4000, seed=0)
    assert np.allclose(i, c, atol=0.02)


def test_cluster_bootstrap_preserves_pairing_across_arrays():
    """Both arrays must get the same row draw, else a perfect paired diff would move."""
    v, g = _clustered(k=10, m=50)
    lo, hi = cluster_bootstrap_ci(
        lambda x, y: float(np.mean(x - y)), v, v + 2.0, groups=g, n_boot=200, seed=0
    )
    assert lo == pytest.approx(-2.0) and hi == pytest.approx(-2.0)


def test_cluster_bootstrap_handles_unequal_cluster_sizes():
    vals = np.concatenate([np.zeros(3), np.ones(97)])
    groups = np.concatenate([np.zeros(3, int), np.ones(97, int)])
    lo, hi = cluster_bootstrap_ci(MEAN, vals, groups=groups, n_boot=500, seed=0)
    assert lo >= 0.0 and hi <= 1.0 and lo < hi


def test_cluster_bootstrap_group_labels_may_be_strings():
    v = np.arange(60, dtype=float)
    g = np.array([f"dev_{i % 6}" for i in range(60)])
    lo, hi = cluster_bootstrap_ci(MEAN, v, groups=g, n_boot=300, seed=0)
    assert lo < float(np.mean(v)) < hi


def test_cluster_bootstrap_rejects_length_mismatch():
    with pytest.raises(ValueError, match="groups has length"):
        cluster_bootstrap_ci(MEAN, np.zeros(10), groups=np.zeros(9), n_boot=10)
    with pytest.raises(ValueError, match="same length"):
        cluster_bootstrap_ci(MEAN, np.zeros(10), np.zeros(9), groups=np.zeros(10), n_boot=10)


def test_cluster_bootstrap_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        cluster_bootstrap_ci(MEAN, np.zeros(0), groups=np.zeros(0), n_boot=10)


# ----------------------------------------------------------- legacy bootstrap frozen
def test_legacy_bootstrap_ci_is_unchanged():
    """Published numbers must not move: pin bootstrap_ci's exact output."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=200)
    lo, hi = bootstrap_ci(MEAN, v, n_boot=2000, seed=0)
    assert (lo, hi) == pytest.approx((-0.12222803189838999, 0.1434441244034861), abs=1e-12)


# ----------------------------------------------------------- cell-level Wilcoxon
def _cells(n_cells=36, rows=2000, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for c in range(n_cells):
        base = rng.normal(0.0, 1.0)  # cell-level offset shared by every row
        a = base + rng.normal(0.0, 0.01, size=rows)
        out[str(c)] = (a + shift, a.copy())
    return out


def test_cell_wilcoxon_reports_cell_count_not_row_count():
    w = cell_paired_wilcoxon(_cells(n_cells=36, rows=2000))
    assert int(w.iloc[0]["n_cells"]) == 36


def test_cell_wilcoxon_is_far_less_significant_than_row_level_on_nested_data():
    """A tiny shift replicated across thousands of nested rows fakes significance."""
    cells = _cells(n_cells=36, rows=2000, shift=0.002)
    row_level = paired_wilcoxon_holm(cells)
    per_cell_p = row_level[row_level["cell"] != "__summary__"]["wilcoxon_p"].to_numpy()
    cell_level_p = float(cell_paired_wilcoxon(cells).iloc[0]["wilcoxon_p"])
    assert np.median(per_cell_p) < 1e-6      # row-level: overwhelming "evidence"
    assert cell_level_p > np.median(per_cell_p)


def test_cell_wilcoxon_all_zero_differences_give_p_one():
    cells = {"a": (np.arange(5.0), np.arange(5.0)), "b": (np.ones(5), np.ones(5))}
    w = cell_paired_wilcoxon(cells)
    assert float(w.iloc[0]["wilcoxon_p"]) == 1.0
    assert np.isnan(float(w.iloc[0]["statistic"]))


def test_cell_wilcoxon_detects_a_consistent_uniform_shift():
    w = cell_paired_wilcoxon(_cells(n_cells=36, rows=50, shift=0.5))
    assert float(w.iloc[0]["wilcoxon_p"]) < 1e-6
    assert float(w.iloc[0]["frac_cells_a_worse"]) == 1.0
    assert float(w.iloc[0]["cliffs_delta"]) > 0.0


def test_cell_wilcoxon_rejects_ragged_cells():
    with pytest.raises(ValueError, match="differ in shape"):
        cell_paired_wilcoxon({"x": (np.zeros(4), np.zeros(3))})


def test_cell_wilcoxon_rejects_no_cells():
    with pytest.raises(ValueError, match="no cells"):
        cell_paired_wilcoxon({})


def test_cliffs_delta_sign_convention_unchanged():
    assert cliffs_delta(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert cliffs_delta(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == -1.0
