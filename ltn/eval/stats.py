"""Statistical protocol machinery (master plan §2.7).

Guarantee: implements the pre-registered protocol exactly — paired Wilcoxon
signed-rank per device×observable cell (scipy, zero_method='pratt'), Holm
step-down correction across cells, Cliff's δ effect sizes, and percentile
bootstrap CIs with joint row resampling. Deterministic under fixed seed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

__all__ = [
    "cliffs_delta",
    "holm_adjust",
    "paired_wilcoxon_holm",
    "bootstrap_ci",
    "cluster_bootstrap_ci",
    "cell_paired_wilcoxon",
]

SUMMARY_ROW = "__summary__"


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's δ ∈ [-1, 1]: P(a > b) − P(a < b) over all (i, j) pairs, exact O(n·m)."""
    a = np.asarray(a, dtype=np.float64)[:, None]
    b = np.asarray(b, dtype=np.float64)[None, :]
    gt = np.count_nonzero(a > b)
    lt = np.count_nonzero(a < b)
    return float((gt - lt) / (a.size * b.size))


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, clipped at 1)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired_wilcoxon_holm(
    cells: dict[str, tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-cell paired Wilcoxon + Holm correction + Cliff's δ (§2.7).

    Each cell maps name → (a, b) paired samples (e.g. per-seed metric of two
    models on one device×observable). Cells whose paired differences are all
    zero get p = 1.0 (scipy's pratt handling is undefined there). The returned
    frame has one row per cell plus a `__summary__` row (n = total pairs,
    cliffs_delta = median δ, significant = fraction of cells with Holm p < alpha).
    """
    names = list(cells)
    raw_p: list[float] = []
    stats_: list[float] = []
    deltas: list[float] = []
    ns: list[int] = []
    for name in names:
        a, b = (np.asarray(x, dtype=np.float64) for x in cells[name])
        if a.shape != b.shape:
            raise ValueError(f"cell {name!r}: paired arrays differ in shape {a.shape} vs {b.shape}")
        ns.append(len(a))
        deltas.append(cliffs_delta(a, b))
        d = a - b
        if np.all(d == 0):
            stats_.append(float("nan"))
            raw_p.append(1.0)
        else:
            res = wilcoxon(a, b, zero_method="pratt")
            stats_.append(float(res.statistic))
            raw_p.append(float(res.pvalue))
    holm = holm_adjust(np.array(raw_p))
    sig = (holm < alpha).astype(np.float64)
    df = pd.DataFrame({
        "cell": names,
        "n": ns,
        "statistic": stats_,
        "wilcoxon_p": raw_p,
        "holm_p": holm,
        "cliffs_delta": deltas,
        "significant": sig,
    })
    summary = pd.DataFrame({
        "cell": [SUMMARY_ROW],
        "n": [int(np.sum(ns))],
        "statistic": [float("nan")],
        "wilcoxon_p": [float("nan")],
        "holm_p": [float("nan")],
        "cliffs_delta": [float(np.median(deltas))],
        "significant": [float(np.mean(sig))],
    })
    return pd.concat([df, summary], ignore_index=True)


def cell_paired_wilcoxon(
    cells: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """ONE Wilcoxon over cell-level means — the unit of analysis is the cell.

    :func:`paired_wilcoxon_holm` tests the raw forecast pairs *inside* each
    cell as if they were independent. They are not: a cell is one series
    observed at many rolling forecast origins, so its effective sample size is
    far below its row count and the per-cell p-values are anti-conservative.
    Collapsing each cell to its mean of `a` and its mean of `b`, then running a
    single paired Wilcoxon across cells, makes the cell the unit of analysis
    and removes the inflated-n problem — at the cost of power (n = number of
    cells, e.g. 36 for E1a and 240 for E1b).

    Returns a one-row frame: `n_cells`, `statistic`, `wilcoxon_p`,
    `cliffs_delta` (over the cell means), and `frac_cells_a_worse`.
    """
    names = list(cells)
    a_mean, b_mean = [], []
    for name in names:
        a, b = (np.asarray(x, dtype=np.float64) for x in cells[name])
        if a.shape != b.shape:
            raise ValueError(f"cell {name!r}: paired arrays differ in shape {a.shape} vs {b.shape}")
        a_mean.append(float(np.mean(a)))
        b_mean.append(float(np.mean(b)))
    am, bm = np.asarray(a_mean), np.asarray(b_mean)
    if am.size == 0:
        raise ValueError("cell_paired_wilcoxon: no cells")
    if np.all(am - bm == 0):
        stat, p = float("nan"), 1.0
    else:
        res = wilcoxon(am, bm, zero_method="pratt")
        stat, p = float(res.statistic), float(res.pvalue)
    return pd.DataFrame({
        "n_cells": [int(am.size)],
        "statistic": [stat],
        "wilcoxon_p": [p],
        "cliffs_delta": [cliffs_delta(am, bm)],
        "frac_cells_a_worse": [float(np.mean(am > bm))],
    })


def bootstrap_ci(
    stat_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for stat_fn(*arrays), resampling rows jointly.

    All arrays must share length n; each replicate draws one index vector and
    applies it to every array (preserves pairing). Deterministic given `seed`.
    """
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise ValueError("bootstrap_ci: all arrays must have the same length")
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[i] = stat_fn(*(a[idx] for a in arrs))
    lo, hi = np.percentile(reps, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)


def cluster_bootstrap_ci(
    stat_fn: Callable[..., float],
    *arrays: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CLUSTER bootstrap CI: resample cluster labels, not rows.

    The rows fed to :func:`bootstrap_ci` are not independent — a (device,
    observable, horizon) cell holds many rolling forecast origins on ONE
    series, so the iid bootstrap treats `n` correlated rows as `n` independent
    ones and reports an interval that is too narrow. This estimator draws `k`
    cluster labels with replacement from the `k` distinct values of `groups`
    and concatenates every member row of each drawn cluster, so the replicate
    variance reflects between-cluster variation only.

    `groups` must align row-for-row with the arrays. Deterministic given
    `seed`. Replicates have varying length when clusters differ in size, so
    `stat_fn` must accept any length (means, medians and rank statistics do).

    Caveat: the cluster bootstrap is asymptotic in `k`, not in `n`. At very
    small `k` it is anti-conservative and can return an interval NARROWER than
    the iid one — a reliable sign the clustering is too coarse or too small.
    Prefer the finest clustering that still absorbs the dependence: the paper
    clusters on (device, observable), giving k = 12 on E1a and k = 60 on E1b.
    """
    if not arrays:
        raise ValueError("cluster_bootstrap_ci: no arrays given")
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise ValueError("cluster_bootstrap_ci: all arrays must have the same length")
    g = np.asarray(groups)
    if len(g) != n:
        raise ValueError(
            f"cluster_bootstrap_ci: groups has length {len(g)}, arrays have length {n}"
        )
    if n == 0:
        raise ValueError("cluster_bootstrap_ci: empty input")
    # member row indices per cluster, in a stable order so the seed reproduces
    uniq, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    members = np.split(order, starts[1:])
    k = len(uniq)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, k, size=k)
        idx = np.concatenate([members[j] for j in pick])
        reps[i] = stat_fn(*(a[idx] for a in arrs))
    lo, hi = np.percentile(reps, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)
