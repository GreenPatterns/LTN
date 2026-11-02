"""Pre-registered evaluation metrics (master plan §2.4 B2, §2.7, §2.8).

Guarantee: every hard metric is numpy-in / float-out and agrees with its
reference implementation (properscoring, scikit-learn, scipy) — these numbers
go straight into paper tables via `ltn.eval.runlog`. `soft_ece` is the
differentiable torch surrogate used as the λ₄ loss term (§2.2 Loss); it
converges to the hard binned `ece` as `temperature → ∞` and is autograd-safe.
"""
from __future__ import annotations

import numpy as np
import properscoring as _ps
import torch
from scipy import stats as _sstats
from sklearn.metrics import average_precision_score as _aps
from sklearn.metrics import roc_auc_score as _ras

__all__ = [
    "mae", "rmse", "crps_gaussian", "auroc", "auprc", "ece", "brier",
    "gaussian_nll", "spearman", "coverage", "regret", "soft_ece",
]


def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(_f64(y) - _f64(yhat))))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((_f64(y) - _f64(yhat)) ** 2)))


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean CRPS of Gaussian forecasts N(mu, sigma^2) against outcomes y."""
    return float(np.mean(_ps.crps_gaussian(_f64(y), mu=_f64(mu), sig=_f64(sigma))))


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (binary labels, real-valued scores)."""
    return float(_ras(_f64(y_true).astype(int), _f64(scores)))


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    return float(_aps(_f64(y_true).astype(int), _f64(scores)))


def ece(y_true_binary: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: equal-width bins on [0,1], Σ_b (n_b/N)·|acc_b − conf_b|."""
    y = _f64(y_true_binary)
    p = _f64(p)
    # p == 1.0 lands in the last bin rather than an (empty) extra one.
    idx = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(total)


def brier(y_true_binary: np.ndarray, p: np.ndarray) -> float:
    """Brier score: mean squared error of probabilities against binary outcomes."""
    return float(np.mean((_f64(p) - _f64(y_true_binary)) ** 2))


def gaussian_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean Gaussian negative log-likelihood of y under N(mu, sigma^2)."""
    y, mu, sigma = _f64(y), _f64(mu), _f64(sigma)
    return float(np.mean(
        0.5 * np.log(2.0 * np.pi * sigma**2) + (y - mu) ** 2 / (2.0 * sigma**2)
    ))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation coefficient (E5 zero-shot ranking metric)."""
    return float(_sstats.spearmanr(_f64(a), _f64(b)).statistic)


def coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float = 0.9) -> float:
    """Empirical coverage of the central `level` Gaussian prediction interval."""
    z = _sstats.norm.ppf(0.5 + level / 2.0)
    inside = np.abs(_f64(y) - _f64(mu)) <= z * _f64(sigma)
    return float(np.mean(inside))


def regret(chosen_value: np.ndarray | float, best_value: np.ndarray | float) -> np.ndarray | float:
    """Regret = best_value − chosen_value (higher-is-better values; E6 policy replay)."""
    r = _f64(best_value) - _f64(chosen_value)
    return float(r) if r.ndim == 0 else r


def soft_ece(
    p: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 15,
    temperature: float = 50.0,
) -> torch.Tensor:
    """Differentiable ECE via soft binning (λ₄ loss term, §2.2).

    Bin membership uses sigmoid kernels of sharpness `temperature` at the
    internal bin edges; memberships sum to exactly 1 per sample, so the
    hard `ece` is recovered in the temperature → ∞ limit. Returns a scalar
    tensor connected to `p`'s autograd graph.
    """
    if p.ndim != 1:
        p = p.reshape(-1)
    y = y.reshape(-1).to(p.dtype)
    n = p.shape[0]
    edges = torch.linspace(0.0, 1.0, n_bins + 1, dtype=p.dtype, device=p.device)[1:-1]
    # cum[:, j] ≈ 1{p > edge_j}; padding with exact 1/0 makes memberships a partition of unity
    cum = torch.sigmoid(temperature * (p.unsqueeze(1) - edges.unsqueeze(0)))
    ones = torch.ones(n, 1, dtype=p.dtype, device=p.device)
    zeros = torch.zeros(n, 1, dtype=p.dtype, device=p.device)
    w = torch.cat([ones, cum], dim=1) - torch.cat([cum, zeros], dim=1)  # (n, n_bins)
    mass = w.sum(dim=0)
    eps = torch.finfo(p.dtype).eps
    conf = (w * p.unsqueeze(1)).sum(dim=0) / (mass + eps)
    acc = (w * y.unsqueeze(1)).sum(dim=0) / (mass + eps)
    return ((mass / n) * (conf - acc).abs()).sum()
