"""Reliability endpoints derived in closed form from stored Gaussian forecasts.

Motivation (IEEE Trans. Reliability framing): the pre-registered endpoints of this
study are forecasting metrics (NCRPS, Spearman rho, RMSE, AUROC). Reliability
engineering asks different questions of the same predictive distribution:

  * SLA / threshold-violation probability -- will the device be out of spec at
    t+h?  With a Gaussian predictive law N(mu, sigma^2) this is a closed-form
    transform of the *already stored* (mu, sigma), scored as a probabilistic
    classifier against the stored outcome y (Brier, ECE, Brier skill score).
  * Discrete-time reliability R(h) = P(no violation through horizon h) and the
    associated hazard 1 - R(h)/R(h-1), evaluated on the stored horizon grid.
  * Expected wasted shots N_shots * (1 - p(S)) and availability, for a policy
    that selects hardware to run a job on.

Nothing here refits or retrains anything: every quantity is a deterministic
function of columns already present in runs/**/predictions.parquet (mu, sigma, y)
or of the reward column of the E6 replay parquets. That is deliberate -- it means
the reliability endpoints cannot flatter the model relative to the endpoints
already reported.

Sign convention.  A "violation" always means *quality below spec*:

  * ``direction="upper"``  -- larger y is worse (error rates).  Violation is
    y > threshold, so P = 1 - Phi((thr - mu)/sigma) = Phi((mu - thr)/sigma).
  * ``direction="lower"``  -- smaller y is worse (coherence times T1/T2).
    Violation is y < threshold, so P = Phi((thr - mu)/sigma).

Both are the same closed form up to a sign flip of the standardised margin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _sstats

from ltn.eval.metrics import brier, ece

__all__ = [
    "DEFAULT_DIRECTIONS",
    "direction_for",
    "violation_probability",
    "violation_label",
    "sla_thresholds",
    "attach_violations",
    "sla_scores",
    "reliability_curve",
    "wasted_shots",
    "paired_bootstrap_diff",
]

#: Which tail of each stored observable counts as "out of spec".
DEFAULT_DIRECTIONS: dict[str, str] = {
    "gate_error_2q": "upper",
    "epc_1q": "upper",
    "readout_error": "upper",
    "t1_us": "lower",
    "t2_us": "lower",
}

_SIGMA_FLOOR = 1e-12


def direction_for(observable: str, directions: dict[str, str] | None = None) -> str:
    """Tail convention for `observable`; unknown names default to 'upper'."""
    d = DEFAULT_DIRECTIONS if directions is None else directions
    return d.get(str(observable), "upper")


def _standardised_margin(mu, sigma, threshold, direction: str) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), _SIGMA_FLOOR)
    threshold = np.asarray(threshold, dtype=np.float64)
    if direction == "lower":
        return (threshold - mu) / sigma
    if direction == "upper":
        return (mu - threshold) / sigma
    raise ValueError(f"direction must be 'lower' or 'upper', got {direction!r}")


def violation_probability(mu, sigma, threshold, direction: str = "lower") -> np.ndarray:
    """P(out of spec) under N(mu, sigma^2): Phi of the standardised margin."""
    return _sstats.norm.cdf(_standardised_margin(mu, sigma, threshold, direction))


def violation_label(y, threshold, direction: str = "lower") -> np.ndarray:
    """Realised out-of-spec indicator (int array) for outcomes `y`."""
    y = np.asarray(y, dtype=np.float64)
    threshold = np.asarray(threshold, dtype=np.float64)
    if direction == "lower":
        return (y < threshold).astype(int)
    if direction == "upper":
        return (y > threshold).astype(int)
    raise ValueError(f"direction must be 'lower' or 'upper', got {direction!r}")


def sla_thresholds(
    df: pd.DataFrame,
    base_rate: float = 0.10,
    group_cols: tuple[str, ...] = ("device", "observable"),
    directions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Empirical-quantile SLA thresholds with a nominal `base_rate` violation rate.

    The threshold for each group is the quantile of the *stored outcomes* y that
    puts `base_rate` of the observed mass out of spec: the (1 - base_rate)
    quantile for 'upper' observables, the `base_rate` quantile for 'lower' ones.
    Deriving it from y (not from any model's mu) is what makes it identical for
    every model, so a threshold choice cannot advantage one forecaster.

    Returns a frame with `group_cols` plus `direction`, `threshold`, `n`.
    """
    if not 0.0 < base_rate < 1.0:
        raise ValueError(f"base_rate must be in (0,1), got {base_rate}")
    rows = []
    for keys, g in df.groupby(list(group_cols), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        obs = dict(zip(group_cols, keys)).get("observable", "")
        direction = direction_for(obs, directions)
        q = 1.0 - base_rate if direction == "upper" else base_rate
        rows.append(
            dict(zip(group_cols, keys))
            | {
                "direction": direction,
                "threshold": float(np.quantile(g["y"].to_numpy(dtype=np.float64), q)),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def attach_violations(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    group_cols: tuple[str, ...] = ("device", "observable"),
) -> pd.DataFrame:
    """Join `thresholds` onto `df` and add `p_viol` (predicted) and `viol` (realised)."""
    out = df.merge(thresholds[list(group_cols) + ["direction", "threshold"]],
                   on=list(group_cols), how="inner")
    if len(out) != len(df):
        raise ValueError(
            f"threshold join changed row count ({len(df)} -> {len(out)}); "
            "thresholds must cover every group exactly once"
        )
    out = out.reset_index(drop=True)
    p = np.empty(len(out), dtype=np.float64)
    v = np.empty(len(out), dtype=int)
    dirs = out["direction"].to_numpy()
    for direction in np.unique(dirs):
        m = dirs == direction
        sl = out.loc[m]
        p[m] = violation_probability(sl["mu"], sl["sigma"], sl["threshold"], direction)
        v[m] = violation_label(sl["y"], sl["threshold"], direction)
    out["p_viol"] = p
    out["viol"] = v
    return out


def sla_scores(df: pd.DataFrame, n_bins: int = 15) -> dict[str, float]:
    """Brier / ECE / Brier-skill for the violation probabilities in `df`.

    `df` must carry `p_viol` and `viol` (see :func:`attach_violations`). The
    reference forecast for the skill score is the constant climatology (the
    realised base rate), whose Brier score is base*(1-base).
    """
    p = df["p_viol"].to_numpy(dtype=np.float64)
    v = df["viol"].to_numpy(dtype=np.float64)
    base = float(v.mean())
    bs = brier(v, p)
    clim = base * (1.0 - base)
    return {
        "n": int(len(p)),
        "base_rate": base,
        "mean_p": float(p.mean()),
        "brier": float(bs),
        "brier_climatology": float(clim),
        "brier_skill": float(1.0 - bs / clim) if clim > 0 else float("nan"),
        "ece": float(ece(v, p, n_bins=n_bins)),
    }


def reliability_curve(
    df: pd.DataFrame,
    origin_cols: tuple[str, ...] = ("device", "observable", "s_idx", "t_idx"),
    horizon_col: str = "horizon",
) -> pd.DataFrame:
    """Discrete-time reliability R(h) and hazard over the stored horizon grid.

    R(h) = P(no violation at any checkpoint through h). The predicted survival
    multiplies per-horizon (1 - p_viol) across checkpoints, i.e. it assumes
    violations at different horizons are conditionally independent given the
    forecast -- an approximation, since the same latent drift drives them. The
    empirical survival makes no such assumption. Divergence between the two
    columns is therefore itself informative and must not be read as pure
    miscalibration.

    Only origins observed at *every* horizon are used, so R is a survival curve
    on a fixed cohort. Returns horizon, R_pred, R_emp, hazard_pred, hazard_emp,
    n_origins.
    """
    horizons = sorted(df[horizon_col].unique())
    piv_p = df.pivot_table(index=list(origin_cols), columns=horizon_col,
                           values="p_viol", aggfunc="mean")
    piv_v = df.pivot_table(index=list(origin_cols), columns=horizon_col,
                           values="viol", aggfunc="max")
    complete = piv_p.notna().all(axis=1) & piv_v.notna().all(axis=1)
    piv_p, piv_v = piv_p[complete], piv_v[complete]
    n_origins = int(len(piv_p))
    if n_origins == 0:
        raise ValueError("no origin observed at every horizon; cannot form a survival curve")

    surv_p = np.cumprod(1.0 - piv_p[horizons].to_numpy(dtype=np.float64), axis=1)
    surv_v = np.cumprod(1.0 - piv_v[horizons].to_numpy(dtype=np.float64), axis=1)
    R_pred = surv_p.mean(axis=0)
    R_emp = surv_v.mean(axis=0)

    rows = []
    for k, h in enumerate(horizons):
        prev_p = R_pred[k - 1] if k else 1.0
        prev_e = R_emp[k - 1] if k else 1.0
        rows.append({
            "horizon": int(h),
            "R_pred": float(R_pred[k]),
            "R_emp": float(R_emp[k]),
            "hazard_pred": float(1.0 - R_pred[k] / prev_p) if prev_p > 0 else float("nan"),
            "hazard_emp": float(1.0 - R_emp[k] / prev_e) if prev_e > 0 else float("nan"),
            "n_origins": n_origins,
        })
    return pd.DataFrame(rows)


def wasted_shots(rewards, n_shots: int = 4096, best_rewards=None) -> dict[str, float]:
    """Expected wasted shots N_shots*(1 - p(S)) and availability for a replay.

    `rewards` is the per-round realised job-success probability p(S) (the
    `reward` column written by ltn.eval.regret_sim.replay). Availability is the
    expected usable-shot fraction, i.e. mean p(S). If `best_rewards` is given
    (the per-round oracle p(S)), the excess wasted shots against the oracle --
    the quantity an operator actually pays for -- is reported too.
    """
    r = np.asarray(rewards, dtype=np.float64)
    out = {
        "n_rounds": int(r.size),
        "n_shots": int(n_shots),
        "availability": float(r.mean()),
        "wasted_shots_mean": float(n_shots * (1.0 - r).mean()),
        "wasted_frac_mean": float((1.0 - r).mean()),
        "wasted_shots_worst_round": float(n_shots * (1.0 - r).max()),
    }
    if best_rewards is not None:
        b = np.asarray(best_rewards, dtype=np.float64)
        out["excess_wasted_shots_vs_oracle"] = float(n_shots * (b - r).mean())
        out["excess_wasted_frac_vs_oracle"] = float((b - r).mean())
    return out


def paired_bootstrap_diff(
    loss_a,
    loss_b,
    clusters,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Cluster bootstrap CI for mean(loss_a) - mean(loss_b) on paired rows.

    Rows of `loss_a` and `loss_b` are the per-row losses (e.g. squared error of a
    violation probability) of two models on the *same* row, so the comparison is
    paired. Resampling whole clusters (devices) rather than rows respects the
    strong within-device correlation of telemetry, which an i.i.d. row bootstrap
    would ignore and so produce implausibly tight intervals.

    Negative `diff` means model A has the lower loss (A is better). The interval
    excludes 0 iff the two models separate at the `1-alpha` level.
    """
    a = np.asarray(loss_a, dtype=np.float64)
    b = np.asarray(loss_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired losses must have equal shape, got {a.shape} vs {b.shape}")
    d = a - b
    codes, _ = pd.factorize(np.asarray(clusters))
    n_clusters = int(codes.max()) + 1
    if n_clusters < 2:
        raise ValueError("need at least 2 clusters to bootstrap")
    # Per-cluster sums/counts make each bootstrap replicate O(n_clusters).
    sums = np.bincount(codes, weights=d, minlength=n_clusters)
    counts = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    reps = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "diff": float(d.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_clusters": n_clusters,
        "n_boot": int(n_boot),
        "separates": bool(lo > 0.0 or hi < 0.0),
    }
