"""Stochastic processes driving the synthetic fleet [DEV-2] — numpy only.

Guarantees implemented (unit-tested in tests/test_sim.py):

* **Exact OU/AR(1) discretization.** For dX = theta (mu - X) dt + sigma dW the
  transition over a step dt is X(t+dt) = mu + phi (X(t) - mu) + eps with
  phi = exp(-theta dt) and eps ~ N(0, sigma^2 (1 - phi^2) / (2 theta)) — exact
  for any dt, so lag-1 autocorrelation of a stationary series equals
  exp(-theta dt) and the stationary std is sigma / sqrt(2 theta).
* **Bitwise determinism.** Every function that draws randomness takes an
  explicit ``numpy.random.Generator``; identical (arguments, generator state)
  give bitwise-identical outputs.
* **TLS episodes** are a per-qubit 2-state Markov chain (on/off rates per day)
  whose "on" state multiplies T1 by a per-episode factor ~ Uniform(lo, hi).
* **Recalibration renewal**: inter-recal intervals ~ LogNormal with the given
  median; the re-pin update is x <- (1-kappa) x + kappa mu + N(0, jump_sigma).
* **inject_gaps** removes exactly the rows inside the requested windows and
  nothing else (E7 masking utility).
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- OU

def ou_phi(theta: float, dt: float) -> float:
    """AR(1) coefficient of the exact OU discretization: exp(-theta*dt)."""
    return float(np.exp(-theta * dt))


def ou_eps_std(theta: float, sigma: float, dt: float) -> float:
    """Std of the exact OU innovation: sqrt(sigma^2 (1-phi^2) / (2 theta))."""
    phi = ou_phi(theta, dt)
    return float(sigma * np.sqrt(max(1.0 - phi * phi, 0.0) / (2.0 * theta)))


def ou_stationary_std(theta: float, sigma: float) -> float:
    """Stationary std of the OU process: sigma / sqrt(2 theta)."""
    return float(sigma / np.sqrt(2.0 * theta))


def ou_init(mu: np.ndarray, theta: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Draw from the stationary law N(mu, sigma^2 / (2 theta)) elementwise."""
    mu = np.asarray(mu, dtype=np.float64)
    return mu + ou_stationary_std(theta, sigma) * rng.standard_normal(mu.shape)


def ou_step(
    x: np.ndarray,
    mu: np.ndarray | float,
    theta: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One exact AR(1) step of the OU process (vectorized, float64)."""
    x = np.asarray(x, dtype=np.float64)
    phi = ou_phi(theta, dt)
    eps = ou_eps_std(theta, sigma, dt) * rng.standard_normal(x.shape)
    return mu + phi * (x - mu) + eps


# --------------------------------------------------------------------- TLS chain

def tls_step(
    on: np.ndarray,
    factor: np.ndarray,
    on_rate: float,
    off_rate: float,
    factor_lo: float,
    factor_hi: float,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the per-qubit TLS 2-state Markov chain by dt (days).

    ``on`` is a bool array; ``factor`` the current multiplicative T1 factor
    (1.0 while off). Switching probabilities over dt are 1 - exp(-rate*dt).
    A qubit that turns on draws a fresh episode factor ~ U(lo, hi); a qubit
    that turns off resets its factor to 1. Draw counts are shape-fixed so the
    generator stream does not depend on the chain state.
    """
    on = np.asarray(on, dtype=bool)
    p_on = 1.0 - np.exp(-on_rate * dt)
    p_off = 1.0 - np.exp(-off_rate * dt)
    u = rng.random(on.shape)
    new_on = np.where(on, u >= p_off, u < p_on)
    fresh = rng.uniform(factor_lo, factor_hi, on.shape)
    new_factor = np.where(new_on & ~on, fresh, np.where(new_on, factor, 1.0))
    return new_on, new_factor.astype(np.float64)


# ------------------------------------------------------------------ degradation

def sample_degradation_episodes(
    n_days: float,
    arrival_rate_per_day: float,
    mean_duration_days: float,
    drift_per_day: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rare device-level degradation episodes: Poisson arrivals, geometric
    duration (days), linear downward drift on the device mean (log units/day).

    Returns an array of shape (n_episodes, 3): [start_day, duration_days, slope].
    """
    n = int(rng.poisson(arrival_rate_per_day * n_days))
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)
    starts = np.sort(rng.uniform(0.0, n_days, n))
    p = min(max(1.0 / max(mean_duration_days, 1.0), 1e-9), 1.0)
    durations = rng.geometric(p, n).astype(np.float64)
    slopes = -np.abs(rng.normal(0.0, drift_per_day, n))
    return np.column_stack([starts, durations, slopes])


def degradation_offset(episodes: np.ndarray, t_day: float) -> float:
    """Cumulative drift contributed by all episodes up to time t (days)."""
    if len(episodes) == 0:
        return 0.0
    start, dur, slope = episodes[:, 0], episodes[:, 1], episodes[:, 2]
    active = np.clip(t_day - start, 0.0, dur)
    return float(np.sum(slope * active))


# ---------------------------------------------------------------- recalibration

def sample_recal_times(
    t_end_h: float,
    median_h: float,
    sigma_log: float,
    rng: np.random.Generator,
    t0_h: float = 0.0,
) -> np.ndarray:
    """Renewal recalibration times in (t0, t_end) hours; intervals ~ LogNormal
    with the given median (exp of the underlying normal mean)."""
    times: list[float] = []
    t = float(t0_h)
    mu_log = np.log(median_h)
    max_iter = int(50 * (t_end_h - t0_h) / max(median_h, 1e-6)) + 100
    for _ in range(max_iter):
        t += float(rng.lognormal(mu_log, sigma_log))
        if t >= t_end_h:
            break
        times.append(t)
    return np.asarray(times, dtype=np.float64)


def recal_repin(
    x: np.ndarray,
    mu: np.ndarray | float,
    kappa: float,
    jump_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Recalibration partially re-pins the latent: (1-kappa) x + kappa mu + jump."""
    x = np.asarray(x, dtype=np.float64)
    return (1.0 - kappa) * x + kappa * np.asarray(mu, dtype=np.float64) + jump_sigma * rng.standard_normal(x.shape)


# ------------------------------------------------------------------------- gaps

def sample_offline_days(
    n_days: int,
    p_mcar: float,
    burst_rate_per_day: float,
    burst_geo_p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Device-offline day mask: MCAR single days plus geometric-length bursts."""
    n_days = int(n_days)
    mask = rng.random(n_days) < p_mcar
    n_bursts = int(rng.poisson(burst_rate_per_day * n_days))
    if n_bursts:
        starts = rng.integers(0, n_days, n_bursts)
        lengths = rng.geometric(min(max(burst_geo_p, 1e-9), 1.0), n_bursts)
        for s, ln in zip(starts, lengths):
            mask[int(s): int(s) + int(ln)] = True
    return mask


def inject_gaps(events_df, spec):
    """Post-hoc masking utility (E7): drop rows inside the requested windows.

    ``spec`` is an iterable of dicts with keys:
      * ``start``, ``end`` — timestamp-likes; a row is masked iff
        start <= ts < end;
      * optional ``device``, ``observables`` (list), ``channels`` (list),
        ``entity_ids`` (list) restricting the window's scope.

    Returns a new DataFrame with exactly the matching rows removed (index
    reset); the input is not modified.
    """
    import pandas as pd  # local import: this module is otherwise numpy-only

    drop = np.zeros(len(events_df), dtype=bool)
    ts = events_df["ts"]
    for w in spec:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w["end"])
        if start.tz is None:
            start = start.tz_localize("UTC")
        if end.tz is None:
            end = end.tz_localize("UTC")
        m = (ts >= start) & (ts < end)
        if w.get("device") is not None:
            m &= events_df["device"] == w["device"]
        if w.get("observables") is not None:
            m &= events_df["observable"].isin(list(w["observables"]))
        if w.get("channels") is not None:
            m &= events_df["channel"].isin(list(w["channels"]))
        if w.get("entity_ids") is not None:
            m &= events_df["entity_id"].isin(list(w["entity_ids"]))
        drop |= m.to_numpy()
    return events_df.loc[~drop].reset_index(drop=True)
