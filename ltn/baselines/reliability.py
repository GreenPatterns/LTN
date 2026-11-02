"""Classical reliability / degradation baselines for the E1 forecasting task.

They are included because ARIMA, EWMA and persistence do not represent the
reliability-modelling literature. Every model
here implements the same :class:`ltn.baselines.common.Forecaster` protocol as the
existing baselines (``fit(panel, train_mask, val_mask, cfg)`` /
``predict(pairs, panel) -> (mu, sigma)``) and therefore runs through the *identical*
evaluation path in :mod:`ltn.exp.run`: same frozen splits, same forecast pairs, same
per-cell NCRPS, same ``results.jsonl`` + ``predictions.parquet``.

Models
------
``kalman``   Local linear-trend / local-level linear-Gaussian state-space model, one
             per series, variances by per-series maximum likelihood (prediction-error
             decomposition) on the TRAIN fold only. The classical reliability
             workhorse and the direct classical competitor to a learned latent state.
``gp``       Gaussian-process degradation model, RBF (+ optional linear trend) plus a
             white-noise term, hyperparameters by marginal-likelihood maximisation
             (scikit-learn ``GaussianProcessRegressor``) on the TRAIN fold, posterior
             conditioned on a causal sliding window of history at each origin.
``wiener``   Wiener-process degradation (drift + diffusion), MLE per series. Optionally
             fitted on a log scale for strictly positive observables, in which case the
             lognormal predictive law is moment-matched back to (mu, sigma).
``gamma``    Gamma-process degradation. Only meaningful for monotone degradation; see
             :func:`monotonicity_report`, which is what decides whether it is
             admissible on this telemetry.
``hmm``      Markov-switching (2-3 regime) state-space model on the increment process,
             Baum-Welch per series; exact h-step moment recursion over the regime chain.

No-lookahead
------------
Every model's parameters come from ``train_mask`` only, and every per-origin quantity
is a function of ``panel.values[:t+1, j]``. The Kalman/HMM filters are forward
recursions, so the state at origin ``t`` is by construction conditioned on
``y_{0..t}``. This is unit-tested by perturbing future values and asserting identical
predictions (``tests/test_reliability_baselines.py``).

Tuning parity
-------------
Each model exposes exactly THREE tunable hyperparameters (:data:`SEARCH_SPACE`), the
same count as ARIMA's (p, d, q), and is tuned with the same Optuna protocol as every
other baseline: TPE sampler seeded 0, 50-trial budget, objective = mean per-cell NCRPS
on the split's validation region, on the same tuning-device subset for e1b. The driver
lives in this module (``python -m ltn.baselines.reliability tune ...``) so that
``ltn/exp/tune.py`` is untouched; it reuses ``ltn.exp.tune``'s TUNE_DEVICES/schedule
and writes the same ``runs/tuning/<split>_<model>.json`` format that ``--tuned`` reads.
Note that the *per-series* parameters of all five models (state variances, kernel
hyperparameters, drift/diffusion, regime parameters) are fitted by maximum likelihood
per series rather than searched, which is the standard practice for these models and
is strictly more adaptive than a global grid.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ltn.baselines.common import sigma_floor
from ltn.data.series import Panel

__all__ = [
    "KalmanLocalTrend",
    "GPDegradation",
    "WienerDegradation",
    "GammaDegradation",
    "SwitchingStateSpace",
    "monotonicity_report",
    "cusum_scores",
    "page_hinkley_scores",
    "binseg_scores",
    "SEARCH_SPACE",
    "RELIABILITY_MODELS",
]

_MIN_TRAIN = 8  # fewer usable train points than this -> per-series fallback


# --------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------
def _train_series(panel: Panel, train_mask: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    """(times, values) of the observed train points of series j, times = grid indices."""
    idx = np.nonzero(train_mask)[0]
    v = panel.values[idx, j]
    ok = ~np.isnan(v)
    return idx[ok].astype(np.float64), v[ok]


def _series_scale(panel: Panel, train_mask: np.ndarray, j: int) -> float:
    v = panel.values[train_mask, j]
    v = v[~np.isnan(v)]
    s = float(np.std(v)) if len(v) > 1 else 0.0
    return s if s > 0 else 1.0


class _PerSeriesForecaster:
    """Shared plumbing: per-series fit + a persistence-style fallback."""

    name = "reliability-base"

    def fit(self, panel: Panel, train_mask: np.ndarray, val_mask: np.ndarray,
            cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self._configure(cfg)
        self._train_std = {j: _series_scale(panel, train_mask, j) for j in range(panel.n_series)}
        self._n_series = panel.n_series
        self._fitted: dict[int, object] = {}
        for j in range(panel.n_series):
            t, v = _train_series(panel, train_mask, j)
            if len(v) < _MIN_TRAIN:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f = self._fit_series(j, t, v)
            except Exception:
                f = None  # falls back to persistence-style prediction for this series
            if f is not None:
                self._fitted[j] = f

    # -- subclass hooks ----------------------------------------------------------
    def _configure(self, cfg: dict) -> None:
        self.sigma_scale = float(cfg.get("sigma_scale", 1.0))

    def _fit_series(self, j: int, t: np.ndarray, v: np.ndarray):  # pragma: no cover - abstract
        raise NotImplementedError

    def _predict_series(self, j: int, fitted, hist: np.ndarray, t: int,
                        horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(mu, sigma) for each horizon in `horizons` at origin index `t`.

        `hist` is panel.values[:t+1, j] -- the causal history, NaNs included.
        """
        raise NotImplementedError  # pragma: no cover - abstract

    # -- driver ------------------------------------------------------------------
    def predict(self, pairs: pd.DataFrame, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
        mu = np.empty(len(pairs), dtype=np.float64)
        sig = np.empty(len(pairs), dtype=np.float64)
        t_arr = pairs["t_idx"].to_numpy(dtype=int)
        j_arr = pairs["s_idx"].to_numpy(dtype=int)
        h_arr = pairs["horizon"].to_numpy(dtype=int)
        x_last = pairs["x_last"].to_numpy(dtype=float)
        order = np.lexsort((t_arr, j_arr))
        V = panel.values
        cur_j = -1
        state = None
        for pos in order:
            j, t, h = int(j_arr[pos]), int(t_arr[pos]), int(h_arr[pos])
            std = self._train_std[j]
            fitted = self._fitted.get(j)
            if fitted is None:
                mu[pos] = x_last[pos]
                sig[pos] = max(std * np.sqrt(h), sigma_floor(std))
                continue
            if j != cur_j:
                cur_j, state = j, self._new_series_state(j, fitted, V[:, j])
            m, s = self._predict_one(j, fitted, state, V[: t + 1, j], t, h)
            if not np.isfinite(m):
                m = x_last[pos]
            s = s * self.sigma_scale
            if not np.isfinite(s) or s <= 0:
                s = std * np.sqrt(h)
            mu[pos] = m
            sig[pos] = max(s, sigma_floor(std))
        return mu, sig

    def _new_series_state(self, j: int, fitted, column: np.ndarray):
        """Optional per-series precomputation reused across that series' origins.

        `column` is the FULL panel column. Implementations must only ever use
        `column[:t+1]` when answering for origin t; the forward filters below
        satisfy this by construction (their value at index t depends on y_{0..t}).
        """
        return None

    def _predict_one(self, j, fitted, state, hist: np.ndarray, t: int, h: int) -> tuple[float, float]:
        raise NotImplementedError  # pragma: no cover - abstract


# --------------------------------------------------------------------------------
# 1. Local linear-trend Kalman filter (linear-Gaussian state space)
# --------------------------------------------------------------------------------
@dataclass
class _KalmanPars:
    s2_obs: float
    q_level: float
    q_slope: float
    trend: bool


def _kf_loglik(y: np.ndarray, s2_obs: float, q_level: float, q_slope: float,
               trend: bool, kappa: float = 1e6) -> float:
    """Gaussian log-likelihood of a local-level / local-linear-trend model via the
    prediction-error decomposition. `y` may contain NaN (missing update skipped)."""
    n = len(y)
    y0 = y[np.isfinite(y)]
    if len(y0) < 3:
        return -np.inf
    a0, a1 = float(y0[0]), 0.0
    scale = float(np.var(y0)) or 1.0
    p00, p01, p11 = kappa * scale, 0.0, (kappa * scale if trend else 0.0)
    ll = 0.0
    n_used = 0
    for i in range(n):
        if i > 0:  # predict
            if trend:
                a0 = a0 + a1
                p00 = p00 + 2.0 * p01 + p11 + q_level
                p01 = p01 + p11
                p11 = p11 + q_slope
            else:
                p00 = p00 + q_level
        yi = y[i]
        if np.isfinite(yi):
            f = p00 + s2_obs
            if f <= 0:
                return -np.inf
            e = yi - a0
            ll += -0.5 * (np.log(2.0 * np.pi * f) + e * e / f)
            n_used += 1
            k0 = p00 / f
            k1 = p01 / f
            a0 = a0 + k0 * e
            a1 = a1 + k1 * e
            new00 = p00 - k0 * p00
            new01 = p01 - k0 * p01
            new11 = p11 - k1 * p01
            p00, p01, p11 = new00, new01, new11
    if n_used < 3 or not np.isfinite(ll):
        return -np.inf
    return ll


def _kf_filter_path(y: np.ndarray, pars: _KalmanPars, kappa: float = 1e6):
    """Forward filter over the whole column; returns filtered (a0,a1,P00,P01,P11)[t].

    Entry t is conditioned on y_{0..t} only -- this is what makes per-origin
    prediction lookahead-free without refiltering per origin.
    """
    n = len(y)
    A0 = np.empty(n); A1 = np.empty(n)
    P00a = np.empty(n); P01a = np.empty(n); P11a = np.empty(n)
    fin = np.isfinite(y)
    first = int(np.argmax(fin)) if fin.any() else 0
    scale = float(np.nanvar(y[fin])) if fin.any() else 1.0
    scale = scale or 1.0
    a0 = float(y[first]) if fin.any() else 0.0
    a1 = 0.0
    p00, p01, p11 = kappa * scale, 0.0, (kappa * scale if pars.trend else 0.0)
    for i in range(n):
        if i > 0:
            if pars.trend:
                a0 = a0 + a1
                p00 = p00 + 2.0 * p01 + p11 + pars.q_level
                p01 = p01 + p11
                p11 = p11 + pars.q_slope
            else:
                p00 = p00 + pars.q_level
        yi = y[i]
        if np.isfinite(yi):
            f = p00 + pars.s2_obs
            e = yi - a0
            k0 = p00 / f
            k1 = p01 / f
            a0 = a0 + k0 * e
            a1 = a1 + k1 * e
            new00 = p00 - k0 * p00
            new01 = p01 - k0 * p01
            new11 = p11 - k1 * p01
            p00, p01, p11 = new00, new01, new11
        A0[i], A1[i] = a0, a1
        P00a[i], P01a[i], P11a[i] = p00, p01, p11
    return A0, A1, P00a, P01a, P11a


class KalmanLocalTrend(_PerSeriesForecaster):
    """Local linear-trend (or local-level) linear-Gaussian state-space model per series.

    State x_t = [level_t, slope_t]:
        level_{t+1} = level_t + slope_t + w^L_t,   w^L ~ N(0, q_level)
        slope_{t+1} = slope_t + w^S_t,             w^S ~ N(0, q_slope)
        y_t         = level_t + v_t,               v   ~ N(0, s2_obs)
    (`trend=0` pins slope_t == 0, giving the local-level / random-walk-plus-noise model.)

    The three variances are estimated per series by maximising the exact Gaussian
    likelihood on the TRAIN fold. The h-step predictive law is closed form:
        mu    = level_t + h * slope_t
        sigma^2 = Z T^h P_t (T^h)' Z' + sum_{i<h} Z T^i Q (T^i)' Z' + s2_obs.

    Tunables (3, matching ARIMA's p/d/q): trend, q_slope_cap, sigma_scale.
    """

    name = "kalman"

    def _configure(self, cfg: dict) -> None:
        super()._configure(cfg)
        self.trend = bool(int(cfg.get("trend", 1)))
        # cap on slope-innovation variance as a fraction of the series variance;
        # without it the MLE occasionally runs away on short, near-flat series.
        self.q_slope_cap = float(cfg.get("q_slope_cap", 1e-2))

    def _fit_series(self, j: int, t: np.ndarray, v: np.ndarray):
        from scipy.optimize import minimize

        var = float(np.var(v)) or 1.0
        cap = self.q_slope_cap * var
        # place the train points back on the integer grid so gaps are real gaps
        tt = t.astype(int)
        y = np.full(int(tt[-1] - tt[0]) + 1, np.nan)
        y[tt - tt[0]] = v

        def nll(theta):
            s2 = np.exp(theta[0]) * var
            ql = np.exp(theta[1]) * var
            qs = min(np.exp(theta[2]) * var, cap) if self.trend else 0.0
            ll = _kf_loglik(y, s2, ql, qs, self.trend)
            return -ll if np.isfinite(ll) else 1e12

        best = None
        for th0 in ([np.log(0.5), np.log(0.05), np.log(1e-3)],
                    [np.log(0.05), np.log(0.5), np.log(1e-4)]):
            r = minimize(nll, np.asarray(th0), method="Nelder-Mead",
                         options={"maxfev": 200, "xatol": 1e-3, "fatol": 1e-3})
            if best is None or r.fun < best.fun:
                best = r
        if best is None or not np.isfinite(best.fun) or best.fun >= 1e11:
            return None
        s2 = float(np.exp(best.x[0]) * var)
        ql = float(np.exp(best.x[1]) * var)
        qs = float(min(np.exp(best.x[2]) * var, cap)) if self.trend else 0.0
        return _KalmanPars(s2_obs=s2, q_level=ql, q_slope=qs, trend=self.trend)

    def _new_series_state(self, j, fitted, column):
        return _kf_filter_path(column, fitted)

    def _predict_one(self, j, fitted, state, hist, t, h):
        A0, A1, P00, P01, P11 = state
        p = fitted
        a0, a1 = A0[t], A1[t]
        p00, p01, p11 = P00[t], P01[t], P11[t]
        if p.trend:
            # exact h-step propagation with T^h = [[1,h],[0,1]]
            mu = a0 + h * a1
            var = p00 + 2.0 * h * p01 + h * h * p11
            # accumulated process noise: sum_{i=0}^{h-1} [q_level + i^2 q_slope]
            k = np.arange(h, dtype=np.float64)
            var += h * p.q_level + float(np.sum(k * k)) * p.q_slope
        else:
            mu = a0
            var = p00 + h * p.q_level
        var += p.s2_obs
        return float(mu), float(np.sqrt(max(var, 0.0)))


# --------------------------------------------------------------------------------
# 2. Gaussian-process degradation model
# --------------------------------------------------------------------------------
@dataclass
class _GPPars:
    amp: float          # constant (signal) variance
    ls: float           # RBF length scale, in grid steps
    noise: float        # white-noise variance
    dot: float          # DotProduct sigma_0^2 coefficient (0 = pure RBF)
    mean: float
    scale: float


class GPDegradation(_PerSeriesForecaster):
    """Gaussian-process degradation model, one GP per series.

    Kernel: C * RBF(length_scale) [+ DotProduct] + WhiteKernel, fitted by
    marginal-likelihood maximisation with scikit-learn's ``GaussianProcessRegressor``
    on the standardised TRAIN fold. At each forecast origin the posterior is
    recomputed in closed form on a causal sliding window of the most recent
    ``window`` observed points at or before the origin, giving a native predictive
    mean and variance (observation noise included).

    Because the time axis is the integer snapshot grid, the window Gram matrix
    depends only on the *relative* offsets of the retained points, so its Cholesky
    factor is cached and reused across origins with the same gap pattern.

    Tunables (3): window, kernel, sigma_scale.
    """

    name = "gp"

    def _configure(self, cfg: dict) -> None:
        super()._configure(cfg)
        self.window = int(cfg.get("window", 48))
        self.kernel = str(cfg.get("kernel", "rbf"))  # "rbf" | "rbf_dot"

    def _fit_series(self, j: int, t: np.ndarray, v: np.ndarray):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel, DotProduct, RBF, WhiteKernel,
        )

        mean = float(np.mean(v))
        scale = float(np.std(v)) or 1.0
        yz = (v - mean) / scale
        # last `window` train points keep the hyperparameter fit affordable and
        # match the window the posterior will actually use
        if len(t) > self.window:
            t, yz = t[-self.window:], yz[-self.window:]
        X = (t - t[-1]).reshape(-1, 1)  # offsets relative to the last train point
        k = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(5.0, (0.5, 1e3))
        if self.kernel == "rbf_dot":
            k = k + DotProduct(1.0, (1e-3, 1e3))
        k = k + WhiteKernel(0.1, (1e-6, 1e2))
        gp = GaussianProcessRegressor(kernel=k, normalize_y=False, optimizer="fmin_l_bfgs_b",
                                      n_restarts_optimizer=0, alpha=0.0)
        gp.fit(X, yz)
        p = gp.kernel_.get_params()
        amp = float(p["k1__k1__k1__constant_value"]) if self.kernel == "rbf_dot" \
            else float(p["k1__k1__constant_value"])
        ls = float(p["k1__k1__k2__length_scale"]) if self.kernel == "rbf_dot" \
            else float(p["k1__k2__length_scale"])
        dot = float(p["k1__k2__sigma_0"] ** 2) if self.kernel == "rbf_dot" else 0.0
        noise = float(p["k2__noise_level"])
        return _GPPars(amp=amp, ls=max(ls, 1e-3), noise=max(noise, 1e-8), dot=dot,
                       mean=mean, scale=scale)

    def _kern(self, p: _GPPars, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a.reshape(-1, 1) - b.reshape(1, -1)
        k = p.amp * np.exp(-0.5 * (d / p.ls) ** 2)
        if p.dot:
            k = k + p.dot * (a.reshape(-1, 1) * b.reshape(1, -1))
        return k

    def _new_series_state(self, j, fitted, column):
        return {}  # cache keyed by (offsets pattern, h)

    def _predict_one(self, j, fitted, state, hist, t, h):
        p: _GPPars = fitted
        obs = np.nonzero(np.isfinite(hist))[0]
        if len(obs) == 0:
            return np.nan, np.nan
        obs = obs[-self.window:]
        off = (obs - t).astype(np.float64)  # <= 0, relative to the origin
        key = (off.tobytes(), h)
        cached = state.get(key)
        if cached is None:
            K = self._kern(p, off, off) + p.noise * np.eye(len(off))
            ks = self._kern(p, np.array([float(h)]), off).ravel()
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K = K + 1e-6 * np.eye(len(off))
                L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, ks))  # K^{-1} k_*
            kss = float(self._kern(p, np.array([float(h)]), np.array([float(h)]))[0, 0])
            var = kss + p.noise - float(ks @ alpha)
            cached = (alpha, max(var, 1e-12))
            if len(state) < 200_000:
                state[key] = cached
        alpha, var = cached
        yz = (hist[obs] - p.mean) / p.scale
        mu = float(alpha @ yz) * p.scale + p.mean
        sd = float(np.sqrt(var)) * p.scale
        return mu, sd


# --------------------------------------------------------------------------------
# 3. Wiener-process degradation
# --------------------------------------------------------------------------------
@dataclass
class _WienerPars:
    drift: float
    diffusion: float   # sigma_B, per unit time step
    log: bool
    mean_dt: float


class WienerDegradation(_PerSeriesForecaster):
    """Wiener (Brownian-motion-with-drift) degradation model, MLE per series.

    X(t) = X(0) + mu t + sigma_B B(t). With increments observed on the snapshot grid,
    the MLE is mu_hat = sum(dX)/sum(dt) and sigma_B^2 = mean over increments of
    (dX - mu dt)^2 / dt. The h-step predictive law from origin t is exactly
    N(x_t + mu h, sigma_B^2 h).

    ``log_scale=1`` fits the process to log X, which is the standard device for
    strictly positive degradation observables (T1/T2, error rates) because it keeps
    the forecast positive. The predictive law is then lognormal; since the study's
    NCRPS is the *Gaussian* CRPS, the lognormal law is moment-matched to
    (mean, sd) in the original units --
        mean = exp(m + s^2/2),  sd = sqrt(exp(s^2) - 1) * exp(m + s^2/2)
    -- and this approximation is stated explicitly rather than hidden.

    ``drift_window`` > 0 re-estimates the drift causally from the last k increments
    at or before the origin (diffusion stays at its train MLE); 0 uses the train MLE
    throughout.

    Tunables (3): log_scale, drift_window, sigma_scale.
    """

    name = "wiener"

    def _configure(self, cfg: dict) -> None:
        super()._configure(cfg)
        self.log = bool(int(cfg.get("log_scale", 1)))
        self.drift_window = int(cfg.get("drift_window", 0))

    @staticmethod
    def _mle(t: np.ndarray, x: np.ndarray) -> tuple[float, float]:
        dt = np.diff(t)
        dx = np.diff(x)
        ok = dt > 0
        dt, dx = dt[ok], dx[ok]
        if len(dx) < 2:
            return 0.0, 1.0
        drift = float(np.sum(dx) / np.sum(dt))
        resid = (dx - drift * dt) ** 2 / dt
        diff = float(np.sqrt(max(np.mean(resid), 1e-24)))
        return drift, diff

    def _fit_series(self, j: int, t: np.ndarray, v: np.ndarray):
        use_log = self.log and bool(np.all(v > 0))
        x = np.log(v) if use_log else v
        drift, diff = self._mle(t, x)
        if not (np.isfinite(drift) and np.isfinite(diff)):
            return None
        return _WienerPars(drift=drift, diffusion=diff, log=use_log,
                           mean_dt=float(np.mean(np.diff(t))) if len(t) > 1 else 1.0)

    def _predict_one(self, j, fitted, state, hist, t, h):
        p: _WienerPars = fitted
        obs = np.nonzero(np.isfinite(hist))[0]
        if len(obs) == 0:
            return np.nan, np.nan
        xt = hist[obs[-1]]
        if p.log:
            if xt <= 0:
                return np.nan, np.nan
            xt = np.log(xt)
        drift = p.drift
        if self.drift_window > 0 and len(obs) >= 3:
            w = obs[-(self.drift_window + 1):]
            xs = hist[w]
            if p.log:
                if np.any(xs <= 0):
                    xs = None
                else:
                    xs = np.log(xs)
            if xs is not None:
                d, _ = self._mle(w.astype(np.float64), xs)
                if np.isfinite(d):
                    drift = d
        # gap between the last observation and the origin costs extra diffusion
        lag = float(t - obs[-1])
        m = xt + drift * (h + lag)
        s2 = (p.diffusion ** 2) * (h + lag)
        if not p.log:
            return float(m), float(np.sqrt(max(s2, 0.0)))
        # moment-matched Gaussian for the lognormal predictive law
        s2 = min(s2, 50.0)  # exp overflow guard on pathological series
        mean = float(np.exp(m + 0.5 * s2))
        sd = float(np.sqrt(max(np.exp(s2) - 1.0, 0.0)) * mean)
        return mean, sd


# --------------------------------------------------------------------------------
# 4. Gamma-process degradation (admissible only for monotone degradation)
# --------------------------------------------------------------------------------
def monotonicity_report(panel: Panel, train_mask: np.ndarray,
                        keys: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-observable monotonicity diagnostics for the gamma-process assumption.

    A gamma process has non-negative, independent increments: the degradation path
    is non-decreasing by construction. This measures, per observable, the fraction
    of grid increments that move in the degrading direction, the mean fraction of
    the path range that is given back by reversals, and the number of series that
    are exactly monotone. The degrading direction is 'up' for error rates and
    'down' for coherence times, so both are folded onto a common 'damage' axis.
    """
    from ltn.eval.reliability import direction_for

    keys = panel.keys if keys is None else keys
    rows = []
    for obs, grp in keys.groupby("observable", observed=True):
        direction = direction_for(str(obs))  # 'upper' => larger is worse
        sign = 1.0 if direction == "upper" else -1.0
        fracs, reversals, exact, n_inc = [], [], 0, 0
        for j in grp.index:
            v = panel.values[train_mask, int(j)]
            v = v[~np.isnan(v)]
            if len(v) < 3:
                continue
            d = sign * np.diff(v)  # damage increments
            fracs.append(float(np.mean(d >= 0)))
            up = float(np.sum(d[d > 0]))
            down = float(-np.sum(d[d < 0]))
            reversals.append(down / up if up > 0 else np.nan)
            exact += int(np.all(d >= 0))
            n_inc += len(d)
        rows.append({
            "observable": str(obs),
            "damage_direction": "increase" if sign > 0 else "decrease",
            "n_series": int(len(fracs)),
            "n_increments": int(n_inc),
            "frac_nonneg_increments_mean": float(np.mean(fracs)) if fracs else np.nan,
            "frac_nonneg_increments_min": float(np.min(fracs)) if fracs else np.nan,
            "frac_nonneg_increments_max": float(np.max(fracs)) if fracs else np.nan,
            "reversal_ratio_median": float(np.nanmedian(reversals)) if reversals else np.nan,
            "n_series_exactly_monotone": int(exact),
        })
    return pd.DataFrame(rows)


@dataclass
class _GammaPars:
    shape_rate: float   # alpha per unit time
    scale: float        # beta
    sign: float
    base: float


class GammaDegradation(_PerSeriesForecaster):
    """Gamma-process degradation: increments ~ Gamma(alpha*dt, beta), non-negative.

    Fitted by moment matching on the damage increments (alpha = mean^2/var per unit
    time, beta = var/mean), which is the standard estimator for the stationary gamma
    process. The h-step predictive law is Gamma(alpha*h, beta) added to the current
    damage level, moment-matched to (mu, sigma) for the study's Gaussian NCRPS.

    IMPORTANT: this model is only admissible if the observable really is monotone
    degradation. On telemetry that recalibrates (which resets the observable), the
    monotone assumption is violated and the negative increments simply have to be
    discarded, which biases the drift upward. :func:`monotonicity_report` is the
    check; the paper reports its result and states whether the model was admitted.

    Tunables (3): min_frac_nonneg (admissibility gate), drift_window, sigma_scale.
    """

    name = "gamma"

    def _configure(self, cfg: dict) -> None:
        super()._configure(cfg)
        self.min_frac_nonneg = float(cfg.get("min_frac_nonneg", 0.9))
        self.drift_window = int(cfg.get("drift_window", 0))
        self._directions: dict[int, float] = {}

    def fit(self, panel, train_mask, val_mask, cfg=None):
        from ltn.eval.reliability import direction_for
        obs_of = panel.keys["observable"].to_numpy()
        self._sign_of = {j: (1.0 if direction_for(str(obs_of[j])) == "upper" else -1.0)
                         for j in range(panel.n_series)}
        super().fit(panel, train_mask, val_mask, cfg)

    def _fit_series(self, j: int, t: np.ndarray, v: np.ndarray):
        sign = self._sign_of[j]
        d = sign * np.diff(v)
        if len(d) < 3:
            return None
        if float(np.mean(d >= 0)) < self.min_frac_nonneg:
            return None  # not monotone: model is inadmissible for this series
        dpos = d[d > 0]
        if len(dpos) < 3:
            return None
        m, vv = float(np.mean(dpos)), float(np.var(dpos))
        if not (m > 0 and vv > 0):
            return None
        return _GammaPars(shape_rate=m * m / vv, scale=vv / m, sign=sign, base=0.0)

    def _predict_one(self, j, fitted, state, hist, t, h):
        p: _GammaPars = fitted
        obs = np.nonzero(np.isfinite(hist))[0]
        if len(obs) == 0:
            return np.nan, np.nan
        xt = float(hist[obs[-1]])
        lag = float(t - obs[-1])
        dt = h + lag
        dmg_mean = p.shape_rate * dt * p.scale
        dmg_var = p.shape_rate * dt * p.scale ** 2
        return xt + p.sign * dmg_mean, float(np.sqrt(max(dmg_var, 0.0)))


# --------------------------------------------------------------------------------
# 5. Markov-switching (hidden-Markov) state-space model
# --------------------------------------------------------------------------------
@dataclass
class _HMMPars:
    A: np.ndarray       # (K, K) transition matrix
    m: np.ndarray       # (K,) increment means
    v: np.ndarray       # (K,) increment variances
    pi: np.ndarray      # (K,) initial distribution


def _gauss_pdf(x: np.ndarray, m: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (x[:, None] - m[None, :]) ** 2 / v[None, :]) / np.sqrt(2 * np.pi * v[None, :])


def _baum_welch(d: np.ndarray, K: int, var_floor: float, n_iter: int = 30,
                seed: int = 0) -> _HMMPars:
    """Baum-Welch for a K-regime Gaussian HMM on the increment series `d`."""
    n = len(d)
    rng = np.random.default_rng(seed)
    qs = np.quantile(d, np.linspace(0.15, 0.85, K))
    m = qs + 1e-9 * rng.standard_normal(K)
    v = np.full(K, max(float(np.var(d)), var_floor))
    A = np.full((K, K), 0.1 / max(K - 1, 1))
    np.fill_diagonal(A, 0.9)
    pi = np.full(K, 1.0 / K)
    for _ in range(n_iter):
        B = _gauss_pdf(d, m, v)                       # (n, K)
        B = np.maximum(B, 1e-300)
        alpha = np.empty((n, K)); c = np.empty(n)
        alpha[0] = pi * B[0]
        c[0] = alpha[0].sum() or 1e-300
        alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * B[t]
            c[t] = alpha[t].sum() or 1e-300
            alpha[t] /= c[t]
        beta = np.empty((n, K)); beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        g = alpha * beta
        g /= np.maximum(g.sum(axis=1, keepdims=True), 1e-300)
        xi = np.zeros((K, K))
        for t in range(n - 1):
            x = (alpha[t][:, None] * A) * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
            xi += x
        pi = g[0] / max(g[0].sum(), 1e-300)
        A = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
        w = np.maximum(g.sum(axis=0), 1e-300)
        m = (g * d[:, None]).sum(axis=0) / w
        v = np.maximum(((d[:, None] - m[None, :]) ** 2 * g).sum(axis=0) / w, var_floor)
    return _HMMPars(A=A, m=m, v=v, pi=pi)


class SwitchingStateSpace(_PerSeriesForecaster):
    """Markov-switching state-space model (2-3 regimes) on the increment process.

    The observable is modelled as x_t = x_{t-1} + eps_t with eps_t ~ N(m_{S_t}, v_{S_t})
    and S_t a K-state Markov chain -- a discrete-regime (switching drift/volatility)
    degradation model, the classical way to represent a device that alternates between
    a quiet regime and a drifting or noisy one. Parameters are fitted per series by
    Baum-Welch on the TRAIN increments; the regime posterior at each origin comes from
    the forward pass, so it is causal.

    The h-step predictive law is the exact first two moments of the cumulative
    increment over the regime chain, obtained from the recursion
        M0_k(s+1) = sum_j A_jk M0_j(s)
        M1_k(s+1) = sum_j A_jk (M1_j(s) + m_k M0_j(s))
        M2_k(s+1) = sum_j A_jk (M2_j(s) + 2 m_k M1_j(s) + (m_k^2 + v_k) M0_j(s))
    moment-matched to a Gaussian (the mixture is not Gaussian; this is stated).

    Tunables (3): n_states, var_floor_frac, sigma_scale.
    """

    name = "hmm"

    def _configure(self, cfg: dict) -> None:
        super()._configure(cfg)
        self.n_states = int(cfg.get("n_states", 2))
        self.var_floor_frac = float(cfg.get("var_floor_frac", 1e-3))

    def _fit_series(self, j: int, t: np.ndarray, v: np.ndarray):
        d = np.diff(v)
        if len(d) < max(10, 3 * self.n_states):
            return None
        vf = max(self.var_floor_frac * float(np.var(v)), 1e-24)
        if float(np.var(d)) <= 0:
            return None
        return _baum_welch(d, self.n_states, vf, seed=0)

    def _new_series_state(self, j, fitted, column):
        """Forward regime posterior at every t (conditioned on increments up to t)."""
        p: _HMMPars = fitted
        n = len(column)
        K = len(p.m)
        post = np.empty((n, K))
        obs = np.isfinite(column)
        cur = p.pi.copy()
        last = -1
        for i in range(n):
            if obs[i]:
                if last >= 0:
                    d = column[i] - column[last]
                    steps = i - last
                    Ak = np.linalg.matrix_power(p.A, steps)
                    prior = cur @ Ak
                    lik = np.exp(-0.5 * (d - p.m * steps) ** 2 / (p.v * steps)) \
                        / np.sqrt(2 * np.pi * p.v * steps)
                    post_t = prior * np.maximum(lik, 1e-300)
                    s = post_t.sum()
                    cur = post_t / s if s > 0 else prior
                last = i
            post[i] = cur
        return post

    def _predict_one(self, j, fitted, state, hist, t, h):
        p: _HMMPars = fitted
        obs = np.nonzero(np.isfinite(hist))[0]
        if len(obs) == 0:
            return np.nan, np.nan
        xt = float(hist[obs[-1]])
        steps = int(h + (t - obs[-1]))
        p0 = state[t]
        K = len(p.m)
        M0 = p0.astype(np.float64).copy()
        M1 = np.zeros(K); M2 = np.zeros(K)
        A, m, v = p.A, p.m, p.v
        for _ in range(steps):
            n0 = A.T @ M0
            n1 = A.T @ M1 + m * n0
            n2 = A.T @ M2 + 2.0 * m * (A.T @ M1) + (m * m + v) * n0
            M0, M1, M2 = n0, n1, n2
        mean = float(M1.sum())
        var = float(M2.sum() - mean * mean)
        return xt + mean, float(np.sqrt(max(var, 1e-24)))


# --------------------------------------------------------------------------------
# 6. Change-point / drift detectors (NOT forecasters -- scored on E3's AUROC endpoint)
# --------------------------------------------------------------------------------
def cusum_scores(x: np.ndarray, k: float = 0.5) -> np.ndarray:
    """Causal two-sided CUSUM statistic max(S+, S-) at each t.

    x is standardised online against the running mean/sd of the history strictly
    before t, so the statistic at t uses only x_{0..t} -- it is a legitimate
    anticipation score. k is the reference value (allowance) in sd units.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.zeros(n)
    sp = sm = 0.0
    s1 = s2 = 0.0
    for i in range(n):
        xi = x[i]
        if i >= 2:
            mu = s1 / i
            var = max(s2 / i - mu * mu, 1e-24)
            z = (xi - mu) / np.sqrt(var)
        else:
            z = 0.0
        sp = max(0.0, sp + z - k)
        sm = max(0.0, sm - z - k)
        out[i] = max(sp, sm)
        s1 += xi
        s2 += xi * xi
    return out


def page_hinkley_scores(x: np.ndarray, delta: float = 0.005) -> np.ndarray:
    """Causal Page-Hinkley drift statistic (cumulative deviation minus its running min)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.zeros(n)
    s = 0.0
    smin = 0.0
    run = 0.0
    for i in range(n):
        run += x[i]
        mu = run / (i + 1)
        s += x[i] - mu - delta
        smin = min(smin, s)
        out[i] = s - smin
    return out


def binseg_scores(x: np.ndarray, n_bkps: int = 3) -> np.ndarray:
    """OFFLINE binary segmentation: distance to the nearest detected change point.

    NOT causal -- it sees the whole series. Reported only as an optimistic upper
    bound on what a change-point detector could achieve on this signal; it must
    never be compared to the causal forecasters as an equal.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return np.zeros(n)

    def cost(a: int, b: int) -> float:
        seg = x[a:b]
        return float(np.sum((seg - seg.mean()) ** 2)) if len(seg) else 0.0

    bkps: list[int] = []
    segs = [(0, n)]
    for _ in range(n_bkps):
        best = None
        for (a, b) in segs:
            if b - a < 4:
                continue
            base = cost(a, b)
            for c in range(a + 2, b - 1):
                gain = base - cost(a, c) - cost(c, b)
                if best is None or gain > best[0]:
                    best = (gain, a, b, c)
        if best is None or best[0] <= 0:
            break
        _, a, b, c = best
        bkps.append(c)
        segs = [s for s in segs if s != (a, b)] + [(a, c), (c, b)]
    if not bkps:
        return np.zeros(n)
    idx = np.arange(n)
    d = np.min(np.abs(idx[:, None] - np.asarray(bkps)[None, :]), axis=1)
    return -d.astype(np.float64)  # closer to a change point = higher score


RELIABILITY_MODELS = {
    "kalman": KalmanLocalTrend,
    "gp": GPDegradation,
    "wiener": WienerDegradation,
    "gamma": GammaDegradation,
    "hmm": SwitchingStateSpace,
}

#: Optuna search spaces -- three tunables each, matching ARIMA's (p, d, q) budget.
SEARCH_SPACE = {
    "kalman": lambda tr: {
        "trend": tr.suggest_categorical("trend", [0, 1]),
        "q_slope_cap": tr.suggest_float("q_slope_cap", 1e-6, 1e-1, log=True),
        "sigma_scale": tr.suggest_float("sigma_scale", 0.5, 2.0),
    },
    "gp": lambda tr: {
        "window": tr.suggest_int("window", 12, 96, log=True),
        "kernel": tr.suggest_categorical("kernel", ["rbf", "rbf_dot"]),
        "sigma_scale": tr.suggest_float("sigma_scale", 0.5, 2.0),
    },
    "wiener": lambda tr: {
        "log_scale": tr.suggest_categorical("log_scale", [0, 1]),
        "drift_window": tr.suggest_categorical("drift_window", [0, 4, 8, 16, 32]),
        "sigma_scale": tr.suggest_float("sigma_scale", 0.5, 2.0),
    },
    "gamma": lambda tr: {
        "min_frac_nonneg": tr.suggest_float("min_frac_nonneg", 0.5, 0.95),
        "drift_window": tr.suggest_categorical("drift_window", [0, 8, 32]),
        "sigma_scale": tr.suggest_float("sigma_scale", 0.5, 2.0),
    },
    "hmm": lambda tr: {
        "n_states": tr.suggest_categorical("n_states", [2, 3]),
        "var_floor_frac": tr.suggest_float("var_floor_frac", 1e-5, 1e-1, log=True),
        "sigma_scale": tr.suggest_float("sigma_scale", 0.5, 2.0),
    },
}


# --------------------------------------------------------------------------------
# tuning driver (same protocol as ltn/exp/tune.py, which this module does not touch)
# --------------------------------------------------------------------------------
def tune(split_name: str, model_name: str, trials: int = 50) -> dict:
    """Optuna study identical in protocol to ltn.exp.tune: TPE(seed=0), mean per-cell
    NCRPS on the split's validation region, e1b tuned on the frozen TUNE_DEVICES
    subset. Writes runs/tuning/<split>_<model>.json so --tuned picks it up."""
    import json
    from pathlib import Path

    import optuna

    from ltn.data.series import forecast_pairs, subset_panel
    from ltn.eval.splits import load_split
    from ltn.exp.run import build_data, evaluate, model_cfg, parse_norm_stats
    from ltn.exp.tune import TUNE_DEVICES

    split = load_split(split_name)
    uts, panel, (tr, va, te) = build_data(split)
    tune_devs = TUNE_DEVICES.get(split_name)
    norm = parse_norm_stats(split)
    if tune_devs is not None:
        panel = subset_panel(panel, tune_devs)
    val_pairs = forecast_pairs(panel, split["horizons_steps"], va)

    def objective(trial: "optuna.Trial") -> float:
        cfg = model_cfg(split, seed=0, extra=SEARCH_SPACE[model_name](trial))
        cfg["uts"] = uts
        if tune_devs is not None:
            cfg["devices"] = tune_devs
        m = RELIABILITY_MODELS[model_name]()
        m.fit(panel, tr, va, cfg)
        mu, sigma = m.predict(val_pairs, panel)
        cells, _ = evaluate(val_pairs, panel, mu, sigma, norm)
        val = float(cells["ncrps"].mean())
        return val if np.isfinite(val) else 1e6

    out = Path("runs/tuning")
    out.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{split_name}_{model_name}",
        storage=f"sqlite:///{out}/optuna_{split_name}_{model_name}.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=0),
        direction="minimize",
    )
    n_done = len([t for t in study.trials if t.state.is_finished()])
    remaining = max(0, trials - n_done)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    best = {"value": study.best_value, "params": study.best_params,
            "n_trials": len([t for t in study.trials if t.state.name == "COMPLETE"]),
            "n_trial_records": len(study.trials)}
    (out / f"{split_name}_{model_name}.json").write_text(json.dumps(best, indent=1))
    return best


def main() -> None:  # pragma: no cover - CLI
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["tune", "monotonicity"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()
    if args.cmd == "tune":
        print(json.dumps(tune(args.split, args.model, args.trials)))
    else:
        from ltn.eval.splits import load_split
        from ltn.exp.run import build_data
        split = load_split(args.split)
        _, panel, (tr, _, _) = build_data(split)
        print(monotonicity_report(panel, tr).to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
