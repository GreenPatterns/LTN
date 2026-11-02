"""Post-hoc calibration on validation data (master plan §2.2 Readout: "Temperature-scale on validation").

Guarantee: `TemperatureScaler` finds the scalar t minimizing the Gaussian NLL
of (mu, t·sigma) — if forecast sigmas are understated by a factor c, fit
recovers t ≈ c. `PlattBinary` is standard Platt scaling (logistic regression
on logit(p)) for the E3 divergence probabilities. Both are fit on validation
only and never touch test targets.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

from ltn.eval.metrics import gaussian_nll

__all__ = ["TemperatureScaler", "PlattBinary"]

_EPS = 1e-12


class TemperatureScaler:
    """Scalar variance calibration: sigma → t·sigma, t = argmin Gaussian NLL on validation."""

    def __init__(self) -> None:
        self.t_: float | None = None

    def fit(self, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> "TemperatureScaler":
        y = np.asarray(y, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        # optimize log t so the bounded search is scale-free and t stays positive
        res = minimize_scalar(
            lambda log_t: gaussian_nll(y, mu, np.exp(log_t) * sigma),
            bounds=(-10.0, 10.0),
            method="bounded",
        )
        self.t_ = float(np.exp(res.x))
        return self

    def transform(self, sigma: np.ndarray) -> np.ndarray:
        if self.t_ is None:
            raise RuntimeError("TemperatureScaler.transform called before fit")
        return self.t_ * np.asarray(sigma, dtype=np.float64)


class PlattBinary:
    """Platt scaling for binary probabilities: p → sigmoid(a·logit(p) + b)."""

    def __init__(self) -> None:
        self.a_: float | None = None
        self.b_: float | None = None
        self._lr: LogisticRegression | None = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
        return np.log(p / (1.0 - p))

    def fit(self, y: np.ndarray, p: np.ndarray) -> "PlattBinary":
        y = np.asarray(y).astype(int)
        x = self._logit(p).reshape(-1, 1)
        # near-unregularized MLE (huge C) = classic Platt fit
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(x, y)
        self.a_ = float(self._lr.coef_[0, 0])
        self.b_ = float(self._lr.intercept_[0])
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.a_ is None or self.b_ is None:
            raise RuntimeError("PlattBinary.transform called before fit")
        z = self.a_ * self._logit(p) + self.b_
        return 1.0 / (1.0 + np.exp(-z))
