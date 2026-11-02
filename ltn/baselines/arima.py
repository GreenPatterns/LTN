"""ARIMA(p,d,q) per series (statsmodels), fit once on train history, then rolled
forward through each forecast origin via ARIMAResults.apply (filter-only, no refit)
so per-origin prediction stays cheap and lookahead-free.

Tunables (Optuna): p, d, q."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from ltn.baselines.common import sigma_floor
from ltn.data.series import Panel


class ARIMAForecaster:
    name = "arima"

    def fit(self, panel: Panel, train_mask: np.ndarray, val_mask: np.ndarray, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.order = (int(cfg.get("p", 1)), int(cfg.get("d", 1)), int(cfg.get("q", 0)))
        self._models: dict[int, object] = {}
        self._train_std: dict[int, float] = {}
        for j in range(panel.n_series):
            v = panel.values[train_mask, j]
            v = v[~np.isnan(v)]
            self._train_std[j] = float(np.std(v)) or 1.0
            if len(v) < max(8, sum(self.order) + 3):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self._models[j] = ARIMA(v, order=self.order).fit()
                except Exception:
                    pass  # falls back to persistence-style prediction for this series

    def predict(self, pairs: pd.DataFrame, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
        mu = np.empty(len(pairs))
        sig = np.empty(len(pairs))
        # group by (series, origin) so each history extension is filtered once
        cache: dict[tuple[int, int], object] = {}
        for i, (t, j, h) in enumerate(zip(pairs["t_idx"], pairs["s_idx"], pairs["horizon"])):
            t, j, h = int(t), int(j), int(h)
            base = self._models.get(j)
            if base is None:
                mu[i] = pairs["x_last"].iloc[i]
                sig[i] = max(self._train_std[j] * np.sqrt(h), sigma_floor(self._train_std[j]))
                continue
            key = (t, j)
            if key not in cache:
                v = panel.values[: t + 1, j]
                v = v[~np.isnan(v)]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cache[key] = base.apply(v)  # re-filter on history up to origin
            res = cache[key]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = res.get_forecast(steps=h)
            mu[i] = float(np.asarray(fc.predicted_mean)[-1])
            se = float(np.asarray(fc.se_mean)[-1])
            sig[i] = max(se, sigma_floor(self._train_std[j]))
        return mu, sig
