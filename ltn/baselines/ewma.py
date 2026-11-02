"""EWMA: exponentially weighted mean forecast with EW variance predictive sigma.

Tunables (Optuna): alpha (smoothing), var_scale (per-horizon variance inflation
exponent q: sigma_h = sigma * h**q)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ltn.baselines.common import history, sigma_floor
from ltn.data.series import Panel


class EWMA:
    name = "ewma"

    def fit(self, panel: Panel, train_mask: np.ndarray, val_mask: np.ndarray, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.alpha = float(cfg.get("alpha", 0.4))
        self.q = float(cfg.get("var_scale", 0.5))
        self._train_std = {
            j: float(np.nanstd(panel.values[train_mask, j])) or 1.0
            for j in range(panel.n_series)
        }

    def _ew(self, v: np.ndarray) -> tuple[float, float]:
        m, s2 = v[0], 0.0
        for x in v[1:]:
            e = x - m
            m += self.alpha * e
            s2 = (1 - self.alpha) * (s2 + self.alpha * e * e)
        return m, np.sqrt(max(s2, 0.0))

    def predict(self, pairs: pd.DataFrame, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
        mu = np.empty(len(pairs))
        sig = np.empty(len(pairs))
        cache: dict[tuple[int, int], tuple[float, float]] = {}
        for i, (t, j, h) in enumerate(zip(pairs["t_idx"], pairs["s_idx"], pairs["horizon"])):
            key = (int(t), int(j))
            if key not in cache:
                cache[key] = self._ew(history(panel, int(t), int(j)))
            m, s = cache[key]
            mu[i] = m
            sig[i] = max(s * (int(h) ** self.q), sigma_floor(self._train_std[int(j)]))
        return mu, sig
