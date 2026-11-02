"""Persistence: mu = last observed value; sigma = train-history std of the
h-step difference per (series, horizon). No hyperparameters (Optuna-exempt,
declared in prereg)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ltn.baselines.common import sigma_floor
from ltn.data.series import Panel, forecast_pairs


class Persistence:
    name = "persistence"

    def fit(self, panel: Panel, train_mask: np.ndarray, val_mask: np.ndarray, cfg: dict | None = None) -> None:
        self._sigma: dict[tuple[int, int], float] = {}
        self._fallback: dict[int, float] = {}
        horizons = (cfg or {}).get("horizons", [1, 3, 9, 21])
        tr = forecast_pairs(panel, horizons, train_mask)
        for (j, h), g in tr.groupby(["s_idx", "horizon"]):
            d = (g["y"] - g["x_last"]).to_numpy()
            self._sigma[(int(j), int(h))] = float(np.std(d)) if len(d) > 1 else np.nan
        for j in range(panel.n_series):
            v = panel.values[train_mask, j]
            v = v[~np.isnan(v)]
            self._fallback[j] = float(np.std(np.diff(v))) if len(v) > 2 else 1.0

    def predict(self, pairs: pd.DataFrame, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
        mu = pairs["x_last"].to_numpy(dtype=float)
        sig = np.empty(len(pairs))
        for i, (j, h) in enumerate(zip(pairs["s_idx"], pairs["horizon"])):
            s = self._sigma.get((int(j), int(h)), np.nan)
            if not np.isfinite(s) or s <= 0:
                s = self._fallback[int(j)] * np.sqrt(int(h))
            sig[i] = max(s, sigma_floor(self._fallback[int(j)]))
        return mu, sig
