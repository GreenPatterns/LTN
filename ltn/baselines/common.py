"""Forecaster interface shared by all E1 baselines and (via an adapter) LTN itself.

Contract: `predict(pairs, panel)` may use panel values only at times <= each pair's
origin t_idx (no lookahead). The no-lookahead property is unit-tested by perturbing
future values and asserting identical predictions.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from ltn.data.series import Panel


class Forecaster(Protocol):
    name: str

    def fit(self, panel: Panel, train_mask: np.ndarray, val_mask: np.ndarray, cfg: dict) -> None: ...

    def predict(self, pairs: pd.DataFrame, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
        """-> (mu, sigma) arrays aligned to pairs rows."""
        ...


def history(panel: Panel, t_idx: int, s_idx: int) -> np.ndarray:
    """Observed values of series s up to and including origin t (never beyond)."""
    v = panel.values[: t_idx + 1, s_idx]
    return v[~np.isnan(v)]


SIGMA_FLOOR_FRAC = 1e-3  # of the series' train std; keeps CRPS/NLL finite


def sigma_floor(train_std: float) -> float:
    return max(train_std * SIGMA_FLOOR_FRAC, 1e-12)
