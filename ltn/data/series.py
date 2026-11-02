"""UTS events -> per-entity time series panels for forecasting experiments (E1/E4).

A *series* is one (device, entity_id, observable, channel) value sequence on the
device's snapshot grid. Duplicate observations at one (series, ts) — e.g. DAQEC's two
strategy arms probing the same calibration — are averaged for telemetry channels;
`delivered` outcomes are NOT averaged here (they stay per-run in the lake; E3/E6 read
them directly).

Horizons are expressed in *snapshots* (grid steps). The experiment configs translate
days -> steps per source (DAQEC grid = 3/day, sim default = 3/day).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TELEMETRY_CHANNELS = ("claimed", "probe", "derived")

SERIES_KEY = ["device", "entity_id", "observable", "channel"]


def to_series(
    uts: pd.DataFrame,
    observables: list[str] | None = None,
    channels: list[str] | None = None,
    entity_types: list[str] | None = None,
) -> pd.DataFrame:
    """Long series frame [device, entity_id, observable, channel, ts, value, sigma],
    duplicates averaged, sorted by ts within series."""
    df = uts
    if observables is not None:
        df = df[df["observable"].isin(observables)]
    if channels is not None:
        df = df[df["channel"].isin(channels)]
    if entity_types is not None:
        df = df[df["entity_type"].isin(entity_types)]
    g = (
        df.groupby(SERIES_KEY + ["ts"], observed=True)
        .agg(value=("value", "mean"), sigma=("sigma", "mean"))
        .reset_index()
        .sort_values(SERIES_KEY + ["ts"])
        .reset_index(drop=True)
    )
    return g


@dataclass
class Panel:
    """Wide panel: values [T, S] on a common ts grid; series metadata aligned to columns."""

    ts: pd.DatetimeIndex          # [T]
    keys: pd.DataFrame            # [S] rows of SERIES_KEY
    values: np.ndarray            # [T, S] float64, NaN = unobserved
    platform: pd.Series           # [S] platform per series (for normalization keys)

    @property
    def n_series(self) -> int:
        return len(self.keys)

    def series_name(self, j: int) -> str:
        k = self.keys.iloc[j]
        return f"{k.device}/{k.entity_id}/{k.observable}/{k.channel}"


def make_panel(series: pd.DataFrame, uts: pd.DataFrame | None = None) -> Panel:
    """Pivot the long series frame onto the union grid of observed timestamps."""
    wide = series.pivot_table(
        index="ts", columns=SERIES_KEY, values="value", aggfunc="mean", observed=True
    ).sort_index()
    keys = pd.DataFrame(list(wide.columns), columns=SERIES_KEY)
    if uts is not None:
        dev2plat = uts.drop_duplicates("device").set_index("device")["platform"]
        platform = keys["device"].map(dev2plat)
    else:
        platform = pd.Series(["unknown"] * len(keys))
    return Panel(
        ts=pd.DatetimeIndex(wide.index),
        keys=keys,
        values=wide.to_numpy(dtype=np.float64),
        platform=platform,
    )


def subset_panel(panel: Panel, devices: list[str]) -> Panel:
    """Column-subset a panel to the given devices (tuning-protocol helper)."""
    keep = panel.keys["device"].isin(devices).to_numpy()
    return Panel(
        ts=panel.ts,
        keys=panel.keys[keep].reset_index(drop=True),
        values=panel.values[:, keep],
        platform=panel.platform[keep].reset_index(drop=True),
    )


def split_panel_by_date(
    panel: Panel, train_frac: float = 0.7, val_frac: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temporal index masks (train, val, test) over panel.ts — boundaries by row count."""
    T = len(panel.ts)
    n_tr = int(round(T * train_frac))
    n_va = int(round(T * val_frac))
    idx = np.arange(T)
    return idx < n_tr, (idx >= n_tr) & (idx < n_tr + n_va), idx >= n_tr + n_va


def forecast_pairs(
    panel: Panel, horizons: list[int], t_mask: np.ndarray
) -> pd.DataFrame:
    """All (origin t, series j, horizon h) rows with observed origin AND target,
    where the TARGET time falls inside t_mask (the split owns the target, so test
    targets are never visible from train origins).

    Columns: t_idx, s_idx, horizon, x_last (value at origin), y (value at t+h)."""
    rows = []
    V = panel.values
    T = len(panel.ts)
    for h in horizons:
        for t in range(T - h):
            if not t_mask[t + h]:
                continue
            obs = ~np.isnan(V[t]) & ~np.isnan(V[t + h])
            for j in np.nonzero(obs)[0]:
                rows.append((t, int(j), h, V[t, j], V[t + h, j]))
    return pd.DataFrame(rows, columns=["t_idx", "s_idx", "horizon", "x_last", "y"])
