"""Panel + UTS -> per-device tensor sequences for LTN and neural baselines.

Feature layout per entity per snapshot (must match LTNModel's docstring):
  x = [z-scored value per target observable (F_obs), observed-mask (F_obs),
       log1p(cal_age_h), log1p(dt_h), platform one-hot (P)]

Graph edges: qubit-level entities get topology edges (ltn.data.graphbuild registry);
subgraph/device entities get the DAQEC star graph (subgraph<->device + overlap edges).
Edge features: [z-scored gate_error_2q on the coupler (0 if absent), presence flag].
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from ltn.data.series import Panel

PLATFORMS_ORDER = ["ibm_sc", "google_sc", "si_spin", "iqm_sc", "ionq_trap", "rigetti_sc", "sim_sc"]


@dataclass
class DeviceSequence:
    device: str
    ts: pd.DatetimeIndex               # [T] this device's snapshot grid
    entity_ids: list[str]              # [n]
    observables: list[str]             # [F_obs] (observable, channel) label strings
    x: torch.Tensor                    # [T, n, F] float64
    cal_age_h: torch.Tensor            # [T, n] float64 raw hours
    dt_steps: torch.Tensor             # [T] float64 (grid units; dt[0] = 1)
    edge_index: torch.Tensor           # [2, E]
    edge_attr: torch.Tensor            # [E, 2]
    y: torch.Tensor                    # [T, n, F_obs] z-scored targets (NaN = unobserved)
    norm: dict = field(default_factory=dict)   # (obs_label) -> (mean, std) for inversion
    panel_col: dict = field(default_factory=dict)  # (entity_id, obs_label) -> panel s_idx


def _obs_label(observable: str, channel: str) -> str:
    return f"{observable}/{channel}"


def _entity_star_edges(entity_ids: list[str], qubit_sets: dict[str, set]) -> list[tuple[int, int]]:
    """DAQEC-style: device node <-> every subgraph node; subgraph-subgraph when overlapping."""
    idx = {e: i for i, e in enumerate(entity_ids)}
    edges = []
    dev = idx.get("device")
    for e, i in idx.items():
        if e != "device" and dev is not None:
            edges += [(i, dev), (dev, i)]
    subs = [e for e in entity_ids if e != "device"]
    for a in range(len(subs)):
        for b in range(a + 1, len(subs)):
            qa, qb = qubit_sets.get(subs[a], set()), qubit_sets.get(subs[b], set())
            if qa & qb:
                ia, ib = idx[subs[a]], idx[subs[b]]
                edges += [(ia, ib), (ib, ia)]
    return edges


def _qubit_coupler_edges(entity_ids: list[str]) -> list[tuple[int, int]]:
    """Qubit/coupler entity graph: coupler c<a>-<b> nodes attach to their endpoint
    qubits q<a>, q<b>, and endpoints get a direct edge — the coupler set enumerates
    the device topology exactly (sim/harvester entity-id convention). Directed both ways."""
    idx = {e: i for i, e in enumerate(entity_ids)}
    edges: list[tuple[int, int]] = []

    def add(a: int, b: int) -> None:
        edges.append((a, b))
        edges.append((b, a))

    for e, i in idx.items():
        if e.startswith("c") and "-" in e:
            try:
                a, b = e[1:].split("-")
                qa, qb = idx.get(f"q{int(a)}"), idx.get(f"q{int(b)}")
            except ValueError:
                continue
            if qa is not None:
                add(i, qa)
            if qb is not None:
                add(i, qb)
            if qa is not None and qb is not None:
                add(qa, qb)
    return sorted(set(edges))


def build_device_sequence(
    uts: pd.DataFrame,
    panel: Panel,
    device: str,
    observables: list[tuple[str, str]],
    norm_stats: dict,
    time_unit_h: float = 8.0,
) -> DeviceSequence:
    """norm_stats: {(platform, observable, channel): (mean, std)} from TRAIN data only."""
    obs_labels = [_obs_label(o, c) for o, c in observables]
    du = uts[uts["device"] == device]
    platform = du["platform"].iloc[0]

    # device-local grid = union of ts across this device's panel series
    mask_dev = (panel.keys["device"] == device).to_numpy()
    col_idx = np.nonzero(mask_dev)[0]
    keys_dev = panel.keys.iloc[col_idx]
    t_obs = ~np.isnan(panel.values[:, col_idx]).all(axis=1)
    ts = panel.ts[t_obs]
    t_map = np.nonzero(t_obs)[0]

    entity_ids = sorted(keys_dev["entity_id"].unique())
    n, T, F_obs = len(entity_ids), len(ts), len(obs_labels)
    e_idx = {e: i for i, e in enumerate(entity_ids)}

    vals = np.full((T, n, F_obs), np.nan)
    panel_col: dict[tuple[str, str], int] = {}
    norm: dict[str, tuple[float, float]] = {}
    for local_j, (gj, k) in enumerate(zip(col_idx, keys_dev.itertuples())):
        lbl = _obs_label(k.observable, k.channel)
        if lbl not in obs_labels:
            continue
        f = obs_labels.index(lbl)
        m, s = norm_stats.get((platform, k.observable, k.channel), (0.0, 1.0))
        s = s if s and np.isfinite(s) else 1.0
        norm[lbl] = (m, s)
        vals[:, e_idx[k.entity_id], f] = (panel.values[t_obs, gj] - m) / s
        panel_col[(k.entity_id, lbl)] = int(gj)

    obs_mask = ~np.isnan(vals)
    x_vals = np.nan_to_num(vals, nan=0.0)

    # cal_age per (entity, ts) from raw UTS cal_ts (mean across that entity's events)
    cal = np.zeros((T, n))
    dcal = du.dropna(subset=["cal_ts"])
    if len(dcal):
        age_h = (dcal["ts"] - dcal["cal_ts"]).dt.total_seconds() / 3600.0
        g = dcal.assign(age=age_h).groupby(["entity_id", "ts"])['age'].mean()
        ts_pos = {t: i for i, t in enumerate(ts)}
        for (ent, t), a in g.items():
            if ent in e_idx and t in ts_pos:
                cal[ts_pos[t], e_idx[ent]] = a

    dt_h = np.diff(ts.asi8 / 3.6e12, prepend=ts.asi8[0] / 3.6e12 - time_unit_h)
    dt_steps = np.clip(dt_h / time_unit_h, 0.05, 30.0)

    plat_onehot = np.zeros(len(PLATFORMS_ORDER))
    if platform in PLATFORMS_ORDER:
        plat_onehot[PLATFORMS_ORDER.index(platform)] = 1.0

    F = 2 * F_obs + 2 + len(PLATFORMS_ORDER)
    x = np.zeros((T, n, F))
    x[:, :, :F_obs] = x_vals
    x[:, :, F_obs : 2 * F_obs] = obs_mask.astype(float)
    x[:, :, 2 * F_obs] = np.log1p(np.maximum(cal, 0.0))
    x[:, :, 2 * F_obs + 1] = np.log1p(dt_steps)[:, None]
    x[:, :, 2 * F_obs + 2 :] = plat_onehot[None, None, :]

    # graph
    qubit_sets: dict[str, set] = {}
    for _, r in du.drop_duplicates("entity_id").iterrows():
        try:
            ctx = json.loads(r["context"]) if isinstance(r["context"], str) else {}
            q = ctx.get("data_qubits")
            if isinstance(q, str):
                q = json.loads(q)
            if q:
                qubit_sets[r["entity_id"]] = set(int(v) for v in q)
        except Exception:
            pass
    if set(entity_ids) - {"device"} and all(e.startswith("sub:") or e == "device" for e in entity_ids):
        edges = _entity_star_edges(entity_ids, qubit_sets)
    else:
        edges = _qubit_coupler_edges(entity_ids)
    ei = torch.tensor(edges, dtype=torch.long).T if edges else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.zeros(ei.shape[1], 2, dtype=torch.float64)  # gate_error z + presence
    ge = du[(du["observable"] == "gate_error_2q")]
    if len(ge) and ei.numel():
        m, s = norm_stats.get((platform, "gate_error_2q", "claimed"), (0.0, 1.0))
        mean_ge = float(((ge["value"].mean()) - m) / (s or 1.0))
        ea[:, 0] = mean_ge
        ea[:, 1] = 1.0

    return DeviceSequence(
        device=device, ts=ts, entity_ids=entity_ids, observables=obs_labels,
        x=torch.tensor(x, dtype=torch.float64),
        cal_age_h=torch.tensor(cal, dtype=torch.float64),
        dt_steps=torch.tensor(dt_steps, dtype=torch.float64),
        edge_index=ei, edge_attr=ea,
        y=torch.tensor(vals, dtype=torch.float64),
        norm=norm, panel_col=panel_col,
    )
