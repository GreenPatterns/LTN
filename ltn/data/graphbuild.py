"""UTS -> temporal graph snapshots (master plan A4, consumed by LTN + TGN baselines).

Converts Universal Telemetry Schema event frames (:mod:`ltn.data.uts`) into
ordered lists of :class:`Snapshot` graphs. Guarantees implemented here:

* **Leakage guard** — normalization statistics come exclusively from the frame
  passed to :func:`compute_norm_stats` (fit on the TRAIN split only);
  :func:`build_snapshots` never updates them, and the feature vocabulary
  (observable, channel) is defined by those train-time stats.
* **Symmetric edges** — every undirected coupler/star edge emits both
  directions in ``edge_index``; ``edge_attr`` rows are duplicated to match.
* **Node features** — z-scored *latest* value per (observable, channel) with
  an explicit observed mask (missing -> 0.0 feature + 0 mask), plus
  log1p(calibration-age hours) and log1p(hours since the entity's previous
  event) as the final two columns of ``x``.
* **Forecast targets** — ``y_next`` holds the *next* snapshot's freshly
  observed z-scored values per entity (NaN wherever the next snapshot has no
  fresh observation for that entity/column), aligned column-for-column with
  the z-score block of ``x``.
* **Edge features** — latest ``gate_error_2q`` z-score per physical coupler
  in column 0 of ``edge_attr`` with an observed flag in column 1
  (0.0 + flag 0 while unobserved).

Graph structure: devices with qubit-level entities take their topology from a
per-device-name registry (``ibm_*`` -> fake-backend coupling map, falling back
to heavy-hex(7) offline; ``sim_dev_*`` -> topology recorded in event
``context`` else heavy-hex(5); ``google_*`` -> grid; ``simos_*`` ->
linear(2)).  Qubit ids carrying two or more numbers (google-style
``q{row}_{col}``) are treated as grid coordinates and wired by
nearest-Manhattan-neighbour adjacency, which is exactly the grid topology and
avoids first-integer id collisions.  Devices with only subgraph/device
entities (DAQEC) get a star graph (subgraph <-> device hub, hub synthesized
if the frame carries no device entity) plus subgraph-subgraph edges whenever
their ``context.data_qubits`` sets overlap.

Plain torch tensors only — no torch_scatter and no PyG dependency (models
wrap these into their preferred containers). ``truth`` channel rows are never
used for stats or features.
"""
from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch

from ltn.data.uts import PLATFORMS

NormStats = dict[tuple[str, str, str], tuple[float, float]]

_STD_FLOOR = 1e-12
_NS_PER_HOUR = 3.6e12
#: coupling_edges("fake", ...) offline fallback distance (plan A4).
_FALLBACK_HEAVY_HEX_DISTANCE = 7
#: sim_dev_* fallback when no topology is recorded in event context.
_SIM_HEAVY_HEX_DISTANCE = 5
_COORD_TOL = 1e-9

#: Set to False the first time the qiskit fake provider fails to import/resolve
#: (coupling_edges("fake", ...) then falls back to heavy_hex(distance=7)).
FAKE_PROVIDER_OK: bool = True


@dataclass
class Snapshot:
    """One temporal-graph frame: fixed entity set, features at time ``ts``."""

    ts: pd.Timestamp
    entity_ids: list[str]                 # node order for every tensor below
    x: torch.Tensor                       # float32 [n_entities, n_obs + 2]
    edge_index: torch.Tensor              # long [2, E], symmetric
    edge_attr: torch.Tensor               # float32 [E, 2] = (ge2q z-score, observed)
    mask: torch.Tensor                    # float32 [n_entities, n_obs] observed flags
    y_next: torch.Tensor                  # float32 [n_entities, n_obs], NaN-padded
    platform_onehot: torch.Tensor         # float32 [len(PLATFORMS)]


# --------------------------------------------------------------------------- topologies

def _canonical_undirected(edges) -> list[tuple[int, int]]:
    return sorted({(min(int(a), int(b)), max(int(a), int(b)))
                   for a, b in edges if int(a) != int(b)})


def _grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    out = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                out.append((i, i + 1))
            if r + 1 < rows:
                out.append((i, i + cols))
    return out


def _resolve_fake_class(fp_module: Any, name: str):
    stem = name.lower()
    for pre in ("ibm_", "fake_", "fake"):
        if stem.startswith(pre):
            stem = stem[len(pre):]
            break
    key = "fake" + stem.replace("_", "").replace("-", "")
    for attr in dir(fp_module):
        if attr.lower() == key:
            return getattr(fp_module, attr)
    return None


def _fake_edges(name: str) -> list[tuple[int, int]]:
    global FAKE_PROVIDER_OK
    try:
        from qiskit_ibm_runtime import fake_provider
        cls = _resolve_fake_class(fake_provider, name)
        if cls is None:
            raise LookupError(f"no fake backend matching {name!r}")
        cmap = cls().coupling_map
        if cmap is None:
            raise LookupError(f"fake backend {name!r} exposes no coupling map")
        return list(cmap.get_edges())
    except Exception as exc:  # offline-safe fallback, per plan A4
        FAKE_PROVIDER_OK = False
        warnings.warn(
            f"fake provider unavailable for {name!r} ({type(exc).__name__}: {exc}); "
            f"falling back to heavy_hex(distance={_FALLBACK_HEAVY_HEX_DISTANCE})",
            stacklevel=2,
        )
        from qiskit.transpiler import CouplingMap
        return list(CouplingMap.from_heavy_hex(_FALLBACK_HEAVY_HEX_DISTANCE).get_edges())


def coupling_edges(kind: str, **kw) -> list[tuple[int, int]]:
    """Undirected, deduplicated coupling edges (i < j, sorted).

    kinds: ``heavy_hex(distance)``, ``grid(rows, cols)``, ``linear(n)``,
    ``fake(name)`` (name may be ``ibm_brisbane`` / ``fake_brisbane`` /
    ``FakeBrisbane`` / ``brisbane``; falls back to heavy_hex(7) with a
    warning when the fake provider is unavailable or the name is unknown).
    """
    if kind == "heavy_hex":
        from qiskit.transpiler import CouplingMap
        edges = CouplingMap.from_heavy_hex(int(kw["distance"])).get_edges()
    elif kind == "grid":
        edges = _grid_edges(int(kw["rows"]), int(kw["cols"]))
    elif kind == "linear":
        edges = [(i, i + 1) for i in range(int(kw["n"]) - 1)]
    elif kind == "fake":
        edges = _fake_edges(str(kw["name"]))
    else:
        raise ValueError(f"unknown coupling kind {kind!r}")
    return _canonical_undirected(edges)


def _parse_context(val: Any) -> dict:
    if isinstance(val, Mapping):
        return dict(val)
    if isinstance(val, str) and val:
        try:
            obj = json.loads(val)
        except json.JSONDecodeError:
            return {}
        if isinstance(obj, dict):
            return obj
    return {}


def _stored_topology(dev_df: pd.DataFrame) -> list[tuple[int, int]] | None:
    """First topology spec found in row contexts: an edge list, or a dict
    {"kind": ..., **params} forwarded to coupling_edges."""
    for val in dev_df["context"]:
        ctx = _parse_context(val)
        topo = ctx.get("topology", ctx.get("coupling_map"))
        if topo is None:
            continue
        if isinstance(topo, Mapping) and "kind" in topo:
            return coupling_edges(topo["kind"], **{k: v for k, v in topo.items() if k != "kind"})
        if isinstance(topo, (list, tuple)):
            return _canonical_undirected((int(a), int(b)) for a, b in topo)
    return None


def _edges_ibm(device: str, dev_df: pd.DataFrame, n: int) -> list[tuple[int, int]]:
    return coupling_edges("fake", name=device)


def _edges_sim(device: str, dev_df: pd.DataFrame, n: int) -> list[tuple[int, int]]:
    topo = _stored_topology(dev_df)
    if topo is not None:
        return topo
    return coupling_edges("heavy_hex", distance=_SIM_HEAVY_HEX_DISTANCE)


def _edges_google(device: str, dev_df: pd.DataFrame, n: int) -> list[tuple[int, int]]:
    topo = _stored_topology(dev_df)
    if topo is not None:
        return topo
    rows = max(1, round(math.sqrt(n)))
    cols = math.ceil(n / rows)
    return coupling_edges("grid", rows=rows, cols=cols)


def _edges_simos(device: str, dev_df: pd.DataFrame, n: int) -> list[tuple[int, int]]:
    return coupling_edges("linear", n=2)


# Device-name prefix -> topology builder (first match wins).
TOPOLOGY_REGISTRY: list[tuple[str, Callable[[str, pd.DataFrame, int], list[tuple[int, int]]]]] = [
    ("ibm_", _edges_ibm),
    ("sim_dev_", _edges_sim),
    ("google_", _edges_google),
    ("simos_", _edges_simos),
]


def _device_edges(device: str, dev_df: pd.DataFrame, n_qubits: int) -> list[tuple[int, int]]:
    for prefix, builder in TOPOLOGY_REGISTRY:
        if device.startswith(prefix):
            return builder(device, dev_df, n_qubits)
    return _edges_sim(device, dev_df, n_qubits)  # unknown prefix: context else heavy-hex(5)


# --------------------------------------------------------------------------- norm stats

def compute_norm_stats(uts_train: pd.DataFrame) -> NormStats:
    """Per-(platform, observable, channel) mean/std from the TRAIN frame only.

    This is the leakage guard: fit these once on the train split and pass the
    resulting dict to :func:`build_snapshots` for every split. Population std
    (ddof=0); degenerate std (single sample / constant) is floored to 1.0.
    ``truth`` channel rows are excluded defensively.
    """
    df = uts_train[uts_train["channel"] != "truth"]
    stats: NormStats = {}
    for (plat, obs, ch), grp in df.groupby(["platform", "observable", "channel"], observed=True):
        vals = grp["value"].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        if not np.isfinite(std) or std < _STD_FLOOR:
            std = 1.0
        stats[(str(plat), str(obs), str(ch))] = (mean, std)
    return stats


# --------------------------------------------------------------------------- snapshots

_INT_RE = re.compile(r"\d+")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")   # unsigned, so "c0-1" parses as (0, 1)


def _qubit_index(entity_id: str) -> int:
    m = _INT_RE.search(str(entity_id))
    if m is None:
        raise ValueError(f"cannot parse qubit index from entity_id {entity_id!r}")
    return int(m.group())


def _entity_numbers(entity_id: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(str(entity_id))]


def _coupler_pair(entity_id: str) -> tuple[int, int] | None:
    ints = _INT_RE.findall(str(entity_id))
    if len(ints) < 2:
        return None
    a, b = int(ints[0]), int(ints[1])
    return (min(a, b), max(a, b))


def _coupler_pair_coords(entity_id: str):
    nums = _entity_numbers(entity_id)
    if len(nums) < 4:
        return None
    return tuple(sorted(((nums[0], nums[1]), (nums[2], nums[3]))))


def _coord_grid_edges(pts: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Nearest-Manhattan-neighbour edges over 2-D coordinates (= grid adjacency
    for regularly spaced grids, at any spacing)."""
    n = len(pts)
    dists = [abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
             for i in range(n) for j in range(i + 1, n)]
    dists = [d for d in dists if d > _COORD_TOL]
    if not dists:
        return []
    step = min(dists)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
            if abs(d - step) <= _COORD_TOL:
                out.append((i, j))
    return out


def _leaf_qubits(node_rows: pd.DataFrame, entity_id: str) -> frozenset[int]:
    for val in node_rows.loc[node_rows["entity_id"] == entity_id, "context"]:
        ctx = _parse_context(val)
        dq = ctx.get("data_qubits")
        if dq is not None:
            return frozenset(int(q) for q in dq)
    return frozenset()


def _hours_since(ts_ns: int, ref_ns_arr: np.ndarray) -> np.ndarray:
    """log1p(hours) since a per-entity reference (ns since epoch, -1 = never)."""
    out = np.zeros(ref_ns_arr.shape, dtype=np.float64)
    seen = ref_ns_arr >= 0
    out[seen] = np.log1p(np.maximum(0.0, (ts_ns - ref_ns_arr[seen]) / _NS_PER_HOUR))
    return out


def build_snapshots(
    uts: pd.DataFrame,
    device: str,
    norm_stats: Mapping[tuple[str, str, str], tuple[float, float]],
    cadence: str = "event",
) -> list[Snapshot]:
    """Convert one device's UTS rows into an ordered list of graph snapshots.

    cadence="event": one snapshot per distinct event timestamp;
    cadence="1D": one snapshot per calendar day that has events (snapshot
    ``ts`` = latest event time within that day).
    """
    if cadence not in ("event", "1D"):
        raise ValueError(f"cadence must be 'event' or '1D', got {cadence!r}")

    dev = uts[(uts["device"] == device) & (uts["channel"] != "truth")].copy()
    if dev.empty:
        return []
    dev["ts"] = pd.to_datetime(dev["ts"], utc=True)
    dev["cal_ts"] = pd.to_datetime(dev["cal_ts"], utc=True)
    dev = dev.sort_values("ts", kind="stable").reset_index(drop=True)
    dev["_bin"] = dev["ts"] if cadence == "event" else dev["ts"].dt.floor("D")

    platform = str(dev["platform"].iloc[0])
    obs_cols: list[tuple[str, str]] = sorted(
        (obs, ch) for (plat, obs, ch) in norm_stats if plat == platform
    )
    col_ix = {oc: j for j, oc in enumerate(obs_cols)}
    n_obs = len(obs_cols)
    mu = np.array([norm_stats[(platform, o, c)][0] for o, c in obs_cols], dtype=np.float64)
    sd = np.array([norm_stats[(platform, o, c)][1] for o, c in obs_cols], dtype=np.float64)

    coupler_rows = dev[dev["entity_type"] == "coupler"]
    has_qubits = bool((dev["entity_type"] == "qubit").any())

    edge_slot: dict = {}                    # coupler key -> undirected edge slot
    edge_key: Callable[[str], Any] = _coupler_pair
    if has_qubits:
        node_rows = dev[dev["entity_type"] == "qubit"]
        raw_ids = list(pd.unique(node_rows["entity_id"]))
        nums = {e: _entity_numbers(e) for e in raw_ids}
        if raw_ids and all(len(v) >= 2 for v in nums.values()):
            # google-style "q{row}_{col}" coordinate ids: grid adjacency straight
            # from coordinates (first-integer parsing would collide across rows).
            coord = {e: (nums[e][0], nums[e][1]) for e in raw_ids}
            entity_ids = sorted(raw_ids, key=lambda e: coord[e])
            pts = [coord[e] for e in entity_ids]
            und_pos = _coord_grid_edges(pts)
            edge_slot = {tuple(sorted((pts[i], pts[j]))): k
                         for k, (i, j) in enumerate(und_pos)}
            edge_key = _coupler_pair_coords
        else:
            entity_ids = sorted(raw_ids, key=_qubit_index)
            phys = [_qubit_index(e) for e in entity_ids]
            pos_of_phys = {p: k for k, p in enumerate(phys)}
            present = set(phys)
            topo = _device_edges(device, dev, max(phys) + 1)
            und_phys = [(a, b) for a, b in topo if a in present and b in present]
            und_pos = [(pos_of_phys[a], pos_of_phys[b]) for a, b in und_phys]
            edge_slot = {pair: k for k, pair in enumerate(und_phys)}
    else:
        # DAQEC-style star graph: device hub(s) + subgraph/detector leaves.
        node_rows = dev[dev["entity_type"] != "coupler"]
        device_ids = sorted(node_rows.loc[node_rows["entity_type"] == "device", "entity_id"].unique())
        leaf_ids = sorted(node_rows.loc[node_rows["entity_type"] != "device", "entity_id"].unique())
        if not device_ids and leaf_ids:
            device_ids = [f"device:{device}"]  # synthetic hub so the star always exists
        entity_ids = list(device_ids) + list(leaf_ids)
        pos_of_id = {e: k for k, e in enumerate(entity_ids)}
        und_pos = [(pos_of_id[d], pos_of_id[l]) for l in leaf_ids for d in device_ids]
        qsets = {l: _leaf_qubits(node_rows, l) for l in leaf_ids}
        for i, a in enumerate(leaf_ids):
            for b in leaf_ids[i + 1:]:
                if qsets[a] and (qsets[a] & qsets[b]):
                    und_pos.append((pos_of_id[a], pos_of_id[b]))

    n = len(entity_ids)
    row_of_id = {e: k for k, e in enumerate(entity_ids)}
    n_edge = len(und_pos)

    if n_edge:
        ei = np.asarray(und_pos, dtype=np.int64).T          # [2, E_und]
        edge_index = torch.from_numpy(np.hstack([ei, ei[::-1]]).copy())
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    onehot = torch.zeros(len(PLATFORMS), dtype=torch.float32)
    if platform in PLATFORMS:
        onehot[PLATFORMS.index(platform)] = 1.0

    snap_ts = dev.groupby("_bin")["ts"].max().sort_index()
    bins = list(snap_ts.index)                              # ordered group keys
    node_by_bin = {k: g for k, g in node_rows.groupby("_bin", sort=False)}
    coupler_by_bin = {k: g for k, g in coupler_rows.groupby("_bin", sort=False)}

    # Running state (carried forward across snapshots).
    latest_z = np.full((n, n_obs), np.nan)
    last_evt_ns = np.full(n, -1, dtype=np.int64)
    last_cal_ns = np.full(n, -1, dtype=np.int64)
    edge_val = np.zeros(n_edge, dtype=np.float64)
    edge_mask = np.zeros(n_edge, dtype=np.float64)

    fresh_per_bin: list[np.ndarray] = []
    records: list[tuple[pd.Timestamp, np.ndarray, np.ndarray, np.ndarray]] = []

    for b in bins:
        ts = snap_ts.loc[b]
        fresh = np.full((n, n_obs), np.nan)
        nb = node_by_bin.get(b)
        if nb is not None and len(nb):
            nb_last = nb.drop_duplicates(subset=["entity_id", "observable", "channel"], keep="last")
            for r in nb_last.itertuples(index=False):
                j = col_ix.get((r.observable, r.channel))
                if j is None or not np.isfinite(r.value):
                    continue
                fresh[row_of_id[r.entity_id], j] = (float(r.value) - mu[j]) / sd[j]
            for eid, g in nb.groupby("entity_id", sort=False):
                i = row_of_id[eid]
                last_evt_ns[i] = max(last_evt_ns[i], g["ts"].max().value)
                cal_max = g["cal_ts"].max()
                if pd.notna(cal_max):
                    last_cal_ns[i] = max(last_cal_ns[i], cal_max.value)
        upd = ~np.isnan(fresh)
        latest_z[upd] = fresh[upd]

        cb = coupler_by_bin.get(b)
        if cb is not None and len(cb) and edge_slot:
            cb_last = cb[cb["observable"] == "gate_error_2q"].drop_duplicates(
                subset=["entity_id"], keep="last")
            for r in cb_last.itertuples(index=False):
                key = edge_key(r.entity_id)
                slot = edge_slot.get(key) if key is not None else None
                st = norm_stats.get((platform, "gate_error_2q", r.channel))
                if slot is None or st is None or not np.isfinite(r.value):
                    continue
                edge_val[slot] = (float(r.value) - st[0]) / st[1]
                edge_mask[slot] = 1.0

        cal_age = _hours_since(ts.value, last_cal_ns)
        dt_evt = _hours_since(ts.value, last_evt_ns)
        x = np.concatenate(
            [np.nan_to_num(latest_z, nan=0.0), cal_age[:, None], dt_evt[:, None]], axis=1)
        mask = (~np.isnan(latest_z)).astype(np.float64)
        attr_und = np.stack([edge_val, edge_mask], axis=1) if n_edge else np.zeros((0, 2))
        attr = np.vstack([attr_und, attr_und])              # both edge directions
        fresh_per_bin.append(fresh)
        records.append((ts, x, mask, attr))

    snapshots: list[Snapshot] = []
    for k, (ts, x, mask, attr) in enumerate(records):
        y_next = fresh_per_bin[k + 1] if k + 1 < len(records) else np.full((n, n_obs), np.nan)
        snapshots.append(Snapshot(
            ts=ts,
            entity_ids=list(entity_ids),
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=torch.tensor(attr, dtype=torch.float32),
            mask=torch.tensor(mask, dtype=torch.float32),
            y_next=torch.tensor(y_next, dtype=torch.float32),
            platform_onehot=onehot,
        ))
    return snapshots
