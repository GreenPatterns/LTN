"""Universal Telemetry Schema (UTS) — the single event format every data source maps into.

One row = one scalar telemetry observation of one entity of one device at one time.
The lake is parquet, hive-partitioned: data/uts/source=<src>/device=<dev>/part-*.parquet.

Extends the master plan's UTS with `source`, `channel`, and the `subgraph`/`detector`
entity types [DEV-9]; `channel="truth"` rows exist only for the synthetic fleet and are
never model-visible (loaders exclude them by default).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

SOURCES = ["daqec", "daqec_v2", "google_qec", "silicon", "sim", "ibm_live"]
PLATFORMS = ["ibm_sc", "google_sc", "si_spin", "iqm_sc", "ionq_trap", "rigetti_sc", "sim_sc"]
ENTITY_TYPES = ["device", "qubit", "coupler", "subgraph", "detector"]
CHANNELS = ["claimed", "probe", "delivered", "derived", "truth"]

# Canonical observable vocabulary. Loaders may not invent names outside this list.
OBSERVABLES = [
    "t1_us",
    "t2_us",
    "tphi_us",
    "frequency_ghz",
    "readout_error",
    "epc_1q",
    "gate_error_2q",
    "logical_error_rate",
    "logical_error_per_round",
    "detection_event_fraction",
    "detector_error_prob",
    "dem_error_prob",
    "cz_fidelity",
    "burst_count",
    "fano_factor",
    "adjacent_correlation",
    "composite_score",
]

COLUMNS = [
    "source", "platform", "device", "entity_type", "entity_id",
    "observable", "channel", "value", "sigma", "ts", "cal_ts", "context",
]

_CATEGORICALS = ["source", "platform", "entity_type", "observable", "channel"]


def empty_uts() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
    df["value"] = df["value"].astype("float64")
    df["sigma"] = df["sigma"].astype("float64")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["cal_ts"] = pd.to_datetime(df["cal_ts"], utc=True)
    return df


def subgraph_id(qubits: Iterable[int]) -> str:
    """Stable entity id for a qubit subset: sub:<sha1[:8] of sorted indices>."""
    key = ",".join(str(q) for q in sorted(int(q) for q in qubits))
    return "sub:" + hashlib.sha1(key.encode()).hexdigest()[:8]


def make_events(
    *,
    source: str,
    platform: str,
    device: str,
    entity_type: str,
    entity_id: str,
    ts,
    rows: Sequence[tuple[str, str, float, float | None]],
    cal_ts=None,
    context: dict | None = None,
) -> pd.DataFrame:
    """Build UTS rows for one (entity, timestamp): rows = [(observable, channel, value, sigma), ...]."""
    ctx = json.dumps(context or {}, default=str, sort_keys=True)
    recs = [
        {
            "source": source, "platform": platform, "device": device,
            "entity_type": entity_type, "entity_id": entity_id,
            "observable": obs, "channel": ch, "value": float(v),
            "sigma": (None if s is None else float(s)),
            "ts": ts, "cal_ts": cal_ts, "context": ctx,
        }
        for obs, ch, v, s in rows
    ]
    df = pd.DataFrame.from_records(recs, columns=COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["cal_ts"] = pd.to_datetime(df["cal_ts"], utc=True)
    df["value"] = df["value"].astype("float64")
    df["sigma"] = df["sigma"].astype("float64")
    return df


def validate_uts(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError on any schema violation; return df untouched otherwise."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"UTS missing columns: {missing}")
    problems = []
    for col, vocab in [
        ("source", SOURCES), ("platform", PLATFORMS),
        ("entity_type", ENTITY_TYPES), ("observable", OBSERVABLES), ("channel", CHANNELS),
    ]:
        bad = set(df[col].unique()) - set(vocab)
        if bad:
            problems.append(f"{col} has non-canonical values: {sorted(bad)[:5]}")
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        problems.append("ts is not datetime")
    else:
        tz = getattr(df["ts"].dt, "tz", None)
        if tz is None:
            problems.append("ts is not tz-aware UTC")
    if df["value"].isna().any():
        problems.append(f"{int(df['value'].isna().sum())} NaN values in `value`")
    neg_sigma = df["sigma"].dropna() < 0
    if neg_sigma.any():
        problems.append(f"{int(neg_sigma.sum())} negative sigmas")
    truth_bad = df[(df["channel"] == "truth") & (df["source"] != "sim")]
    if len(truth_bad):
        problems.append("channel='truth' present for non-sim source")
    if problems:
        raise ValueError("UTS validation failed: " + "; ".join(problems))
    return df


def write_uts(df: pd.DataFrame, root: str | Path) -> list[Path]:
    """Write validated events into the hive-partitioned lake, one file per (source, device)."""
    validate_uts(df)
    root = Path(root)
    written = []
    for (src, dev), part in df.groupby(["source", "device"], observed=True):
        out_dir = root / f"source={src}" / f"device={dev}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "part-0.parquet"
        if out.exists():  # merge + dedup so loaders are idempotent
            old = pd.read_parquet(out)
            part = pd.concat([old, part.drop(columns=["source", "device"])
                              if "source" not in old.columns else part], ignore_index=True)
        keep = [c for c in COLUMNS if c not in ("source", "device")]
        part = part[keep] if set(keep) <= set(part.columns) else part
        part = part.drop_duplicates(subset=["entity_type", "entity_id", "observable", "channel", "ts", "context"])
        part.to_parquet(out, index=False)
        written.append(out)
    return written


def read_uts(
    root: str | Path,
    sources: Sequence[str] | None = None,
    devices: Sequence[str] | None = None,
    forbid_sources: Sequence[str] = (),
    include_truth: bool = False,
) -> pd.DataFrame:
    """Read back the lake. `truth` channel rows are excluded unless explicitly requested."""
    root = Path(root)
    frames = []
    for src_dir in sorted(root.glob("source=*")):
        src = src_dir.name.split("=", 1)[1]
        if sources is not None and src not in sources:
            continue
        if src in forbid_sources:
            continue
        for dev_dir in sorted(src_dir.glob("device=*")):
            dev = dev_dir.name.split("=", 1)[1]
            if devices is not None and dev not in devices:
                continue
            for f in sorted(dev_dir.glob("*.parquet")):
                part = pd.read_parquet(f)
                part["source"] = src
                part["device"] = dev
                frames.append(part)
    if not frames:
        return empty_uts()
    df = pd.concat(frames, ignore_index=True)[COLUMNS]
    if not include_truth:
        df = df[df["channel"] != "truth"].reset_index(drop=True)
    return df
