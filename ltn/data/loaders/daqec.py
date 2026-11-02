"""DAQEC-Benchmark (Zenodo 18045662) -> UTS loader.

Implements plan step A2 (§2.3): convert the 756 QEC runs in
``data/raw/daqec/master.parquet`` into Universal Telemetry Schema events at two
granularities per run — the recurring data-qubit subset (``entity_type="subgraph"``,
id = ``uts.subgraph_id(data_qubits)``) and a device-level duplicate
(``entity_type="device"``, id ``"device"``) — so downstream code sees both stable
subgraph time series and device-level experiments.

Credibility caveat [DEV-10]: we consume ONLY the telemetry columns of this
dataset (claimed calibration averages, probe measurements, delivered logical
error rates with CIs, derived syndrome statistics). The dataset authors' own
adaptive-vs-baseline conclusions are NOT relied on anywhere in this codebase:
their ibm_fez validation summary shows chance-level logical error rates, the
README is partly template text, and dates are internally inconsistent.

Guarantee: output passes ``uts.validate_uts`` and contains exactly
2 x (number of non-NaN mapped observables) events per master.parquet row.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from ltn.data import uts

# master.parquet column -> (canonical observable, channel). One UTS event per
# non-NaN cell, emitted for both the subgraph entity and the device entity.
OBSERVABLE_MAP: list[tuple[str, str, str]] = [
    ("avg_t1_us", "t1_us", "claimed"),
    ("avg_t2_us", "t2_us", "claimed"),
    ("avg_readout_error", "readout_error", "claimed"),
    ("avg_ecr_error", "gate_error_2q", "claimed"),
    ("probe_t1_us", "t1_us", "probe"),
    ("probe_t2_us", "t2_us", "probe"),
    ("probe_readout_error", "readout_error", "probe"),
    ("logical_error_rate", "logical_error_rate", "delivered"),
    ("syndrome_burst_count", "burst_count", "derived"),
    ("fano_factor", "fano_factor", "derived"),
    ("adjacent_correlation", "adjacent_correlation", "derived"),
]

# 95% normal CI width -> one standard deviation (2 * 1.96).
_CI_TO_SIGMA = 3.92


def _parse_qubits(val: object) -> list[int]:
    """Parse a qubit-index list stored either as a JSON-ish string or a sequence."""
    if isinstance(val, str):
        return [int(q) for q in json.loads(val)]
    return [int(q) for q in val]  # type: ignore[union-attr]


def _row_events(row: pd.Series) -> Sequence[tuple[str, str, float, float | None]]:
    """(observable, channel, value, sigma) tuples for one master.parquet row."""
    ler_sigma = (
        float(row["logical_error_rate_ci_upper"]) - float(row["logical_error_rate_ci_lower"])
    ) / _CI_TO_SIGMA
    out: list[tuple[str, str, float, float | None]] = []
    for col, obs, channel in OBSERVABLE_MAP:
        v = row[col]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue  # probe_* columns are absent for baseline_static runs
        sigma = ler_sigma if obs == "logical_error_rate" else None
        out.append((obs, channel, float(v), sigma))
    return out


def load_daqec(raw_dir: str | Path) -> pd.DataFrame:
    """Load data/raw/daqec/master.parquet into a validated UTS DataFrame."""
    raw_dir = Path(raw_dir)
    master = pd.read_parquet(raw_dir / "master.parquet")

    frames: list[pd.DataFrame] = []
    for _, row in master.iterrows():
        data_qubits = _parse_qubits(row["data_qubits"])
        couplers = _parse_qubits(row["couplers"])
        context = {
            "run_id": str(row["run_id"]),
            "session_id": str(row["session_id"]),
            "strategy": str(row["strategy"]),
            "distance": int(row["distance"]),
            "syndrome_rounds": int(row["syndrome_rounds"]),
            "day": int(row["day"]),
            "data_qubits": data_qubits,
            "couplers": couplers,
        }
        events = _row_events(row)
        common = dict(
            source="daqec",
            platform="ibm_sc",
            device=str(row["backend"]),
            ts=row["timestamp_utc"],
            cal_ts=row["calibration_timestamp"],
            rows=events,
            context=context,
        )
        frames.append(
            uts.make_events(
                entity_type="subgraph", entity_id=uts.subgraph_id(data_qubits), **common
            )
        )
        frames.append(uts.make_events(entity_type="device", entity_id="device", **common))

    df = pd.concat(frames, ignore_index=True)
    return uts.validate_uts(df)
