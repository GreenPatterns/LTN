"""Raw IBM calibration-snapshot schema and the pure snapshot -> UTS mapping [DEV-1].

Guarantee implemented: every persisted raw snapshot is exactly the JSON-serialized
``BackendProperties.to_dict()`` of qiskit-ibm-runtime 0.47.0, and
:func:`from_backend_properties` maps any such dict (fresh from the API *or* re-read
from disk) into rows that pass ``ltn.data.uts.validate_uts`` — no network required.

Raw snapshot JSON layout (one file per (backend, last_update_date), see README.md)::

    {
      "backend_name":     "<str>",
      "backend_version":  "<str>",                       # "X.Y.Z"
      "last_update_date": "<ISO-8601 datetime>",         # snapshot identity / cal_ts
      "qubits": [                                        # index = physical qubit
        [ {"date": "<ISO>", "name": "T1", "unit": "us", "value": 217.3}, ... ],
        ...
      ],
      "gates": [
        {"qubits": [0, 1], "gate": "ecr",
         "parameters": [ {"date": "<ISO>", "name": "gate_error", "unit": "", "value": 7.5e-3}, ... ],
         ...optional extra fields (e.g. "name")... },
        ...
      ],
      "general": [ <Nduv dicts> ]                        # usually empty
    }

Datetime fields may be either ISO-8601 strings (after a JSON round-trip) or
``datetime`` objects (straight from the SDK); both are accepted everywhere here.

UTS mapping (source="ibm_live", platform="ibm_sc", channel="claimed", sigma=None —
IBM publishes point calibration values without uncertainties):

    qubit Nduv "T1" [time unit]        -> entity ("qubit", "q<i>"), observable "t1_us"
    qubit Nduv "T2" [time unit]        -> entity ("qubit", "q<i>"), observable "t2_us"
    qubit Nduv "frequency" [freq unit] -> entity ("qubit", "q<i>"), observable "frequency_ghz"
    qubit Nduv "readout_error"         -> entity ("qubit", "q<i>"), observable "readout_error"
    2q gate param "gate_error"         -> entity ("coupler", "c<a>-<b>"), observable "gate_error_2q"
                                          (qubit order as reported: directed gates stay directed)

``ts`` = the Nduv's own measurement date; ``cal_ts`` = ``last_update_date``.
Faulty sets (from "operational" Nduvs) go into every row's ``context`` JSON.

Unit handling: values are converted to SI via qiskit's ``apply_prefix`` and then to
the UTS unit, with the SI *base* unit checked per observable (``apply_prefix`` is
prefix-only: it rejects bare "Hz" yet reads "parsec" as pico-"arsec"). An Nduv whose
unit cannot be trusted is skipped with a warning, never guessed and never fatal —
the raw snapshot JSON preserves it regardless.
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from qiskit.utils.units import apply_prefix

from ltn.data.uts import empty_uts, make_events, validate_uts

log = logging.getLogger("ltn.harvest.schemas")

REQUIRED_SNAPSHOT_KEYS = ("backend_name", "backend_version", "last_update_date",
                          "qubits", "gates", "general")

# Nduv name -> UTS observable, restricted to the canonical UTS vocabulary.
QUBIT_OBSERVABLES: dict[str, str] = {
    "T1": "t1_us",
    "T2": "t2_us",
    "frequency": "frequency_ghz",
    "readout_error": "readout_error",
}

# SI value (after qiskit's apply_prefix) -> UTS target unit.
_SI_TO_UTS_SCALE: dict[str, float] = {
    "t1_us": 1e6,           # seconds -> microseconds
    "t2_us": 1e6,
    "frequency_ghz": 1e-9,  # Hz -> GHz
    "readout_error": 1.0,   # dimensionless
    "gate_error_2q": 1.0,
}

# Expected SI base unit per observable; None = dimensionless (unit must be "").
# qiskit's apply_prefix is prefix-only: it chokes on the bare base ("Hz") yet
# happily parses garbage like "parsec" as pico-"arsec", so the base is checked here.
_EXPECTED_BASE_UNIT: dict[str, str | None] = {
    "t1_us": "s",
    "t2_us": "s",
    "frequency_ghz": "Hz",
    "readout_error": None,
    "gate_error_2q": None,
}

SOURCE = "ibm_live"
PLATFORM = "ibm_sc"
CHANNEL = "claimed"


def parse_ts(value: Any) -> pd.Timestamp:
    """Parse an ISO string or datetime into a tz-aware UTC Timestamp (naive => UTC).

    Raises ValueError on None/NaT — pd.Timestamp(None) is silently NaT, and a NaT
    ts must never leak into the lake.
    """
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"unparseable timestamp: {value!r}")
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def snapshot_stamp(last_update_date: Any) -> str:
    """Filesystem-safe rendering of last_update_date: '20260712T041500Z'."""
    return parse_ts(last_update_date).strftime("%Y%m%dT%H%M%SZ")


def snapshot_path(raw_root: str | Path, backend_name: str, last_update_date: Any) -> Path:
    """<raw_root>/<backend>/<stamp>.json — the canonical raw-snapshot location."""
    return Path(raw_root) / backend_name / f"{snapshot_stamp(last_update_date)}.json"


def to_json_safe(obj: Any) -> Any:
    """Recursively convert datetimes to ISO strings so json.dump succeeds losslessly."""
    if isinstance(obj, (_dt.datetime, _dt.date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


def validate_raw_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    """Raise ValueError unless `snapshot` has the BackendProperties.to_dict() shape."""
    missing = [k for k in REQUIRED_SNAPSHOT_KEYS if k not in snapshot]
    if missing:
        raise ValueError(f"raw snapshot missing keys: {missing}")
    for key in ("qubits", "gates", "general"):
        if not isinstance(snapshot[key], list):
            raise ValueError(f"raw snapshot field {key!r} must be a list")
    return snapshot


def _to_uts_value(value: Any, unit: str, observable: str) -> float | None:
    """Nduv (value, unit) -> value in the observable's UTS unit; None = skip row.

    Skips (with a warning) rather than raises on an unrecognized/mismatched unit:
    the raw JSON keeps the full record either way, and one odd Nduv must not fail
    the backend — but a value whose scale we cannot trust must never enter the lake.
    """
    if value is None:
        return None
    si = float(value)
    base = _EXPECTED_BASE_UNIT[observable]
    if base is None:
        if unit:  # dimensionless observable carrying a unit: don't guess a scale
            log.warning("skipping %s Nduv with unexpected unit %r", observable, unit)
            return None
    elif unit != base:  # the bare base unit is already SI (apply_prefix chokes on "Hz")
        if not unit.endswith(base):
            log.warning("skipping %s Nduv with unrecognized unit %r", observable, unit)
            return None
        try:
            si = apply_prefix(si, unit)
        except ValueError:
            log.warning("skipping %s Nduv with unrecognized unit %r", observable, unit)
            return None
    out = si * _SI_TO_UTS_SCALE[observable]
    return out if math.isfinite(out) else None


def _faulty_sets(snapshot: Mapping[str, Any]) -> tuple[list[int], list[str]]:
    """Faulty qubits/gates from 'operational' Nduvs (mirrors BackendProperties.faulty_*)."""
    faulty_qubits = []
    for q, nduvs in enumerate(snapshot.get("qubits", [])):
        for nduv in nduvs:
            if nduv.get("name") == "operational" and not bool(nduv.get("value")):
                faulty_qubits.append(q)
    faulty_gates = []
    for gate in snapshot.get("gates", []):
        for param in gate.get("parameters", []):
            if param.get("name") == "operational" and not bool(param.get("value")):
                qubits = "-".join(str(q) for q in gate.get("qubits", []))
                faulty_gates.append(f"{gate.get('gate')}:{qubits}")
    return sorted(faulty_qubits), sorted(faulty_gates)


def from_backend_properties(
    props_dict: Mapping[str, Any],
    backend_name: str | None = None,
) -> pd.DataFrame:
    """Pure function: BackendProperties.to_dict() layout -> validated UTS DataFrame.

    Unmapped Nduvs (anharmonicity, readout_length, prob_meas*, gate_length, 1q gate
    errors, ...) are skipped: UTS forbids observables outside its vocabulary.
    """
    validate_raw_snapshot(props_dict)
    device = backend_name or str(props_dict["backend_name"])
    cal_ts = parse_ts(props_dict["last_update_date"])
    faulty_qubits, faulty_gates = _faulty_sets(props_dict)
    base_ctx = {
        "backend_version": props_dict.get("backend_version"),
        "faulty_qubits": faulty_qubits,
        "faulty_gates": faulty_gates,
    }

    frames: list[pd.DataFrame] = []
    for q, nduvs in enumerate(props_dict.get("qubits", [])):
        # Params of one qubit can carry different measurement dates: one UTS ts each.
        by_ts: dict[pd.Timestamp, list[tuple[str, str, float, None]]] = defaultdict(list)
        for nduv in nduvs:
            obs = QUBIT_OBSERVABLES.get(nduv.get("name", ""))
            if obs is None:
                continue
            value = _to_uts_value(nduv.get("value"), nduv.get("unit", ""), obs)
            if value is None:
                continue
            by_ts[parse_ts(nduv["date"])].append((obs, CHANNEL, value, None))
        for ts, rows in sorted(by_ts.items()):
            frames.append(make_events(
                source=SOURCE, platform=PLATFORM, device=device,
                entity_type="qubit", entity_id=f"q{q}", ts=ts,
                rows=rows, cal_ts=cal_ts, context=base_ctx,
            ))

    for gate in props_dict.get("gates", []):
        qubits = list(gate.get("qubits", []))
        if len(qubits) != 2:  # only two-qubit gates map to couplers in UTS
            continue
        for param in gate.get("parameters", []):
            if param.get("name") != "gate_error":
                continue
            value = _to_uts_value(param.get("value"), param.get("unit", ""), "gate_error_2q")
            if value is None:
                continue
            frames.append(make_events(
                source=SOURCE, platform=PLATFORM, device=device,
                entity_type="coupler", entity_id=f"c{qubits[0]}-{qubits[1]}",
                ts=parse_ts(param["date"]),
                rows=[("gate_error_2q", CHANNEL, value, None)],
                cal_ts=cal_ts, context={**base_ctx, "gate": gate.get("gate")},
            ))

    if not frames:
        return empty_uts()
    df = pd.concat(frames, ignore_index=True)
    return validate_uts(df)
