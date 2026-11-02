"""DAQEC v2 multi-platform validation package (Zenodo 18045661 v2) -> UTS loader.

Converts the per-episode JSON validation results under ``data/raw/daqec_v2/``
into episodic, device-level UTS events (``entity_type="device"``,
``entity_id="device"``). Only canonical observables are emitted — currently
``logical_error_rate`` (channel ``delivered``) with a binomial sigma
sqrt(p*(1-p)/shots) — one event per experiment/condition record.

Pseudo-time: these records lack a real timeline (one-off cloud validation
shots), so ``ts = V2_BASE_TS + file_index days + episode_index minutes`` where
file_index is the file's position in the sorted listing of all ``*.json`` files
in the raw dir. Every event carries ``context.pseudo_time=true``; any original
timestamp found in the JSON is preserved as ``context.recorded_timestamp``.

Platform mapping: ibm_* -> ibm_sc, iqm* -> iqm_sc, ionq* -> ionq_trap,
rigetti* -> rigetti_sc.

Unusable files are listed (with reasons) in the module-level ``SKIPPED`` dict
and never emitted: the two IQM files carry only aggregate interaction
statistics (an LER *difference*, not a canonical observable), and one Rigetti
file is a LocalSimulator circuit-correctness check, not hardware telemetry.

Credibility caveat [DEV-10]: only per-record telemetry (shot counts, LERs) is
consumed; the package authors' adaptive-vs-baseline conclusions are NOT relied
on (their own ibm_fez validation shows chance-level LERs).
"""
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import pandas as pd

from ltn.data import uts

# Documented fixed base date for pseudo-timestamps (after the Dec-2025 runs).
V2_BASE_TS = pd.Timestamp("2026-01-01T00:00:00Z")

# Known-unusable JSON files: filename -> reason. Never converted to events.
SKIPPED: dict[str, str] = {
    "iqm_validation_CANONICAL_RESULT.json": (
        "aggregate interaction statistics only (mean LER difference across 80 runs); "
        "no per-run canonical observable to emit"
    ),
    "iqm_v4_combined_analysis.json": (
        "aggregate interaction statistics only (n_runs/cohens_d summary); "
        "no per-run canonical observable to emit"
    ),
    "rigetti_validation_20251228_143706.json": (
        "device=LocalSimulator circuit-correctness check, not hardware telemetry"
    ),
}

_PLATFORM_PREFIXES: list[tuple[str, str]] = [
    ("ibm", "ibm_sc"),
    ("iqm", "iqm_sc"),
    ("ionq", "ionq_trap"),
    ("rigetti", "rigetti_sc"),
]


def _normalize_device(name: str) -> str:
    """'IonQ Forte-1' -> 'ionq_forte1', 'Rigetti Ankaa-3' -> 'rigetti_ankaa3'."""
    return re.sub(r"\s+", "_", name.lower().replace("-", "").strip())


def _platform_for(device: str) -> str | None:
    for prefix, platform in _PLATFORM_PREFIXES:
        if device.startswith(prefix):
            return platform
    return None


def _binomial_sigma(p: float, shots: float) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / shots) if shots > 0 else 0.0


def _ler_records(payload: dict) -> tuple[str | None, list[dict]]:
    """Extract (device_name_or_None, per-episode records with an 'ler' key)."""
    if "experiments" in payload:  # IonQ interaction-pair files
        return payload.get("device"), list(payload["experiments"])
    if "conditions" in payload:  # Rigetti interaction-effect files
        return payload.get("device"), list(payload["conditions"])
    if "single_test" in payload:  # Rigetti 10-shot smoke test; device from filename
        return None, [payload["single_test"]]
    return None, []


def _episode_context(rec: dict, payload: dict, fname: str) -> dict:
    ctx: dict = {"pseudo_time": True, "source_file": fname}
    recorded = payload.get("timestamp") or rec.get("timestamp")
    if recorded:
        ctx["recorded_timestamp"] = str(recorded)
    for key in (
        "method", "description", "distance", "rounds", "noise_regime",
        "noise_level", "circuit_depth", "qubit_mapping", "task_id",
        "correct_count", "total_shots",
    ):
        if key in rec:
            ctx[key] = rec[key]
    for key in ("n_data_qubits", "n_rounds", "noise_level"):
        if key in payload and key not in ctx:
            ctx[key] = payload[key]
    return ctx


def load_daqec_v2(raw_dir: str | Path) -> pd.DataFrame:
    """Load the daqec_v2 validation JSONs into a validated UTS DataFrame."""
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.json"))
    frames: list[pd.DataFrame] = []

    for file_index, path in enumerate(files):
        if path.name in SKIPPED:
            continue
        payload = json.loads(path.read_text())
        device_name, records = _ler_records(payload)
        if not records:
            warnings.warn(f"daqec_v2: unrecognized JSON structure, skipping {path.name}")
            continue
        if device_name is None:
            # Smoke-test file carries no device field; infer from filename prefix
            # (rigetti_validation_*.json ran on the Ankaa-3 Braket ARN).
            device = "rigetti_ankaa3" if path.name.startswith("rigetti") else _normalize_device(path.stem)
        else:
            device = _normalize_device(device_name)
        platform = _platform_for(device)
        if platform is None:
            warnings.warn(f"daqec_v2: unknown platform for device {device!r}, skipping {path.name}")
            continue

        for episode_index, rec in enumerate(records):
            if "ler" not in rec:
                continue
            ler = float(rec["ler"])
            shots = float(rec.get("total_shots") or rec.get("shots") or 0)
            ctx = _episode_context(rec, payload, path.name)
            if device_name is None:
                ctx["device_inferred_from_filename"] = True
            ts = V2_BASE_TS + pd.Timedelta(days=file_index) + pd.Timedelta(minutes=episode_index)
            frames.append(
                uts.make_events(
                    source="daqec_v2",
                    platform=platform,
                    device=device,
                    entity_type="device",
                    entity_id="device",
                    ts=ts,
                    cal_ts=None,
                    rows=[("logical_error_rate", "delivered", ler, _binomial_sigma(ler, shots))],
                    context=ctx,
                )
            )

    if not frames:
        return uts.empty_uts()
    df = pd.concat(frames, ignore_index=True)
    return uts.validate_uts(df)
