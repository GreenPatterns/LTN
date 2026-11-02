"""IBM fleet calibration harvester — TOKEN-READY ARTIFACT, never executed live [DEV-1].

Guarantee implemented (plan §2.3 A1/A5): given IBM Quantum credentials, one CLI run
enumerates all real operational backends, pulls calibration properties (current, or a
``properties(datetime=...)`` backfill loop), dedups snapshots on
(backend_name, last_update_date), and persists BOTH

  (a) the raw ``props.to_dict()`` JSON at  <out>/raw/<backend>/<stamp>.json, and
  (b) UTS events (source="ibm_live") into the hive lake at <out>/uts/

The exact same code path is exercised offline by ``--dry-run`` via
:class:`DryRunService` (2 fake backends x 3 qubits with realistic properties built
through the real ``qiskit_ibm_runtime.models.BackendProperties``). HTTP 429 is not
retried by qiskit-ibm-runtime 0.47.0, so every API call goes through
:func:`with_backoff` (exponential backoff, injectable sleep).

CLI:
    python -m ltn.harvest.ibm_harvester --out data/ibm_live [--dry-run]
        [--backfill-start 2026-06-01 --backfill-end 2026-07-01 --backfill-step-hours 24]
        [--instance <IBM Cloud CRN>]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

from ltn.data.uts import write_uts
from ltn.harvest.schemas import (
    from_backend_properties,
    snapshot_path,
    snapshot_stamp,
    to_json_safe,
)

log = logging.getLogger("ltn.harvest.ibm")

T = TypeVar("T")

DEFAULT_CHANNEL = "ibm_quantum_platform"  # the only non-cloud channel in runtime 0.47


# --------------------------------------------------------------------------- backoff

def with_backoff(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_status: tuple[int, ...] = (429,),
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> T:
    """Call fn(*args, **kwargs); on an exception whose .status_code is in
    `retry_status` (qiskit_ibm_runtime RequestsApiError carries one; the SDK does NOT
    auto-retry 429), retry with exponential backoff base_delay * 2**attempt, capped
    at max_delay. Any other exception — and exhaustion — re-raises."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — filtered by status_code below
            status = getattr(exc, "status_code", None)
            if status not in retry_status or attempt == max_retries:
                raise
            delay = min(max_delay, base_delay * 2 ** attempt)
            log.warning("http_retry status=%s attempt=%d/%d delay_s=%.1f fn=%s",
                        status, attempt + 1, max_retries, delay,
                        getattr(fn, "__name__", fn))
            sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


# --------------------------------------------------------------------- dry-run fleet

_DRYRUN_NOW = _dt.datetime(2026, 7, 12, 6, 0, 0, tzinfo=_dt.timezone.utc)


def _fake_properties_dict(
    backend_name: str,
    n_qubits: int,
    cal_ts: _dt.datetime,
    twoq_gate: str,
    faulty_qubit: int | None,
) -> dict[str, Any]:
    """Realistic BackendProperties.to_dict() payload, deterministic per (backend, day)."""
    def nduv(name: str, unit: str, value: float, minutes_before: int) -> dict[str, Any]:
        return {"date": cal_ts - _dt.timedelta(minutes=minutes_before),
                "name": name, "unit": unit, "value": value}

    qubits: list[list[dict[str, Any]]] = []
    for q in range(n_qubits):
        rng = random.Random(f"{backend_name}|q{q}|{cal_ts.date()}")
        t1 = rng.uniform(150.0, 350.0)
        t2 = min(rng.uniform(0.5, 1.5) * t1, 2.0 * t1)
        qubits.append([
            nduv("T1", "us", round(t1, 2), 17),
            nduv("T2", "us", round(t2, 2), 17),
            nduv("frequency", "GHz", round(rng.uniform(4.6, 5.2), 5), 5),
            nduv("anharmonicity", "GHz", round(rng.uniform(-0.36, -0.30), 5), 5),
            nduv("readout_error", "", round(rng.uniform(5e-3, 5e-2), 5), 5),
            nduv("prob_meas0_prep1", "", round(rng.uniform(5e-3, 6e-2), 5), 5),
            nduv("prob_meas1_prep0", "", round(rng.uniform(2e-3, 3e-2), 5), 5),
            nduv("readout_length", "ns", round(rng.uniform(660.0, 1500.0), 1), 5),
            nduv("operational", "", 0.0 if q == faulty_qubit else 1.0, 5),
        ])

    gates: list[dict[str, Any]] = []
    for q in range(n_qubits):
        rng = random.Random(f"{backend_name}|sx{q}|{cal_ts.date()}")
        gates.append({
            "qubits": [q], "gate": "sx", "name": f"sx{q}",
            "parameters": [nduv("gate_error", "", round(rng.uniform(1e-4, 8e-4), 7), 9),
                           nduv("gate_length", "ns", 60.0, 9)],
        })
    for a, b in [(0, 1), (1, 2)]:
        rng = random.Random(f"{backend_name}|{twoq_gate}{a}{b}|{cal_ts.date()}")
        operational = 0.0 if faulty_qubit in (a, b) and (a, b) == (1, 2) else 1.0
        gates.append({
            "qubits": [a, b], "gate": twoq_gate, "name": f"{twoq_gate}{a}_{b}",
            "parameters": [nduv("gate_error", "", round(rng.uniform(2e-3, 2e-2), 7), 9),
                           nduv("gate_length", "ns", round(rng.uniform(68.0, 660.0), 1), 9),
                           nduv("operational", "", operational, 9)],
        })

    return {
        "backend_name": backend_name,
        "backend_version": "1.0.0",
        "last_update_date": cal_ts,
        "qubits": qubits,
        "gates": gates,
        "general": [],
    }


class DryRunBackend:
    """Offline stand-in for qiskit_ibm_runtime.IBMBackend (BackendV2 shape)."""

    def __init__(self, name: str, n_qubits: int, twoq_gate: str,
                 now: _dt.datetime, faulty_qubit: int | None = None) -> None:
        self.name = name  # IBMBackend.name is a plain str attribute (BackendV2)
        self._n_qubits = n_qubits
        self._twoq_gate = twoq_gate
        self._now = now
        self._faulty_qubit = faulty_qubit

    def configuration(self) -> Any:
        from types import SimpleNamespace
        return SimpleNamespace(
            backend_name=self.name, n_qubits=self._n_qubits,
            coupling_map=[[0, 1], [1, 2]],
            basis_gates=["id", "rz", "sx", "x", self._twoq_gate],
            processor_type={"family": "DryRun", "revision": "1"},
            online_date=_dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc),
        )

    def properties(self, refresh: bool = False, datetime: _dt.datetime | None = None) -> Any:
        # Historical semantics mirror the real API: the snapshot closest-but-older
        # than `datetime`. Fake fleets recalibrate once a day at 01:23:45 UTC, so
        # sub-daily backfill steps hit the same snapshot (exercises dedup).
        if datetime is not None:
            dt = datetime if datetime.tzinfo else datetime.replace(tzinfo=_dt.timezone.utc)
            cal = dt.replace(hour=1, minute=23, second=45, microsecond=0)
            if cal >= dt:
                cal -= _dt.timedelta(days=1)
        else:
            cal = self._now.replace(hour=1, minute=23, second=45, microsecond=0)
        # Round-trip through the real SDK model so downstream code paths are
        # byte-identical with production (.to_dict(), .last_update_date, accessors).
        from qiskit_ibm_runtime.models import BackendProperties
        return BackendProperties.from_dict(_fake_properties_dict(
            self.name, self._n_qubits, cal, self._twoq_gate, self._faulty_qubit))


class DryRunService:
    """Offline stand-in for QiskitRuntimeService: 2 backends x 3 qubits."""

    def __init__(self, now: _dt.datetime = _DRYRUN_NOW) -> None:
        self._now = now

    def backends(self, simulator: bool = False, operational: bool = True,
                 **kwargs: Any) -> list[DryRunBackend]:
        return [
            DryRunBackend("dryrun_eagle", 3, "ecr", self._now),
            DryRunBackend("dryrun_heron", 3, "cz", self._now, faulty_qubit=2),
        ]

    def instances(self) -> list[dict[str, str]]:
        return [{"crn": "crn:v1:bluemix:public:quantum-computing:us-east:a/0:0::",
                 "plan": "open", "name": "dry-run"}]


# ----------------------------------------------------------------------------- auth

def authenticate(instance: str | None = None) -> Any:
    """Build a live QiskitRuntimeService from saved/env credentials.

    Setup one-liner (writes ~/.qiskit/qiskit-ibm.json; env vars QISKIT_IBM_TOKEN /
    QISKIT_IBM_INSTANCE / QISKIT_IBM_CHANNEL take precedence over the file):

        QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
            token=<API_TOKEN>, instance=<CRN>, set_as_default=True, overwrite=True)

    Passing instance=<CRN> here skips the SDK's instance-discovery call.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs: dict[str, Any] = {}
    if instance:
        kwargs["instance"] = instance
    service = QiskitRuntimeService(**kwargs)
    account = getattr(service, "active_account", lambda: None)() or {}
    log.info("authenticated channel=%s instance=%s",
             account.get("channel", DEFAULT_CHANNEL), account.get("instance"))
    return service


# --------------------------------------------------------------------- harvest core

@dataclass
class HarvestSummary:
    n_backends: int = 0
    n_snapshots: int = 0   # new (backend, last_update_date) snapshots persisted
    n_events: int = 0      # UTS rows written for the new snapshots
    n_deduped: int = 0     # pulls skipped: snapshot already seen (this run or on disk)
    n_missing: int = 0     # pulls that returned no properties (e.g. simulators)
    n_errors: int = 0      # backends that failed after retries


def _pull_properties(backend: Any, dt: _dt.datetime | None) -> Any:
    if dt is None:
        return backend.properties(refresh=True)
    # Docstring in 0.47 claims NotImplementedError for datetime=...; verified stale —
    # the call sends updated_before=<iso> and returns the closest-but-older snapshot.
    return backend.properties(datetime=dt)


def _backend_name(backend: Any) -> str:
    name = getattr(backend, "name", None)
    return name() if callable(name) else str(name)


def backfill_schedule(start: _dt.date, end: _dt.date, step_hours: int) -> list[_dt.datetime]:
    """UTC datetimes from start 00:00 to end 00:00 inclusive, every step_hours."""
    if step_hours <= 0:
        raise ValueError("step_hours must be positive")
    t = _dt.datetime.combine(start, _dt.time(0, 0), tzinfo=_dt.timezone.utc)
    end_t = _dt.datetime.combine(end, _dt.time(0, 0), tzinfo=_dt.timezone.utc)
    if end_t < t:
        raise ValueError("backfill end predates start")
    out = []
    while t <= end_t:
        out.append(t)
        t += _dt.timedelta(hours=step_hours)
    return out


def run_harvest(
    service: Any,
    out_dir: str | Path,
    backfill: Sequence[_dt.datetime] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> HarvestSummary:
    """Harvest every operational non-simulator backend of `service` into `out_dir`.

    Idempotent: snapshots already on disk (same backend_name + last_update_date) are
    skipped, and the UTS lake dedups at row level, so re-runs add nothing twice.
    """
    out = Path(out_dir)
    raw_root = out / "raw"
    lake_root = out / "uts"
    summary = HarvestSummary()
    seen: set[tuple[str, str]] = set()

    backends = with_backoff(service.backends, simulator=False, operational=True,
                            sleep=sleep)
    summary.n_backends = len(backends)
    log.info("fleet n_backends=%d backfill_steps=%d", len(backends),
             len(backfill or []))

    for backend in backends:
        name = _backend_name(backend)
        try:
            for dt in (backfill if backfill else [None]):
                props = with_backoff(_pull_properties, backend, dt, sleep=sleep)
                if props is None:  # simulators expose no properties
                    summary.n_missing += 1
                    log.info("no_properties backend=%s at=%s", name, dt)
                    continue
                stamp = snapshot_stamp(props.last_update_date)
                key = (name, stamp)
                raw_path = snapshot_path(raw_root, name, props.last_update_date)
                if key in seen or raw_path.exists():
                    summary.n_deduped += 1
                    log.info("dedup backend=%s cal=%s", name, stamp)
                    continue
                seen.add(key)

                props_dict = props.to_dict()
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_text(json.dumps(to_json_safe(props_dict), indent=1,
                                               sort_keys=True))
                events = from_backend_properties(props_dict, backend_name=name)
                write_uts(events, lake_root)
                summary.n_snapshots += 1
                summary.n_events += len(events)
                log.info("snapshot backend=%s cal=%s events=%d raw=%s",
                         name, stamp, len(events), raw_path)
        except Exception:
            summary.n_errors += 1
            log.exception("backend_failed backend=%s", name)

    log.info("done %s", " ".join(f"{k}={v}" for k, v in asdict(summary).items()))
    return summary


# ------------------------------------------------------------------------------ CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ibm_harvester",
        description="Harvest IBM fleet calibration properties into raw JSON + the "
                    "UTS parquet lake. Token-ready artifact [DEV-1]; use --dry-run "
                    "without credentials.")
    p.add_argument("--out", required=True, help="output dir (creates raw/ and uts/)")
    p.add_argument("--dry-run", action="store_true",
                   help="use the embedded offline fake fleet instead of IBM")
    p.add_argument("--backfill-start", metavar="YYYY-MM-DD", type=_dt.date.fromisoformat,
                   help="historical backfill window start (UTC)")
    p.add_argument("--backfill-end", metavar="YYYY-MM-DD", type=_dt.date.fromisoformat,
                   help="historical backfill window end (UTC, inclusive)")
    p.add_argument("--backfill-step-hours", type=int, default=24,
                   help="spacing of properties(datetime=...) pulls (default 24)")
    p.add_argument("--instance", default=None,
                   help="IBM Cloud CRN; passing it skips instance discovery")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if (args.backfill_start is None) != (args.backfill_end is None):
        parser.error("--backfill-start and --backfill-end must be given together")
    backfill = None
    if args.backfill_start is not None:
        backfill = backfill_schedule(args.backfill_start, args.backfill_end,
                                     args.backfill_step_hours)

    service = DryRunService() if args.dry_run else authenticate(args.instance)
    summary = run_harvest(service, args.out, backfill=backfill)
    print(json.dumps(asdict(summary)))
    return 0 if summary.n_errors == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
