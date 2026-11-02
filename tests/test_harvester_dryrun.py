"""Offline tests for the token-ready IBM harvester [DEV-1].

Covers: (1) --dry-run end-to-end (raw JSON snapshots on disk, UTS lake validates,
re-run dedups at snapshot level), (2) the pure BackendProperties.to_dict() -> UTS
mapping on a handcrafted snapshot (values, unit conversion, cal_ts, faulty-set
context), (3) the exponential-backoff wrapper around a stubbed HTTP 429.
No network, no credentials, no marks needed — everything runs in tmp_path.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from ltn.data.uts import read_uts, validate_uts
from ltn.harvest.ibm_harvester import (
    DryRunService,
    backfill_schedule,
    main,
    run_harvest,
    with_backoff,
)
from ltn.harvest.schemas import from_backend_properties, parse_ts, snapshot_stamp

UTC = dt.timezone.utc
SNAPSHOT_KEYS = {"backend_name", "backend_version", "last_update_date",
                 "qubits", "gates", "general"}


# ------------------------------------------------------------- dry-run end-to-end

def test_dryrun_end_to_end(tmp_path):
    out = tmp_path / "ibm_live"
    assert main(["--out", str(out), "--dry-run"]) == 0

    # (a) raw snapshots: one JSON per (backend, last_update_date), exact to_dict shape
    raw_files = sorted((out / "raw").glob("*/*.json"))
    assert {p.parent.name for p in raw_files} == {"dryrun_eagle", "dryrun_heron"}
    assert len(raw_files) == 2
    for f in raw_files:
        snap = json.loads(f.read_text())
        assert set(snap) == SNAPSHOT_KEYS
        assert len(snap["qubits"]) >= 3
        assert f.stem == snapshot_stamp(snap["last_update_date"])  # dedup key on disk

    # (b) UTS lake: validates, right vocabulary, right entities
    lake = read_uts(out / "uts")
    validate_uts(lake)
    assert set(lake["source"]) == {"ibm_live"}
    assert set(lake["platform"]) == {"ibm_sc"}
    assert set(lake["channel"]) == {"claimed"}
    assert set(lake["observable"]) == {
        "t1_us", "t2_us", "frequency_ghz", "readout_error", "gate_error_2q"}
    # per backend: 3 qubits x 4 observables + 2 couplers x gate_error_2q
    assert len(lake) == 2 * (3 * 4 + 2)
    qubits = lake[lake["entity_type"] == "qubit"]
    couplers = lake[lake["entity_type"] == "coupler"]
    assert set(qubits["entity_id"]) == {"q0", "q1", "q2"}
    assert set(couplers["entity_id"]) == {"c0-1", "c1-2"}
    assert lake["value"].dtype == "float64"
    assert lake["cal_ts"].notna().all()
    # dryrun_heron ships qubit 2 faulty; the sets land in every heron row's context
    heron_ctx = json.loads(lake.loc[lake["device"] == "dryrun_heron", "context"].iloc[0])
    assert heron_ctx["faulty_qubits"] == [2]
    assert heron_ctx["faulty_gates"] == ["cz:1-2"]
    eagle_ctx = json.loads(lake.loc[lake["device"] == "dryrun_eagle", "context"].iloc[0])
    assert eagle_ctx["faulty_qubits"] == []


def test_dryrun_rerun_dedups(tmp_path):
    out = tmp_path / "ibm_live"
    s1 = run_harvest(DryRunService(), out)
    assert (s1.n_backends, s1.n_snapshots, s1.n_deduped, s1.n_errors) == (2, 2, 0, 0)
    assert s1.n_events == 28
    lake1 = read_uts(out / "uts")
    raw1 = sorted(p.name for p in (out / "raw").glob("*/*.json"))

    s2 = run_harvest(DryRunService(), out)  # identical second run
    assert (s2.n_snapshots, s2.n_events, s2.n_deduped) == (0, 0, 2)
    lake2 = read_uts(out / "uts")
    raw2 = sorted(p.name for p in (out / "raw").glob("*/*.json"))
    assert raw2 == raw1
    pd.testing.assert_frame_equal(lake1, lake2)  # lake unchanged, byte-stable


def test_dryrun_backfill_dedups_subdaily_steps(tmp_path):
    # Fake fleet recalibrates daily at 01:23:45 UTC -> 6h steps hit the same
    # snapshot several times; dedup must collapse them within a single run.
    sched = backfill_schedule(dt.date(2026, 7, 1), dt.date(2026, 7, 3), 6)
    assert len(sched) == 9
    assert sched[0] == dt.datetime(2026, 7, 1, tzinfo=UTC)
    assert sched[-1] == dt.datetime(2026, 7, 3, tzinfo=UTC)
    s = run_harvest(DryRunService(), tmp_path / "o", backfill=sched)
    # per backend: 9 pulls -> 3 distinct calibration days (Jun 30, Jul 1, Jul 2)
    assert (s.n_snapshots, s.n_deduped, s.n_errors) == (6, 12, 0)
    lake = read_uts(tmp_path / "o" / "uts")
    validate_uts(lake)
    assert len(lake) == 6 * 14
    assert lake.groupby("device")["cal_ts"].nunique().eq(3).all()


def test_one_failing_backend_does_not_kill_fleet(tmp_path):
    class BoomBackend:
        name = "boom"

        def properties(self, refresh=False, datetime=None):
            raise RuntimeError("explode")

    class Svc(DryRunService):
        def backends(self, **kwargs):
            return [BoomBackend()] + super().backends(**kwargs)

    s = run_harvest(Svc(), tmp_path / "o")
    assert (s.n_backends, s.n_errors, s.n_snapshots) == (3, 1, 2)


# --------------------------------------------------- pure snapshot -> UTS mapping

def _handcrafted_snapshot() -> dict:
    """Minimal-but-real BackendProperties.to_dict() JSON (post round-trip: ISO strings)."""
    d0 = "2026-07-10T03:50:00+00:00"
    d1 = "2026-07-10T04:00:00+00:00"
    return {
        "backend_name": "fake_kolkata",
        "backend_version": "1.4.2",
        "last_update_date": "2026-07-10T04:15:00+00:00",
        "qubits": [
            [
                {"date": d0, "name": "T1", "unit": "us", "value": 217.3},
                {"date": d0, "name": "T2", "unit": "us", "value": 141.8},
                {"date": d1, "name": "frequency", "unit": "GHz", "value": 5.0123},
                {"date": d1, "name": "readout_error", "unit": "", "value": 0.012},
                {"date": d1, "name": "anharmonicity", "unit": "GHz", "value": -0.34},
                {"date": d1, "name": "operational", "unit": "", "value": 1},
            ],
            [
                {"date": d0, "name": "T1", "unit": "ms", "value": 0.2001},
                {"date": d1, "name": "frequency", "unit": "Hz", "value": 4.9e9},
                {"date": d1, "name": "T2", "unit": "parsec", "value": 1.0},
                {"date": d1, "name": "operational", "unit": "", "value": 0},
            ],
        ],
        "gates": [
            {"qubits": [0], "gate": "sx", "name": "sx0",
             "parameters": [{"date": d1, "name": "gate_error", "unit": "", "value": 3e-4}]},
            {"qubits": [0, 1], "gate": "ecr", "name": "ecr0_1",
             "parameters": [
                 {"date": d0, "name": "gate_error", "unit": "", "value": 7.5e-3},
                 {"date": d0, "name": "gate_length", "unit": "ns", "value": 533.0},
                 {"date": d0, "name": "operational", "unit": "", "value": 0},
             ]},
        ],
        "general": [],
    }


def test_from_backend_properties_values_units_and_context():
    df = from_backend_properties(_handcrafted_snapshot(), backend_name="fake_kolkata")
    validate_uts(df)
    assert set(df["device"]) == {"fake_kolkata"}
    assert (df["channel"] == "claimed").all()
    assert df["sigma"].isna().all()  # IBM publishes point values, no uncertainty
    assert df["value"].dtype == "float64"

    def val(entity_id, obs):
        rows = df[(df["entity_id"] == entity_id) & (df["observable"] == obs)]
        assert len(rows) == 1, f"{entity_id}/{obs}: {len(rows)} rows"
        return rows.iloc[0]

    # q0: native units pass through
    assert val("q0", "t1_us")["value"] == pytest.approx(217.3)
    assert val("q0", "t2_us")["value"] == pytest.approx(141.8)
    assert val("q0", "frequency_ghz")["value"] == pytest.approx(5.0123)
    assert val("q0", "readout_error")["value"] == pytest.approx(0.012)
    # q1: ms -> us, bare-SI "Hz" -> GHz; the "parsec" T2 is skipped, not fatal
    assert val("q1", "t1_us")["value"] == pytest.approx(200.1)
    assert val("q1", "frequency_ghz")["value"] == pytest.approx(4.9)
    assert df[(df["entity_id"] == "q1") & (df["observable"] == "t2_us")].empty
    # coupler: only the 2q gate_error maps; 1q sx and gate_length stay out
    c = val("c0-1", "gate_error_2q")
    assert c["value"] == pytest.approx(7.5e-3)
    assert c["entity_type"] == "coupler"
    assert json.loads(c["context"])["gate"] == "ecr"
    assert set(df["observable"]) == {
        "t1_us", "t2_us", "frequency_ghz", "readout_error", "gate_error_2q"}

    # timestamps: ts = each Nduv's own date, cal_ts = last_update_date everywhere
    assert val("q0", "t1_us")["ts"] == parse_ts("2026-07-10T03:50:00+00:00")
    assert val("q0", "frequency_ghz")["ts"] == parse_ts("2026-07-10T04:00:00+00:00")
    assert (df["cal_ts"] == parse_ts("2026-07-10T04:15:00+00:00")).all()

    # faulty sets (from 'operational' Nduvs) in every row's context
    for ctx_json in df["context"].unique():
        ctx = json.loads(ctx_json)
        assert ctx["faulty_qubits"] == [1]
        assert ctx["faulty_gates"] == ["ecr:0-1"]
        assert ctx["backend_version"] == "1.4.2"


def test_from_backend_properties_matches_real_sdk_roundtrip():
    """The mapping is identical on the SDK-object path (datetime objects, no JSON)."""
    from qiskit_ibm_runtime.models import BackendProperties

    raw = _handcrafted_snapshot()
    # BackendProperties.from_dict itself rejects the bare "Hz" unit (its internal
    # _apply_prefix chokes on it), so the SDK path only ever sees prefixed units.
    raw["qubits"][1] = [n for n in raw["qubits"][1] if n["unit"] not in ("Hz", "parsec")]
    props = BackendProperties.from_dict(raw)
    assert snapshot_stamp(props.last_update_date) == "20260710T041500Z"
    df_obj = from_backend_properties(props.to_dict(), backend_name="fake_kolkata")
    df_json = from_backend_properties(raw, backend_name="fake_kolkata")
    pd.testing.assert_frame_equal(
        df_obj.reset_index(drop=True), df_json.reset_index(drop=True))
    # SDK accessor agrees with our converted value (t1 in seconds -> us)
    assert props.t1(0) * 1e6 == pytest.approx(217.3)


def test_from_backend_properties_rejects_malformed():
    with pytest.raises(ValueError, match="missing keys"):
        from_backend_properties({"backend_name": "x"})
    with pytest.raises(ValueError, match="unparseable timestamp"):
        parse_ts(None)


# ------------------------------------------------------------------ 429 backoff

class _ApiError(Exception):
    """Stub of qiskit_ibm_runtime RequestsApiError: carries .status_code."""

    def __init__(self, status: int) -> None:
        super().__init__(f"http {status}")
        self.status_code = status


def test_with_backoff_retries_429_then_succeeds():
    calls: list[int] = []
    slept: list[float] = []

    def flaky(tag: str) -> str:
        calls.append(1)
        if len(calls) < 3:
            raise _ApiError(429)
        return f"props:{tag}"

    out = with_backoff(flaky, "eagle", base_delay=0.5, sleep=slept.append)
    assert out == "props:eagle"
    assert len(calls) == 3
    assert slept == [0.5, 1.0]  # exponential, no real sleeping in tests


def test_with_backoff_reraises_non_retryable_immediately():
    def denied() -> None:
        raise _ApiError(401)

    with pytest.raises(_ApiError):
        with_backoff(denied, sleep=lambda s: pytest.fail("must not sleep on 401"))


def test_with_backoff_exhausts_and_caps_delay():
    slept: list[float] = []

    def always_throttled() -> None:
        raise _ApiError(429)

    with pytest.raises(_ApiError):
        with_backoff(always_throttled, max_retries=3, base_delay=1.0, max_delay=2.0,
                     sleep=slept.append)
    assert slept == [1.0, 2.0, 2.0]  # capped at max_delay; final attempt re-raises
