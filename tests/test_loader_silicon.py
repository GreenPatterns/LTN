"""Tests for the silicon spin-qubit (Zenodo 15653466) -> UTS loader (plan A3, §2.3).

Checks the loader contract: validate_uts-clean output, devices simos_a and
simos_b both present with >=10-point time-series observables, physical ranges
(cz_fidelity in (0.9, 1], gate_error_2q / epc_1q in [0, 0.1), t2_us in
microseconds), real vs pseudo timestamps, context provenance (source file +
measurement type), and idempotent lazy extraction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import requires_raw
from ltn.data import uts
from ltn.data.loaders.silicon import (
    DEVICES,
    FBT_TIME_STEP_MIN,
    PSEUDO_BASE_TS,
    clip_fidelity,
    load_silicon,
    parse_anchor_ts,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
SILICON = RAW / "silicon"

silicon_data = [pytest.mark.data, requires_raw("silicon/AncillaryFiles.zip")]


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    return load_silicon(SILICON)


# ---------------------------------------------------------------- unit (no raw data)


def test_parse_anchor_ts() -> None:
    ts = parse_anchor_ts("alder_26008_2022_06_02_17_38_13_2Q_RBM_single_cptp_go")
    assert ts == pd.Timestamp("2022-06-02 17:38:13", tz="UTC")
    assert parse_anchor_ts("no_date_here") is None


def test_clip_fidelity() -> None:
    assert clip_fidelity(0.9937) == (0.9937, None)
    clipped, raw = clip_fidelity(1.0001)
    assert clipped == 1.0 and raw == pytest.approx(1.0001)


def test_missing_raw_dir_returns_empty(tmp_path: Path) -> None:
    with pytest.warns(UserWarning, match=r"\[silicon\] missing archive"):
        df = load_silicon(tmp_path)
    assert len(df) == 0
    uts.validate_uts(df)


# ------------------------------------------------------------------------ with data


class TestSiliconData:
    pytestmark = silicon_data

    def test_loads_clean(self, events: pd.DataFrame) -> None:
        assert len(events) > 1000
        uts.validate_uts(events)
        assert set(events["source"]) == {"silicon"}
        assert set(events["platform"]) == {"si_spin"}
        assert set(events["entity_type"]) <= {"device", "qubit"}

    def test_both_devices_present(self, events: pd.DataFrame) -> None:
        devices = set(events["device"])
        assert {"simos_a", "simos_b"} <= devices
        assert devices <= {name for name, _ in DEVICES.values()}

    def test_time_series_per_device(self, events: pd.DataFrame) -> None:
        """>=1 time-series observable with >=10 distinct-time points per device."""
        for dev in ("simos_a", "simos_b"):
            sub = events[events["device"] == dev]
            n_ts = sub.groupby("observable", observed=True)["ts"].nunique()
            assert (n_ts >= 10).any(), f"{dev}: no >=10-point series ({n_ts.to_dict()})"
            # The FBT CZ fidelity trace specifically must be a long series.
            cz = sub[(sub["observable"] == "cz_fidelity") & (sub["channel"] == "delivered")]
            assert cz["ts"].nunique() >= 100

    def test_fidelity_physical_range(self, events: pd.DataFrame) -> None:
        fid = events.loc[events["observable"] == "cz_fidelity", "value"]
        assert len(fid) > 0
        assert (fid > 0.9).all() and (fid <= 1.0).all()

    def test_error_physical_range(self, events: pd.DataFrame) -> None:
        for obs in ("gate_error_2q", "epc_1q"):
            err = events.loc[events["observable"] == obs, "value"]
            assert len(err) > 0, f"no {obs} events"
            assert (err >= 0.0).all() and (err < 0.1).all(), obs

    def test_t2_in_microseconds(self, events: pd.DataFrame) -> None:
        t2 = events[events["observable"] == "t2_us"]
        assert {"simos_a", "simos_b"} <= set(t2["device"])
        assert (t2["value"] > 0.1).all() and (t2["value"] < 1e4).all()
        assert set(t2["channel"]) == {"probe"}

    def test_channels_and_entities(self, events: pd.DataFrame) -> None:
        by_obs = events.groupby("observable", observed=True)
        assert set(by_obs.get_group("cz_fidelity")["channel"]) == {"delivered"}
        assert set(by_obs.get_group("gate_error_2q")["channel"]) == {"derived"}
        epc = by_obs.get_group("epc_1q")
        assert set(epc["entity_type"]) == {"qubit"}
        assert set(epc["entity_id"]) == {"q1", "q2"}

    def test_device_b_trace_has_real_timestamps(self, events: pd.DataFrame) -> None:
        cz_b = events[
            (events["device"] == "simos_b")
            & (events["observable"] == "cz_fidelity")
            & (events["channel"] == "delivered")
        ]
        ctx = cz_b["context"].map(json.loads)
        real = cz_b[~ctx.map(lambda c: c["pseudo_time"])]
        assert len(real) >= 400  # 500-point FBT trace with real lab-time minutes
        # Anchored at the alder run start (2022-06-02 17:38:13 UTC), ~256 min span.
        assert real["ts"].min() == pd.Timestamp("2022-06-02 17:38:13", tz="UTC")
        span_min = (real["ts"].max() - real["ts"].min()).total_seconds() / 60.0
        assert 200.0 < span_min < 300.0

    def test_pseudo_time_flag_and_base(self, events: pd.DataFrame) -> None:
        ctx = events["context"].map(json.loads)
        assert ctx.map(lambda c: "pseudo_time" in c).all()
        pseudo = events[ctx.map(lambda c: c["pseudo_time"])]
        assert len(pseudo) > 0
        # Episodic (non-trace) pseudo events sit on the mandated 2023 base grid.
        t2 = pseudo[pseudo["observable"] == "t2_us"]
        assert (t2["ts"] >= PSEUDO_BASE_TS).all()
        assert ((t2["ts"] - PSEUDO_BASE_TS).dt.total_seconds() % 86400 == 0).all()

    def test_context_provenance(self, events: pd.DataFrame) -> None:
        ctx = events["context"].map(json.loads)
        assert ctx.map(lambda c: bool(c.get("file"))).all()
        assert ctx.map(lambda c: bool(c.get("measurement"))).all()

    def test_sigma_present_for_device_b_cz_trace(self, events: pd.DataFrame) -> None:
        cz_b = events[
            (events["device"] == "simos_b")
            & (events["observable"] == "cz_fidelity")
            & (events["context"].str.contains("fbt_fidelity_trace"))
        ]
        assert cz_b["sigma"].notna().all()
        assert (cz_b["sigma"] > 0).all() and (cz_b["sigma"] < 0.05).all()

    def test_extraction_lazy_and_idempotent(self, events: pd.DataFrame) -> None:
        extracted = SILICON / "extracted"
        assert extracted.is_dir()
        mtimes = {p: p.stat().st_mtime_ns for p in extracted.rglob("*") if p.is_file()}
        assert mtimes, "nothing was extracted"
        again = load_silicon(SILICON)
        # No re-extraction on a second run, and the result is identical.
        assert {p: p.stat().st_mtime_ns for p in extracted.rglob("*") if p.is_file()} == mtimes
        pd.testing.assert_frame_equal(again, events)

    def test_fbt_pseudo_traces_use_borrowed_cadence(self, events: pd.DataFrame) -> None:
        cz_a = events[
            (events["device"] == "simos_a")
            & (events["observable"] == "cz_fidelity")
            & (events["channel"] == "delivered")
            & (events["context"].str.contains("fbt_fidelity_trace"))
        ].sort_values("ts")
        assert len(cz_a) >= 100
        ctx0 = json.loads(cz_a["context"].iloc[0])
        assert ctx0["pseudo_time"] is True
        assert ctx0["time_step_min"] == FBT_TIME_STEP_MIN
        deltas = cz_a["ts"].diff().dropna().dt.total_seconds() / 60.0
        assert (abs(deltas - FBT_TIME_STEP_MIN) < 1e-6).all()
