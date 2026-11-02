"""Tests for the DAQEC / DAQEC-v2 -> UTS loaders (plan step A2, §2.3).

Checks the loader contract: canonical vocabulary (validate_uts), exact event
counts derived independently from the observable mapping, claimed/probe channel
pairing, stable subgraph ids, LER sigmas from the 95% CI, and v2 pseudo-time
episodic events.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from conftest import requires_raw
from ltn.data import uts
from ltn.data.loaders.daqec import OBSERVABLE_MAP, _parse_qubits, load_daqec
from ltn.data.loaders.daqec_v2 import SKIPPED, V2_BASE_TS, load_daqec_v2

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

daqec_data = [pytest.mark.data, requires_raw("daqec/master.parquet")]
daqec_v2_data = [pytest.mark.data, requires_raw("daqec_v2")]


@pytest.fixture(scope="module")
def daqec_events() -> pd.DataFrame:
    return load_daqec(RAW / "daqec")


@pytest.fixture(scope="module")
def master() -> pd.DataFrame:
    return pd.read_parquet(RAW / "daqec" / "master.parquet")


@pytest.fixture(scope="module")
def v2_events() -> pd.DataFrame:
    return load_daqec_v2(RAW / "daqec_v2")


# ---------------------------------------------------------------- unit (no raw data)


def test_subgraph_id_stable_and_order_invariant() -> None:
    a = uts.subgraph_id([12, 14, 16])
    assert a == uts.subgraph_id([16, 12, 14]) == uts.subgraph_id((14, 16, 12))
    assert a != uts.subgraph_id([12, 14, 17])
    assert a.startswith("sub:")


def test_parse_qubits_string_and_sequence() -> None:
    assert _parse_qubits("[0, 1, 2]") == [0, 1, 2]
    assert _parse_qubits([3, 4]) == [3, 4]


# ---------------------------------------------------------------------------- daqec


def test_observable_map_targets_canonical_vocab() -> None:
    for _, obs, channel in OBSERVABLE_MAP:
        assert obs in uts.OBSERVABLES
        assert channel in uts.CHANNELS


class TestDaqec:
    pytestmark = daqec_data

    def test_validates_and_vocab(self, daqec_events: pd.DataFrame) -> None:
        uts.validate_uts(daqec_events)  # raises on any vocab/schema violation
        assert set(daqec_events["source"]) == {"daqec"}
        assert set(daqec_events["platform"]) == {"ibm_sc"}
        assert set(daqec_events["device"]) == {"ibm_brisbane", "ibm_kyoto", "ibm_osaka"}
        assert set(daqec_events["entity_type"]) == {"device", "subgraph"}
        assert set(daqec_events["channel"]) == {"claimed", "probe", "delivered", "derived"}

    def test_exact_event_count_from_mapping(
        self, daqec_events: pd.DataFrame, master: pd.DataFrame
    ) -> None:
        assert len(master) == 756
        mapped_cols = [col for col, _, _ in OBSERVABLE_MAP]
        # one event per non-NaN mapped cell, duplicated at subgraph + device level
        expected = 2 * int(master[mapped_cols].notna().sum().sum())
        assert len(daqec_events) == expected
        by_ent = daqec_events["entity_type"].value_counts()
        assert by_ent["device"] == by_ent["subgraph"] == expected // 2

    def test_claimed_and_probe_channels_pair(self, daqec_events: pd.DataFrame) -> None:
        dev = daqec_events[daqec_events["entity_type"] == "device"]
        probe = dev[dev["channel"] == "probe"]
        assert len(probe) > 0
        claimed_keys = set(
            map(tuple, dev[dev["channel"] == "claimed"][
                ["device", "ts", "observable", "context"]
            ].itertuples(index=False, name=None))
        )
        # every probe observation has a claimed counterpart from the same run
        for key in probe[["device", "ts", "observable", "context"]].itertuples(
            index=False, name=None
        ):
            assert tuple(key) in claimed_keys

    def test_subgraph_ids_stable_across_repeated_subsets(
        self, daqec_events: pd.DataFrame, master: pd.DataFrame
    ) -> None:
        sub = daqec_events[daqec_events["entity_type"] == "subgraph"]
        raw_subsets = {tuple(sorted(_parse_qubits(s))) for s in master["data_qubits"]}
        # recurring subsets collapse onto exactly one entity id each
        assert sub["entity_id"].nunique() == len(raw_subsets)
        expected_ids = {uts.subgraph_id(q) for q in raw_subsets}
        assert set(sub["entity_id"]) == expected_ids

    def test_ler_sigma_from_ci(self, daqec_events: pd.DataFrame, master: pd.DataFrame) -> None:
        ler = daqec_events[daqec_events["observable"] == "logical_error_rate"]
        assert ler["channel"].eq("delivered").all()
        assert ler["sigma"].notna().all()
        assert (ler["sigma"] >= 0).all()
        row = master.iloc[0]
        expect = (
            row["logical_error_rate_ci_upper"] - row["logical_error_rate_ci_lower"]
        ) / 3.92
        got = ler[
            ler["context"].str.contains(f'"run_id": "{row["run_id"]}"')
        ]["sigma"].unique()
        assert len(got) == 1 and got[0] == pytest.approx(expect)
        # non-LER events carry no sigma in this source
        assert daqec_events[daqec_events["observable"] != "logical_error_rate"]["sigma"].isna().all()

    def test_context_fields_and_timestamps(self, daqec_events: pd.DataFrame) -> None:
        ctx = json.loads(daqec_events["context"].iloc[0])
        for key in ("run_id", "session_id", "strategy", "distance", "syndrome_rounds",
                    "day", "data_qubits", "couplers"):
            assert key in ctx
        assert daqec_events["ts"].notna().all()
        assert daqec_events["cal_ts"].notna().all()
        assert str(daqec_events["ts"].dt.tz) == "UTC"
        # cal_ts precedes the run timestamp (calibration happened before execution)
        assert (daqec_events["cal_ts"] <= daqec_events["ts"]).all()


# ------------------------------------------------------------------------- daqec_v2


class TestDaqecV2:
    pytestmark = daqec_v2_data

    def test_loads_and_validates(self, v2_events: pd.DataFrame) -> None:
        uts.validate_uts(v2_events)
        assert len(v2_events) > 0
        assert set(v2_events["source"]) == {"daqec_v2"}
        assert set(v2_events["entity_type"]) == {"device"}
        assert set(v2_events["entity_id"]) == {"device"}
        assert set(v2_events["observable"]) <= set(uts.OBSERVABLES)
        assert set(v2_events["platform"]) <= {"ibm_sc", "iqm_sc", "ionq_trap", "rigetti_sc"}

    def test_pseudo_time_and_context(self, v2_events: pd.DataFrame) -> None:
        assert (v2_events["ts"] >= V2_BASE_TS).all()
        contexts = [json.loads(c) for c in v2_events["context"]]
        assert all(c.get("pseudo_time") is True for c in contexts)
        assert all("source_file" in c for c in contexts)

    def test_skipped_files_not_emitted(self, v2_events: pd.DataFrame) -> None:
        emitted = {json.loads(c)["source_file"] for c in v2_events["context"]}
        assert emitted.isdisjoint(SKIPPED.keys())
        # skip reasons documented for every skipped file
        assert all(reason for reason in SKIPPED.values())

    def test_ler_events_have_sigma(self, v2_events: pd.DataFrame) -> None:
        ler = v2_events[v2_events["observable"] == "logical_error_rate"]
        assert len(ler) == len(v2_events)  # only canonical LER emitted from v2
        assert ler["channel"].eq("delivered").all()
        assert ler["sigma"].notna().all()
        assert (ler["sigma"] >= 0).all()
        assert ler["value"].between(0.0, 1.0).all()
