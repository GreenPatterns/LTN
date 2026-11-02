"""Tests for the Google QEC -> UTS loader (plan step A3, §2.2/§2.8-E5).

Three layers:
1. Pure-unit math (per-round conversion, sigmas, subgraph-coordinate hashing).
2. Synthetic end-to-end (no raw data): stim-generated surface-code circuits ->
   DEM aggregation, m2d conversion, pymatching decode — both as direct function
   calls and through ``load_google`` on a miniature 2022-layout zip built on
   the fly (provided-files run + measurements-only run).
3. ``@pytest.mark.data`` integration on the real zips: >=3 Sycamore-2022 runs
   and one 105Q run (.b8 pathway), asserting the loaded UTS events, parsed
   parameters, and physically sensible values.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import stim

from conftest import requires_raw
from ltn.data import uts
from ltn.data.loaders.google import (
    GOOGLE_RECORDS,
    binomial_sigma,
    coords_subgraph_id,
    decode_logical_error,
    dem_per_qubit_error_probs,
    load_google,
    measurements_to_detections,
    per_round_error_rate,
    per_round_sigma,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
GOOGLE_RAW = RAW / "google"

sycamore_data = [pytest.mark.data, requires_raw("google/google_qec3v5_experiment_data.zip")]
q105_data = [pytest.mark.data, requires_raw("google/google_105Q_surface_code_d3_d5_d7.zip")]


def _noisy_surface_code(distance: int = 3, rounds: int = 5) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.008,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
        before_round_data_depolarization=0.008,
    )


def _pack_b8(bits: np.ndarray) -> bytes:
    """Shot-major little-endian bit packing == stim's b8 format."""
    return np.packbits(bits.astype(np.uint8), axis=1, bitorder="little").tobytes()


# ------------------------------------------------------------------- unit: math


def test_per_round_error_rate_roundtrip_and_clamps() -> None:
    # r=1 is the identity on (0, 0.5)
    assert per_round_error_rate(0.1, 1) == pytest.approx(0.1)
    # inverse relation: P = 0.5*(1-(1-2*eps)^r)
    eps = per_round_error_rate(0.3, 7)
    assert 0.5 * (1.0 - (1.0 - 2.0 * eps) ** 7) == pytest.approx(0.3)
    # more rounds at fixed per-shot P -> smaller per-round rate
    assert per_round_error_rate(0.2, 10) < per_round_error_rate(0.2, 2)
    # boundary clamps
    assert per_round_error_rate(0.0, 5) == 0.0
    assert per_round_error_rate(-0.01, 5) == 0.0
    assert per_round_error_rate(0.5, 5) == 0.5
    assert per_round_error_rate(0.7, 5) == 0.5
    with pytest.raises(ValueError):
        per_round_error_rate(0.1, 0)


def test_sigmas_positive_and_scaled() -> None:
    s = binomial_sigma(0.1, 1000)
    assert s == pytest.approx(np.sqrt(0.1 * 0.9 / 1000))
    assert binomial_sigma(0.0, 100) > 0  # floored, never exactly 0
    with pytest.raises(ValueError):
        binomial_sigma(0.1, 0)
    # delta method: r=1 passes sigma through; more rounds shrink it
    s1 = per_round_sigma(0.1, s, 1)
    assert s1 == pytest.approx(s)
    assert per_round_sigma(0.1, s, 10) < s1
    assert per_round_sigma(0.499, s, 5) > 0  # finite near the 0.5 divergence


def test_coords_subgraph_id_stable_and_injective() -> None:
    coords = [(1.0, 2.0), (3.0, 4.0), (5.5, 6.5)]
    a = coords_subgraph_id(coords)
    assert a == coords_subgraph_id(list(reversed(coords)))  # order-invariant
    assert a.startswith("sub:")
    assert a != coords_subgraph_id([(1.0, 2.0), (3.0, 4.0), (5.5, 7.0)])
    # half-integer coordinates stay distinct from swapped integer ones
    assert coords_subgraph_id([(1.0, 2.5)]) != coords_subgraph_id([(2.5, 1.0)])


# ------------------------------------------------- synthetic: stim -> functions


class TestSyntheticPipeline:
    """Direct function-level pipeline on a stim-generated noisy surface code."""

    @pytest.fixture(scope="class")
    def circuit(self) -> stim.Circuit:
        return _noisy_surface_code()

    @pytest.fixture(scope="class")
    def dem(self, circuit: stim.Circuit) -> stim.DetectorErrorModel:
        return circuit.detector_error_model(decompose_errors=True)

    def test_dem_aggregation(self, circuit: stim.Circuit, dem: stim.DetectorErrorModel) -> None:
        per_qubit = dem_per_qubit_error_probs(dem)
        assert len(per_qubit) > 0
        det_xy = {
            (c[0], c[1]) for c in circuit.get_detector_coordinates().values() if len(c) >= 2
        }
        assert set(per_qubit) <= det_xy  # aggregation lands on detector coordinates
        assert all(0.0 < p < 1.0 for p in per_qubit.values())

    def test_m2d_and_decode(self, circuit: stim.Circuit, dem: stim.DetectorErrorModel) -> None:
        n = 4000
        meas = circuit.compile_sampler(seed=7).sample(n)
        dets, obs = measurements_to_detections(circuit, meas)
        assert dets.shape == (n, circuit.num_detectors)
        assert obs.shape == (n, circuit.num_observables)
        frac = float(dets.mean())
        assert 0.0 < frac < 1.0
        p_shot, n_cmp = decode_logical_error(dem, dets, obs)
        assert n_cmp == n
        assert 0.0 < p_shot < 0.5  # decoder beats chance on its own noise model
        eps = per_round_error_rate(p_shot, 5)
        assert 0.0 < eps < 0.5
        # m2d agrees with stim's reference detector sampling statistics
        ref = circuit.compile_detector_sampler(seed=7).sample(n).mean()
        assert frac == pytest.approx(float(ref), abs=0.01)


# ------------------------------------------- synthetic: load_google on mini-zip


@pytest.fixture(scope="module")
def mini_raw(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fake raw_dir holding a 2022-layout zip with two synthetic runs.

    run A (provided-files style): circuit + dem + detection_events.b8 +
        obs_flips_actual.01.
    run B (measurements-only style): circuit + dem + measurements.b8, forcing
        the m2d pathway.
    """
    raw = tmp_path_factory.mktemp("google_raw")
    circuit = _noisy_surface_code()
    dem = circuit.detector_error_model(decompose_errors=True)
    n = 2000
    dets, obs = circuit.compile_detector_sampler(seed=3).sample(n, separate_observables=True)
    meas = circuit.compile_sampler(seed=5).sample(n)
    zpath = raw / GOOGLE_RECORDS["sycamore_2022"].zip_name
    with zipfile.ZipFile(zpath, "w") as zf:
        a = "surface_code_bZ_d3_r05_center_1_1"
        zf.writestr(f"{a}/circuit_ideal.stim", str(circuit))
        zf.writestr(f"{a}/circuit_detector_error_model.dem", str(dem))
        zf.writestr(f"{a}/detection_events.b8", _pack_b8(dets))
        zf.writestr(f"{a}/obs_flips_actual.01",
                    b"".join(b"1\n" if o else b"0\n" for o in obs[:, 0]))
        b = "surface_code_bZ_d3_r05_center_3_3"
        zf.writestr(f"{b}/circuit_ideal.stim", str(circuit))
        zf.writestr(f"{b}/circuit_detector_error_model.dem", str(dem))
        zf.writestr(f"{b}/measurements.b8", _pack_b8(meas))
    return raw


class TestLoadGoogleSyntheticZip:
    @pytest.fixture(scope="class")
    def events(self, mini_raw: Path) -> pd.DataFrame:
        return load_google(mini_raw, records="sycamore_2022", max_shots=1500)

    def test_validates_and_shape(self, events: pd.DataFrame) -> None:
        uts.validate_uts(events)
        assert set(events["source"]) == {"google_qec"}
        assert set(events["platform"]) == {"google_sc"}
        assert set(events["device"]) == {"google_sycamore_2022"}
        assert set(events["entity_type"]) == {"subgraph", "qubit", "device"}
        assert set(events["observable"]) == {
            "dem_error_prob", "detection_event_fraction", "logical_error_per_round"}

    def test_both_pathways_covered(self, events: pd.DataFrame) -> None:
        sub = events[events["entity_type"] == "subgraph"]
        ctx = {json.loads(c)["dir"]: json.loads(c) for c in sub["context"].unique()}
        assert set(ctx) == {"surface_code_bZ_d3_r05_center_1_1",
                            "surface_code_bZ_d3_r05_center_3_3"}
        assert ctx["surface_code_bZ_d3_r05_center_1_1"]["det_source"] == "provided"
        assert ctx["surface_code_bZ_d3_r05_center_3_3"]["det_source"] == "m2d"
        for c in ctx.values():
            assert c["decoder"] == "pymatching"
            assert c["decode_fallback"] is False
            assert c["pseudo_time"] is True
            assert c["basis"] == "Z" and c["distance"] == 3 and c["rounds"] == 5
            assert c["n_shots_used"] == 1500  # max_shots honoured

    def test_values_physical(self, events: pd.DataFrame) -> None:
        sub = events[events["entity_type"] == "subgraph"]
        det = sub[sub["observable"] == "detection_event_fraction"]
        ler = sub[sub["observable"] == "logical_error_per_round"]
        assert len(det) == len(ler) == 2
        assert det["channel"].eq("derived").all() and ler["channel"].eq("delivered").all()
        assert det["value"].between(0, 1, inclusive="neither").all()
        assert ler["value"].between(0, 0.5, inclusive="neither").all()
        assert (ler["sigma"] > 0).all()
        # same circuit -> same subgraph entity for both runs
        assert sub["entity_id"].nunique() == 1
        qub = events[events["entity_type"] == "qubit"]
        assert qub["observable"].eq("dem_error_prob").all()
        assert qub["channel"].eq("claimed").all()
        assert qub["value"].between(0, 1, inclusive="neither").all()
        # device mean equals mean of the per-qubit claims of the same run
        dev = events[(events["entity_type"] == "device")
                     & (events["observable"] == "dem_error_prob")]
        one_ctx = dev["context"].iloc[0]
        assert dev[dev["context"] == one_ctx]["value"].iloc[0] == pytest.approx(
            qub[qub["context"] == one_ctx]["value"].mean())

    def test_device_rows_duplicate_subgraph_rows(self, events: pd.DataFrame) -> None:
        sub = events[events["entity_type"] == "subgraph"]
        dev = events[events["entity_type"] == "device"]
        for obs in ("detection_event_fraction", "logical_error_per_round"):
            s = sub[sub["observable"] == obs].sort_values("ts")["value"].tolist()
            d = dev[dev["observable"] == obs].sort_values("ts")["value"].tolist()
            assert s == d

    def test_max_runs_and_decode_flag_and_filters(self, mini_raw: Path) -> None:
        one = load_google(mini_raw, records="sycamore_2022", max_shots=500, max_runs=1)
        assert {json.loads(c)["dir"] for c in one["context"]} == {
            "surface_code_bZ_d3_r05_center_1_1"}
        nodecode = load_google(mini_raw, records="sycamore_2022", max_shots=500, decode=False)
        assert "logical_error_per_round" not in set(nodecode["observable"])
        assert {"detection_event_fraction", "dem_error_prob"} <= set(nodecode["observable"])
        with pytest.raises(ValueError):
            load_google(mini_raw, records="not_a_record")
        # absent zips of other records are skipped, not fatal
        assert len(load_google(mini_raw)) == len(
            load_google(mini_raw, records="sycamore_2022"))


# ------------------------------------------------------------ real data: 2022


@pytest.mark.parametrize("dummy", [0])  # keep module importable without data
def test_records_registry_complete(dummy: int) -> None:
    assert set(GOOGLE_RECORDS) == {"sycamore_2022", "105q", "dynamic", "color"}
    assert GOOGLE_RECORDS["sycamore_2022"].device == "google_sycamore_2022"
    assert not GOOGLE_RECORDS["color"].matchable  # color code is not MWPM-decodable


class TestSycamore2022:
    pytestmark = sycamore_data

    @pytest.fixture(scope="class")
    def events(self) -> pd.DataFrame:
        return load_google(GOOGLE_RAW, records="sycamore_2022", max_shots=2000,
                           max_runs=3, run_filter="surface_code_bZ_d3_r09")

    def test_three_runs_validate(self, events: pd.DataFrame) -> None:
        uts.validate_uts(events)
        dirs = {json.loads(c)["dir"] for c in events["context"]}
        assert len(dirs) == 3  # >= 3 distinct experiment dirs loaded
        assert all(d.startswith("surface_code_bZ_d3_r09") for d in dirs)
        assert set(events["device"]) == {"google_sycamore_2022"}
        # distinct centers -> distinct subgraph entities
        assert events[events["entity_type"] == "subgraph"]["entity_id"].nunique() == 3

    def test_params_parsed(self, events: pd.DataFrame) -> None:
        for c in events["context"].unique():
            ctx = json.loads(c)
            assert ctx["distance"] == 3
            assert ctx["rounds"] == 9
            assert ctx["basis"] == "Z"
            assert ctx["code"] == "surface_code"
            assert ctx["pseudo_time"] is True
            assert ctx["n_shots_used"] == 2000
            assert ctx["decoder"] == "pymatching"
            assert ctx["dem"] == "circuit_detector_error_model.dem"

    def test_values_physical(self, events: pd.DataFrame) -> None:
        sub = events[events["entity_type"] == "subgraph"]
        ler = sub[sub["observable"] == "logical_error_per_round"]
        det = sub[sub["observable"] == "detection_event_fraction"]
        assert len(ler) == len(det) == 3
        assert ler["value"].between(0, 0.5, inclusive="neither").all()
        assert ler["sigma"].gt(0).all()
        # d3 Sycamore memory: ~3-4% per round, ~11-17% detection fraction
        assert ler["value"].between(0.01, 0.10).all()
        assert det["value"].between(0.05, 0.30).all()
        qub = events[events["entity_type"] == "qubit"]
        assert len(qub) > 0 and qub["observable"].eq("dem_error_prob").all()
        assert qub["channel"].eq("claimed").all()
        # 8 measure qubits per d3 patch -> 8 per-qubit claims per run
        assert len(qub) == 3 * 8

    def test_pseudo_timestamps(self, events: pd.DataFrame) -> None:
        base = pd.Timestamp("2022-07-01", tz="UTC")
        assert (events["ts"] >= base).all()
        assert (events["ts"] < base + pd.Timedelta(days=30)).all()
        assert events.groupby(
            events["context"].map(lambda c: json.loads(c)["dir"])
        )["ts"].nunique().eq(1).all()  # one timestamp per run


# ------------------------------------------------------------ real data: 105Q


class Test105Q:
    pytestmark = q105_data

    @pytest.fixture(scope="class")
    def events(self) -> pd.DataFrame:
        return load_google(GOOGLE_RAW, records="105q", max_shots=1000,
                           max_runs=1, run_filter="d3_at_q4_5/X/r13")

    def test_one_run_loads_b8_path(self, events: pd.DataFrame) -> None:
        uts.validate_uts(events)
        assert len(events) > 0
        assert set(events["device"]) == {"google_105q"}
        ctx = json.loads(events["context"].iloc[0])
        assert ctx["basis"] == "X" and ctx["distance"] == 3 and ctx["rounds"] == 13
        assert ctx["det_source"] == "provided"  # .b8 detection events consumed
        assert ctx["decoder"] == "pymatching"
        assert "si1000" in ctx["dem"]  # claim DEM is the a-priori noise model
        assert ctx["n_shots_used"] == 1000
        sub = events[events["entity_type"] == "subgraph"]
        ler = sub[sub["observable"] == "logical_error_per_round"]
        det = sub[sub["observable"] == "detection_event_fraction"]
        assert len(ler) == len(det) == 1
        assert 0.0 < ler["value"].iloc[0] < 0.5
        assert 0.0 < det["value"].iloc[0] < 1.0
        assert events[events["entity_type"] == "qubit"]["value"].between(
            0, 1, inclusive="neither").all()
        assert (events["ts"] >= pd.Timestamp("2024-08-01", tz="UTC")).all()
