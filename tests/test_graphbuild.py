"""Tests for ltn.data.graphbuild: topology registry, norm stats (leakage guard),
and UTS -> Snapshot conversion on fully synthetic frames (no raw data needed)."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import torch

from ltn.data import graphbuild as gb
from ltn.data import uts

T0 = pd.Timestamp("2026-01-01 00:00", tz="UTC")
H = pd.Timedelta(hours=1)


def ev(source, platform, device, etype, eid, ts, rows, cal_ts=None, context=None):
    return uts.make_events(
        source=source, platform=platform, device=device, entity_type=etype,
        entity_id=eid, ts=ts, rows=rows, cal_ts=cal_ts, context=context,
    )


def und_pairs(snap: gb.Snapshot) -> set[tuple[int, int]]:
    """Unordered node-position pairs present in edge_index."""
    cols = snap.edge_index.t().tolist()
    return {(min(a, b), max(a, b)) for a, b in cols}


def und_slots(snap: gb.Snapshot) -> list[tuple[int, int]]:
    """First half of edge_index = one row per undirected slot, in slot order."""
    e = snap.edge_index.shape[1] // 2
    return [tuple(snap.edge_index[:, k].tolist()) for k in range(e)]


def assert_nan_equal(t: torch.Tensor, expected: np.ndarray) -> None:
    e = torch.tensor(np.asarray(expected), dtype=torch.float32)
    assert t.shape == e.shape
    assert torch.equal(torch.isnan(t), torch.isnan(e))
    m = ~torch.isnan(e)
    assert torch.allclose(t[m], e[m], atol=1e-6)


# --------------------------------------------------------------------- coupling_edges

def test_coupling_edges_grid_3x3():
    edges = gb.coupling_edges("grid", rows=3, cols=3)
    expected = {(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
                (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8)}
    assert set(edges) == expected
    assert edges == sorted(edges)                       # canonical order
    assert all(a < b for a, b in edges)


def test_coupling_edges_linear():
    assert gb.coupling_edges("linear", n=4) == [(0, 1), (1, 2), (2, 3)]
    assert gb.coupling_edges("linear", n=1) == []


def test_coupling_edges_heavy_hex():
    edges = gb.coupling_edges("heavy_hex", distance=3)
    assert len(edges) > 0
    assert all(a < b for a, b in edges)
    assert len(edges) == len(set(edges))                # deduplicated
    n_qubits = max(max(e) for e in edges) + 1
    assert n_qubits == 19                               # heavy-hex d=3 has 19 qubits


def test_coupling_edges_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown coupling kind"):
        gb.coupling_edges("torus", n=4)


def test_coupling_edges_fake_real_provider():
    pytest.importorskip("qiskit_ibm_runtime")
    edges = gb.coupling_edges("fake", name="ibm_brisbane")
    assert len(edges) > 100                             # 127-qubit heavy-hex device
    assert all(a < b for a, b in edges)


def test_coupling_edges_fake_import_error_falls_back(monkeypatch):
    monkeypatch.setattr(gb, "FAKE_PROVIDER_OK", True)
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)  # forces ImportError
    with pytest.warns(UserWarning, match="falling back to heavy_hex"):
        edges = gb.coupling_edges("fake", name="ibm_brisbane")
    assert edges == gb.coupling_edges("heavy_hex", distance=7)
    assert gb.FAKE_PROVIDER_OK is False


def test_coupling_edges_fake_unknown_name_falls_back(monkeypatch):
    pytest.importorskip("qiskit_ibm_runtime")
    monkeypatch.setattr(gb, "FAKE_PROVIDER_OK", True)
    with pytest.warns(UserWarning, match="falling back to heavy_hex"):
        edges = gb.coupling_edges("fake", name="ibm_no_such_backend_xyz")
    assert edges == gb.coupling_edges("heavy_hex", distance=7)


# ------------------------------------------------------------------ compute_norm_stats

def _stats_frame(offset: float = 0.0, device: str = "devA") -> pd.DataFrame:
    i = dict(source="daqec", platform="ibm_sc", device=device)
    frames = [
        ev(**i, etype="qubit", eid="q0", ts=T0 + k * H,
           rows=[("t1_us", "claimed", v + offset, None)])
        for k, v in enumerate([10.0, 20.0, 30.0])
    ]
    frames.append(ev(**i, etype="qubit", eid="q0", ts=T0,
                     rows=[("readout_error", "claimed", 0.5, None)]))
    return pd.concat(frames, ignore_index=True)


def test_norm_stats_hand_computed():
    stats = gb.compute_norm_stats(_stats_frame())
    mean, std = stats[("ibm_sc", "t1_us", "claimed")]
    assert mean == pytest.approx(20.0)
    assert std == pytest.approx(np.sqrt(200.0 / 3.0))   # population std, ddof=0


def test_norm_stats_degenerate_std_floored_to_one():
    stats = gb.compute_norm_stats(_stats_frame())
    mean, std = stats[("ibm_sc", "readout_error", "claimed")]
    assert mean == pytest.approx(0.5)
    assert std == 1.0                                   # single sample -> floor


def test_norm_stats_excludes_truth_channel():
    base = _stats_frame()
    truth = ev(source="sim", platform="ibm_sc", device="devA", etype="qubit",
               eid="q0", ts=T0, rows=[("t1_us", "truth", 1e9, None)])
    stats = gb.compute_norm_stats(pd.concat([base, truth], ignore_index=True))
    assert ("ibm_sc", "t1_us", "truth") not in stats
    assert stats[("ibm_sc", "t1_us", "claimed")][0] == pytest.approx(20.0)


def test_norm_stats_derive_only_from_passed_frame():
    train = _stats_frame()
    other = ev(source="google_qec", platform="google_sc", device="g1", etype="qubit",
               eid="q0", ts=T0, rows=[("t1_us", "claimed", 999.0, None)])
    stats_train = gb.compute_norm_stats(train)
    assert all(k[0] == "ibm_sc" for k in stats_train)   # no keys from unseen frames
    shifted = gb.compute_norm_stats(_stats_frame(offset=100.0))
    assert shifted[("ibm_sc", "t1_us", "claimed")][0] == pytest.approx(120.0)
    both = gb.compute_norm_stats(pd.concat([train, other], ignore_index=True))
    assert ("google_sc", "t1_us", "claimed") in both    # sanity: input drives keys


# ------------------------------------------------------------- 3x3 grid qubit device

GRID_STATS = {
    ("google_sc", "t1_us", "claimed"): (100.0, 10.0),
    ("google_sc", "gate_error_2q", "claimed"): (0.02, 0.01),
}
# feature columns sorted by (observable, channel):
COL_GE, COL_T1 = 0, 1


@pytest.fixture()
def grid_frame() -> pd.DataFrame:
    g = dict(source="google_qec", platform="google_sc", device="google_toy")
    t0, t1, t2 = T0, T0 + 2 * H, T0 + 4 * H
    frames = [
        ev(**g, etype="qubit", eid="q0", ts=t0,
           rows=[("t1_us", "claimed", 110.0, None)], cal_ts=t0 - 0.5 * H),
        ev(**g, etype="qubit", eid="q1", ts=t0, rows=[("t1_us", "claimed", 90.0, None)]),
    ]
    frames += [
        ev(**g, etype="qubit", eid=f"q{q}", ts=t0, rows=[("t1_us", "claimed", 120.0, None)])
        for q in range(2, 8)
    ]
    frames += [
        ev(**g, etype="coupler", eid="c0-1", ts=t0,
           rows=[("gate_error_2q", "claimed", 0.03, None)]),
        ev(**g, etype="qubit", eid="q0", ts=t1, rows=[("t1_us", "claimed", 95.0, None)]),
        ev(**g, etype="qubit", eid="q8", ts=t1, rows=[("t1_us", "claimed", 130.0, None)]),
        ev(**g, etype="coupler", eid="c1-2", ts=t1,
           rows=[("gate_error_2q", "claimed", 0.01, None)]),
        ev(**g, etype="qubit", eid="q1", ts=t2, rows=[("t1_us", "claimed", 100.0, None)]),
    ]
    return pd.concat(frames, ignore_index=True)


def test_grid_shapes_dtypes_and_order(grid_frame):
    snaps = gb.build_snapshots(grid_frame, "google_toy", GRID_STATS)
    assert len(snaps) == 3                              # event cadence: one per ts
    s = snaps[0]
    assert s.entity_ids == [f"q{q}" for q in range(9)]
    assert all(sn.entity_ids == s.entity_ids for sn in snaps)  # stable order
    assert s.x.shape == (9, 4) and s.x.dtype == torch.float32
    assert s.mask.shape == (9, 2) and s.mask.dtype == torch.float32
    assert s.y_next.shape == (9, 2)
    assert s.edge_index.shape == (2, 24) and s.edge_index.dtype == torch.long
    assert s.edge_attr.shape == (24, 2) and s.edge_attr.dtype == torch.float32
    assert s.platform_onehot.shape == (len(uts.PLATFORMS),)
    assert s.platform_onehot.sum() == 1.0
    assert s.platform_onehot[uts.PLATFORMS.index("google_sc")] == 1.0
    assert s.ts == T0 and snaps[1].ts == T0 + 2 * H


def test_grid_edges_symmetric_and_grid_shaped(grid_frame):
    s = gb.build_snapshots(grid_frame, "google_toy", GRID_STATS)[0]
    directed = {tuple(c) for c in s.edge_index.t().tolist()}
    assert len(directed) == 24
    assert directed == {(b, a) for a, b in directed}    # symmetric
    assert und_pairs(s) == set(gb.coupling_edges("grid", rows=3, cols=3))


def test_grid_zscores_and_mask(grid_frame):
    snaps = gb.build_snapshots(grid_frame, "google_toy", GRID_STATS)
    # t0: (110-100)/10=1, (90-100)/10=-1, six qubits at (120-100)/10=2, q8 missing.
    z0 = np.array([1, -1, 2, 2, 2, 2, 2, 2, 0], dtype=np.float64)
    np.testing.assert_allclose(snaps[0].x[:, COL_T1].numpy(), z0, atol=1e-6)
    np.testing.assert_allclose(snaps[0].x[:, COL_GE].numpy(), 0.0)   # never observed
    m0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.float64)
    np.testing.assert_allclose(snaps[0].mask[:, COL_T1].numpy(), m0)
    np.testing.assert_allclose(snaps[0].mask[:, COL_GE].numpy(), 0.0)
    # t1: q0 refreshed to -0.5, q8 now 3.0, others carried forward with mask 1.
    z1 = np.array([-0.5, -1, 2, 2, 2, 2, 2, 2, 3], dtype=np.float64)
    np.testing.assert_allclose(snaps[1].x[:, COL_T1].numpy(), z1, atol=1e-6)
    np.testing.assert_allclose(snaps[1].mask[:, COL_T1].numpy(), 1.0)


def test_grid_time_features(grid_frame):
    snaps = gb.build_snapshots(grid_frame, "google_toy", GRID_STATS)
    cal_col, dt_col = 2, 3
    # t0: only q0 has a cal_ts (30 min old); every present qubit's event is "now".
    cal0 = np.zeros(9); cal0[0] = np.log1p(0.5)
    np.testing.assert_allclose(snaps[0].x[:, cal_col].numpy(), cal0, atol=1e-6)
    np.testing.assert_allclose(snaps[0].x[:, dt_col].numpy(), 0.0, atol=1e-6)
    # t1 (=t0+2h): q0/q8 have fresh events (dt=0), q1..q7 are 2h stale.
    dt1 = np.array([0.0] + [np.log1p(2.0)] * 7 + [0.0])
    np.testing.assert_allclose(snaps[1].x[:, dt_col].numpy(), dt1, atol=1e-6)
    cal1 = np.zeros(9); cal1[0] = np.log1p(2.5)         # cal_ts not refreshed at t1
    np.testing.assert_allclose(snaps[1].x[:, cal_col].numpy(), cal1, atol=1e-6)
    # t2 (=t0+4h): q1 fresh, q0/q8 2h stale, q2..q7 4h stale.
    dt2 = np.array([np.log1p(2.0), 0.0] + [np.log1p(4.0)] * 6 + [np.log1p(2.0)])
    np.testing.assert_allclose(snaps[2].x[:, dt_col].numpy(), dt2, atol=1e-6)


def test_grid_edge_attr_zscores_and_flags(grid_frame):
    snaps = gb.build_snapshots(grid_frame, "google_toy", GRID_STATS)
    slots = und_slots(snaps[0])
    e = len(slots)
    s01, s12 = slots.index((0, 1)), slots.index((1, 2))
    # t0: only coupler (0,1) observed, z = (0.03-0.02)/0.01 = 1.
    exp0 = np.zeros((e, 2)); exp0[s01] = (1.0, 1.0)
    np.testing.assert_allclose(snaps[0].edge_attr[:e].numpy(), exp0, atol=1e-6)
    # t1: (1,2) arrives at z=-1; (0,1) carried forward.
    exp1 = exp0.copy(); exp1[s12] = (-1.0, 1.0)
    np.testing.assert_allclose(snaps[1].edge_attr[:e].numpy(), exp1, atol=1e-6)
    np.testing.assert_allclose(snaps[2].edge_attr[:e].numpy(), exp1, atol=1e-6)
    for s in snaps:  # attr rows duplicated for the reversed directions
        assert torch.equal(s.edge_attr[:e], s.edge_attr[e:])


def test_grid_y_next_alignment(grid_frame):
    snaps = gb.build_snapshots(grid_frame, "google_toy", GRID_STATS)
    y0 = np.full((9, 2), np.nan)
    y0[0, COL_T1] = -0.5                                # q0 fresh at t1
    y0[8, COL_T1] = 3.0                                 # q8 fresh at t1
    assert_nan_equal(snaps[0].y_next, y0)
    y1 = np.full((9, 2), np.nan)
    y1[1, COL_T1] = 0.0                                 # q1 fresh at t2, z=(100-100)/10
    assert_nan_equal(snaps[1].y_next, y1)
    assert torch.isnan(snaps[2].y_next).all()           # last snapshot: no next


def test_build_snapshots_uses_supplied_stats_not_frame(grid_frame):
    """Leakage guard: features come from the *given* stats, not the frame's own."""
    other_stats = {("google_sc", "t1_us", "claimed"): (0.0, 1.0)}
    s = gb.build_snapshots(grid_frame, "google_toy", other_stats)[0]
    assert s.x.shape == (9, 3)                          # vocabulary = stats keys only
    assert s.x[0, 0].item() == pytest.approx(110.0)     # raw value under (mu=0, sd=1)


# ---------------------------------------------------------------- DAQEC star device

DAQEC_STATS = {
    ("ibm_sc", "composite_score", "derived"): (0.0, 1.0),
    ("ibm_sc", "logical_error_rate", "delivered"): (0.02, 0.01),
}
SUB_A, SUB_B, SUB_C = (uts.subgraph_id(q) for q in ([0, 1, 2], [2, 3, 4], [5, 6]))


def _daqec_frame(with_device: bool = True) -> pd.DataFrame:
    d = dict(source="daqec", platform="ibm_sc", device="daqec_toy")
    t0, t1 = T0, T0 + 6 * H
    frames = []
    if with_device:
        frames.append(ev(**d, etype="device", eid="device", ts=t0,
                         rows=[("composite_score", "derived", 0.5, None)]))
    frames += [
        ev(**d, etype="subgraph", eid=SUB_A, ts=t0,
           rows=[("logical_error_rate", "delivered", 0.01, None)],
           context={"data_qubits": [0, 1, 2]}),
        ev(**d, etype="subgraph", eid=SUB_B, ts=t0,
           rows=[("logical_error_rate", "delivered", 0.02, None)],
           context={"data_qubits": [2, 3, 4]}),
        ev(**d, etype="subgraph", eid=SUB_A, ts=t1,
           rows=[("logical_error_rate", "delivered", 0.03, None)],
           context={"data_qubits": [0, 1, 2]}),
        ev(**d, etype="subgraph", eid=SUB_C, ts=t1,
           rows=[("logical_error_rate", "delivered", 0.05, None)],
           context={"data_qubits": [5, 6]}),
    ]
    return pd.concat(frames, ignore_index=True)


def test_daqec_star_and_overlap_edges():
    snaps = gb.build_snapshots(_daqec_frame(), "daqec_toy", DAQEC_STATS)
    assert len(snaps) == 2
    s = snaps[0]
    assert s.entity_ids[0] == "device" and len(s.entity_ids) == 4
    p = {e: s.entity_ids.index(e) for e in (SUB_A, SUB_B, SUB_C)}
    star = {(0, p[SUB_A]), (0, p[SUB_B]), (0, p[SUB_C])}
    overlap = {(min(p[SUB_A], p[SUB_B]), max(p[SUB_A], p[SUB_B]))}  # A∩B={2}
    assert und_pairs(s) == star | overlap               # and no A-C / B-C edge
    assert s.edge_index.shape == (2, 8)
    assert s.edge_attr.shape == (8, 2)
    assert torch.all(s.edge_attr == 0)                  # no couplers on this device
    directed = {tuple(c) for c in s.edge_index.t().tolist()}
    assert directed == {(b, a) for a, b in directed}    # symmetric


def test_daqec_features_and_y_next():
    snaps = gb.build_snapshots(_daqec_frame(), "daqec_toy", DAQEC_STATS)
    s0 = snaps[0]
    a, b, c = (s0.entity_ids.index(e) for e in (SUB_A, SUB_B, SUB_C))
    col_cs, col_ler = 0, 1                              # sorted (observable, channel)
    assert s0.x[0, col_cs].item() == pytest.approx(0.5)
    assert s0.x[a, col_ler].item() == pytest.approx(-1.0)   # (0.01-0.02)/0.01
    assert s0.x[b, col_ler].item() == pytest.approx(0.0)
    assert s0.mask[b, col_ler] == 1.0                   # z=0 but observed
    assert torch.all(s0.x[c, :2] == 0) and torch.all(s0.mask[c] == 0)
    y0 = np.full((4, 2), np.nan)
    y0[a, col_ler] = 1.0                                # (0.03-0.02)/0.01 at t1
    y0[c, col_ler] = 3.0                                # (0.05-0.02)/0.01 at t1
    assert_nan_equal(s0.y_next, y0)
    s1 = snaps[1]
    assert s1.x[b, col_ler].item() == pytest.approx(0.0)     # carried forward
    assert s1.mask[b, col_ler] == 1.0
    assert s1.x[0, col_cs].item() == pytest.approx(0.5)      # device carried
    dt_col = 3
    assert s1.x[0, dt_col].item() == pytest.approx(np.log1p(6.0), abs=1e-6)
    assert s1.x[a, dt_col].item() == pytest.approx(0.0)


def test_daqec_synthesizes_hub_when_no_device_entity():
    snaps = gb.build_snapshots(_daqec_frame(with_device=False), "daqec_toy", DAQEC_STATS)
    s = snaps[0]
    assert s.entity_ids[0] == "device:daqec_toy"
    assert len(s.entity_ids) == 4
    assert s.edge_index.shape == (2, 8)                 # 3 star + 1 overlap, both dirs
    assert torch.all(s.x[0] == 0) and torch.all(s.mask[0] == 0)
    assert torch.isnan(s.y_next[0]).all()


# -------------------------------------------------------- registry + cadence variants

SIMOS_STATS = {("si_spin", "t1_us", "claimed"): (10.0, 2.0)}


def _simos_frame() -> pd.DataFrame:
    s = dict(source="silicon", platform="si_spin", device="simos_toy")
    d1a = pd.Timestamp("2026-02-01 09:00", tz="UTC")
    d1b = pd.Timestamp("2026-02-01 15:00", tz="UTC")
    d2 = pd.Timestamp("2026-02-02 10:00", tz="UTC")
    return pd.concat([
        ev(**s, etype="qubit", eid="q0", ts=d1a, rows=[("t1_us", "claimed", 12.0, None)]),
        ev(**s, etype="qubit", eid="q0", ts=d1b, rows=[("t1_us", "claimed", 8.0, None)]),
        ev(**s, etype="qubit", eid="q1", ts=d2, rows=[("t1_us", "claimed", 14.0, None)]),
    ], ignore_index=True)


def test_simos_linear_topology_and_event_cadence():
    snaps = gb.build_snapshots(_simos_frame(), "simos_toy", SIMOS_STATS)
    assert len(snaps) == 3
    assert und_pairs(snaps[0]) == {(0, 1)}              # simos_* -> linear(2)
    assert snaps[0].edge_index.shape == (2, 2)


def test_daily_cadence_bins_and_targets():
    snaps = gb.build_snapshots(_simos_frame(), "simos_toy", SIMOS_STATS, cadence="1D")
    assert len(snaps) == 2                              # two calendar days
    s0, s1 = snaps
    assert s0.ts == pd.Timestamp("2026-02-01 15:00", tz="UTC")  # latest event in day
    assert s0.x[0, 0].item() == pytest.approx(-1.0)     # last-in-day wins: (8-10)/2
    y0 = np.full((2, 1), np.nan)
    y0[1, 0] = 2.0                                      # q1 fresh next day: (14-10)/2
    assert_nan_equal(s0.y_next, y0)
    assert s1.ts == pd.Timestamp("2026-02-02 10:00", tz="UTC")
    assert s1.x[0, 2].item() == pytest.approx(np.log1p(19.0), abs=1e-6)  # q0 19h stale
    assert s1.x[1, 2].item() == pytest.approx(0.0)


SIM_STATS = {("sim_sc", "t1_us", "claimed"): (0.0, 1.0)}


def test_sim_dev_topology_from_context():
    s = dict(source="sim", platform="sim_sc", device="sim_dev_ring")
    frame = pd.concat([
        ev(**s, etype="qubit", eid="q0", ts=T0, rows=[("t1_us", "claimed", 1.0, None)],
           context={"topology": [[0, 1], [1, 2], [0, 2]]}),
        ev(**s, etype="qubit", eid="q1", ts=T0, rows=[("t1_us", "claimed", 2.0, None)]),
        ev(**s, etype="qubit", eid="q2", ts=T0, rows=[("t1_us", "claimed", 3.0, None)]),
    ], ignore_index=True)
    snap = gb.build_snapshots(frame, "sim_dev_ring", SIM_STATS)[0]
    assert und_pairs(snap) == {(0, 1), (0, 2), (1, 2)}


def test_sim_dev_falls_back_to_heavy_hex_5():
    s = dict(source="sim", platform="sim_sc", device="sim_dev_hh")
    frame = pd.concat([
        ev(**s, etype="qubit", eid=f"q{q}", ts=T0, rows=[("t1_us", "claimed", float(q), None)])
        for q in range(5)
    ], ignore_index=True)
    snap = gb.build_snapshots(frame, "sim_dev_hh", SIM_STATS)[0]
    expected = {(a, b) for a, b in gb.coupling_edges("heavy_hex", distance=5)
                if a < 5 and b < 5}
    assert und_pairs(snap) == expected


def test_google_coordinate_qubit_ids_no_collision():
    g = dict(source="google_qec", platform="google_sc", device="google_coord")
    frame = pd.concat([
        ev(**g, etype="qubit", eid=f"q{r}_{c}", ts=T0,
           rows=[("t1_us", "claimed", 100.0 + 3 * r + c, None)])
        for r in range(3) for c in range(3)
    ], ignore_index=True)
    snap = gb.build_snapshots(frame, "google_coord", GRID_STATS)[0]
    assert len(snap.entity_ids) == 9                    # no first-int id collisions
    assert snap.entity_ids == [f"q{r}_{c}" for r in range(3) for c in range(3)]
    assert len(und_pairs(snap)) == 12                   # 3x3 grid adjacency
    ix = {e: k for k, e in enumerate(snap.entity_ids)}
    pairs = und_pairs(snap)
    assert (min(ix["q0_0"], ix["q0_1"]), max(ix["q0_0"], ix["q0_1"])) in pairs
    assert (min(ix["q0_0"], ix["q1_0"]), max(ix["q0_0"], ix["q1_0"])) in pairs
    assert (min(ix["q0_0"], ix["q1_1"]), max(ix["q0_0"], ix["q1_1"])) not in pairs
    assert snap.x[ix["q1_2"], COL_T1].item() == pytest.approx(0.5)  # (105-100)/10


def test_unknown_device_and_bad_cadence(grid_frame):
    assert gb.build_snapshots(grid_frame, "no_such_device", GRID_STATS) == []
    with pytest.raises(ValueError, match="cadence"):
        gb.build_snapshots(grid_frame, "google_toy", GRID_STATS, cadence="2D")
