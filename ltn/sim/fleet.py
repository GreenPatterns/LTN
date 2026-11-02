"""Synthetic quantum-device-fleet telemetry generator [DEV-2].

Produces longitudinal UTS telemetry (source="sim", platform="sim_sc", devices
"sim_dev_00".."sim_dev_NN") replacing the unrecoverable live IBM backbone.

Guarantees implemented (unit-tested in tests/test_sim.py):

* **Physics**: per-qubit log-T1 and log-Tphi follow exact-discretization OU
  processes (ltn.sim.processes); T2 is derived as 1/T2 = 1/(2 T1) + 1/Tphi in
  every channel, so T2 <= 2 T1 holds everywhere, including noisy channels.
* **Channels**: ``truth`` = model-invisible ground-truth latents (written to a
  separate root by the CLI); ``probe`` = truth + probe noise at every
  snapshot; ``claimed`` = snapshot of truth + claimed noise frozen at the most
  recent recalibration (stale in between); ``cal_ts`` = last recal time.
* **Recalibration renewal** per device (LogNormal intervals, median ~8 h to
  match DAQEC cadence) refreshes the claimed snapshot and partially re-pins
  the truth latents.
* **Recalibration-reset process** (opt-in; the defaults reproduce the original
  fleet bit-for-bit): ``recal_groups`` splits each device into independently
  recalibrated groups (one renewal stream and one ``cal_ts`` per group), and a
  daily full recalibration at ``reset_hour`` resets a log-T1 decline of
  ``reset_decay_per_h`` per hour, centred on the day.
* **Bitwise determinism**: all randomness flows through per-device
  ``numpy.random.default_rng([seed, device_index])`` generators, so identical
  (config, seed) give bitwise-identical outputs.

CLI::

    python -m ltn.sim.fleet --config cfg.json --days 180 --seed 20260712 \
        --out data/uts --truth data/uts_truth
    python -m ltn.sim.fleet --write-default-config cfg.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ltn.data import uts
from . import processes as proc
from .topology import make_topology

SOURCE = "sim"
PLATFORM = "sim_sc"

#: qubit observables emitted on model-visible channels (claimed + probe)
VISIBLE_QUBIT_OBS = ("t1_us", "t2_us", "readout_error", "epc_1q")
#: qubit observables emitted on the truth channel
TRUTH_QUBIT_OBS = ("t1_us", "t2_us", "tphi_us", "readout_error", "epc_1q")
EDGE_OBS = "gate_error_2q"


def default_fleet_mix() -> list[dict]:
    """10x heavy_hex(d=5), 3x heavy_hex(d=7), 2x grid(6,12), 1x linear(12)."""
    return (
        [{"kind": "heavy_hex", "distance": 5} for _ in range(10)]
        + [{"kind": "heavy_hex", "distance": 7} for _ in range(3)]
        + [{"kind": "grid", "rows": 6, "cols": 12} for _ in range(2)]
        + [{"kind": "linear", "n": 12}]
    )


@dataclass
class FleetConfig:
    """Every knob of the synthetic fleet. Units noted per field."""

    # fleet composition and epoch
    topologies: list[dict] = field(default_factory=default_fleet_mix)
    start: str = "2026-01-01T00:00:00+00:00"

    # log-T1 OU (theta per day; sigma per sqrt(day), log-units)
    t1_median_us: float = 180.0
    t1_device_spread: float = 0.25
    t1_qubit_spread: float = 0.35
    ou_theta_t1: float = 0.35
    ou_sigma_t1: float = 0.18

    # log-Tphi OU (its own process; T2 derived, never simulated directly)
    tphi_median_us: float = 150.0
    tphi_qubit_spread: float = 0.35
    ou_theta_tphi: float = 0.5
    ou_sigma_tphi: float = 0.25

    # TLS defect episodes (rates per day; factor multiplies T1 while on)
    tls_on_rate: float = 0.03
    tls_off_rate: float = 0.5
    tls_factor_lo: float = 0.3
    tls_factor_hi: float = 0.8

    # rare per-device degradation trends (Poisson arrivals, geometric duration)
    degrade_rate_per_day: float = 1.0 / 120.0
    degrade_mean_days: float = 10.0
    degrade_drift_per_day: float = 0.02

    # recalibration renewal (LogNormal intervals; median ~8 h = DAQEC cadence)
    recal_median_h: float = 8.0
    recal_sigma_log: float = 0.35
    recal_kappa: float = 0.35
    recal_jump_sigma: float = 0.05

    # recalibration-reset process (defaults: one device-wide stream, no daily reset)
    recal_groups: int = 1            # independent recalibration streams per device
    reset_hour: float = -1.0         # hour of the daily full recalibration; < 0 = off
    reset_decay_per_h: float = 0.0   # log-T1 decline per hour since the daily reset

    # observation noise (relative)
    claimed_noise: float = 0.02
    probe_noise: float = 0.04

    # coupled error channels
    t_gate_1q_us: float = 0.035
    t_gate_2q_us: float = 0.30
    epc_a1: float = 1.5e-4
    epc_b1: float = 0.6
    err2q_a2: float = 2.5e-3
    err2q_b2: float = 0.8
    err2q_c: float = 2.0e-3
    coupler_theta: float = 0.4
    coupler_sigma: float = 0.5
    err_noise_rel: float = 0.03

    # readout error AR(1), weakly correlated with T1 via log-space coupling
    readout_median: float = 1.5e-2
    readout_qubit_spread: float = 0.4
    readout_theta: float = 0.4
    readout_sigma: float = 0.12
    readout_t1_coupling: float = -0.3

    # device-offline gaps (MCAR + geometric bursts)
    offline_p_mcar: float = 0.01
    offline_burst_rate: float = 1.0 / 90.0
    offline_burst_geo_p: float = 0.35

    # ------------------------------------------------------------- serialization
    def to_json(self, path: str | Path | None = None) -> str:
        text = json.dumps(asdict(self), indent=2, sort_keys=True)
        if path is not None:
            Path(path).write_text(text + "\n")
        return text

    @classmethod
    def from_json(cls, text: str) -> "FleetConfig":
        d = json.loads(text)
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown FleetConfig fields: {sorted(unknown)}")
        return cls(**d)

    @property
    def n_devices(self) -> int:
        return len(self.topologies)

    def device_names(self) -> list[str]:
        return [f"sim_dev_{i:02d}" for i in range(self.n_devices)]


class FleetSim:
    """Simulate a fleet: ``run(n_days)`` -> (uts_events, truth_events)."""

    def __init__(self, cfg: FleetConfig, seed: int):
        self.cfg = cfg
        self.seed = int(seed)

    # ------------------------------------------------------------------- physics
    def _truth_obs(
        self,
        x_t1: np.ndarray,
        x_tphi: np.ndarray,
        x_ro: np.ndarray,
        x_cp: np.ndarray,
        tls_fac: np.ndarray,
        mu_t1: np.ndarray,
        edge_u: np.ndarray,
        edge_v: np.ndarray,
        rng: np.random.Generator,
        t1_offset: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        cfg = self.cfg
        if t1_offset is not None:
            x_t1 = x_t1 + t1_offset
        t1 = np.exp(x_t1) * tls_fac
        tphi = np.exp(x_tphi)
        t2 = 1.0 / (1.0 / (2.0 * t1) + 1.0 / tphi)
        ro = np.exp(x_ro + cfg.readout_t1_coupling * (x_t1 - mu_t1))
        ro = np.clip(ro, 1e-5, 0.5)
        epc = cfg.epc_a1 + cfg.epc_b1 * (cfg.t_gate_1q_us / t1)
        epc = epc * (1.0 + cfg.err_noise_rel * rng.standard_normal(t1.shape))
        epc = np.clip(epc, 1e-7, 0.5)
        g2 = (
            cfg.err2q_a2
            + cfg.err2q_b2 * (cfg.t_gate_2q_us / 2.0)
            * (1.0 / t1[edge_u] + 1.0 / t1[edge_v] + 1.0 / tphi[edge_u] + 1.0 / tphi[edge_v])
            + cfg.err2q_c * x_cp
        )
        g2 = g2 * (1.0 + cfg.err_noise_rel * rng.standard_normal(x_cp.shape))
        g2 = np.clip(g2, 1e-7, 0.9)
        return {
            "t1_us": t1, "tphi_us": tphi, "t2_us": t2,
            "readout_error": ro, "epc_1q": epc, "gate_error_2q": g2,
        }

    def _observe(
        self, truth: dict[str, np.ndarray], rel: float, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        """Noisy view of truth. T2 is re-derived from the noisy T1/Tphi so
        T2 <= 2 T1 holds in every channel, not just in truth."""
        t1 = np.maximum(truth["t1_us"] * (1.0 + rel * rng.standard_normal(truth["t1_us"].shape)), 1e-3)
        tphi = np.maximum(truth["tphi_us"] * (1.0 + rel * rng.standard_normal(truth["tphi_us"].shape)), 1e-3)
        t2 = 1.0 / (1.0 / (2.0 * t1) + 1.0 / tphi)
        ro = np.clip(truth["readout_error"] * (1.0 + rel * rng.standard_normal(truth["readout_error"].shape)), 1e-6, 0.5)
        epc = np.clip(truth["epc_1q"] * (1.0 + rel * rng.standard_normal(truth["epc_1q"].shape)), 1e-8, 0.5)
        g2 = np.clip(truth["gate_error_2q"] * (1.0 + rel * rng.standard_normal(truth["gate_error_2q"].shape)), 1e-8, 0.9)
        return {"t1_us": t1, "t2_us": t2, "readout_error": ro, "epc_1q": epc, "gate_error_2q": g2}

    # ---------------------------------------------------------------- simulation
    def _simulate_device(
        self, dev_idx: int, n_days: int, snapshots_per_day: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.cfg
        spec = dict(cfg.topologies[dev_idx])
        kind = spec.pop("kind")
        n_q, edges = make_topology(kind, **spec)
        edge_u = np.array([e[0] for e in edges], dtype=np.int64)
        edge_v = np.array([e[1] for e in edges], dtype=np.int64)
        n_e = len(edges)
        dev = f"sim_dev_{dev_idx:02d}"
        context = json.dumps({"topology": kind, **spec}, sort_keys=True)
        rng = np.random.default_rng([self.seed, dev_idx])

        # static per-device draws
        dev_off = float(rng.normal(0.0, cfg.t1_device_spread))
        mu_t1 = np.log(cfg.t1_median_us) + dev_off + rng.normal(0.0, cfg.t1_qubit_spread, n_q)
        mu_tphi = np.log(cfg.tphi_median_us) + dev_off + rng.normal(0.0, cfg.tphi_qubit_spread, n_q)
        mu_ro = np.log(cfg.readout_median) + rng.normal(0.0, cfg.readout_qubit_spread, n_q)
        mu_cp = np.zeros(n_e)

        x_t1 = proc.ou_init(mu_t1, cfg.ou_theta_t1, cfg.ou_sigma_t1, rng)
        x_tphi = proc.ou_init(mu_tphi, cfg.ou_theta_tphi, cfg.ou_sigma_tphi, rng)
        x_ro = proc.ou_init(mu_ro, cfg.readout_theta, cfg.readout_sigma, rng)
        x_cp = proc.ou_init(mu_cp, cfg.coupler_theta, cfg.coupler_sigma, rng)
        tls_on = np.zeros(n_q, dtype=bool)
        tls_fac = np.ones(n_q, dtype=np.float64)

        n_groups = max(int(cfg.recal_groups), 1)
        q_group = np.arange(n_q) % n_groups
        e_group = q_group[edge_u]  # a coupler is recalibrated with its lower qubit
        recal_streams = [
            proc.sample_recal_times(n_days * 24.0, cfg.recal_median_h, cfg.recal_sigma_log, rng) / 24.0
            for _ in range(n_groups)]
        episodes = proc.sample_degradation_episodes(
            n_days, cfg.degrade_rate_per_day, cfg.degrade_mean_days,
            cfg.degrade_drift_per_day, rng)
        offline = proc.sample_offline_days(
            n_days, cfg.offline_p_mcar, cfg.offline_burst_rate, cfg.offline_burst_geo_p, rng)
        # daily full recalibration: a sweep that recalibrates the groups one after
        # another within the hour starting at reset_hour, each anchoring its own reset
        reset_h = cfg.reset_hour + np.arange(n_groups) / n_groups
        if cfg.reset_hour >= 0:
            for g in range(n_groups):
                recal_streams[g] = np.sort(np.concatenate(
                    [recal_streams[g], (np.arange(n_days) * 24.0 + reset_h[g]) / 24.0]))
        # recalibration events in time order as (day, group)
        events = sorted((float(t), g) for g, ts in enumerate(recal_streams) for t in ts)

        n_snap = int(n_days * snapshots_per_day)
        dt = 1.0 / snapshots_per_day
        snap_t = np.arange(n_snap, dtype=np.float64) * dt  # days

        state = {"t1": x_t1, "tphi": x_tphi, "ro": x_ro, "cp": x_cp,
                 "on": tls_on, "fac": tls_fac}

        def advance(dtau: float, t_end: float) -> None:
            if dtau <= 0.0:
                return
            mu_eff = mu_t1 + proc.degradation_offset(episodes, t_end)
            state["t1"] = proc.ou_step(state["t1"], mu_eff, cfg.ou_theta_t1, cfg.ou_sigma_t1, dtau, rng)
            state["tphi"] = proc.ou_step(state["tphi"], mu_tphi, cfg.ou_theta_tphi, cfg.ou_sigma_tphi, dtau, rng)
            state["ro"] = proc.ou_step(state["ro"], mu_ro, cfg.readout_theta, cfg.readout_sigma, dtau, rng)
            state["cp"] = proc.ou_step(state["cp"], mu_cp, cfg.coupler_theta, cfg.coupler_sigma, dtau, rng)
            state["on"], state["fac"] = proc.tls_step(
                state["on"], state["fac"], cfg.tls_on_rate, cfg.tls_off_rate,
                cfg.tls_factor_lo, cfg.tls_factor_hi, dtau, rng)

        def reset_offset(t_day: float) -> np.ndarray | None:
            """Per-qubit log-T1 sawtooth: declines after its group's daily reset, centred on the day."""
            if cfg.reset_hour < 0 or not cfg.reset_decay_per_h:
                return None
            since_h = (t_day * 24.0 - reset_h[q_group]) % 24.0
            return -cfg.reset_decay_per_h * (since_h - 12.0)

        def truth_now(t_day: float) -> dict[str, np.ndarray]:
            return self._truth_obs(
                state["t1"], state["tphi"], state["ro"], state["cp"], state["fac"],
                mu_t1 + proc.degradation_offset(episodes, t_day),
                edge_u, edge_v, rng, t1_offset=reset_offset(t_day))

        def repin(t_day: float, gq: np.ndarray | None = None, ge: np.ndarray | None = None) -> None:
            """Partially re-pin the latents; gq/ge restrict it to one calibration group."""
            mu_eff = mu_t1 + proc.degradation_offset(episodes, t_day)
            for key, mu, sel in (("t1", mu_eff, gq), ("tphi", mu_tphi, gq),
                                 ("ro", mu_ro, gq), ("cp", mu_cp, ge)):
                new = proc.recal_repin(state[key], mu, cfg.recal_kappa, cfg.recal_jump_sigma, rng)
                state[key] = new if sel is None else np.where(sel, new, state[key])

        # initial calibration at t = 0
        claimed = self._observe(truth_now(0.0), cfg.claimed_noise, rng)
        last_cal_q = np.zeros(n_q, dtype=np.float64)
        last_cal_e = np.zeros(n_e, dtype=np.float64)

        traj: dict[str, dict[str, list[np.ndarray]]] = {
            "truth": {o: [] for o in (*TRUTH_QUBIT_OBS, EDGE_OBS)},
            "probe": {o: [] for o in (*VISIBLE_QUBIT_OBS, EDGE_OBS)},
            "claimed": {o: [] for o in (*VISIBLE_QUBIT_OBS, EDGE_OBS)},
        }
        cal_q_arr = np.empty((n_snap, n_q), dtype=np.float64)
        cal_e_arr = np.empty((n_snap, n_e), dtype=np.float64)

        t_cur = 0.0
        ri = 0
        for k in range(n_snap):
            t_target = snap_t[k]
            while ri < len(events) and events[ri][0] <= t_target:
                tr, g = events[ri]
                advance(tr - t_cur, tr)
                t_cur = tr
                if n_groups == 1:  # device-wide recalibration
                    repin(tr)
                    claimed = self._observe(truth_now(tr), cfg.claimed_noise, rng)
                    last_cal_q[:] = tr
                    last_cal_e[:] = tr
                else:  # one calibration group
                    gq, ge = q_group == g, e_group == g
                    repin(tr, gq, ge)
                    fresh = self._observe(truth_now(tr), cfg.claimed_noise, rng)
                    claimed = {o: np.where(ge if o == EDGE_OBS else gq, fresh[o], v)
                               for o, v in claimed.items()}
                    last_cal_q[gq] = tr
                    last_cal_e[ge] = tr
                ri += 1
            advance(t_target - t_cur, t_target)
            t_cur = t_target

            truth = truth_now(t_target)
            probe = self._observe(truth, cfg.probe_noise, rng)
            for o in traj["truth"]:
                traj["truth"][o].append(truth[o])
            for o in traj["probe"]:
                traj["probe"][o].append(probe[o])
            for o in traj["claimed"]:
                traj["claimed"][o].append(claimed[o])
            cal_q_arr[k] = last_cal_q
            cal_e_arr[k] = last_cal_e

        # ----------------------------------------------------------- row assembly
        q_ids = np.array([f"q{i}" for i in range(n_q)], dtype=object)
        c_ids = np.array([f"c{a}-{b}" for a, b in edges], dtype=object)
        day_idx = np.minimum(snap_t.astype(np.int64), n_days - 1)
        keep_vis = ~offline[day_idx]
        keep_all = np.ones(n_snap, dtype=bool)

        cols: dict[str, list] = {k: [] for k in (
            "entity_type", "entity_id", "observable", "channel", "value", "sigma", "ts_h", "cal_h")}

        def emit(channel: str, obs: str, ent_type: str, ids: np.ndarray, cal: np.ndarray,
                 keep: np.ndarray, sig_rel: float | None) -> None:
            vals = np.stack(traj[channel][obs], axis=0)[keep]  # (n_keep, n_ent)
            n_keep, n_ent = vals.shape
            if n_keep == 0 or n_ent == 0:
                return
            flat = vals.ravel().astype(np.float64)
            cols["value"].append(flat)
            cols["sigma"].append(
                np.full(flat.shape, np.nan) if sig_rel is None else sig_rel * np.abs(flat))
            cols["ts_h"].append(np.repeat(snap_t[keep] * 24.0, n_ent))
            cols["cal_h"].append((cal[keep] * 24.0).ravel())
            cols["entity_id"].append(np.tile(ids, n_keep))
            n_rows = n_keep * n_ent
            cols["entity_type"].append(np.full(n_rows, ent_type, dtype=object))
            cols["observable"].append(np.full(n_rows, obs, dtype=object))
            cols["channel"].append(np.full(n_rows, channel, dtype=object))

        for obs in TRUTH_QUBIT_OBS:
            emit("truth", obs, "qubit", q_ids, cal_q_arr, keep_all, None)
        emit("truth", EDGE_OBS, "coupler", c_ids, cal_e_arr, keep_all, None)
        for channel, rel in (("claimed", cfg.claimed_noise), ("probe", cfg.probe_noise)):
            for obs in VISIBLE_QUBIT_OBS:
                emit(channel, obs, "qubit", q_ids, cal_q_arr, keep_vis, rel)
            emit(channel, EDGE_OBS, "coupler", c_ids, cal_e_arr, keep_vis, rel)

        start_ts = pd.Timestamp(cfg.start)
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize("UTC")

        def build(channels: set[str]) -> pd.DataFrame:
            sel = np.concatenate([np.isin(a, list(channels)) for a in cols["channel"]])
            data = {k: np.concatenate(cols[k])[sel] for k in cols}
            df = pd.DataFrame({
                "source": SOURCE, "platform": PLATFORM, "device": dev,
                "entity_type": data["entity_type"], "entity_id": data["entity_id"],
                "observable": data["observable"], "channel": data["channel"],
                "value": data["value"].astype(np.float64),
                "sigma": data["sigma"].astype(np.float64),
                "ts": start_ts + pd.to_timedelta(data["ts_h"], unit="h"),
                "cal_ts": start_ts + pd.to_timedelta(data["cal_h"], unit="h"),
                "context": context,
            })
            return df[uts.COLUMNS]

        return build({"claimed", "probe"}), build({"truth"})

    def run(
        self, n_days: int, snapshots_per_day: int = 3
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Simulate the whole fleet.

        Returns ``(uts_events, truth_events)``: the first holds the
        model-visible ``claimed``/``probe`` channels (validate_uts-clean, with
        device-offline gaps applied), the second the ``channel="truth"``
        ground-truth latents at every snapshot (no gaps).
        """
        vis_frames, truth_frames = [], []
        for i in range(self.cfg.n_devices):
            v, t = self._simulate_device(i, int(n_days), int(snapshots_per_day))
            vis_frames.append(v)
            truth_frames.append(t)
        vis = pd.concat(vis_frames, ignore_index=True)
        truth = pd.concat(truth_frames, ignore_index=True)
        return uts.validate_uts(vis), uts.validate_uts(truth)


# ------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m ltn.sim.fleet",
        description="Generate synthetic fleet telemetry into a UTS lake [DEV-2].")
    p.add_argument("--config", type=Path, default=None, help="FleetConfig JSON (default: built-in defaults)")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--snapshots-per-day", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--out", type=Path, default=None, help="UTS root for model-visible events")
    p.add_argument("--truth", type=Path, default=None, help="separate UTS root for channel='truth' latents")
    p.add_argument("--write-default-config", type=Path, default=None, metavar="PATH",
                   help="write the default FleetConfig JSON to PATH and exit")
    args = p.parse_args(argv)

    if args.write_default_config is not None:
        FleetConfig().to_json(args.write_default_config)
        print(f"wrote default config -> {args.write_default_config}")
        return 0

    if args.out is None or args.truth is None:
        p.error("--out and --truth are both required (truth latents live in a separate root)")

    cfg = FleetConfig.from_json(args.config.read_text()) if args.config else FleetConfig()
    sim = FleetSim(cfg, args.seed)
    events, truth = sim.run(args.days, args.snapshots_per_day)
    files_out = uts.write_uts(events, args.out)
    files_truth = uts.write_uts(truth, args.truth)
    print(f"visible: {len(events)} rows -> {len(files_out)} files under {args.out}")
    print(f"truth  : {len(truth)} rows -> {len(files_truth)} files under {args.truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
