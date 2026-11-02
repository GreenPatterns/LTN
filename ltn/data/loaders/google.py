"""UTS loader for Google's open QEC memory-experiment releases (source="google_qec").

Implements Phase A task A3 of the master plan: every experiment directory ("run")
inside the released zips is mapped into Universal Telemetry Schema events
(:mod:`ltn.data.uts`). Guarantees, per run:

1. A ``subgraph`` entity keyed by the circuit's qubit coordinates
   (``uts.subgraph_id`` over integer-encoded QUBIT_COORDS; stable dir-name hash
   if a circuit carries no coordinates), with the key observables duplicated at
   ``entity_type="device"``.
2. ``dem_error_prob`` (channel="claimed"): the detector-error-model probabilities
   aggregated per qubit coordinate touched by each error's detectors — the
   calibration-analog input for zero-shot experiment E5 — plus a device-mean event.
   The DEM used for the claim prefers a-priori noise models
   (circuit_detector_error_model.dem in 2022; *_si1000_prior in 2024) over
   data-fitted ones (pij_*, *_rl_optimized_*).
3. ``detection_event_fraction`` (channel="derived") from provided detection-event
   files, else an m2d conversion of the raw measurements.
4. ``logical_error_per_round`` (channel="delivered"): pymatching decode of a
   ``max_shots`` subsample; per-shot error P over r rounds is converted with
   eps = 0.5*(1-(1-2P)**(1/r)); sigma is binomial (delta method for eps). When
   live decoding fails, or the code is not MWPM-decodable (the color-code
   record, where pymatching runs but at near-chance accuracy), the loader uses
   the dataset's own predicted-vs-actual observable flips and notes it in
   context ("decoder": "provided:<name>"; "decode_fallback" is true only when
   matching should have worked).

Timestamps are pseudo-time — a fixed base date per record (sycamore_2022:
2022-07-01, 105q: 2024-08-01, dynamic/color: 2024-12-01) plus run-index hours —
and are flagged with context["pseudo_time"]=true.

Two on-disk layouts are handled:
- 2022 Sycamore (``google_qec3v5_experiment_data.zip``): flat top-level dirs
  ``{code}_b{basis}_d{distance}_r{rounds}_center_{row}_{col}`` with
  circuit_ideal.stim, circuit_detector_error_model.dem, detection_events.b8,
  obs_flips_actual.01, obs_flips_predicted_by_<decoder>.01, properties.yml.
- 2024 releases (105Q / dynamic / color): nested ``<placement>/<basis>/r<NN>/``
  dirs with circuit_ideal.stim, circuit_noisy_si1000.stim, metadata.json,
  measurements.b8, sweep_bits.b8, optionally detection_events.b8 and
  obs_flips_actual.b8, and decoding_results/<decoder>/{error_model.dem,
  obs_flips_predicted.b8}.

Zip members are extracted lazily and idempotently to
``<raw_dir>/extracted/<record>/...`` (bit-files truncated to the requested shot
head) because ``stim.read_shot_data_file`` requires real files.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pymatching
import stim

from ltn.data import uts

__all__ = [
    "GOOGLE_RECORDS",
    "load_google",
    "coords_subgraph_id",
    "dem_per_qubit_error_probs",
    "measurements_to_detections",
    "decode_logical_error",
    "per_round_error_rate",
    "per_round_sigma",
    "binomial_sigma",
]


@dataclass(frozen=True)
class GoogleRecord:
    key: str
    zip_name: str
    device: str
    base_ts: str  # pseudo-time base date (documented in module docstring)
    # False for codes MWPM cannot decode meaningfully (color code): pymatching
    # still *runs* on those DEMs but at near-chance accuracy, so the loader uses
    # the dataset's provided decoder predictions instead.
    matchable: bool = True


GOOGLE_RECORDS: dict[str, GoogleRecord] = {
    r.key: r
    for r in [
        GoogleRecord("sycamore_2022", "google_qec3v5_experiment_data.zip",
                     "google_sycamore_2022", "2022-07-01"),
        GoogleRecord("105q", "google_105Q_surface_code_d3_d5_d7.zip",
                     "google_105q", "2024-08-01"),
        GoogleRecord("dynamic", "google_dynamic_circuits_d3_d5.zip",
                     "google_dynamic", "2024-12-01"),
        GoogleRecord("color", "google_superdense_color_code_d3_d5.zip",
                     "google_color", "2024-12-01", matchable=False),
    ]
}

_NAME_2022 = re.compile(
    r"(?P<code>[a-z_]+)_b(?P<basis>[XZ])_d(?P<distance>\d+)_r(?P<rounds>\d+)"
    r"_center_(?P<row>\d+)_(?P<col>\d+)$"
)


# --------------------------------------------------------------------------- math
def per_round_error_rate(p_shot: float, rounds: int) -> float:
    """Per-round logical error rate from per-shot error probability over `rounds`.

    eps = 0.5*(1-(1-2P)**(1/r)); clamped so P outside [0, 0.5] maps to the boundary
    (P >= 0.5 means the decoder carries no information beyond coin-flipping).
    """
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")
    if p_shot <= 0.0:
        return 0.0
    if p_shot >= 0.5:
        return 0.5
    return 0.5 * (1.0 - (1.0 - 2.0 * p_shot) ** (1.0 / rounds))


def binomial_sigma(p: float, n: int) -> float:
    """Binomial standard error of a proportion, floored at 1/n so p in {0,1} is not sigma=0."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return math.sqrt(max(p * (1.0 - p), 1.0 / n) / n)


def per_round_sigma(p_shot: float, sigma_shot: float, rounds: int) -> float:
    """Delta-method sigma of per_round_error_rate given binomial sigma of the per-shot P."""
    if rounds <= 0:
        raise ValueError(f"rounds must be positive, got {rounds}")
    # Clamp away from 0.5 where the derivative diverges.
    p = min(max(p_shot, 0.0), 0.499)
    deriv = (1.0 / rounds) * (1.0 - 2.0 * p) ** (1.0 / rounds - 1.0)
    return abs(deriv) * sigma_shot


def coords_subgraph_id(coords: Iterable[Sequence[float]]) -> str:
    """Stable subgraph id from 2D qubit coordinates.

    Each (row, col) is encoded as an integer 2*row*100003 + 2*col (doubling keeps
    half-integer grid coords injective) and hashed via uts.subgraph_id.
    """
    ints = [
        int(round(2.0 * float(c[0]))) * 100003 + int(round(2.0 * float(c[1])))
        for c in coords
    ]
    return uts.subgraph_id(ints)


def dem_per_qubit_error_probs(
    dem: stim.DetectorErrorModel,
    detector_coords: Mapping[int, Sequence[float]] | None = None,
) -> dict[tuple[float, float], float]:
    """Mean error probability per qubit coordinate touched by each DEM error's detectors.

    The DEM is flattened; for every `error(p)` instruction, p contributes to the
    (row, col) of every detector the error flips (first two coordinate dims).
    Returns {(row, col): mean probability}.
    """
    flat = dem.flattened()
    if detector_coords is None:
        detector_coords = flat.get_detector_coordinates()
    acc: dict[tuple[float, float], list[float]] = {}
    for inst in flat:
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        touched: set[tuple[float, float]] = set()
        for t in inst.targets_copy():
            if not t.is_relative_detector_id():
                continue
            c = detector_coords.get(t.val)
            if c is not None and len(c) >= 2:
                touched.add((float(c[0]), float(c[1])))
        for qc in touched:
            acc.setdefault(qc, []).append(p)
    return {qc: float(np.mean(ps)) for qc, ps in sorted(acc.items())}


def measurements_to_detections(
    circuit: stim.Circuit,
    measurements: np.ndarray,
    sweep_bits: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw measurement shots to (detection_events, observable_flips) via stim m2d."""
    converter = circuit.compile_m2d_converter()
    kwargs: dict = {"measurements": measurements.astype(np.bool_), "separate_observables": True}
    if sweep_bits is not None:
        kwargs["sweep_bits"] = sweep_bits.astype(np.bool_)
    dets, obs = converter.convert(**kwargs)
    return dets, obs


def decode_logical_error(
    dem: stim.DetectorErrorModel,
    detection_events: np.ndarray,
    obs_actual: np.ndarray,
) -> tuple[float, int]:
    """Decode with pymatching; return (per-shot logical error P, n shots compared).

    A shot counts as a logical error when ANY observable prediction mismatches.
    Raises whatever pymatching raises for non-graphlike DEMs (caller handles fallback).
    """
    matching = pymatching.Matching.from_detector_error_model(dem)
    n = min(len(detection_events), len(obs_actual))
    pred = matching.decode_batch(detection_events[:n].astype(np.bool_))
    actual = np.atleast_2d(obs_actual[:n].astype(np.bool_))
    pred = np.asarray(pred, dtype=np.bool_).reshape(n, -1)
    return float(np.mean(np.any(pred != actual.reshape(n, -1), axis=1))), n


# ---------------------------------------------------------------- zip plumbing
def _extract_head(zf: zipfile.ZipFile, member: str, dest: Path, n_bytes: int) -> Path:
    """Idempotently extract the first `n_bytes` of a zip member to `dest`."""
    want = min(n_bytes, zf.getinfo(member).file_size)
    if dest.exists() and dest.stat().st_size >= want:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with zf.open(member) as src, open(tmp, "wb") as out:
        remaining = want
        while remaining > 0:
            chunk = src.read(min(1 << 20, remaining))
            if not chunk:
                break
            out.write(chunk)
            remaining -= len(chunk)
    tmp.replace(dest)  # atomic within the same filesystem
    return dest


def _read_bits_head(
    zf: zipfile.ZipFile,
    member: str,
    extract_dir: Path,
    fmt: str,
    bits_per_shot: int,
    max_shots: int,
) -> np.ndarray:
    """Read up to max_shots shots of a .b8/.01 member as a bool array (shots, bits)."""
    if bits_per_shot <= 0:
        raise ValueError(f"bits_per_shot must be positive for {member}")
    # b8 pads each shot to a byte boundary; 01 stores one ASCII char per bit + newline.
    bytes_per_shot = (bits_per_shot + 7) // 8 if fmt == "b8" else bits_per_shot + 1
    size = zf.getinfo(member).file_size
    n_avail = size // bytes_per_shot
    n_use = min(max_shots, n_avail)
    if n_use == 0:
        raise ValueError(f"no complete shots in {member} ({size} bytes)")
    path = _extract_head(zf, member, extract_dir / member, n_use * bytes_per_shot)
    # read_shot_data_file only needs the total bits-per-shot; declare them as measurements.
    data = stim.read_shot_data_file(
        path=str(path), format=fmt, num_measurements=bits_per_shot
    )
    return np.asarray(data, dtype=np.bool_)[:n_use]


# ------------------------------------------------------------- run-level logic
def _run_params(rec: GoogleRecord, zf: zipfile.ZipFile, run: str,
                circuit: stim.Circuit, rel_members: Sequence[str]) -> dict:
    """Parse code/basis/distance/rounds from the dir name (2022) or path+metadata (2024)."""
    m = _NAME_2022.match(run) if "/" not in run else None
    params: dict = {"code": None, "basis": None, "distance": None, "rounds": None}
    if m:
        params.update(
            code=m.group("code"), basis=m.group("basis"),
            distance=int(m.group("distance")), rounds=int(m.group("rounds")),
        )
        return params
    if "metadata.json" in rel_members:
        meta = json.loads(zf.read(f"{run}/metadata.json"))
        params["basis"] = meta.get("basis")
        params["rounds"] = meta.get("rounds")
        if "distance" in meta:
            params["distance"] = int(meta["distance"])
    for comp in run.split("/"):
        if params["basis"] is None and comp in ("X", "Z"):
            params["basis"] = comp
        dm = re.match(r"^d(\d+)", comp)
        if params["distance"] is None and dm:
            params["distance"] = int(dm.group(1))
        rm = re.fullmatch(r"r(\d+)", comp)
        if params["rounds"] is None and rm:
            params["rounds"] = int(rm.group(1))
    if params["rounds"] is None:
        # Last resort: DETECTOR time coordinates run 0..rounds in these releases.
        ts = [c[2] for c in circuit.get_detector_coordinates().values() if len(c) >= 3]
        if ts:
            params["rounds"] = int(max(ts))
    params["code"] = params["code"] or ("color_code" if rec.key == "color" else "surface_code")
    return params


def _dem_rank(rel: str) -> tuple:
    """Claim-preference order for DEMs.

    A-priori noise-model claims first — circuit_detector_error_model.dem (2022),
    then decoder DEMs with an si1000 prior (2024) — ahead of data-fitted models
    (pij_*, rl-optimized, frequency-calibrated). Within a tier, matching-decoder
    DEMs are preferred because they are graphlike (pymatching-decodable).
    """
    name = rel.lower()
    matching = 0 if "matching" in name else 1
    if rel == "circuit_detector_error_model.dem":
        return (0, 0, rel)
    if "si1000" in name:
        return (1, matching, rel)
    if name.startswith("pij_"):
        return (2, 0, rel)
    if "/" not in rel:
        return (3, 0, rel)
    return (4, matching, rel)


def _pick_dems(rel_members: Sequence[str]) -> list[str]:
    return sorted((m for m in rel_members if m.endswith(".dem")), key=_dem_rank)


def _load_run(
    zf: zipfile.ZipFile,
    rec: GoogleRecord,
    run: str,
    run_idx: int,
    extract_dir: Path,
    max_shots: int,
    decode: bool,
) -> pd.DataFrame:
    prefix = run + "/"
    rel_members = [n[len(prefix):] for n in zf.namelist()
                   if n.startswith(prefix) and not n.endswith("/")]
    circuit = stim.Circuit(zf.read(f"{run}/circuit_ideal.stim").decode())
    params = _run_params(rec, zf, run, circuit, rel_members)
    qubit_coords = circuit.get_final_qubit_coordinates()
    # Coordinate hash normally; stable dir-name hash if a circuit has no coords.
    sub_id = (coords_subgraph_id(qubit_coords.values()) if qubit_coords
              else "sub:" + hashlib.sha1(run.encode()).hexdigest()[:8])
    ts = pd.Timestamp(rec.base_ts, tz="UTC") + pd.Timedelta(hours=run_idx)

    context: dict = {
        "record": rec.key, "dir": run, "code": params["code"],
        "basis": params["basis"], "distance": params["distance"],
        "rounds": params["rounds"], "pseudo_time": True,
    }

    # -- (2) DEM -> per-qubit claimed error probs + device mean
    dems = _pick_dems(rel_members)
    per_qubit: dict[tuple[float, float], float] = {}
    dem_obj: stim.DetectorErrorModel | None = None
    if dems:
        dem_obj = stim.DetectorErrorModel(zf.read(f"{run}/{dems[0]}").decode())
        det_coords = circuit.get_detector_coordinates() or dem_obj.get_detector_coordinates()
        per_qubit = dem_per_qubit_error_probs(dem_obj, det_coords)
        context["dem"] = dems[0]

    # -- (3) detection events + actual observable flips
    dets: np.ndarray | None = None
    obs_actual: np.ndarray | None = None
    det_source = None
    if "detection_events.b8" in rel_members and circuit.num_detectors > 0:
        dets = _read_bits_head(zf, f"{run}/detection_events.b8", extract_dir,
                               "b8", circuit.num_detectors, max_shots)
        det_source = "provided"
    for obs_file, fmt in (("obs_flips_actual.01", "01"), ("obs_flips_actual.b8", "b8")):
        if obs_file in rel_members and circuit.num_observables > 0:
            obs_actual = _read_bits_head(zf, f"{run}/{obs_file}", extract_dir,
                                         fmt, circuit.num_observables, max_shots)
            break
    if dets is None or (decode and obs_actual is None):
        m2d = _try_m2d(zf, rec, run, rel_members, circuit, extract_dir, max_shots)
        if m2d is not None:
            if dets is None:
                dets, det_source = m2d[0], "m2d"
            if obs_actual is None:
                obs_actual = m2d[1]

    frames: list[pd.DataFrame] = []
    n_used = 0
    sub_rows: list[tuple[str, str, float, float | None]] = []

    if dets is not None:
        n_used = len(dets)
        frac = float(dets.mean())
        # Conservative binomial n: shots (detector bits within a shot are correlated).
        sub_rows.append(("detection_event_fraction", "derived", frac,
                         binomial_sigma(frac, n_used)))
        context["det_source"] = det_source

    # -- (4) delivered logical error per round
    if decode and dets is not None and obs_actual is not None and params["rounds"]:
        p_shot = n_cmp = None
        for dem_rel in (dems if rec.matchable else []):
            try:
                cand = (dem_obj if dem_rel == dems[0]
                        else stim.DetectorErrorModel(zf.read(f"{run}/{dem_rel}").decode()))
                p_shot, n_cmp = decode_logical_error(cand, dets, obs_actual)
            except Exception:  # non-graphlike DEM etc. -> try next, then fallback
                continue
            context["decoder"] = "pymatching"
            context["decode_dem"] = dem_rel
            context["decode_fallback"] = False
            break
        if p_shot is None:
            fb = _provided_prediction_error(zf, run, rel_members, circuit,
                                            extract_dir, obs_actual, max_shots)
            if fb is not None:
                p_shot, n_cmp, dec_name = fb
                context["decoder"] = f"provided:{dec_name}"
                # Only a true fallback when matching *should* have worked;
                # for non-matchable codes provided predictions are the plan.
                context["decode_fallback"] = rec.matchable
        if p_shot is not None and n_cmp:
            n_used = max(n_used, n_cmp)
            eps = per_round_error_rate(p_shot, params["rounds"])
            sig = per_round_sigma(p_shot, binomial_sigma(p_shot, n_cmp), params["rounds"])
            sub_rows.append(("logical_error_per_round", "delivered", eps, sig))
            context["p_logical_per_shot"] = p_shot

    context["n_shots_used"] = n_used

    if sub_rows:
        frames.append(uts.make_events(
            source="google_qec", platform="google_sc", device=rec.device,
            entity_type="subgraph", entity_id=sub_id, ts=ts, cal_ts=ts,
            rows=sub_rows, context=context,
        ))
    dev_rows = list(sub_rows)
    if per_qubit:
        for (r, c), p in per_qubit.items():
            frames.append(uts.make_events(
                source="google_qec", platform="google_sc", device=rec.device,
                entity_type="qubit", entity_id=f"q{r:g}_{c:g}", ts=ts, cal_ts=ts,
                rows=[("dem_error_prob", "claimed", p, None)], context=context,
            ))
        dev_rows.append(("dem_error_prob", "claimed",
                         float(np.mean(list(per_qubit.values()))), None))
    if dev_rows:
        frames.append(uts.make_events(
            source="google_qec", platform="google_sc", device=rec.device,
            entity_type="device", entity_id=rec.device, ts=ts, cal_ts=ts,
            rows=dev_rows, context=context,
        ))
    if not frames:
        return uts.empty_uts()
    return pd.concat(frames, ignore_index=True)


def _try_m2d(
    zf: zipfile.ZipFile,
    rec: GoogleRecord,
    run: str,
    rel_members: Sequence[str],
    circuit: stim.Circuit,
    extract_dir: Path,
    max_shots: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """m2d-convert a shot head of measurements(.b8) [+ sweep bits] to (dets, obs)."""
    if "measurements.b8" not in rel_members or circuit.num_measurements == 0:
        return None
    meas = _read_bits_head(zf, f"{run}/measurements.b8", extract_dir,
                           "b8", circuit.num_measurements, max_shots)
    sweep = None
    if circuit.num_sweep_bits > 0:
        sweep_file = next((f for f in ("sweep.b8", "sweep_bits.b8") if f in rel_members), None)
        if sweep_file is None:
            return None  # sweep-controlled gates without sweep data -> wrong detectors
        sweep = _read_bits_head(zf, f"{run}/{sweep_file}", extract_dir,
                                "b8", circuit.num_sweep_bits, max_shots)
        n = min(len(meas), len(sweep))
        meas, sweep = meas[:n], sweep[:n]
    try:
        return measurements_to_detections(circuit, meas, sweep)
    except Exception as e:
        warnings.warn(f"[google_qec] m2d failed for {rec.key}:{run}: {e}")
        return None


def _provided_prediction_error(
    zf: zipfile.ZipFile,
    run: str,
    rel_members: Sequence[str],
    circuit: stim.Circuit,
    extract_dir: Path,
    obs_actual: np.ndarray,
    max_shots: int,
) -> tuple[float, int, str] | None:
    """Fallback: per-shot error from dataset-provided predicted vs actual obs flips."""
    cands: list[tuple[int, int, str, str, str]] = []  # (pref, si_pref, name, member, fmt)
    for m in rel_members:
        low = m.lower()
        m01 = re.fullmatch(r"obs_flips_predicted_by_(\w+)\.01", m)
        if m01:
            name = m01.group(1)
            cands.append((0 if "pymatching" in name else 1, 0, name, m, "01"))
        elif low.endswith("obs_flips_predicted.b8") and "/" in m:
            name = m.split("/")[-2]
            cands.append((0 if "matching" in name else 1,
                          0 if "si1000" in name else 1, name, m, "b8"))
    if not cands or circuit.num_observables == 0:
        return None
    _, _, name, member, fmt = sorted(cands)[0]
    pred = _read_bits_head(zf, f"{run}/{member}", extract_dir, fmt,
                           circuit.num_observables, max_shots)
    n = min(len(pred), len(obs_actual))
    if n == 0:
        return None
    p = float(np.mean(np.any(pred[:n] != obs_actual[:n], axis=1)))
    return p, n, name


# ------------------------------------------------------------------ public API
def _match_run(run_filter, run: str) -> bool:
    if run_filter is None:
        return True
    if callable(run_filter):
        return bool(run_filter(run))
    if isinstance(run_filter, str):
        return run_filter in run
    return any(f in run for f in run_filter)


def load_google(
    raw_dir: str | Path,
    records: Sequence[str] | str | None = None,
    max_shots: int = 20000,
    max_runs: int | None = None,
    decode: bool = True,
    run_filter: str | Sequence[str] | Callable[[str], bool] | None = None,
) -> pd.DataFrame:
    """Load Google QEC zips under `raw_dir` (data/raw/google) into a validated UTS frame.

    records: keys from GOOGLE_RECORDS ("sycamore_2022", "105q", "dynamic", "color");
        None = every record whose zip is present and readable.
    max_shots: per-run shot-subsample cap (first N shots, deterministic).
    max_runs: cap on run directories loaded per record (applied after run_filter).
    decode: run the decode step (4); detection/DEM events emit regardless.
    run_filter: substring, list of substrings, or predicate over the run dir path.
    """
    if max_shots < 1:
        raise ValueError("max_shots must be >= 1")
    raw_dir = Path(raw_dir)
    if isinstance(records, str):
        records = [records]
    if records is None:
        recs = list(GOOGLE_RECORDS.values())
    else:
        unknown = set(records) - set(GOOGLE_RECORDS)
        if unknown:
            raise ValueError(
                f"unknown records {sorted(unknown)}; available: {sorted(GOOGLE_RECORDS)}")
        recs = [GOOGLE_RECORDS[k] for k in records]

    frames: list[pd.DataFrame] = []
    for rec in recs:
        zpath = raw_dir / rec.zip_name
        if not zpath.exists():
            if records is not None:
                warnings.warn(f"[google_qec] {rec.key}: missing zip {zpath}")
            continue
        try:
            zf = zipfile.ZipFile(zpath)
        except zipfile.BadZipFile:
            warnings.warn(f"[google_qec] {rec.key}: {zpath.name} unreadable "
                          "(download still in progress?) — skipped")
            continue
        with zf:
            run_dirs = sorted({n[: -len("/circuit_ideal.stim")] for n in zf.namelist()
                               if n.endswith("/circuit_ideal.stim")})
            extract_dir = raw_dir / "extracted" / rec.key
            n_loaded = 0
            # run_idx over the FULL sorted list so pseudo-timestamps are stable
            # regardless of the filter used.
            for run_idx, run in enumerate(run_dirs):
                if not _match_run(run_filter, run):
                    continue
                if max_runs is not None and n_loaded >= max_runs:
                    break
                try:
                    df = _load_run(zf, rec, run, run_idx, extract_dir, max_shots, decode)
                except Exception as e:
                    warnings.warn(f"[google_qec] failed run {rec.key}:{run}: {e!r}")
                    continue
                if len(df):
                    frames.append(df)
                    n_loaded += 1
    if not frames:
        return uts.empty_uts()
    out = pd.concat(frames, ignore_index=True)
    return uts.validate_uts(out)
