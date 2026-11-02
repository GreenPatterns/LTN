"""UTS loader for the silicon spin-qubit CZ-gate dataset (source="silicon").

Implements Phase A task A3 of the master plan for Zenodo record 15653466
(Tanttu et al., "Assessment of error variation in high-fidelity two-qubit
gates in silicon", Nature Physics 2024; arXiv:2303.04090): two-qubit CZ gate
error variation across SiMOS devices A/B/C, mapped to UTS devices
``simos_a`` / ``simos_b`` / ``simos_c`` (platform="si_spin").

Guarantees (what is emitted, and from which file):

1. ``cz_fidelity`` (channel="delivered", entity_type="device") — the FBT
   (fast Bayesian tomography) CZ/DCZ fidelity traces from
   ``AncillaryFiles/fidelity_statistics_allcoldboot.mat`` (A: 300 pts,
   B: 500 pts, C: 592 pts); device B's trace is taken from
   ``AncillaryFiles/DeviceB_26008.mat`` which additionally provides per-point
   1-sigma errors and real lab-time minute offsets.  Per-run interleaved-RB /
   FBT-summary / GST CZ fidelities from ``AncillaryFiles/table_of_RBruns.txt``
   are added as episodic events (context["method"] in {"IRB","FBT","GST"}).
2. ``gate_error_2q`` (channel="derived") — 1 - cz_fidelity for every
   cz_fidelity event, clipped at 0.
3. ``epc_1q`` (channel="derived", entity_type="qubit", entity_id="q1"/"q2")
   — 1 - fidelity of the single-qubit X gates (x1/x2) from the same FBT
   traces and RB-table rows.
4. ``t2_us`` (channel="probe", entity_type="device") — exchange-oscillation
   decay times (seconds -> microseconds) versus J-gate voltage from
   ``AncillaryFiles/Exchange_in_31_analysed_measDevice{A40769,B19927,C1608}.mat``
   (context carries the J voltage and exchange frequency per point).

Timestamps.  Every FBT trace is anchored at the real experiment start
datetime embedded in its ``data_source`` string (e.g.
``alder_26008_2022_06_02_17_38_13_...``).  Device B carries real per-point
lab-time offsets in minutes (``time_stamp``; context["pseudo_time"]=false);
devices A and C have no per-point clock, so points are spaced by device B's
documented cadence of 0.514 min per FBT update and flagged
context["pseudo_time"]=true.  Episodic events with no timestamp at all
(RB-table rows, exchange-decay points) use the mandated fallback: base date
2023-01-01 UTC + 1 day per per-device experiment index, pseudo_time=true.
Fidelities quoted marginally above 1 (e.g. IRB6B at 100.01%) are clipped to
1.0 with the raw value kept in context["raw_value"].

Zip members are extracted lazily and idempotently into
``<raw_dir>/extracted/`` before parsing.  Every context records the source
file and measurement type.  The returned frame passes ``uts.validate_uts``.
"""
from __future__ import annotations

import json
import re
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

from ltn.data import uts

__all__ = [
    "DEVICES",
    "FBT_TIME_STEP_MIN",
    "PSEUDO_BASE_TS",
    "clip_fidelity",
    "load_silicon",
    "parse_anchor_ts",
]

SOURCE = "silicon"
PLATFORM = "si_spin"

#: dataset letter -> (UTS device name, codename used inside the release)
DEVICES: dict[str, tuple[str, str]] = {
    "A": ("simos_a", "Sunshine"),
    "B": ("simos_b", "alder"),
    "C": ("simos_c", "MV25_OG31"),
}

#: Fallback base date for events with no real timestamp (mandated pseudo-time).
PSEUDO_BASE_TS = pd.Timestamp("2023-01-01", tz="UTC")

#: Device B's FBT cadence (minutes per fidelity-trace point, DeviceB_26008.mat
#: ``time_per_run``); reused as the assumed spacing for the A/C traces.
FBT_TIME_STEP_MIN = 0.514

_ANCILLARY_ZIP = "AncillaryFiles.zip"
_FBT_STATS_MEMBER = "AncillaryFiles/fidelity_statistics_allcoldboot.mat"
_DEVICE_B_TRACE_MEMBER = "AncillaryFiles/DeviceB_26008.mat"
_RB_TABLE_MEMBER = "AncillaryFiles/table_of_RBruns.txt"
_EXCHANGE_MEMBERS = {
    "A": "AncillaryFiles/Exchange_in_31_analysed_measDeviceA40769.mat",
    "B": "AncillaryFiles/Exchange_in_31_analysed_measDeviceB19927.mat",
    "C": "AncillaryFiles/Exchange_in_31_analysed_measDeviceC1608.mat",
}

_ANCHOR_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")


# ------------------------------------------------------------------ small utils
def parse_anchor_ts(data_source: str) -> pd.Timestamp | None:
    """Parse the ``YYYY_MM_DD_HH_MM_SS`` datetime embedded in an FBT id string.

    E.g. ``"alder_26008_2022_06_02_17_38_13_2Q_RBM_..."`` ->
    ``Timestamp("2022-06-02 17:38:13", tz="UTC")``; None when absent.
    """
    m = _ANCHOR_RE.search(data_source)
    if m is None:
        return None
    y, mo, d, h, mi, s = (int(g) for g in m.groups())
    try:
        return pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi,
                            second=s, tz="UTC")
    except ValueError:
        return None


def clip_fidelity(f: float) -> tuple[float, float | None]:
    """Clip a fidelity to <= 1.0; return (clipped, raw-if-clipped-else-None)."""
    f = float(f)
    if f > 1.0:
        return 1.0, f
    return f, None


def _extract_member(raw_dir: Path, zip_name: str, member: str) -> Path | None:
    """Lazily and idempotently extract one zip member to raw_dir/extracted/.

    Returns the extracted path, or None (with a warning) when the zip or the
    member is missing/unreadable.  A re-run is a no-op when the file already
    exists with the recorded size.
    """
    zpath = raw_dir / zip_name
    if not zpath.exists():
        warnings.warn(f"[silicon] missing archive {zpath}")
        return None
    dest = raw_dir / "extracted" / member
    try:
        with zipfile.ZipFile(zpath) as zf:
            try:
                info = zf.getinfo(member)
            except KeyError:
                warnings.warn(f"[silicon] {zip_name} has no member {member!r}")
                return None
            if dest.exists() and dest.stat().st_size == info.file_size:
                return dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with zf.open(member) as src, open(tmp, "wb") as out:
                while chunk := src.read(1 << 20):
                    out.write(chunk)
            tmp.replace(dest)  # atomic within the filesystem
    except zipfile.BadZipFile:
        warnings.warn(f"[silicon] unreadable archive {zpath} — skipped")
        return None
    return dest


def _field(struct: np.ndarray, name: str) -> np.ndarray:
    """Read field `name` of a scipy-loadmat MATLAB struct array (shape (1,1))."""
    return np.asarray(struct).ravel()[0][name]


def _event(
    records: list[dict],
    *,
    device: str,
    entity_type: str,
    entity_id: str,
    observable: str,
    channel: str,
    value: float,
    sigma: float | None,
    ts: pd.Timestamp,
    cal_ts: pd.Timestamp | None,
    context: dict,
) -> None:
    """Append one UTS row (context serialized exactly like uts.make_events)."""
    records.append({
        "source": SOURCE, "platform": PLATFORM, "device": device,
        "entity_type": entity_type, "entity_id": entity_id,
        "observable": observable, "channel": channel,
        "value": float(value), "sigma": None if sigma is None else float(sigma),
        "ts": ts, "cal_ts": cal_ts,
        "context": json.dumps(context, default=str, sort_keys=True),
    })


# --------------------------------------------------------------- FBT trace events
def _fbt_trace_events(raw_dir: Path, records: list[dict]) -> None:
    """cz_fidelity/gate_error_2q/epc_1q from the cold-boot FBT fidelity traces."""
    stats_path = _extract_member(raw_dir, _ANCILLARY_ZIP, _FBT_STATS_MEMBER)
    if stats_path is None:
        return
    mat = sio.loadmat(stats_path)

    # Device B extras: real per-point lab time (minutes) + 1-sigma errors.
    b_minutes: np.ndarray | None = None
    b_sigma: np.ndarray | None = None
    b_trace: np.ndarray | None = None
    b_path = _extract_member(raw_dir, _ANCILLARY_ZIP, _DEVICE_B_TRACE_MEMBER)
    if b_path is not None:
        bmat = sio.loadmat(b_path)
        b_minutes = np.asarray(bmat["time_stamp"], dtype=np.float64).ravel()
        b_sigma = np.asarray(bmat["f_1_sigma"], dtype=np.float64).ravel()
        b_trace = np.asarray(bmat["f_trace"], dtype=np.float64).ravel()

    for gate in ("CZ", "x1", "x2"):
        if gate not in mat:
            warnings.warn(f"[silicon] {_FBT_STATS_MEMBER} lacks gate {gate!r}")
            continue
        for letter, (device, codename) in DEVICES.items():
            try:
                rec = _field(mat[gate], letter)
                trace = np.asarray(
                    _field(_field(rec, "fidelity_trace"), "f"), dtype=np.float64
                ).ravel()
                data_source = str(np.asarray(_field(rec, "data_source")).ravel()[0])
            except (IndexError, KeyError, ValueError) as e:
                warnings.warn(f"[silicon] FBT trace {gate}/{letter} unreadable: {e!r}")
                continue
            anchor = parse_anchor_ts(data_source) or PSEUDO_BASE_TS
            real_time = (
                letter == "B" and b_minutes is not None and len(b_minutes) == len(trace)
            )
            if real_time:
                minutes = b_minutes
                if gate == "CZ" and b_trace is not None and len(b_trace) == len(trace):
                    trace = b_trace  # bitwise-identical source of the same run
            else:
                minutes = FBT_TIME_STEP_MIN * np.arange(len(trace), dtype=np.float64)
            sigmas = (
                b_sigma
                if (gate == "CZ" and real_time and b_sigma is not None
                    and len(b_sigma) == len(trace))
                else None
            )
            context = {
                "file": (_DEVICE_B_TRACE_MEMBER if (gate == "CZ" and real_time)
                         else _FBT_STATS_MEMBER),
                "archive": _ANCILLARY_ZIP,
                "measurement": "fbt_fidelity_trace",
                "gate": gate,
                "codename": codename,
                "data_source": data_source,
                "pseudo_time": not real_time,
            }
            if not real_time:
                context["time_step_min"] = FBT_TIME_STEP_MIN
                context["time_step_note"] = "cadence borrowed from DeviceB_26008 time_per_run"
            ts_arr = anchor + pd.to_timedelta(minutes, unit="m")
            for i, f in enumerate(trace):
                if not np.isfinite(f):
                    continue
                fid, raw = clip_fidelity(f)
                ctx = dict(context, point_index=i)
                if raw is not None:
                    ctx["raw_value"] = raw
                sig = float(sigmas[i]) if sigmas is not None else None
                if gate == "CZ":
                    _event(records, device=device, entity_type="device",
                           entity_id=device, observable="cz_fidelity",
                           channel="delivered", value=fid, sigma=sig,
                           ts=ts_arr[i], cal_ts=anchor, context=ctx)
                    _event(records, device=device, entity_type="device",
                           entity_id=device, observable="gate_error_2q",
                           channel="derived", value=max(0.0, 1.0 - fid),
                           sigma=sig, ts=ts_arr[i], cal_ts=anchor, context=ctx)
                else:  # x1 / x2 -> single-qubit error per Clifford proxy
                    qubit = "q1" if gate == "x1" else "q2"
                    _event(records, device=device, entity_type="qubit",
                           entity_id=qubit, observable="epc_1q",
                           channel="derived", value=max(0.0, 1.0 - fid),
                           sigma=sig, ts=ts_arr[i], cal_ts=anchor, context=ctx)


# ----------------------------------------------------------------- RB table events
def _rb_table_events(raw_dir: Path, records: list[dict],
                     day_counter: dict[str, int]) -> None:
    """Episodic IRB / FBT-summary / GST fidelities from table_of_RBruns.txt.

    Row formats (values in percent, quoted error after '±'):
      ``IRB<k><D> <clifford> ± <err> <cz> ± <err> - -``
      ``FBT<D> - <cz> ± <err> <x1> ± <err> <x2> ± <err>`` (same for GST<D>)
    """
    path = _extract_member(raw_dir, _ANCILLARY_ZIP, _RB_TABLE_MEMBER)
    if path is None:
        return
    text = path.read_text(encoding="utf-8", errors="replace")

    def emit_cz(device: str, fid_pct: float, err_pct: float, ctx: dict, ts) -> None:
        fid, raw = clip_fidelity(fid_pct / 100.0)
        if raw is not None:
            ctx = dict(ctx, raw_value=raw)
        sig = err_pct / 100.0
        _event(records, device=device, entity_type="device", entity_id=device,
               observable="cz_fidelity", channel="delivered", value=fid,
               sigma=sig, ts=ts, cal_ts=None, context=ctx)
        _event(records, device=device, entity_type="device", entity_id=device,
               observable="gate_error_2q", channel="derived",
               value=max(0.0, 1.0 - fid), sigma=sig, ts=ts, cal_ts=None,
               context=ctx)

    for line in text.splitlines():
        tok = line.split()
        if not tok or not re.fullmatch(r"(IRB\d+|FBT|GST)[ABC]", tok[0]):
            continue  # skip mean/std summary rows and blanks
        name = tok[0]
        letter = name[-1]
        device, codename = DEVICES[letter]
        method = "IRB" if name.startswith("IRB") else name[:3]
        idx = day_counter.get(device, 0)
        day_counter[device] = idx + 1
        ts = PSEUDO_BASE_TS + pd.Timedelta(days=idx)
        base_ctx = {
            "file": _RB_TABLE_MEMBER, "archive": _ANCILLARY_ZIP,
            "run": name, "method": method, "codename": codename,
            "pseudo_time": True,
        }
        try:
            if method == "IRB":
                # tok: [name, clifford, ±, err, cz, ±, err, -, -]
                clifford, cliff_err = float(tok[1]), float(tok[3])
                cz, cz_err = float(tok[4]), float(tok[6])
                ctx = dict(base_ctx, measurement="interleaved_2q_rb",
                           clifford_2q_fidelity_pct=clifford,
                           clifford_2q_error_pct=cliff_err)
                emit_cz(device, cz, cz_err, ctx, ts)
            else:
                # tok: [name, -, cz, ±, err, x1, ±, err, x2, ±, err]
                cz, cz_err = float(tok[2]), float(tok[4])
                meas = ("fbt_summary_fidelity" if method == "FBT"
                        else "gate_set_tomography")
                ctx = dict(base_ctx, measurement=meas)
                emit_cz(device, cz, cz_err, ctx, ts)
                for qubit, f_pct, e_pct in (
                    ("q1", float(tok[5]), float(tok[7])),
                    ("q2", float(tok[8]), float(tok[10])),
                ):
                    fid, raw = clip_fidelity(f_pct / 100.0)
                    qctx = dict(ctx, gate="x1" if qubit == "q1" else "x2")
                    if raw is not None:
                        qctx["raw_value"] = raw
                    _event(records, device=device, entity_type="qubit",
                           entity_id=qubit, observable="epc_1q",
                           channel="derived", value=max(0.0, 1.0 - fid),
                           sigma=e_pct / 100.0, ts=ts, cal_ts=None, context=qctx)
        except (IndexError, ValueError) as e:
            warnings.warn(f"[silicon] unparseable RB-table row {name!r}: {e!r}")


# ----------------------------------------------------------------- T2 decay events
def _t2_events(raw_dir: Path, records: list[dict],
               day_counter: dict[str, int]) -> None:
    """t2_us/probe from exchange-oscillation decay times vs J-gate voltage."""
    for letter, member in _EXCHANGE_MEMBERS.items():
        device, codename = DEVICES[letter]
        path = _extract_member(raw_dir, _ANCILLARY_ZIP, member)
        if path is None:
            continue
        mat = sio.loadmat(path)
        try:
            decay_s = np.asarray(mat[f"decayTimesDevice{letter}"],
                                 dtype=np.float64).ravel()
            vj = np.asarray(mat[f"VJDevice{letter}"], dtype=np.float64).ravel()
            exchange_hz = np.asarray(mat[f"exchangeDevice{letter}"],
                                     dtype=np.float64).ravel()
        except KeyError as e:
            warnings.warn(f"[silicon] {member} lacks expected variable: {e!r}")
            continue
        for i, t_s in enumerate(decay_s):
            if not np.isfinite(t_s) or t_s <= 0.0:
                continue
            idx = day_counter.get(device, 0)
            day_counter[device] = idx + 1
            ctx = {
                "file": member, "archive": _ANCILLARY_ZIP,
                "measurement": "exchange_oscillation_decay",
                "codename": codename, "pseudo_time": True,
                "point_index": i,
                "vj_volt": float(vj[i]) if i < len(vj) else None,
                "exchange_hz": (float(exchange_hz[i])
                                if i < len(exchange_hz) else None),
            }
            _event(records, device=device, entity_type="device",
                   entity_id=device, observable="t2_us", channel="probe",
                   value=t_s * 1e6, sigma=None,
                   ts=PSEUDO_BASE_TS + pd.Timedelta(days=idx), cal_ts=None,
                   context=ctx)


# ---------------------------------------------------------------------- public API
def load_silicon(raw_dir: str | Path) -> pd.DataFrame:
    """Load the silicon spin-qubit raw release under `raw_dir` into UTS events.

    raw_dir: directory holding AncillaryFiles.zip (and siblings), e.g.
        ``data/raw/silicon``.  Needed members are extracted lazily and
        idempotently into ``raw_dir/extracted/``.

    Returns a validated UTS frame (see module docstring for the observable
    mapping); empty frame when no archive is readable.
    """
    raw_dir = Path(raw_dir)
    records: list[dict] = []
    # Per-device sequential experiment index for pseudo-time (episodic events).
    day_counter: dict[str, int] = {}
    _fbt_trace_events(raw_dir, records)
    _rb_table_events(raw_dir, records, day_counter)
    _t2_events(raw_dir, records, day_counter)
    if not records:
        return uts.empty_uts()
    df = pd.DataFrame.from_records(records, columns=uts.COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["cal_ts"] = pd.to_datetime(df["cal_ts"], utc=True)
    df["value"] = df["value"].astype("float64")
    df["sigma"] = df["sigma"].astype("float64")
    return uts.validate_uts(df)
