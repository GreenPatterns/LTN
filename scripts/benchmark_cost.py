"""Computational-cost benchmark for LTN, LTN-v2 and every E1 baseline.

Measures the cost of each model: training wall-clock, inference latency,
peak memory, and scaling in d / nodes / edges. Nothing here is estimated --
every number is measured by this file and written to runs/COST/.

Sections (subcommands):
  train    harvest duration_s from the existing runs/E1{a,b}/<model>/<seed>/status.yaml
           (real campaign wall-clock; no retraining)
  latency  inference latency + peak memory per model, CPU and GPU
  dscale   d in {2,4,8,16,32}: forward+backward time/memory of the Lindblad drift,
           with an empirical exponent fit against the O(d^6) matrix_exp prediction
  gscale   nodes in {16,64,256,1024} x mean degree: KrausProp message-passing cost
  table    emit paper/tables/T7_cost.tex from the JSON artifacts

CONTENTION.  This box is shared. Every timing records wall-clock AND process CPU
time; `wall_over_cpu` near 1.0 means the pinned worker got a full core and the
median is trustworthy. GPU timings cannot be protected this way, so we report the
minimum (least-contaminated sample, i.e. an upper bound on true speed) alongside
the median and the interquartile spread, and `contended` flags any measurement
whose IQR exceeds 25% of its median. Peak GPU memory is a per-process allocator
counter and is unaffected by other tenants.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np

# single-threaded: makes CPU wall-clock reproducible on a loaded box
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch  # noqa: E402

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "COST"
MODELS = ["ltn", "ltn_v2", "lstm", "tgn", "transformer", "arima", "ewma", "persistence"]

# Representative inference workloads, read off the real splits by
# ltn.exp.dataset.build_device_sequence (see runs/COST/workload.json):
#   E1a: 3 devices x 4 entities, 12 edges, F=17, 4 observables, T=42
#   E1b: 12 devices x 121 entities (2 x 247), 384 edges, F=19, 5 observables, T~175
WORKLOADS = {
    "e1a": dict(n=4, n_edges=12, feat=17, n_obs=4, platform_dim=7),
    "e1b": dict(n=121, n_edges=384, feat=19, n_obs=5, platform_dim=7),
}
# tuned hyperparameters actually used by the campaign (runs/E1{a,b}/<m>/0/config.yaml)
TUNED = {
    "e1a": {
        "ltn": dict(d=4, carry_dim=59, hidden=94, evidence_rank=2, rounds=1),
        "ltn_v2": dict(carry_dim=31, hidden=126),
        "lstm": dict(width=50),
        "tgn": dict(width=157),
        "transformer": dict(width=37, n_layers=2),
    },
    "e1b": {
        "ltn": dict(d=4, carry_dim=52, hidden=109, evidence_rank=4, rounds=1),
        "ltn_v2": dict(carry_dim=42, hidden=99),
        "lstm": dict(width=76),
        "tgn": dict(width=183),
        "transformer": dict(width=164, n_layers=1),
    },
}


# --------------------------------------------------------------------- timing --
def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def timeit(fn, repeats: int = 100, warmup: int = 10, cuda: bool = False) -> dict:
    """Median-of-`repeats` wall-clock with warmup. Also records process CPU time
    so contention on this shared box is visible rather than silently absorbed."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    wall, cpu = [], []
    for _ in range(repeats):
        t0, p0 = time.perf_counter(), time.process_time()
        fn()
        if cuda:
            torch.cuda.synchronize()
        wall.append(time.perf_counter() - t0)
        cpu.append(time.process_time() - p0)
    med = statistics.median(wall)
    q1, q3 = _pct(wall, 0.25), _pct(wall, 0.75)
    cpu_med = statistics.median(cpu)
    return {
        "median_ms": med * 1e3,
        "min_ms": min(wall) * 1e3,
        "p25_ms": q1 * 1e3,
        "p75_ms": q3 * 1e3,
        "max_ms": max(wall) * 1e3,
        "cpu_median_ms": cpu_med * 1e3,
        "wall_over_cpu": (med / cpu_med) if cpu_med > 0 else float("nan"),
        "iqr_over_median": (q3 - q1) / med if med > 0 else float("nan"),
        "contended": bool(med > 0 and (q3 - q1) / med > 0.25),
        "repeats": repeats,
    }


def cuda_peak_mb(fn, device: str) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / 2**20


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def hardware() -> dict:
    hw = {
        "cpu_model": platform.processor() or platform.machine(),
        "n_cores_visible": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "loadavg_at_start": os.getloadavg(),
    }
    if torch.cuda.is_available():
        hw["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        hw["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
    return hw


# ------------------------------------------------------------ model builders --
def build_ltn(kind: str, wl: dict, cfg: dict, device: str):
    from ltn.models.ltn import LTNModel
    from ltn.models.ltn_v2 import LTNv2Model
    kw = dict(n_obs=wl["n_obs"], platform_dim=wl["platform_dim"], **cfg)
    m = (LTNv2Model(**kw) if kind == "ltn_v2" else LTNModel(**kw)).to(device)
    return m.eval()


def build_baseline(kind: str, wl: dict, cfg: dict, device: str):
    from ltn.baselines.neural import _CORES
    cfg = dict(cfg)
    width = cfg.pop("width")
    if kind == "transformer":
        n_heads = 4
        width = n_heads * max(1, round(width / n_heads))
        core = _CORES[kind](wl["feat"], width, wl["n_obs"], n_layers=cfg.get("n_layers", 2),
                            n_heads=n_heads, window=64)
    else:
        core = _CORES[kind](wl["feat"], width, wl["n_obs"])
    return core.to(device).eval()


def synth_inputs(wl: dict, device: str, seed: int = 0):
    """Random inputs with the exact shapes/dtypes the real pipeline produces."""
    g = torch.Generator().manual_seed(seed)
    n, e = wl["n"], wl["n_edges"]
    x = torch.randn(n, wl["feat"], dtype=torch.float64, generator=g).to(device)
    cal = torch.rand(n, dtype=torch.float64, generator=g).mul(48).to(device)
    dt = torch.ones(n, dtype=torch.float64, device=device)
    ei = torch.randint(0, n, (2, e), generator=g).to(device)
    ea = torch.randn(e, 2, dtype=torch.float64, generator=g).to(device)
    return x, cal, dt, ei, ea


def ltn_infer_closure(model, wl, device, horizon: float = 3.0):
    """One inference batch = advance all n entities by one snapshot and read out a
    forecast at one horizon -> n x n_obs (mu, sigma) pairs, exactly what
    LTNForecaster.predict emits per (device, origin, horizon) group."""
    x, cal, dt, ei, ea = synth_inputs(wl, device)
    state0 = model.init_state(wl["n"], device=device)
    h = torch.full((wl["n"],), horizon, dtype=torch.float64, device=device)

    def run():
        with torch.no_grad():
            st = model.step(state0, x, cal, dt, ei, ea)
            return model.forecast(st, x, cal, h)
    return run


def baseline_infer_closure(core, wl, device, horizon: float = 3.0):
    x, _, _, ei, ea = synth_inputs(wl, device)
    st0 = core.init_state(wl["n"], device)
    if isinstance(st0, list):  # transformer needs a filled history window
        for _ in range(64):
            st0 = core.step(st0, x, ei, ea)
    h = torch.full((wl["n"],), horizon, dtype=torch.float64, device=device)

    def run():
        with torch.no_grad():
            st = core.step(st0, x, ei, ea)
            return core.head(core.readout(st), h)
    return run


# ------------------------------------------------------------- 1. train time --
def cmd_train(args) -> dict:
    import yaml
    out = {}
    for exp in ("E1a", "E1b"):
        out[exp] = {}
        for m in MODELS:
            per_seed = {}
            for s in range(5):
                f = ROOT / "runs" / exp / m / str(s) / "status.yaml"
                if not f.exists():
                    continue
                y = yaml.safe_load(f.read_text())
                if y.get("duration_s") is None:
                    continue
                per_seed[s] = float(y["duration_s"])
            if not per_seed:
                out[exp][m] = {"missing": True}
                continue
            v = sorted(per_seed.values())
            out[exp][m] = {
                "per_seed_s": per_seed,
                "median_s": statistics.median(v),
                "min_s": v[0],
                "max_s": v[-1],
                "n_seeds": len(v),
            }
    _write("train_time.json", out)
    return out


# ---------------------------------------------------------- 2. latency + mem --
def cmd_latency(args) -> dict:
    res = {"hardware": hardware(), "note": (
        "One inference batch = one snapshot advanced for all n entities plus one "
        "horizon readout, producing n x n_obs forecasts. per_forecast_us = "
        "batch median / (n * n_obs).")}
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() and not args.cpu_only else [])
    for split, wl in WORKLOADS.items():
        res[split] = {"workload": wl, "models": {}}
        n_fc = wl["n"] * wl["n_obs"]
        for m in ["ltn", "ltn_v2", "lstm", "tgn", "transformer"]:
            cfg = TUNED[split][m]
            res[split]["models"][m] = entry = {"config": cfg}
            for dev in devices:
                mdl = (build_ltn(m, wl, cfg, dev) if m.startswith("ltn")
                       else build_baseline(m, wl, cfg, dev))
                entry["n_params"] = sum(p.numel() for p in mdl.parameters() if p.requires_grad)
                run = (ltn_infer_closure(mdl, wl, dev) if m.startswith("ltn")
                       else baseline_infer_closure(mdl, wl, dev))
                t = timeit(run, repeats=args.repeats, cuda=dev.startswith("cuda"))
                t["per_forecast_us"] = t["median_ms"] * 1e3 / n_fc
                if dev.startswith("cuda"):
                    t["peak_mem_mb"] = cuda_peak_mb(run, dev)
                entry[dev] = t
                del mdl
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()
    if not args.no_classical:
        res["classical_e1a"] = classical_latency()
    _write("latency.json", res)
    return res


def classical_latency() -> dict:
    """Classical baselines on the REAL e1a panel: fit once, then time predict over
    the real test pairs. Resident-set peak is measured for the whole fit+predict."""
    from ltn.baselines.arima import ARIMAForecaster
    from ltn.baselines.ewma import EWMA
    from ltn.baselines.persistence import Persistence
    from ltn.data.series import forecast_pairs
    from ltn.eval.splits import load_split
    from ltn.exp.run import build_data, model_cfg

    split = load_split("e1a")
    uts, panel, (tr, va, te) = build_data(split)
    cfg = model_cfg(split, 0, {})
    cfg["uts"] = uts
    pairs = forecast_pairs(panel, split["horizons_steps"], te)
    out = {"n_pairs": int(len(pairs)), "split": "e1a"}
    for name, cls in [("persistence", Persistence), ("ewma", EWMA), ("arima", ARIMAForecaster)]:
        rss0 = rss_mb()
        m = cls()
        t0 = time.perf_counter()
        m.fit(panel, tr, va, cfg)
        fit_s = time.perf_counter() - t0
        reps = 3 if name == "arima" else 20
        t = timeit(lambda: m.predict(pairs, panel), repeats=reps, warmup=1)
        out[name] = {
            "fit_s": fit_s,
            "predict": t,
            "per_forecast_us": t["median_ms"] * 1e3 / len(pairs),
            "rss_peak_mb": rss_mb(),
            "rss_delta_mb": rss_mb() - rss0,
        }
    return out


# --------------------------------------------------------------- 3. d-scaling --
def cmd_dscale(args) -> dict:
    """Forward+backward cost of one LTN training step as d grows.

    The Lindblad propagator vectorizes each node's d x d state into a d^2 x d^2
    superoperator and calls torch.matrix_exp on it, so the drift alone is
    O(d^6) per node; we fit the measured exponent against that prediction.
    """
    wl = dict(WORKLOADS["e1b"])
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() and not args.cpu_only else [])
    res = {"hardware": hardware(), "workload": wl, "theory": {
        "matrix_exp_on_d2_by_d2": "O((d^2)^3) = O(d^6) per node",
        "superoperator_memory": "O(d^4) complex128 per node = 16 d^4 bytes"}}
    for dev in devices:
        rows = []
        for d in args.dims:
            cfg = dict(TUNED["e1b"]["ltn"]); cfg["d"] = d
            mdl = build_ltn("ltn", wl, cfg, dev).train()
            x, cal, dt, ei, ea = synth_inputs(wl, dev)
            st0 = mdl.init_state(wl["n"], device=dev)
            h = torch.full((wl["n"],), 3.0, dtype=torch.float64, device=dev)

            def fwdbwd():
                mdl.zero_grad(set_to_none=True)
                st = mdl.step(st0, x, cal, dt, ei, ea)
                mu, sigma, _ = mdl.forecast(st, x, cal, h)
                (mu.pow(2).mean() + sigma.pow(2).mean()).backward()

            # separate the drift alone, to isolate matrix_exp from the rest
            def drift_only():
                with torch.no_grad():
                    mdl.drift(st0, x, cal, dt)

            reps = max(10, args.repeats // max(1, d // 2))
            row = {"d": d,
                   "n_params": sum(p.numel() for p in mdl.parameters() if p.requires_grad),
                   "fwd_bwd": timeit(fwdbwd, repeats=reps, warmup=3, cuda=dev.startswith("cuda")),
                   "drift_fwd": timeit(drift_only, repeats=reps, warmup=3, cuda=dev.startswith("cuda"))}
            if dev.startswith("cuda"):
                row["peak_mem_mb"] = cuda_peak_mb(fwdbwd, dev)
            else:
                row["rss_peak_mb"] = rss_mb()
            row["superop_bytes_per_node"] = 16 * d**4
            row["superop_total_mb"] = 16 * d**4 * wl["n"] / 2**20
            rows.append(row)
            print(f"[dscale {dev}] d={d:3d} fwd+bwd {row['fwd_bwd']['median_ms']:9.2f} ms "
                  f"drift {row['drift_fwd']['median_ms']:9.2f} ms", flush=True)
            del mdl
            if dev.startswith("cuda"):
                torch.cuda.empty_cache()
        res[dev] = {"rows": rows,
                    "fit_fwd_bwd": powerfit([r["d"] for r in rows],
                                            [r["fwd_bwd"]["median_ms"] for r in rows]),
                    "fit_drift": powerfit([r["d"] for r in rows],
                                          [r["drift_fwd"]["median_ms"] for r in rows])}
    _write("dscale.json", res)
    return res


def powerfit(xs, ys, tail: int | None = None) -> dict:
    """Least-squares fit of log y = a + k log x. `tail` restricts to the largest
    points, where the asymptotic regime holds and fixed overheads have washed out."""
    x, y = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    full = np.polyfit(x, y, 1)
    out = {"exponent_all": float(full[0]), "prefactor_all": float(np.exp(full[1]))}
    t = tail or max(3, len(xs) // 2)
    if len(xs) >= t + 1:
        tf = np.polyfit(x[-t:], y[-t:], 1)
        out["exponent_tail"] = float(tf[0])
        out["tail_points"] = int(t)
    resid = y - np.polyval(full, x)
    out["r2"] = float(1 - (resid**2).sum() / max(((y - y.mean())**2).sum(), 1e-30))
    return out


# ----------------------------------------------------------- 4. graph scaling --
def cmd_gscale(args) -> dict:
    """KrausProp message-passing cost vs node count and mean degree, d fixed at 4."""
    from ltn.models.krausprop import KrausProp
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() and not args.cpu_only else [])
    d, feat = 4, WORKLOADS["e1b"]["feat"]
    res = {"hardware": hardware(), "d": d, "note": (
        "KrausProp with rounds=1, rank=4, hidden=64; edges drawn uniformly. "
        "mean_degree = E / N.")}
    for dev in devices:
        rows = []
        for n in args.nodes:
            for deg in args.degrees:
                e = n * deg
                g = torch.Generator().manual_seed(0)
                prop = KrausProp(d, edge_dim=2, node_dim=feat, rounds=1).to(dev).eval()
                rho = torch.eye(d, dtype=torch.complex128, device=dev).expand(n, d, d).contiguous() / d
                nf = torch.randn(n, feat, dtype=torch.float64, generator=g).to(dev)
                ei = torch.randint(0, n, (2, e), generator=g).to(dev)
                ea = torch.randn(e, 2, dtype=torch.float64, generator=g).to(dev)

                def run():
                    with torch.no_grad():
                        prop(rho, nf, ei, ea)

                t = timeit(run, repeats=args.repeats, warmup=5, cuda=dev.startswith("cuda"))
                row = {"nodes": n, "mean_degree": deg, "edges": e, "time": t}
                if dev.startswith("cuda"):
                    row["peak_mem_mb"] = cuda_peak_mb(run, dev)
                rows.append(row)
                print(f"[gscale {dev}] N={n:5d} deg={deg} E={e:6d} "
                      f"{t['median_ms']:8.3f} ms", flush=True)
                del prop
                if dev.startswith("cuda"):
                    torch.cuda.empty_cache()
        res[dev] = {"rows": rows}
        for deg in args.degrees:
            sub = [r for r in rows if r["mean_degree"] == deg]
            if len(sub) >= 2:
                res[dev][f"fit_vs_nodes_deg{deg}"] = powerfit(
                    [r["nodes"] for r in sub], [r["time"]["median_ms"] for r in sub])
        for n in args.nodes:
            sub = [r for r in rows if r["nodes"] == n]
            if len(sub) >= 2:
                res[dev][f"fit_vs_degree_n{n}"] = powerfit(
                    [r["mean_degree"] for r in sub], [r["time"]["median_ms"] for r in sub])
    _write("gscale.json", res)
    return res


# ------------------------------------------------------------------- output --
def _write(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    print(f"wrote {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["train", "latency", "dscale", "gscale", "all"])
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--dims", type=int, nargs="*", default=[2, 4, 8, 16, 32])
    ap.add_argument("--nodes", type=int, nargs="*", default=[16, 64, 256, 1024])
    ap.add_argument("--degrees", type=int, nargs="*", default=[2, 4, 8])
    args = ap.parse_args()
    fns = {"train": cmd_train, "latency": cmd_latency, "dscale": cmd_dscale, "gscale": cmd_gscale}
    for c in (["train", "latency", "dscale", "gscale"] if args.cmd == "all" else [args.cmd]):
        fns[c](args)


if __name__ == "__main__":
    main()
