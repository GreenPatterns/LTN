"""Auto-generated paper tables — the ONLY numbers pipeline (§2.6: nothing hand-copied).

  python -m ltn.paper_gen.tables

Reads runs/**/results.jsonl (+ predictions.parquet for pairing) and data/uts.
Missing experiments degrade to a \\todopaper{} marker, never a crash.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ltn.eval.metrics import crps_gaussian
from ltn.eval.runlog import read_results
from ltn.eval.stats import (
    cell_paired_wilcoxon,
    cluster_bootstrap_ci,
    paired_wilcoxon_holm,
)
from ltn.paper_gen.style import MODEL_LABELS, fmt3

OUT = Path("paper/tables")


def _write(name: str, content: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(content)
    print(f"[tables] wrote {name}")


def _placeholder(name: str, note: str) -> None:
    _write(name, "\\todopaper{%s}\n" % note)


# ---------------------------------------------------------------- T1: data inventory
def t1_data() -> None:
    from ltn.data.uts import read_uts
    lake = read_uts("data/uts")
    if lake.empty:
        return _placeholder("T1_data.tex", "T1: lake not built")
    rows = []
    for (src, dev), g in lake.groupby(["source", "device"], observed=True):
        rows.append({
            "source": src, "device": dev.replace("_", "\\_"),
            "events": len(g),
            "entities": g["entity_id"].nunique(),
            "obs": g["observable"].nunique(),
            "span": f"{g['ts'].min():%Y-%m-%d}--{g['ts'].max():%Y-%m-%d}",
            "kind": "SIM" if src == "sim" else "real",
        })
    df = pd.DataFrame(rows).sort_values(["kind", "source", "device"])
    lines = [
        "\\begin{tabular}{llrrrll}", "\\toprule",
        "Source & Device & Events & Entities & Obs. & Span & Type \\\\", "\\midrule",
    ]
    for _, r in df.iterrows():
        src = r['source'].replace('_', '\\_')
        note = "\\textsuperscript{a}" if r["source"] == "daqec" else ""
        lines.append(f"{src}{note} & {r['device']} & {r['events']:,} & "
                     f"{r['entities']} & {r['obs']} & {r['span']} & {r['kind']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "% \\textsuperscript{a} telemetry columns used; dataset's own conclusions not relied on [DEV-10]"]
    _write("T1_data.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- pairing helpers
def _pair_crps(exp: str, model: str, root=Path("runs")) -> pd.DataFrame | None:
    """Per-pair CRPS, averaged over seeds, keyed by (s_idx, t_idx, horizon)."""
    frames = []
    for seed_dir in sorted((root / exp / model).glob("[0-9]*")):
        f = seed_dir / "predictions.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f)
        import properscoring as ps
        p["crps"] = ps.crps_gaussian(p["y"], mu=p["mu"], sig=p["sigma"])
        frames.append(p[["s_idx", "t_idx", "horizon", "device", "observable", "crps"]])
    if not frames:
        return None
    allp = pd.concat(frames)
    return (allp.groupby(["s_idx", "t_idx", "horizon", "device", "observable"], observed=True)
            ["crps"].mean().reset_index())


def _crps_cells(exp: str, model_a: str, model_b: str) -> dict | None:
    """Paired CRPS arrays per (device, observable, horizon) cell."""
    a, b = _pair_crps(exp, model_a), _pair_crps(exp, model_b)
    if a is None or b is None:
        return None
    m = a.merge(b, on=["s_idx", "t_idx", "horizon", "device", "observable"],
                suffixes=("_a", "_b"))
    cells = {}
    for (dev, obs, h), g in m.groupby(["device", "observable", "horizon"], observed=True):
        if len(g) >= 5:
            cells[f"{dev}|{obs}|{h}"] = (g["crps_a"].to_numpy(), g["crps_b"].to_numpy())
    return cells or None


def _wilcoxon_vs(exp: str, model_a: str, model_b: str) -> pd.DataFrame | None:
    """LEGACY row-level test, kept only so the old numbers stay reproducible.

    It treats the rolling forecast origins inside a cell as iid, which inflates
    the effective n by ~3 orders of magnitude (E1b averages 2298 pairs/cell, all
    from ONE series). `_cell_wilcoxon_vs` is what the paper should quote.
    """
    cells = _crps_cells(exp, model_a, model_b)
    return None if cells is None else paired_wilcoxon_holm(cells)


def _cell_wilcoxon_vs(exp: str, model_a: str, model_b: str) -> pd.DataFrame | None:
    """ONE Wilcoxon over cell-mean CRPS differences (the cell is the unit)."""
    cells = _crps_cells(exp, model_a, model_b)
    return None if cells is None else cell_paired_wilcoxon(cells)


# ---------------------------------------------------------------- T2: main results
def t2_main() -> None:
    res = read_results("runs")
    if res.empty:
        return _placeholder("T2_main.tex", "T2: no runs yet")
    blocks = []
    for exp, label in (("E1a", "E1a (real, DAQEC)"), ("E1b", "E1b (Sim-Fleet, synthetic)")):
        df = res[(res.get("exp") == exp) & res.get("ncrps").notna()] if "ncrps" in res else pd.DataFrame()
        if df.empty:
            blocks.append(f"% {exp} pending\n\\todopaper{{T2 {exp} rows}}")
            continue
        # CLUSTER bootstrap. The rows of one (device, observable) series are
        # rolling forecast origins, not independent draws, so the iid bootstrap
        # understates the interval. The cluster IS the series, hence
        # (device, observable): k = 12 on E1a, k = 60 on E1b.
        #
        # Coarser clusterings were checked and rejected. Device-only is
        # degenerate here: E1a has 3 devices (k = 3 came out NARROWER than iid),
        # and on E1b device-only clustering also narrows the interval for the
        # deterministic baselines (EWMA 0.59x, persistence 0.60x of the iid
        # width) because the 12 sim devices are near-exchangeable by
        # construction while the observables are not -- it absorbs the wrong
        # dependence. (device, observable) widens every model on both splits
        # (1.97-4.62x on E1b) and never narrows.
        clust_keys = ["device", "observable"]
        agg = []
        for model, g in df.groupby("model"):
            per_seed = g.groupby("seed")["ncrps"].mean()
            vals = g["ncrps"].to_numpy()
            groups = g[clust_keys].astype(str).agg("|".join, axis=1).to_numpy()
            lo, hi = cluster_bootstrap_ci(
                lambda v: float(np.mean(v)), vals, groups=groups)
            n_params = None
            cfgs = sorted(Path(f"runs/{exp}/{model}").glob("*/config.yaml"))
            if cfgs:
                import yaml
                c = yaml.safe_load(cfgs[0].read_text())
                n_params = (c.get("config") or {}).get("n_params") or c.get("n_params")
            agg.append({"model": model, "ncrps": float(np.mean(vals)), "lo": lo, "hi": hi,
                        "seeds": g["seed"].nunique(), "n_params": n_params})
        agg = pd.DataFrame(agg).sort_values("ncrps")
        best = agg.iloc[0]["model"]
        # significance of ltn vs the best non-ltn baseline
        base = agg[~agg["model"].str.startswith("ltn")].iloc[0]["model"] if (~agg["model"].str.startswith("ltn")).any() else None
        sig = ""
        if base and "ltn" in set(agg["model"]):
            w = _wilcoxon_vs(exp, "ltn", base)
            if w is not None and "__summary__" in set(w["cell"]):
                s = w[w["cell"] == "__summary__"].iloc[0]
                sig = (f"% [legacy, row-level, inflated n] Wilcoxon LTN vs {base}: "
                       f"{s['significant']:.0%} of cells Holm-significant, "
                       f"median Cliff's d = {s['cliffs_delta']:.2f}\n")
            cw = _cell_wilcoxon_vs(exp, "ltn", base)
            if cw is not None:
                c = cw.iloc[0]
                sig += (f"% [reported] cell-paired Wilcoxon LTN vs {base}: "
                        f"n = {int(c['n_cells'])} cells, p = {c['wilcoxon_p']:.3g}, "
                        f"Cliff's d = {c['cliffs_delta']:.2f}, "
                        f"LTN worse in {c['frac_cells_a_worse']:.0%} of cells\n")
        rows = [f"\\multicolumn{{4}}{{l}}{{\\emph{{{label}}}}} \\\\"]
        for _, r in agg.iterrows():
            name = MODEL_LABELS.get(r["model"], r["model"])
            val = f"{fmt3(r['ncrps'])} [{fmt3(r['lo'])}, {fmt3(r['hi'])}]"
            if r["model"] == best:
                name, val = f"\\textbf{{{name}}}", f"\\textbf{{{val}}}"
            np_s = f"{int(r['n_params']):,}" if pd.notna(r["n_params"]) and r["n_params"] else "--"
            rows.append(f"{name} & {val} & {int(r['seeds'])} & {np_s} \\\\")
        blocks.append(sig + "\n".join(rows))
    body = ("\\begin{tabular}{lccr}\n\\toprule\n"
            "Model & NCRPS [95\\% CI] & Seeds & Params \\\\\n\\midrule\n"
            + "\n\\midrule\n".join(blocks)
            + "\n\\bottomrule\n\\end{tabular}\n")
    _write("T2_main.tex", body)


# ---------------------------------------------------------------- T3: ablations
ABL_LABELS = {
    "ltn": "LTN (full)", "ltn_gaussian": "$-$DM $\\to$ Gaussian state",
    "ltn_ode": "$-$Lindblad $\\to$ neural ODE", "ltn_nograph": "$-$graph",
    "ltn_nophys": "$-$physics-$\\gamma$", "ltn_noH": "$-$Hamiltonian",
}


def t3_ablations() -> None:
    res = read_results("runs")
    df = (res[res["exp"].astype(str).str.startswith("E2") & res.get("ncrps").notna()]
          if not res.empty and "exp" in res and "ncrps" in res else pd.DataFrame())
    if df.empty:
        return _placeholder("T3_ablations.tex", "T3: E2 not run yet")
    # E1a additionally reports the SEED MEDIAN delta and the seed spread: its full
    # model has one diverging seed, so every mean delta on that split has the
    # opposite sign to the median one. Reporting only the mean would credit each
    # ablation with an improvement that a single run manufactured.
    rows = ["\\begin{tabular}{lcccc|cc}", "\\toprule",
            " & \\multicolumn{4}{c|}{E1a (real)} & \\multicolumn{2}{c}{E1b (sim)} \\\\",
            "Variant & NCRPS & $\\Delta$ & $\\Delta_{\\mathrm{med}}$ & "
            "$\\sigma_{\\mathrm{seed}}$ & NCRPS & $\\Delta$ \\\\", "\\midrule"]

    def _seed_means(exp: str, model: str) -> np.ndarray:
        g = df[(df["exp"] == exp) & (df["model"] == model)]
        return g.groupby("seed")["ncrps"].mean().to_numpy() if not g.empty else np.array([])

    full_a, full_b = _seed_means("E2-e1a", "ltn"), _seed_means("E2-e1b", "ltn")
    for model in ["ltn", "ltn_ode", "ltn_nophys", "ltn_gaussian", "ltn_nograph", "ltn_noH"]:
        a, b = _seed_means("E2-e1a", model), _seed_means("E2-e1b", model)
        if a.size:
            d = "--" if model == "ltn" else f"{a.mean() - full_a.mean():+.3f}"
            dm = "--" if model == "ltn" else f"{np.median(a) - np.median(full_a):+.3f}"
            sd = f"{a.std(ddof=1):.3f}" if a.size > 1 else "--"
            ca = [fmt3(a.mean()), d, dm, sd]
        else:
            ca = ["--", "--", "--", "--"]
        cb = ["--", "--"] if not b.size else [
            fmt3(b.mean()), "--" if model == "ltn" else f"{b.mean() - full_b.mean():+.3f}"]
        rows.append(f"{ABL_LABELS.get(model, model)} & " + " & ".join(ca + cb) + " \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write("T3_ablations.tex", "\n".join(rows) + "\n")


# ---------------------------------------------------------------- T4: E8 appendix
def t4_e8() -> None:
    res = read_results("runs")
    df = res[res.get("exp") == "E8"] if not res.empty and "exp" in res else pd.DataFrame()
    if df.empty or "rmse" not in df:
        return _placeholder("T4_e8.tex", "T4: E8 not run yet")
    rows = ["\\begin{tabular}{llcc}", "\\toprule",
            "Dataset & Arm & RMSE & sign-AUROC \\\\", "\\midrule"]
    df = df.copy()
    df["dataset"] = df["model"].str.rsplit("-", n=1).str[-1]
    df["arm"] = df["model"].str.rsplit("-", n=1).str[0]
    for (ds, arm), g in df.groupby(["dataset", "arm"]):
        rows.append(f"{ds} & {arm.replace('_', ' ')} & {fmt3(g['rmse'].mean())} & "
                    f"{fmt3(g['auroc'].mean())} \\\\")
    rows += ["\\bottomrule", "\\end{tabular}"]
    _write("T4_e8.tex", "\n".join(rows) + "\n")


def main() -> None:
    t1_data()
    t2_main()
    t3_ablations()
    t4_e8()


if __name__ == "__main__":
    main()
