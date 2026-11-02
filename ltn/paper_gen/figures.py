"""Auto-generated paper figures F1-F7 (vector PDF, dataviz-skill styling).

  python -m ltn.paper_gen.figures [--only F3,F6]

Every figure reads only runs/** artifacts and the UTS lake; missing inputs produce
a console note and no file — never a crash. Colors: fixed entity assignment from
ltn.paper_gen.style (never cycled); persistence/reference in neutral ink.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ltn.paper_gen.style import (GRID, INK, INK2, MODEL_COLORS, MODEL_LABELS,
                                 POLICY_COLORS, apply_style)

import matplotlib.pyplot as plt

OUT = Path("paper/figures")


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"[figures] wrote {name}")


def _note(name: str, why: str) -> None:
    print(f"[figures] SKIP {name}: {why}")


# --------------------------------------------------------------------------- F1
def f1_concept() -> None:
    from ltn.data.uts import read_uts
    uts = read_uts("data/uts", sources=["daqec"])
    if uts.empty:
        return _note("F1", "daqec lake missing")
    d = uts[(uts["device"] == "ibm_brisbane") & (uts["entity_type"] == "device")]
    t1 = d[d["observable"] == "t1_us"].pivot_table(index="ts", columns="channel",
                                                   values="value", aggfunc="mean", observed=True)
    cal = d.dropna(subset=["cal_ts"]).groupby("ts")["cal_ts"].max()
    age = (cal.index.to_series() - cal).dt.total_seconds() / 3600
    recals = cal.index[cal.diff() > pd.Timedelta(0)]

    # F1 stays a data-only exhibit: the E1a-tuned model's state entropy hovers near
    # the vacuous maximum on this densely-recalibrated grid (S in [1.33, 1.36] of
    # log 4 = 1.386), so a model panel here would show nothing; the model's
    # epistemic dynamics are measured where they exist — F7's gap injections.
    ent = None
    ckpt = Path("runs/E1a/ltn/0/model.pt")
    if False and ckpt.exists():
        try:
            import torch

            from ltn.eval.splits import load_split
            from ltn.exp.run import build_data, model_cfg
            from ltn.exp.train import LTNForecaster
            split = load_split("e1a")
            u2, panel, (tr, va, te) = build_data(split)
            cfg = model_cfg(split, 0, {"epochs": 0, "init_from": str(ckpt), "device": "cpu"})
            cfg["uts"] = u2
            m = LTNForecaster()
            m.fit(panel, tr, va, cfg)
            # epistemic decoherence gap: S(rho) now vs after an evidence-free 1-day roll
            model, seq = m.model, m.seqs["ibm_brisbane"]
            model.eval()
            rows = []
            with torch.no_grad():
                st = model.init_state(len(seq.entity_ids))
                for t in range(len(seq.ts)):
                    st = model.step(st, seq.x[t], seq.cal_age_h[t],
                                    seq.dt_steps[t].expand(seq.x.shape[1]), seq.edge_index, seq.edge_attr)
                    _, _, ent_h = model.forecast(
                        st, seq.x[t], seq.cal_age_h[t],
                        torch.full((seq.x.shape[1],), 3.0, dtype=torch.float64))
                    rows.append((seq.ts[t], float(st.entropy.mean()), float(ent_h.mean())))
            ent = pd.DataFrame(rows, columns=["ts", "now", "rolled"]).set_index("ts")
        except Exception as e:  # fall back to the two-panel data-only story
            print(f"[figures] F1 model overlay failed ({e}); two-panel fallback")

    import matplotlib.dates as mdates

    n = 3 if ent is not None else 2
    fig, ax = plt.subplots(n, 1, figsize=(3.5, 1.15 * n + 0.9), sharex=True,
                           gridspec_kw={"height_ratios": [2] + [1] * (n - 1)})
    ax[0].plot(t1.index, t1["claimed"], drawstyle="steps-post", color=INK2, lw=1.2,
               label="claimed $T_1$")
    ax[0].plot(t1.index, t1["probe"], "o-", ms=1.8, lw=0.9,
               color=MODEL_COLORS["ltn"], label="delivered $T_1$ (probe)")
    ax[0].set_ylabel("$T_1$ ($\\mu$s)")
    ax[0].legend(loc="lower left", ncols=2)
    ax[1].plot(age.index, age.values, color=MODEL_COLORS["arima"], lw=1.0)
    ax[1].set_ylabel("claim age (h)")
    if ent is not None:
        ax[2].fill_between(ent.index, ent["now"], ent["rolled"],
                           color=MODEL_COLORS["ltn"], alpha=0.18, lw=0)
        ax[2].plot(ent.index, ent["now"], color=MODEL_COLORS["ltn"], lw=1.0,
                   label="$S(\\rho_t)$ with evidence")
        ax[2].plot(ent.index, ent["rolled"], color=MODEL_COLORS["ltn_gaussian"], lw=1.0,
                   ls="--", label="$S(\\rho_t)$, no evidence for 1 d")
        ax[2].set_ylabel("$S(\\rho)$")
        ax[2].legend(loc="center right", fontsize=6)
    ax[-1].set_xlabel("date (2025)")
    ax[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.align_ylabels(ax)
    _save(fig, "F1_concept.pdf")


# --------------------------------------------------------------------------- F3
def f3_crps() -> None:
    from ltn.eval.runlog import read_results
    res = read_results("runs")
    if res.empty or "ncrps" not in res:
        return _note("F3", "no E1 results")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for ax, exp, title in ((axes[0], "E1a", "E1a — real (DAQEC)"),
                           (axes[1], "E1b", "E1b — Sim-Fleet (SYNTHETIC)")):
        df = res[(res["exp"] == exp) & res["ncrps"].notna()]
        if df.empty:
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=INK2)
            ax.set_title(title)
            continue
        grid_h = 8.0 if exp == "E1a" else 24.0
        for model in [m for m in MODEL_COLORS if m in set(df["model"])]:
            g = df[df["model"] == model]
            per = g.groupby(["horizon", "seed"])["ncrps"].mean().reset_index()
            m_ = per.groupby("horizon")["ncrps"].mean()
            sd = per.groupby("horizon")["ncrps"].std().fillna(0)
            days = m_.index * grid_h / 24.0
            ax.plot(days, m_.values, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            ax.fill_between(days, m_ - sd, m_ + sd, color=MODEL_COLORS[model], alpha=0.12, lw=0)
        ax.set_xlabel("horizon (days)")
        ax.set_title(title)
    axes[0].set_ylabel("NCRPS (lower is better)")
    axes[1].legend(ncols=2, loc="upper left")
    _save(fig, "F3_crps.pdf")


# --------------------------------------------------------------------------- F4
def f4_ablations() -> None:
    from ltn.eval.runlog import read_results
    res = read_results("runs")
    df = (res[res["exp"].astype(str).str.startswith("E2") & res["ncrps"].notna()]
          if not res.empty and "exp" in res and "ncrps" in res else pd.DataFrame())
    df = df[df["exp"] == "E2-e1b"] if not df.empty else df  # primary panel: sim (powered)
    if df.empty:
        return _note("F4", "E2 not run")
    # CLUSTER bootstrap on (device, observable), k = 60: rows within one series
    # are rolling forecast origins, so the iid bootstrap over-narrows. Device-only
    # clustering is not enough here -- the 12 sim devices are near-exchangeable by
    # construction, so it absorbs the wrong dependence (see tables.t2_main).
    from ltn.eval.stats import cluster_bootstrap_ci
    full = df[df["model"] == "ltn"]["ncrps"].mean()
    names, vals, errs, cols = [], [], [], []
    for model, g in df.groupby("model"):
        v = g["ncrps"].to_numpy()
        groups = g[["device", "observable"]].astype(str).agg("|".join, axis=1).to_numpy()
        lo, hi = cluster_bootstrap_ci(lambda x: float(np.mean(x)), v, groups=groups)
        names.append(MODEL_LABELS.get(model, model))
        vals.append(np.mean(v) - full)
        errs.append((np.mean(v) - lo + hi - np.mean(v)) / 2)
        cols.append(MODEL_COLORS.get(model, INK2))
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y = np.arange(len(names))
    ax.barh(y, vals, xerr=errs, color=cols, height=0.55, error_kw={"lw": 0.8})
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, names)
    ax.set_xlabel("$\\Delta$NCRPS vs. full LTN (higher = worse)")
    _save(fig, "F4_ablations.pdf")


# --------------------------------------------------------------------------- F5
def f5_transfer() -> None:
    from ltn.eval.runlog import read_results
    res = read_results("runs")
    have_e4 = not res.empty and "exp" in res and res["exp"].isin(["E4a", "E4b"]).any()
    e5_dir = Path("runs/E5/ltn")
    have_e5 = e5_dir.exists() and any(e5_dir.glob("*/scored.parquet"))
    if not (have_e4 or have_e5):
        return _note("F5", "E4/E5 not run")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    fig.subplots_adjust(wspace=0.42)
    for ax, exp, title in ((axes[0], "E4a", "E4a: real"),
                           (axes[1], "E4b", "E4b: sim (SYNTH)")):
        df = res[res.get("exp") == exp] if have_e4 else pd.DataFrame()
        if df.empty:
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=INK2)
            ax.set_title(title)
            continue
        for arm, color in (("pretrained", MODEL_COLORS["ltn"]), ("scratch", INK2)):
            g = df[df["arm"] == arm]
            per = g.groupby(["k_days", "seed"])["ncrps"].mean().reset_index()
            m_ = per.groupby("k_days")["ncrps"].mean()
            sd = per.groupby("k_days")["ncrps"].std().fillna(0)
            ax.plot(m_.index, m_.values, "o-", ms=3, color=color, label=arm)
            ax.fill_between(m_.index, m_ - sd, m_ + sd, color=color, alpha=0.12, lw=0)
        ax.set_xlabel("adaptation days $k$")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("NCRPS")
    ax = axes[2]
    if have_e5:
        sc = pd.concat([pd.read_parquet(f) for f in e5_dir.glob("*/scored.parquet")])
        sc = sc.groupby(["device", "entity_id"], observed=True).mean().reset_index()
        # WITHIN-DEVICE percentile ranks (the pre-registered pooled-rank Spearman,
        # matching ltn.exp.e5.spearman_with_null); a naive cross-device pooled rank
        # mixes device-level offsets and flips the sign (Simpson's paradox).
        from scipy.stats import spearmanr
        sc["x"] = sc.groupby("device")["ltn_score"].rank(pct=True)
        sc["y"] = sc.groupby("device")["ler"].rank(pct=True, ascending=False)
        rho = spearmanr(sc["x"], sc["y"]).statistic
        for d, g in sc.groupby("device"):
            ax.scatter(g["x"], g["y"], s=8, alpha=0.7)
        ax.set_xlabel("trust score (rank)")
        ax.set_ylabel("reliability ($-$LER, rank)")
        ax.set_title(f"E5: zero-shot ($\\rho$={rho:.2f})")
    else:
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=INK2)
        ax.set_title("E5 — zero-shot Google")
    _save(fig, "F5_transfer.pdf")


# --------------------------------------------------------------------------- F6
def f6_regret() -> None:
    files = sorted(Path("runs/E6a").glob("replay_*_s0.parquet")) if Path("runs/E6a").exists() else []
    if not files:
        return _note("F6", "E6a not run")
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for f in files:
        pname = f.stem.replace("replay_", "").rsplit("_w", 1)[0]
        w = f.stem.rsplit("_w", 1)[1].split("_")[0]
        if w != "12":  # headline width; others in appendix
            continue
        df = pd.read_parquet(f)
        style = {"ls": "--"} if pname == "oracle" else {}
        ax.plot(range(len(df)), df["cum_regret"], color=POLICY_COLORS.get(pname, INK2),
                label=pname.replace("_", " "), **style)
    ax.set_xlabel("replay round (test days, SIM)")
    ax.set_ylabel("cumulative regret")
    ax.legend()
    _save(fig, "F6_regret.pdf")


# --------------------------------------------------------------------------- F7
def f7_event() -> None:
    f = Path("runs/E7/event_study.parquet")
    if not f.exists():
        return _note("F7", "E7 not run")
    ev = pd.read_parquet(f)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    gaps = ev[ev["event_type"] == "gap"]
    if len(gaps):
        ax = axes[0]
        # pool only positions still inside each event's gap (t_rel <= gap_len);
        # mixing post-gap recovery into the tail previously produced artifacts
        within = gaps[(gaps["t_rel"] <= gaps["gap_len"]) & (gaps["t_rel"] <= 4)]
        cols = ["entropy_pred", "err_emp"] + (["sigma_pred"] if "sigma_pred" in gaps else [])
        g = within.groupby("t_rel")[cols].mean()
        base = g.loc[:-1].mean()  # pre-gap reference level
        ax.plot(g.index, 100 * (g["err_emp"] / base["err_emp"] - 1), "s--", ms=3,
                color=MODEL_COLORS["arima"], label="empirical $|$error$|$")
        if "sigma_pred" in g:
            ax.plot(g.index, 100 * (g["sigma_pred"] / base["sigma_pred"] - 1), "^-", ms=3,
                    color=MODEL_COLORS["ltn"], label="v1 predictive $\\sigma$ (post hoc)")
        ax.plot(g.index, 100 * (g["entropy_pred"] / base["entropy_pred"] - 1), "o-", ms=3,
                color=MODEL_COLORS["ltn_gaussian"], label="v1 state entropy $S(\\rho_t)$ (prereg)")
        # the repaired model's sigma, on the SAME injected gaps: the Born-rule readout
        # is the one line here that moves with the error rather than ignoring it
        f2 = Path("runs/E7-v2/event_study.parquet")
        if f2.exists():
            e2 = pd.read_parquet(f2)
            g2 = e2[(e2["event_type"] == "gap") & (e2["t_rel"] <= e2["gap_len"])
                    & (e2["t_rel"] <= 4)]
            if len(g2) and "sigma_pred" in g2:
                a2 = g2.groupby("t_rel")[["sigma_pred"]].mean()
                b2 = a2.loc[:-1].mean()
                ax.plot(a2.index, 100 * (a2["sigma_pred"] / b2["sigma_pred"] - 1), "v-", ms=3,
                        color=MODEL_COLORS["ltn_v2"], lw=2.0,
                        label="v2 predictive $\\sigma$ (Born readout)")
        ax.axvspan(0, 4, color=GRID, alpha=0.5, lw=0)
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_xlabel("snapshots since gap start")
        ax.set_ylabel("growth vs. pre-gap (%)")
        ax.set_title("telemetry gap (SIM): uncertainty vs. error", fontsize=7.5)
        ax.legend(loc="upper left", fontsize=5.6)
    for etype, color, label in (("recal_real", MODEL_COLORS["ltn"], "recal (real, DAQEC)"),
                                ("recal_sim", MODEL_COLORS["ltn_gaussian"], "recal (SIM)"),
                                ("divergence", MODEL_COLORS["ltn_ode"], "divergence (real)")):
        g = ev[ev["event_type"] == etype]
        if len(g):
            m = g.groupby("t_rel")["dissonance"].mean()
            axes[1].plot(m.index, m.values, "o-", ms=3, color=color, label=label)
    axes[1].axvline(0, color=INK, lw=0.8)
    axes[1].set_xlabel("snapshots relative to event")
    axes[1].set_ylabel("mean dissonance")
    axes[1].set_title("dissonance around events", fontsize=7.5)
    axes[1].legend()
    _save(fig, "F7_event.pdf")


# --------------------------------------------------------------------------- F2
def f2_architecture() -> None:
    """Two-panel system diagram: (a) end-to-end pipeline from public telemetry to
    experiment outputs; (b) per-timestep LTN cell in plain language, each stage named
    by what it does with its Sec. III module name underneath, so it reads without a
    quantum-computing background. The formulas and the per-stage validity guarantees
    are in the text (Sec. III), not the figure."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    BLUE = MODEL_COLORS["ltn"]
    VIOLET = MODEL_COLORS["ltn_gaussian"]
    AQUA = MODEL_COLORS["lstm"]
    ORANGE = MODEL_COLORS["arima"]

    # Explicit axes, in inches: panel (a) keeps the 5.58 x 1.633 in of the original
    # two-row gridspec; panel (b), without formula and guarantee rows, needs 0.90 in.
    W, HA, HB, GAP, PAD = 7.2, 1.633, 0.90, 0.06, 0.05
    H = PAD + HB + GAP + HA + PAD
    fig = plt.figure(figsize=(W, H))
    b = fig.add_axes([0.9 / W, PAD / H, 5.58 / W, HB / H])
    a = fig.add_axes([0.9 / W, (PAD + HB + GAP) / H, 5.58 / W, HA / H])
    for ax in (a, b):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def box(ax, xc, yc, w, h, txt, color, fs=6.3, fc="white", lw=1.2):
        ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                                    boxstyle="round,pad=0.010", fc=fc, ec=color, lw=lw))
        ax.text(xc, yc, txt, ha="center", va="center", fontsize=fs)

    def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.0, style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=9, color=color, lw=lw, linestyle=ls))

    # ---------------- panel (a): system pipeline -------------------------------
    a.set_ylim(0, 1.14)
    a.text(0.005, 1.12, "(a) System pipeline", fontsize=7.5, fontweight="bold", va="top")
    srcs = [("DAQEC\n3 IBM Eagle, 14 d", 0.86), ("Google QEC\nStim/DEM, 2{,}552 runs"
             .replace("{,}", ","), 0.64), ("Silicon spin\n3 SiMOS devices", 0.42),
            ("Sim-Fleet\nsynthetic, labeled", 0.20)]
    for txt, yc in srcs:
        box(a, 0.075, yc, 0.13, 0.17, txt, INK2, fs=5.6)
    a.text(0.075, 0.03, "token-ready live harvester (unused)", fontsize=5.2,
           color=INK2, style="italic", ha="center")
    box(a, 0.29, 0.53, 0.14, 0.40,
        "UTS event lake\n(source, device,\nentity, observable,\nchannel, value,\n$\\sigma$, $t$, $t_{\\mathrm{cal}}$)",
        ORANGE, fs=5.6)
    for _, yc in srcs:
        arrow(a, 0.142, yc, 0.218, min(max(yc, 0.38), 0.68))
    box(a, 0.475, 0.68, 0.14, 0.20, "graph builder\ncoupling maps,\nz-scored features", AQUA, fs=5.6)
    box(a, 0.475, 0.34, 0.14, 0.20, "frozen splits\nSHA-256 sealed,\npre-registered", AQUA, fs=5.6)
    arrow(a, 0.362, 0.60, 0.403, 0.66)
    arrow(a, 0.362, 0.46, 0.403, 0.38)
    box(a, 0.655, 0.53, 0.13, 0.34, "LTN\ntemporal model\n(panel b)", BLUE, fs=6.4, lw=1.6)
    arrow(a, 0.547, 0.66, 0.588, 0.58)
    arrow(a, 0.547, 0.38, 0.588, 0.47)
    outs = [("forecasts $(\\mu,\\sigma)$\nE1, E4, E5", 0.82),
            ("trust scores +\nvacuity/dissonance\nE3, E7", 0.53),
            ("selection policy\n(LCB) E6", 0.22)]
    for txt, yc in outs:
        box(a, 0.885, yc, 0.16, 0.20, txt, VIOLET, fs=5.6)
        arrow(a, 0.722, 0.53 + (yc - 0.53) * 0.35, 0.803, yc)

    # ---------------- panel (b): per-timestep LTN cell -------------------------
    b.set_ylim(0, HB)  # one data unit = one inch vertically
    b.text(0.005, HB - 0.01, "(b) LTN cell (one node $v$, one timestep)", fontsize=7.5,
           fontweight="bold", va="top")
    cells = [  # (what the stage does, its Sec. III module, x centre, colour)
        ("trust state\n$\\rho_v$", "DM-State", 0.08, BLUE),
        ("evolve between\nobservations", "Lindblad drift", 0.275, BLUE),
        ("absorb new\ntelemetry", "evidence channel", 0.47, VIOLET),
        ("blend with\nneighbors", "KrausProp", 0.665, VIOLET),
        ("forecast $(\\mu,\\sigma)$\n+ uncertainty", "readout", 0.86, AQUA),
    ]
    yc, w, h = 0.37, 0.16, 0.33
    for main, module, xc, color in cells:
        box(b, xc, yc, w, h, "", color)
        b.text(xc, yc + 0.05, main, ha="center", va="center", fontsize=7.0)
        b.text(xc, yc - 0.118, module, ha="center", va="center", fontsize=6.0,
               color=INK2, style="italic")
    for i in range(len(cells) - 1):
        arrow(b, cells[i][2] + w / 2, yc, cells[i + 1][2] - w / 2, yc)
    # recurrence: the state and the GRU carry pass to the next timestep
    arrow(b, 0.90, yc - h / 2, 0.90, 0.10, color=INK2, lw=0.8)
    b.annotate("", xy=(0.08, yc - h / 2), xytext=(0.08, 0.10),
               arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    b.plot([0.08, 0.90], [0.10, 0.10], color=INK2, lw=0.8)
    b.text(0.49, 0.035, "trust state and GRU carry $\\rightarrow$ next timestep",
           fontsize=6.3, color=INK2, ha="center", va="center")
    # what enters each stage
    for xc, txt in ((0.275, "time gap; $T_1$, $T_2$, cal. age"),
                    (0.47, "new telemetry"), (0.665, "neighbors' states")):
        b.text(xc, 0.71, txt, fontsize=6.3, color=INK2, ha="center", va="center")
        arrow(b, xc, 0.665, xc, yc + h / 2, lw=0.8)

    _save(fig, "F2_architecture.pdf")


FIGS = {"F1": f1_concept, "F2": f2_architecture, "F3": f3_crps, "F4": f4_ablations,
        "F5": f5_transfer, "F6": f6_regret, "F7": f7_event}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    apply_style()
    names = args.only.split(",") if args.only else list(FIGS)
    for n in names:
        try:
            FIGS[n]()
        except Exception as e:
            print(f"[figures] ERROR {n}: {e}")


if __name__ == "__main__":
    main()
