"""A4-carry: is the density matrix rho load-bearing, or does the GRU carry bypass it?

Tier 0 (this file, --tier 0): EVAL-ONLY interventions on the ALREADY-TRAINED
runs/E1{a,b}/ltn_v2/{0..4}/model.pt checkpoints. No retraining, no weight updates.
Four arms, all scored with the same frozen test split and the same NCRPS pipeline
as ltn.exp.run:

  baseline    untouched checkpoint (must reproduce runs/*/results.jsonl exactly)
  zero_resid  BornReadout.resid output forced to 0 at readout time. v2's ONLY
              carry -> output path is this residual (sigma is a functional of rho
              alone, ltn_v2.py:213-214, and self.hermitian is None), so this
              measures the exact share of mu the carry supplies.
  perm_rho    rho permuted across the entity batch at readout, carry intact.
              If NCRPS barely moves, predictions do not depend on rho.
  perm_carry  the converse: carry permuted, rho intact.

Both permutations are applied INSIDE BornReadout.forward, i.e. strictly at
readout: the recurrence (step/drift) is untouched, so each entity's state was
still built from its own history. A fixed seeded derangement is used, identical
across arms and seeds.

CAVEAT that must accompany any "the carry dominates" reading: ltn.py:147 feeds
rho_to_real(rho) into the GRU at every step, so the carry is itself a function of
rho's history. carry dominance therefore does NOT prove rho is inert; it proves
the *readout* reaches the state through the GRU's real-vector compression rather
than through the Born term. perm_rho is the arm that speaks to the density-matrix
readout path directly.

Tier 1 (--tier 1 is not run here; see ltn/exp/variants.py::LTNv2NoCarryForecaster
and scripts run with `--model ltn_v2_nocarry`): a trained carry-free arm.

Usage:
  python scripts/carry_ablation.py --exp e1a --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ltn.data.series import forecast_pairs
from ltn.exp.run import build_data, evaluate, parse_norm_stats
from ltn.eval.splits import load_split
from ltn.exp.variants import LTNv2Forecaster
from ltn.models.ltn_v2 import LTNv2Model
from ltn.models.param_match import count_trainable, match_width

ARMS = ("baseline", "zero_resid", "perm_rho", "perm_carry")


def _derangement(n: int, seed: int = 1234) -> torch.Tensor:
    """Fixed random permutation with no fixed point (n >= 2)."""
    g = torch.Generator().manual_seed(seed)
    for _ in range(1000):
        p = torch.randperm(n, generator=g)
        if n < 2 or bool((p != torch.arange(n)).all()):
            return p
    return torch.roll(torch.arange(n), 1)  # guaranteed derangement fallback


class ReadoutProbe:
    """Wraps BornReadout.forward: applies one intervention and records magnitudes."""

    def __init__(self, head, arm: str):
        self.head, self.arm = head, arm
        self._orig = head.forward
        self.born_abs, self.resid_abs = [], []
        self.born_std, self.resid_std = [], []

    def __enter__(self):
        head, arm = self.head, self.arm

        def fwd(rho, carry):
            if arm == "perm_rho":
                p = _derangement(rho.shape[0]).to(rho.device)
                rho = rho[p]
            elif arm == "perm_carry":
                p = _derangement(carry.shape[0]).to(carry.device)
                carry = carry[p]
            mu, sigma = self._orig(rho, carry)
            r = (head.resid(carry.to(torch.float64)) if head.resid is not None
                 else torch.zeros_like(mu))
            born = mu - r
            self.resid_abs.append(float(r.abs().mean()))
            self.born_abs.append(float(born.abs().mean()))
            # spread ACROSS the entity batch: how much of each term actually
            # discriminates between entities, vs. acting as a shared offset
            if mu.shape[0] > 1:
                self.resid_std.append(float(r.std(dim=0).mean()))
                self.born_std.append(float(born.std(dim=0).mean()))
            if arm == "zero_resid":
                mu = born
            return mu, sigma

        head.forward = fwd
        return self

    def __exit__(self, *a):
        self.head.forward = self._orig
        return False


def run_arm(split, uts, panel, masks, ckpt: Path, arm: str, dev: str) -> dict:
    tr, va, te = masks
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    cfg["uts"] = uts
    cfg["epochs"] = 0            # no training: fit() just builds + loads weights
    cfg["init_from"] = str(ckpt)
    cfg["device"] = dev
    m = LTNv2Forecaster()
    m.fit(panel, tr, va, cfg)
    pairs = forecast_pairs(panel, split["horizons_steps"], te)
    with ReadoutProbe(m.model.forecast_head, arm) as probe:
        mu, sigma = m.predict(pairs, panel)
    cells, _ = evaluate(pairs, panel, mu, sigma, parse_norm_stats(split))
    return {
        "arm": arm,
        "ncrps": float(cells["ncrps"].mean()),
        "crps": float(cells["crps"].mean()),
        "mae": float(cells["mae"].mean()),
        "nll": float(cells["nll"].mean()),
        "coverage90": float(cells["coverage90"].mean()),
        "n_cells": int(len(cells)),
        "mean_abs_born_mu": float(np.mean(probe.born_abs)),
        "mean_abs_resid_mu": float(np.mean(probe.resid_abs)),
        "entity_std_born_mu": float(np.mean(probe.born_std)) if probe.born_std else None,
        "entity_std_resid_mu": float(np.mean(probe.resid_std)) if probe.resid_std else None,
        "n_params": int(m.n_params()),
    }


class LTNv2NoCarryForecaster(LTNv2Forecaster):
    """Tier 1: LTNv2 with use_carry=False, EQUAL-CAPACITY matched to the full v2
    model built from the same cfg (ltn.tex commits ablation variants to +/-10%
    parameter matching), by binary-searching `hidden` exactly as the E2 variants
    in ltn/exp/variants.py do.  `hidden` is the right knob: it is the width of
    GammaNet / the evidence channel / the GRU input MLP, none of which is the
    removed path."""

    name = "ltn_v2_nocarry"

    def _make_model(self, n_obs: int, cfg: dict):
        def build(h: int, use_carry: bool):
            return LTNv2Model(
                n_obs=n_obs, platform_dim=7,
                d=int(cfg["d"]) if cfg.get("d") else None,
                carry_dim=int(cfg.get("carry_dim", 32)), hidden=h,
                evidence_rank=int(cfg.get("evidence_rank", 4)),
                rounds=int(cfg.get("rounds", 2)),
                use_graph=bool(cfg.get("use_graph", True)),
                use_lindblad=bool(cfg.get("use_lindblad", True)),
                physics_gamma=bool(cfg.get("physics_gamma", True)),
                time_unit_h=float(cfg.get("time_unit_h", 8.0)),
                purity_eps=float(cfg.get("purity_eps", 1e-3)),
                use_carry=use_carry,
            )

        target = count_trainable(build(int(cfg.get("hidden", 64)), True))
        model, w, n = match_width(target, lambda h: build(h, False), lo=8, hi=4096)
        print(f"[param-match] full v2 hidden={cfg.get('hidden', 64)} -> {target} params; "
              f"no-carry hidden={w} -> {n} params ({100 * (n - target) / target:+.2f}%)", flush=True)
        return model


def run_tier1(exp: str, seeds, dev: str, d_sweep=()) -> None:
    """Train the carry-free arm on the tuned cfg, 5 seeds, into runs/CARRY/.

    d_sweep re-runs the arm at several state dimensions. This is the point of
    the whole exercise: ltn_v2.py's pre-registered falsifier says NCRPS must MOVE
    with d, and on the FULL model it does not (E1a v2 d=4/5/8 -> .2793/.2791/
    .2795). If the flat sweep is the carry bypass rather than a dead state, the
    sweep should come alive once the bypass is closed."""
    import json as _json
    from ltn.exp.run import MODELS, _register, run_one
    _register()
    MODELS["ltn_v2_nocarry"] = LTNv2NoCarryForecaster
    best = _json.loads(Path(f"runs/tuning/{exp}_ltn_v2.json").read_text())["params"]
    for d in (d_sweep or [None]):
        for seed in seeds:
            cfg = dict(best, device=dev)
            tag = f"ltn_v2_nocarry-{exp}"
            if d is not None:
                cfg["d"] = int(d)
                tag += f"-d{d}"
            cells = run_one(exp, "ltn_v2_nocarry", int(seed), cfg,
                            exp_override="CARRY", label=tag)
            print(f"[tier1 {exp} seed{seed} d={d}] no-carry mean "
                  f"NCRPS={cells['ncrps'].mean():.6f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=0, choices=[0, 1])
    ap.add_argument("--exp", default="e1a", choices=["e1a", "e1b"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--d-sweep", dest="d_sweep", type=int, nargs="*", default=[],
                    help="tier 1 only: also sweep the state dimension d")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.tier == 1:
        run_tier1(args.exp, args.seeds, args.device, args.d_sweep)
        return

    split = load_split(args.exp)
    uts, panel, masks = build_data(split)
    exp_dir = {"e1a": "E1a", "e1b": "E1b"}[args.exp]
    out = Path(args.out or f"runs/CARRY/tier0_{args.exp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        ckpt = Path(f"runs/{exp_dir}/ltn_v2/{seed}/model.pt")
        for arm in args.arms:
            r = run_arm(split, uts, panel, masks, ckpt, arm, args.device)
            r.update(exp=args.exp, seed=seed, ckpt=str(ckpt))
            rows.append(r)
            print(f"[{args.exp} seed{seed}] {arm:11s} NCRPS={r['ncrps']:.6f} "
                  f"|born mu|={r['mean_abs_born_mu']:.4f} |resid mu|={r['mean_abs_resid_mu']:.4f}",
                  flush=True)
            out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")
    base = {r["seed"]: r["ncrps"] for r in rows if r["arm"] == "baseline"}
    for arm in args.arms:
        v = [r["ncrps"] for r in rows if r["arm"] == arm]
        d = [r["ncrps"] - base[r["seed"]] for r in rows if r["arm"] == arm and r["seed"] in base]
        print(f"{arm:11s} mean NCRPS={np.mean(v):.6f}  median={np.median(v):.6f}  "
              f"mean delta vs baseline={np.mean(d):+.6f} ({100*np.mean(d)/np.mean(list(base.values())):+.2f}%)")


if __name__ == "__main__":
    main()
