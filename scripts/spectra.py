"""Generator-spectrum / non-normality diagnostics for LTN Lindblad models.

Answers two questions with measurements, not assertions:

  DEFECT 1 -- Proposition 3 claims Hermitian jump operators make I/d the UNIQUE
  fixed point. Uniqueness needs irreducibility (primitivity), not Hermiticity.
  For a purely dissipative generator built from HERMITIAN jumps,
      D(X) = sum_k gamma_k (L_k X L_k - 1/2 {L_k^2, X})
           = -1/2 sum_k gamma_k [L_k, [L_k, X]]
  is self-adjoint and negative semidefinite in the Hilbert-Schmidt inner product,
  so for any gamma_k > 0
      ker(D) = intersection_k ker(ad_{L_k}) = commutant of {L_k}.
  Uniqueness therefore holds iff {L_k} generates the full matrix algebra, and the
  kernel dimension is INDEPENDENT of the learned rates. `--mode kernel` measures
  the kernel dimension of every fitted v2 checkpoint three ways (SVD rank of the
  generator at gamma=1, at random positive gammas, and the commutant rank) and
  checks that vec(I/d) actually lies in it.

  DEFECT 2 -- the paper attributes v1's seed variance (seed 3 NCRPS 0.3375 vs
  median 0.2808) to "non-normality of the learned Hamiltonian" and calls the
  diagnosis "confirmed". The paired -H ablation establishes the CAUSE; it does
  not establish the MECHANISM. `--mode spectra` rolls each fitted model over the
  real E1a timelines, captures every Lindblad generator the model actually used
  (drift steps and evidence-free horizon rolls), and tabulates the standard
  non-normality diagnostics per seed:
      spectral abscissa, spectral gap, Henrici departure from normality,
      eigenvector condition number kappa(V), numerical abscissa (rightmost point
      of the field of values), ||[H, L_k]||_F, and the matrix_exp argument norm.
  `--mode grads` re-measures gradient norms at the fitted parameters (training-time
  gradient norms were never logged -- ltn/exp/train.py only clips, it does not
  record -- so they are not recoverable from runs/campaign_logs).

The question the tables have to answer is narrow: DOES SEED 3 STAND OUT? If it
does not, the mechanism claim is unsupported even though the causal claim stands.

Usage:
    python scripts/spectra.py --mode all
    python scripts/spectra.py --mode kernel|spectra|grads
Outputs JSON + CSV under runs/SPECTRA/.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from pathlib import Path

# The matrices here are tiny (d^2 <= 64), so BLAS threading is pure overhead:
# unpinned, the eig/svd sweep thrashes across cores and runs ~100x slower.
# Must be set before numpy/torch import their BLAS backends.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import torch

from ltn.models.dmstate import CDTYPE, vec
from ltn.models.lindblad import lindblad_generator

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runs" / "SPECTRA"

# checkpoint cfg keys that determine architecture (must be replayed to rebuild)
ARCH_KEYS = (
    "d", "carry_dim", "hidden", "evidence_rank", "rounds", "use_graph",
    "use_lindblad", "physics_gamma", "use_hamiltonian", "purity_eps",
)
# training keys replayed so the gradient measurement uses the SAME BPTT window
# the run was actually trained with (they do not affect the rebuilt architecture)
TRAIN_KEYS = ("bptt", "lr")

# Relative singular-value tolerance for numerical rank. Generators are rescaled
# to unit spectral norm first, so this is a pure conditioning threshold.
RANK_TOL = 1e-8


# --------------------------------------------------------------------------
# matrix diagnostics
# --------------------------------------------------------------------------
def departure_from_normality(A: np.ndarray, eigvals: np.ndarray | None = None) -> float:
    """Henrici's departure from normality dep_F(A) = sqrt(||A||_F^2 - sum|lambda_i|^2).

    Zero iff A is normal (Schur: ||A||_F^2 = sum|lambda|^2 + ||strictly-upper||^2).
    """
    if eigvals is None:
        eigvals = np.linalg.eigvals(A)
    fro2 = float(np.sum(np.abs(A) ** 2))
    lam2 = float(np.sum(np.abs(eigvals) ** 2))
    return math.sqrt(max(fro2 - lam2, 0.0))


def eigvec_condition(A: np.ndarray) -> float:
    """kappa_2(V) for A = V diag(lambda) V^-1 -- the Bauer-Fike eigenvalue
    sensitivity constant and the standard scalar measure of non-normality.
    Returns inf if the eigenvector matrix is numerically singular (defective A)."""
    try:
        _, V = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        return float("inf")
    s = np.linalg.svd(V, compute_uv=False)
    if s[-1] <= 0 or not np.isfinite(s[-1]):
        return float("inf")
    return float(s[0] / s[-1])


def numerical_abscissa(A: np.ndarray) -> float:
    """Rightmost point of the field of values = lambda_max((A + A^H)/2).

    Bounds the transient growth ||exp(At)|| <= exp(t * numerical_abscissa); for a
    normal A it equals the spectral abscissa, and the GAP between the two is
    exactly the transient-amplification headroom non-normality buys.
    """
    Hp = (A + A.conj().T) / 2
    return float(np.max(np.linalg.eigvalsh(Hp)))


def spectral_stats(A: np.ndarray) -> dict:
    """Full diagnostic bundle for one d^2 x d^2 Liouvillian."""
    lam = np.linalg.eigvals(A)
    re = lam.real
    abscissa = float(np.max(re))
    # spectral gap: distance from the stationary eigenvalue(s) at Re=0 to the
    # slowest genuinely decaying mode. Tolerance is relative to the generator norm.
    nrm = float(np.linalg.norm(A, 2))
    tol = max(1e-10, 1e-9 * max(nrm, 1.0))
    decaying = re[re < -tol]
    gap = float(-np.max(decaying)) if decaying.size else 0.0
    return {
        "abscissa": abscissa,
        "gap": gap,
        "n_zero_eig": int(np.sum(np.abs(re) <= tol)),
        "departure": departure_from_normality(A, lam),
        "departure_rel": departure_from_normality(A, lam) / max(np.linalg.norm(A, "fro"), 1e-300),
        "kappa_V": eigvec_condition(A),
        "num_abscissa": numerical_abscissa(A),
        "fro": float(np.linalg.norm(A, "fro")),
        "spec_norm": nrm,
    }


def kernel_dim(A: np.ndarray, tol: float = RANK_TOL) -> tuple[int, np.ndarray]:
    """Numerical dim ker(A) via SVD of the spectral-norm-normalized generator."""
    s = np.linalg.svd(A, compute_uv=False)
    smax = s[0] if s[0] > 0 else 1.0
    return int(np.sum(s / smax < tol)), s / smax


def commutant_dim(Ls: np.ndarray, tol: float = RANK_TOL) -> int:
    """dim { X : [L_k, X] = 0 for all k }, computed as d^2 - rank(stacked ad_{L_k}).

    ad_L in the column-stacking vec convention is kron(I, L) - kron(L^T, I).
    """
    K, d, _ = Ls.shape
    eye = np.eye(d)
    blocks = [np.kron(eye, L) - np.kron(L.T, eye) for L in Ls]
    M = np.vstack(blocks)
    s = np.linalg.svd(M, compute_uv=False)
    smax = s[0] if s[0] > 0 else 1.0
    return int(d * d - np.sum(s / smax >= tol))


def identity_in_kernel(A: np.ndarray, d: int) -> float:
    """||A vec(I/d)||_2 / ||A||_2 -- 0 iff the maximally mixed state is stationary."""
    v = np.eye(d, dtype=complex).reshape(-1) / d  # vec(I/d): symmetric, convention-free
    nrm = np.linalg.norm(A, 2)
    return float(np.linalg.norm(A @ v) / max(nrm, 1e-300))


# --------------------------------------------------------------------------
# checkpoint plumbing
# --------------------------------------------------------------------------
def v2_jump_ops(state_dict) -> np.ndarray:
    """Rebuild the [K, d, d] Hermitian jump operators of a v2 checkpoint."""
    A = torch.complex(state_dict["jumps.A_real"], state_dict["jumps.A_imag"])
    L = A + A.conj().transpose(-2, -1)
    return L.numpy()


def v1_jump_ops(state_dict) -> np.ndarray:
    A = torch.complex(state_dict["jumps.ops_real"], state_dict["jumps.ops_imag"])
    return A.numpy()


def find_v2_checkpoints() -> list[Path]:
    paths = []
    for p in sorted((REPO / "runs").rglob("model.pt")):
        sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        if "jumps.A_real" in sd:  # UnitalJumpDictionary => v2
            paths.append(p)
    return paths


def build_data_once(split_name: str = "e1a"):
    from ltn.eval.splits import load_split
    from ltn.exp.run import _register, build_data
    _register()
    split = load_split(split_name)
    uts, panel, masks = build_data(split, lake=str(REPO / "data" / "uts"))
    return split, uts, panel, masks


def load_forecaster(ckpt: Path, model_name: str, split, uts, panel, masks, device="cpu"):
    """Rebuild the exact fitted model behind a checkpoint (epochs=0 + init_from)."""
    from ltn.exp.run import MODELS, model_cfg
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    extra = {k: v for k, v in ck["cfg"].items() if k in ARCH_KEYS + TRAIN_KEYS}
    extra.update(epochs=0, init_from=str(ckpt), device=device)
    cfg = model_cfg(split, int(ck["cfg"].get("seed", 0)), extra)
    cfg["uts"] = uts
    m = MODELS[model_name]()
    m.fit(panel, masks[0], masks[1], cfg)
    return m


@contextlib.contextmanager
def capture_generators():
    """Record every (H, Ls, gammas, dt) tuple the model feeds to `evolve`.

    Both LTNModel and LTNv2Model import `evolve` into their own module namespace,
    so both bindings are patched. Captures drift steps AND horizon rolls.
    """
    import ltn.models.ltn as m1
    import ltn.models.ltn_v2 as m2
    recs: list[dict] = []

    def wrap(orig):
        def wrapped(rho, H, Ls, gammas, dt):
            recs.append({
                "H": H.detach().cpu().clone(),
                "Ls": torch.stack([L.detach().cpu().clone() for L in Ls]),
                "gammas": gammas.detach().cpu().clone(),
                "dt": dt.detach().cpu().clone(),
            })
            return orig(rho, H, Ls, gammas, dt)
        return wrapped

    o1, o2 = m1.evolve, m2.evolve
    m1.evolve, m2.evolve = wrap(o1), wrap(o2)
    try:
        yield recs
    finally:
        m1.evolve, m2.evolve = o1, o2


@torch.no_grad()
def roll_and_capture(fc, horizon: int = 21, max_records: int = 400):
    """Roll every device timeline once; return the captured generator batches.

    `max_records` caps the number of captured BATCHES (each batch is [n_entities]
    generators), evenly subsampled over the timeline, to keep eig() cost sane.
    """
    model = fc.model
    model.eval()
    dev = fc._torch_device
    out = []
    for dname, seq in fc.seqs.items():
        x, cal = seq.x.to(dev), seq.cal_age_h.to(dev)
        ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
        n = len(seq.entity_ids)
        T = len(seq.ts)
        stride = max(1, T // max(1, max_records // max(1, len(fc.seqs))))
        st = model.init_state(n, device=dev)
        for t in range(T):
            with capture_generators() as recs:
                st = model.step(st, x[t], cal[t],
                                seq.dt_steps[t].expand(n).to(dev), ei, ea)
            if t % stride == 0:
                for r in recs:
                    r.update(device=dname, t=t, kind="drift")
                out.extend(recs)
                with capture_generators() as frecs:
                    model.forecast(st, x[t], cal[t],
                                   torch.full((n,), float(horizon),
                                              dtype=torch.float64, device=dev))
                for r in frecs:
                    r.update(device=dname, t=t, kind=f"forecast_h{horizon}")
                out.extend(frecs)
    return out


# --------------------------------------------------------------------------
# mode: kernel  (DEFECT 1)
# --------------------------------------------------------------------------
def mode_kernel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for p in find_v2_checkpoints():
        sd = torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        Ls = v2_jump_ops(sd)
        K, d, _ = Ls.shape
        Lt = [torch.tensor(L, dtype=CDTYPE) for L in Ls]
        H0 = torch.zeros(1, d, d, dtype=CDTYPE)

        def gen_for(g):
            G = lindblad_generator(H0, Lt, torch.tensor([g], dtype=torch.float64))
            return G[0].numpy()

        # (a) unit rates
        G1 = gen_for(np.ones(K))
        kd1, sv1 = kernel_dim(G1)
        # (b) random positive rates, log-uniform over 6 decades -- if the kernel
        #     dimension is rate-independent (as the ad-operator argument predicts)
        #     every draw must agree with (a)
        kd_rand = []
        for _ in range(8):
            g = 10.0 ** rng.uniform(-3, 3, size=K)
            kd_rand.append(kernel_dim(gen_for(g))[0])
        # (c) commutant rank -- the rate-free theoretical answer
        kc = commutant_dim(Ls)
        # Hermiticity of the learned jumps (structural, should be exact)
        herm_err = float(max(np.linalg.norm(L - L.conj().T) for L in Ls))
        rows.append({
            "ckpt": str(p.relative_to(REPO)),
            "d": d, "K": K,
            "kernel_dim_gamma1": kd1,
            "kernel_dim_random_min": int(min(kd_rand)),
            "kernel_dim_random_max": int(max(kd_rand)),
            "commutant_dim": kc,
            "unique_fixed_point": bool(kd1 == 1 and kc == 1),
            "identity_residual": identity_in_kernel(G1, d),
            "hermiticity_err": herm_err,
            "sv_smallest": float(sv1[-1]),
            "sv_second_smallest": float(sv1[-2]),
            "sv_gap_ratio": float(sv1[-2] / max(sv1[-1], 1e-300)),
        })
    return pd.DataFrame(rows)


def mode_kernel_empirical(split, uts, panel, masks) -> pd.DataFrame:
    """Same kernel question but with the REAL learned rates the model produces on
    the real E1a timelines, not synthetic gammas."""
    rows = []
    for seed in range(5):
        ckpt = REPO / f"runs/E1a/ltn_v2/{seed}/model.pt"
        if not ckpt.exists():
            continue
        fc = load_forecaster(ckpt, "ltn_v2", split, uts, panel, masks)
        recs = roll_and_capture(fc, max_records=60)
        kds, ids, gmin, gmax = [], [], [], []
        for r in recs:
            Lt = [r["Ls"][k] for k in range(r["Ls"].shape[0])]
            G = lindblad_generator(r["H"], Lt, r["gammas"]).numpy()
            for i in range(G.shape[0]):
                kd, _ = kernel_dim(G[i])
                kds.append(kd)
                ids.append(identity_in_kernel(G[i], r["H"].shape[-1]))
            gmin.append(float(r["gammas"].min()))
            gmax.append(float(r["gammas"].max()))
        rows.append({
            "seed": seed, "n_generators": len(kds),
            "kernel_dim_min": int(min(kds)), "kernel_dim_max": int(max(kds)),
            "kernel_dim_mode": int(pd.Series(kds).mode()[0]),
            "identity_residual_max": float(max(ids)),
            "gamma_min": float(min(gmin)), "gamma_max": float(max(gmax)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# mode: spectra  (DEFECT 2)
# --------------------------------------------------------------------------
def analyse_records(recs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in recs:
        Ls_t = r["Ls"]
        K = Ls_t.shape[0]
        Lt = [Ls_t[k] for k in range(K)]
        G = lindblad_generator(r["H"], Lt, r["gammas"])
        Gn = G.numpy()
        # dissipator-only generator (H zeroed, everything else identical): the
        # paper claims H_v "inflates the generator's norm from 1.0 to 20.5"
        G0 = lindblad_generator(torch.zeros_like(r["H"]), Lt, r["gammas"]).numpy()
        Hn = r["H"].numpy()
        Lsn = Ls_t.numpy()
        dt = r["dt"].numpy()
        for i in range(Gn.shape[0]):
            st = spectral_stats(Gn[i])
            Hi = Hn[i] if Hn.ndim == 3 else Hn
            hf = float(np.linalg.norm(Hi, "fro"))
            comms = [float(np.linalg.norm(Hi @ L - L @ Hi, "fro")) for L in Lsn]
            # relative commutator: scale-free non-commutativity of H with the jumps
            comms_rel = [
                c / max(hf * float(np.linalg.norm(L, "fro")), 1e-300)
                for c, L in zip(comms, Lsn)
            ]
            dti = float(dt[i]) if dt.ndim else float(dt)
            st.update({
                "device": r["device"], "t": r["t"], "kind": r["kind"], "entity": i,
                "H_fro": hf,
                "comm_max": max(comms), "comm_mean": float(np.mean(comms)),
                "comm_rel_max": max(comms_rel),
                "dt": dti,
                "arg_fro": st["fro"] * dti,        # ||L dt||_F: matrix_exp argument
                "arg_spec": st["spec_norm"] * dti,
                "departure_arg": st["departure"] * dti,
                "gamma_max": float(r["gammas"][i].max()),
                "gamma_sum": float(r["gammas"][i].sum()),
                "fro_noH": float(np.linalg.norm(G0[i], "fro")),
                "fro_ratio_H": (st["fro"]
                                / max(float(np.linalg.norm(G0[i], "fro")), 1e-300)),
            })
            rows.append(st)
    return pd.DataFrame(rows)


def mode_spectra(split, uts, panel, masks) -> pd.DataFrame:
    jobs = [
        ("ltn", "ltn", "runs/E1a/ltn/{s}/model.pt"),
        ("ltn_noH", "ltn", "runs/E2-e1a/ltn_noH/{s}/model.pt"),
        ("ltn_v2", "ltn_v2", "runs/E1a/ltn_v2/{s}/model.pt"),
    ]
    frames = []
    for label, model_name, tmpl in jobs:
        for seed in range(5):
            ckpt = REPO / tmpl.format(s=seed)
            if not ckpt.exists():
                print(f"  MISSING {ckpt}")
                continue
            t0 = time.time()
            fc = load_forecaster(ckpt, model_name, split, uts, panel, masks)
            recs = roll_and_capture(fc, max_records=90)
            df = analyse_records(recs)
            df.insert(0, "seed", seed)
            df.insert(0, "variant", label)
            frames.append(df)
            print(f"  {label} seed {seed}: {len(df)} generators "
                  f"({time.time() - t0:.0f}s)")
    return pd.concat(frames, ignore_index=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per (variant, seed) summary of the non-normality diagnostics."""
    g = df.groupby(["variant", "seed"])
    out = pd.DataFrame({
        "n": g.size(),
        "abscissa_max": g["abscissa"].max(),
        "gap_median": g["gap"].median(),
        "gap_min": g["gap"].min(),
        "departure_mean": g["departure"].mean(),
        "departure_max": g["departure"].max(),
        "departure_rel_mean": g["departure_rel"].mean(),
        "departure_rel_max": g["departure_rel"].max(),
        "kappaV_median": g["kappa_V"].median(),
        "kappaV_p99": g["kappa_V"].quantile(0.99),
        "kappaV_max": g["kappa_V"].max(),
        "num_abscissa_max": g["num_abscissa"].max(),
        "H_fro_mean": g["H_fro"].mean(),
        "H_fro_median": g["H_fro"].median(),
        "H_fro_max": g["H_fro"].max(),
        "fro_median": g["fro"].median(),
        "fro_noH_median": g["fro_noH"].median(),
        "fro_ratio_H_median": g["fro_ratio_H"].median(),
        "comm_max": g["comm_max"].max(),
        "comm_rel_max": g["comm_rel_max"].max(),
        "arg_fro_mean": g["arg_fro"].mean(),
        "arg_fro_max": g["arg_fro"].max(),
        "departure_arg_max": g["departure_arg"].max(),
        "gamma_max": g["gamma_max"].max(),
    }).reset_index()
    return out


# --------------------------------------------------------------------------
# mode: grads
# --------------------------------------------------------------------------
def measure_grad_norms(fc, split, horizons, max_windows: int = 400,
                       bptt_override: int | None = None) -> dict:
    """Gradient norm of the training NLL at the FITTED parameters.

    Reproduces the training inner loop (same loss, same detach schedule) but
    measures ||grad||_2 over all parameters BEFORE clipping -- the quantity
    training never recorded (ltn/exp/train.py only calls clip_grad_norm_).

    bptt_override=k shortens the truncation window to k steps, which yields many
    more independent samples of the gradient magnitude. This is NOT the training
    schedule; it is reported separately as a spread estimate.
    """
    import numpy as np
    model = fc.model
    model.train()
    dev = fc._torch_device
    bptt = int(bptt_override or fc._cfg_cache.get("bptt") or 64)
    norms, norms_H = [], []
    for dname, seq in fc.seqs.items():
        x, cal, y = seq.x.to(dev), seq.cal_age_h.to(dev), seq.y.to(dev)
        ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
        n, T = len(seq.entity_ids), len(seq.ts)
        st = model.init_state(n, device=dev)
        loss_acc, n_acc = None, 0
        for t in range(T):
            st = model.step(st, x[t], cal[t],
                            seq.dt_steps[t].expand(n).to(dev), ei, ea)
            for h in horizons:
                if t + h >= T:
                    continue
                target = y[t + h]
                obs = ~torch.isnan(target)
                if not obs.any():
                    continue
                mu, sigma, _ = model.forecast(
                    st, x[t], cal[t],
                    torch.full((n,), float(h), dtype=torch.float64, device=dev))
                l = 0.5 * (((target[obs] - mu[obs]) / sigma[obs]) ** 2
                           + 2 * sigma[obs].log() + math.log(2 * math.pi))
                l = l.mean()
                loss_acc = l if loss_acc is None else loss_acc + l
                n_acc += 1
            if (t + 1) % bptt == 0 or t == T - 1:
                if loss_acc is not None and len(norms) < max_windows:
                    model.zero_grad(set_to_none=True)
                    (loss_acc / max(n_acc, 1)).backward()
                    tot = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            tot += float(p.grad.detach().pow(2).sum())
                    norms.append(math.sqrt(tot))
                    hg = 0.0
                    if getattr(model, "hermitian", None) is not None:
                        for p in model.hermitian.parameters():
                            if p.grad is not None:
                                hg += float(p.grad.detach().pow(2).sum())
                    norms_H.append(math.sqrt(hg))
                    model.zero_grad(set_to_none=True)
                loss_acc, n_acc = None, 0
                st = type(st)(**{k: (v.detach() if torch.is_tensor(v) else v)
                                 for k, v in st.__dict__.items()})
    a = np.array(norms)
    aH = np.array(norms_H)
    return {
        "n_windows": int(a.size),
        "grad_mean": float(a.mean()) if a.size else float("nan"),
        "grad_median": float(np.median(a)) if a.size else float("nan"),
        "grad_max": float(a.max()) if a.size else float("nan"),
        "grad_std": float(a.std()) if a.size else float("nan"),
        "grad_frac_clipped": float((a > 1.0).mean()) if a.size else float("nan"),
        "gradH_mean": float(aH.mean()) if aH.size else float("nan"),
        "gradH_max": float(aH.max()) if aH.size else float("nan"),
    }


def mode_grads(split, uts, panel, masks) -> pd.DataFrame:
    jobs = [
        ("ltn", "ltn", "runs/E1a/ltn/{s}/model.pt"),
        ("ltn_noH", "ltn", "runs/E2-e1a/ltn_noH/{s}/model.pt"),
        ("ltn_v2", "ltn_v2", "runs/E1a/ltn_v2/{s}/model.pt"),
    ]
    horizons = list(split["horizons_steps"])
    rows = []
    for label, model_name, tmpl in jobs:
        for seed in range(5):
            ckpt = REPO / tmpl.format(s=seed)
            if not ckpt.exists():
                continue
            fc = load_forecaster(ckpt, model_name, split, uts, panel, masks)
            for tag, override in (("train_bptt", None), ("bptt2", 2)):
                r = measure_grad_norms(fc, split, horizons, bptt_override=override)
                r.update(variant=label, seed=seed, schedule=tag,
                         bptt=int(override or fc._cfg_cache.get("bptt") or 64))
                rows.append(r)
                print(f"  grads {label} s{seed} [{tag}]: n={r['n_windows']} "
                      f"mean={r['grad_mean']:.4g} max={r['grad_max']:.4g}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# mode: link  -- do the diagnostics actually track the per-seed outcome?
# --------------------------------------------------------------------------
NCRPS_SOURCES = {
    "ltn": "runs/E1a/ltn/{s}/results.jsonl",
    "ltn_noH": "runs/E2-e1a/ltn_noH/{s}/results.jsonl",
    "ltn_v2": "runs/E1a/ltn_v2/{s}/results.jsonl",
}


def per_seed_ncrps() -> pd.DataFrame:
    rows = []
    for variant, tmpl in NCRPS_SOURCES.items():
        for s in range(5):
            p = REPO / tmpl.format(s=s)
            if not p.exists():
                continue
            cells = pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
            rows.append({"variant": variant, "seed": s, "ncrps": float(cells["ncrps"].mean())})
    return pd.DataFrame(rows)


def jump_nonnormality() -> pd.DataFrame:
    """||[L_k, L_k^dagger]||_F of the fitted jump operators -- the quantity the
    paper already cites as '0.5--1.4 across the fitted E1a seeds' for v1. v2's
    Hermitian jumps must give exactly 0."""
    rows = []
    for variant, tmpl, getter in (
        ("ltn", "runs/E1a/ltn/{s}/model.pt", v1_jump_ops),
        ("ltn_noH", "runs/E2-e1a/ltn_noH/{s}/model.pt", v1_jump_ops),
        ("ltn_v2", "runs/E1a/ltn_v2/{s}/model.pt", v2_jump_ops),
    ):
        for s in range(5):
            p = REPO / tmpl.format(s=s)
            if not p.exists():
                continue
            Ls = getter(torch.load(p, map_location="cpu", weights_only=False)["state_dict"])
            c = [float(np.linalg.norm(L @ L.conj().T - L.conj().T @ L, "fro")) for L in Ls]
            rows.append({"variant": variant, "seed": s,
                         "comm_LLdag_max": max(c), "comm_LLdag_min": min(c),
                         **{f"comm_LLdag_k{k}": v for k, v in enumerate(c)}})
    return pd.DataFrame(rows)


def mode_link() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank-correlate every spectral diagnostic with the per-seed NCRPS.

    n = 5 seeds per variant, so no correlation here can be significant; the point
    is the SIGN and the rank of the diverging seed. If the mechanism were real,
    the diverging seed would sit at the non-normal extreme of these measures.
    """
    from scipy.stats import spearmanr
    summ = pd.read_csv(OUT / "spectra_summary.csv")
    nc = per_seed_ncrps()
    m = summ.merge(nc, on=["variant", "seed"])
    m = m.merge(jump_nonnormality(), on=["variant", "seed"], how="left")
    m.to_csv(OUT / "seed_link.csv", index=False)

    measures = ["departure_mean", "departure_max", "departure_rel_mean",
                "kappaV_median", "kappaV_max", "num_abscissa_max", "H_fro_mean",
                "H_fro_median", "comm_max", "comm_rel_max", "arg_fro_mean",
                "departure_arg_max", "gap_median", "gamma_max", "comm_LLdag_max",
                "fro_median", "fro_ratio_H_median"]
    rows = []
    for variant, g in m.groupby("variant"):
        g = g.sort_values("seed")
        worst = int(g.loc[g["ncrps"].idxmax(), "seed"])
        for meas in measures:
            v = g[meas]
            if v.isna().all() or v.nunique() <= 1:
                continue
            rho, p = spearmanr(g["ncrps"], v)
            # rank of the worst-NCRPS seed on this measure, 1 = most non-normal
            order = g.sort_values(meas, ascending=False)["seed"].tolist()
            rows.append({
                "variant": variant, "measure": meas, "worst_seed": worst,
                "spearman_rho": float(rho), "p": float(p),
                "worst_seed_rank_desc": order.index(worst) + 1, "n_seeds": len(g),
                "worst_seed_value": float(g.loc[g["seed"] == worst, meas].iloc[0]),
                "other_seeds_median": float(g.loc[g["seed"] != worst, meas].median()),
            })
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "seed_link_correlations.csv", index=False)
    return m, corr


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "kernel", "spectra", "grads", "link"])
    ap.add_argument("--split", default="e1a")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)

    need_data = args.mode in ("all", "spectra", "grads", "kernel")  # not "link"
    data = build_data_once(args.split) if need_data else None

    if args.mode in ("all", "kernel"):
        print("== kernel dimension over every fitted v2 checkpoint ==")
        kdf = mode_kernel()
        kdf.to_csv(OUT / "kernel_v2_checkpoints.csv", index=False)
        print(kdf.to_string(index=False))
        edf = mode_kernel_empirical(*data)
        edf.to_csv(OUT / "kernel_v2_empirical.csv", index=False)
        print(edf.to_string(index=False))

    if args.mode in ("all", "spectra"):
        print("== generator spectra across E1a seeds ==")
        df = mode_spectra(*data)
        df.to_csv(OUT / "spectra_raw.csv.gz", index=False, compression="gzip")
        s = summarise(df)
        s.to_csv(OUT / "spectra_summary.csv", index=False)
        print(s.to_string(index=False))

    if args.mode in ("all", "grads"):
        print("== gradient norms at the fitted parameters ==")
        gdf = mode_grads(*data)
        gdf.to_csv(OUT / "grad_norms.csv", index=False)
        print(gdf.to_string(index=False))

    if args.mode in ("all", "link"):
        print("== do the diagnostics track the per-seed outcome? ==")
        m, corr = mode_link()
        print(m[["variant", "seed", "ncrps", "departure_mean", "kappaV_median",
                 "H_fro_mean", "comm_max", "comm_LLdag_max"]].to_string(index=False))
        print(corr.to_string(index=False))

    (OUT / "PROVENANCE.json").write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": args.mode, "split": args.split, "rank_tol": RANK_TOL,
    }, indent=2))


if __name__ == "__main__":
    main()
