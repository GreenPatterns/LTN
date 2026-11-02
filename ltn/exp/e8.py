"""E8 (appendix, generality): DM-State + Lindblad decay in a DGTEN-style pipeline on
Bitcoin-OTC/Alpha vs an equal-parameter vector-state control in the SAME pipeline.

  python -m ltn.exp.e8 [--dataset both] [--seeds 5]

Task: edges are binned into 40 equal-count temporal windows; predict the scaled
weight in [0,1] of window-(w+1) edges from the endpoint node states at the end of
window w. Node state dynamics between a node's interactions:
  DM arm      rho (d=2) decays under the shared Lindblad dictionary with
              gamma = GammaNet([log1p(gap), degree_z]); rated edges enter through
              the CPTP EvidenceChannel; prediction head on POVM(rho_u), POVM(rho_v),
              carries.
  control arm identical wiring with rho replaced by a free 4-real vector updated by
              an MLP (no CPTP/Lindblad structure), width-matched within ±10%.
Cells: (dataset, test window) RMSE + sign-AUROC -> Wilcoxon-pairable across arms.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ltn.eval.metrics import auroc
from ltn.eval.runlog import RunLogger
from ltn.models.dmstate import CDTYPE, RDTYPE
from ltn.models.evidence import EvidenceChannel
from ltn.models.heads import POVMHeads
from ltn.models.lindblad import GammaNet, JumpDictionary, evolve
from ltn.models.param_match import count_trainable, match_width

N_WINDOWS = 40
MAX_NODES = 2000  # most-active nodes; declared subsampling


def load_edges(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["src", "dst", "rating", "ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    top = pd.concat([df["src"], df["dst"]]).value_counts().head(MAX_NODES).index
    df = df[df["src"].isin(top) & df["dst"].isin(top)].reset_index(drop=True)
    nodes = pd.concat([df["src"], df["dst"]]).unique()
    remap = {n: i for i, n in enumerate(nodes)}
    df["u"] = df["src"].map(remap)
    df["v"] = df["dst"].map(remap)
    df["w01"] = (df["rating"] + 10) / 20.0
    df["window"] = pd.qcut(df.index, N_WINDOWS, labels=False)
    return df


class DMArm(nn.Module):
    def __init__(self, d: int = 2, carry: int = 8, hidden: int = 24):
        super().__init__()
        self.d = d
        self.jumps = JumpDictionary(d)
        self.gamma = GammaNet(in_dim=2, K=3, hidden=hidden, cal_age_idx=0)
        self.evidence = EvidenceChannel(feat_dim=2, d=d, rank=2, hidden=hidden)
        self.povm = POVMHeads(d, n_facets=2)
        self.gru = nn.GRUCell(2, carry, dtype=RDTYPE)
        self.head = nn.Sequential(nn.Linear(2 * (2 + carry), hidden, dtype=RDTYPE), nn.SiLU(),
                                  nn.Linear(hidden, 1, dtype=RDTYPE))
        self.carry_dim = carry

    def init_states(self, n: int):
        rho = torch.eye(self.d, dtype=CDTYPE).expand(n, -1, -1).clone() / self.d
        return rho, torch.zeros(n, self.carry_dim, dtype=RDTYPE)

    def decay(self, rho, gaps_z, deg_z):
        feats = torch.stack([torch.log1p(gaps_z.clamp_min(0)), deg_z], -1)
        gammas = self.gamma(feats)
        H = torch.zeros(rho.shape[0], self.d, self.d, dtype=CDTYPE)
        return evolve(rho, H, list(self.jumps()), gammas, gaps_z.to(RDTYPE).clamp(0, 20))

    def absorb(self, rho, carry, rating01, direction):
        x = torch.stack([rating01 * 2 - 1, direction], -1).to(RDTYPE)
        rho = self.evidence(rho, x)
        carry = self.gru(x, carry)
        return rho, carry

    def edge_repr(self, rho, carry):
        return torch.cat([self.povm(rho).to(RDTYPE), carry], -1)

    def predict(self, ru, rv):
        return torch.sigmoid(self.head(torch.cat([ru, rv], -1))).squeeze(-1)


class VectorArm(nn.Module):
    def __init__(self, q: int = 4, carry: int = 8, hidden: int = 24):
        super().__init__()
        self.q = q
        self.decay_mlp = nn.Sequential(nn.Linear(q + 2, hidden, dtype=RDTYPE), nn.SiLU(),
                                       nn.Linear(hidden, q, dtype=RDTYPE))
        self.absorb_mlp = nn.Sequential(nn.Linear(q + 2, hidden, dtype=RDTYPE), nn.SiLU(),
                                        nn.Linear(hidden, q, dtype=RDTYPE))
        self.gru = nn.GRUCell(2, carry, dtype=RDTYPE)
        self.head = nn.Sequential(nn.Linear(2 * (q + carry), hidden, dtype=RDTYPE), nn.SiLU(),
                                  nn.Linear(hidden, 1, dtype=RDTYPE))
        self.carry_dim = carry

    def init_states(self, n: int):
        return torch.zeros(n, self.q, dtype=RDTYPE), torch.zeros(n, 8, dtype=RDTYPE)

    def decay(self, s, gaps_z, deg_z):
        z = torch.cat([s, torch.stack([torch.log1p(gaps_z.clamp_min(0)), deg_z], -1)], -1)
        return s + 0.1 * self.decay_mlp(z)

    def absorb(self, s, carry, rating01, direction):
        x = torch.stack([rating01 * 2 - 1, direction], -1).to(RDTYPE)
        s = s + 0.3 * self.absorb_mlp(torch.cat([s, x], -1))
        return s, self.gru(x, carry)

    def edge_repr(self, s, carry):
        return torch.cat([s, carry], -1)

    def predict(self, ru, rv):
        return torch.sigmoid(self.head(torch.cat([ru, rv], -1))).squeeze(-1)


def run_windows(model, df: pd.DataFrame, n_nodes: int, train_upto: int,
                opt=None) -> pd.DataFrame:
    """Sequential replay over windows; returns per-window test metrics."""
    state, carry = model.init_states(n_nodes)
    last_seen = torch.zeros(n_nodes, dtype=RDTYPE)
    deg = torch.zeros(n_nodes, dtype=RDTYPE)
    results = []
    for w in range(N_WINDOWS - 1):
        cur = df[df["window"] == w]
        nxt = df[df["window"] == w + 1]
        # predictions for next-window edges from current states
        u = torch.tensor(nxt["u"].to_numpy())
        v = torch.tensor(nxt["v"].to_numpy())
        y = torch.tensor(nxt["w01"].to_numpy(), dtype=RDTYPE)
        # decay states by the time since each endpoint's last interaction
        gap_u, gap_v = (w - last_seen[u]), (w - last_seen[v])
        deg_z = (deg - deg.mean()) / (deg.std() + 1e-6)
        su = model.decay(state[u], gap_u, deg_z[u])
        sv = model.decay(state[v], gap_v, deg_z[v])
        pred = model.predict(model.edge_repr(su, carry[u]), model.edge_repr(sv, carry[v]))
        loss = ((pred - y) ** 2).mean()
        if opt is not None and w < train_upto:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        rmse = float(loss.detach().sqrt())
        sign = (nxt["rating"] > 0).astype(int).to_numpy()
        auc = float(auroc(sign, pred.detach().numpy())) if 0 < sign.mean() < 1 else np.nan
        results.append({"window": w + 1, "rmse": rmse, "auroc": auc,
                        "phase": "train" if w < train_upto else "test", "n": len(nxt)})
        # absorb current-window edges into states (detached — replay, not BPTT)
        with torch.no_grad():
            for _, r in cur.iterrows():
                uu, vv = int(r["u"]), int(r["v"])
                sv2, cv2 = model.absorb(state[vv:vv + 1], carry[vv:vv + 1],
                                        torch.tensor([r["w01"]], dtype=RDTYPE),
                                        torch.tensor([1.0], dtype=RDTYPE))
                state[vv], carry[vv] = sv2[0], cv2[0]
                su2, cu2 = model.absorb(state[uu:uu + 1], carry[uu:uu + 1],
                                        torch.tensor([r["w01"]], dtype=RDTYPE),
                                        torch.tensor([-1.0], dtype=RDTYPE))
                state[uu], carry[uu] = su2[0], cu2[0]
                last_seen[uu] = last_seen[vv] = w
                deg[uu] += 1
                deg[vv] += 1
    return pd.DataFrame(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["otc", "alpha", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    paths = {"otc": "data/raw/snap/soc-sign-bitcoinotc.csv.gz",
             "alpha": "data/raw/snap/soc-sign-bitcoinalpha.csv.gz"}
    names = ["otc", "alpha"] if args.dataset == "both" else [args.dataset]
    train_upto = int(N_WINDOWS * 0.8) - 1
    for ds in names:
        df = load_edges(paths[ds])
        n_nodes = int(max(df["u"].max(), df["v"].max())) + 1
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            dm = DMArm()
            target = count_trainable(dm)
            ctrl, _, _ = match_width(target, lambda h: VectorArm(hidden=h), lo=4, hi=512)
            for arm_name, model in (("ltn_dm", dm), ("vector_gru", ctrl)):
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                for ep in range(args.epochs):
                    res = run_windows(model, df, n_nodes, train_upto, opt)
                te = res[res["phase"] == "test"]
                with RunLogger("E8", f"{arm_name}-{ds}", seed,
                               {"dataset": ds, "n_params": count_trainable(model)}) as log:
                    for _, r in te.iterrows():
                        log.log(**r.to_dict())
                print(f"[e8:{ds}] seed{seed} {arm_name}: test RMSE={te['rmse'].mean():.4f} "
                      f"AUROC={te['auroc'].mean():.3f}")


if __name__ == "__main__":
    main()
