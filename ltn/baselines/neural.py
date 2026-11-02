"""Neural baselines (LSTM / TGN-style / causal transformer) on the SAME device
sequences, split masks, evaluation pairs, and z-score normalization as LTN.

Each implements the Forecaster protocol via a shared trainer (mirrors
ltn.exp.train.LTNForecaster's BPTT + masked-target discipline). State is a plain
real vector per entity — these are the equal-capacity controls for H1:
  * LSTMBaseline        — per-entity recurrence, no graph;
  * TGNBaseline         — per-entity GRU memory + neighbor message passing (graph);
  * TransformerBaseline — per-entity causal self-attention over the history window.

Width is the capacity knob for param matching (ltn.models.param_match).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ltn.data.series import Panel
from ltn.exp.dataset import DeviceSequence, build_device_sequence

RD = torch.float64


def _nll(y, mu, sigma):
    return 0.5 * (((y - mu) / sigma) ** 2 + 2 * sigma.log() + math.log(2 * math.pi))


class _GaussHead(nn.Module):
    def __init__(self, state_dim: int, n_obs: int, hidden: int):
        super().__init__()
        self.n_obs = n_obs
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden, dtype=RD), nn.SiLU(),
            nn.Linear(hidden, 2 * n_obs, dtype=RD),
        )

    def forward(self, state, horizon):
        z = torch.cat([state, torch.log1p(horizon.to(RD)).unsqueeze(-1)], dim=-1)
        out = self.net(z)
        mu, raw = out[..., : self.n_obs], out[..., self.n_obs:]
        return mu, torch.nn.functional.softplus(raw) + 1e-4


class _LSTMCore(nn.Module):
    def __init__(self, in_dim, width, n_obs):
        super().__init__()
        self.cell = nn.LSTMCell(in_dim, width, dtype=RD)
        self.head = _GaussHead(width, n_obs, width)
        self.width = width

    def init_state(self, n, device):
        z = torch.zeros(n, self.width, dtype=RD, device=device)
        return (z, z.clone())

    def step(self, state, x, edge_index, edge_attr):
        h, c = self.cell(x, state)
        return (h, c)

    def readout(self, state):
        return state[0]


class _TGNCore(nn.Module):
    """GRU memory + mean-aggregated neighbor messages conditioned on edge features."""

    def __init__(self, in_dim, width, n_obs, edge_dim=2):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(width + edge_dim, width, dtype=RD), nn.SiLU())
        self.gru = nn.GRUCell(in_dim + width, width, dtype=RD)
        self.head = _GaussHead(width, n_obs, width)
        self.width = width

    def init_state(self, n, device):
        return torch.zeros(n, self.width, dtype=RD, device=device)

    def step(self, state, x, edge_index, edge_attr):
        n = state.shape[0]
        agg = torch.zeros_like(state)
        if edge_index.numel():
            src, dst = edge_index[0], edge_index[1]
            m = self.msg(torch.cat([state[src], edge_attr], dim=-1))
            agg.index_add_(0, dst, m)
            deg = torch.zeros(n, dtype=RD, device=state.device).index_add_(
                0, dst, torch.ones(len(dst), dtype=RD, device=state.device)
            ).clamp_min(1.0)
            agg = agg / deg.unsqueeze(-1)
        return self.gru(torch.cat([x, agg], dim=-1), state)

    def readout(self, state):
        return state


class _TransformerCore(nn.Module):
    """Causal self-attention over each entity's own history (window-limited)."""

    def __init__(self, in_dim, width, n_obs, n_layers=2, n_heads=4, window=64):
        super().__init__()
        self.inp = nn.Linear(in_dim, width, dtype=RD)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads, dim_feedforward=2 * width,
            batch_first=True, dtype=RD, dropout=0.0, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos = nn.Parameter(torch.zeros(window, width, dtype=RD))
        self.head = _GaussHead(width, n_obs, width)
        self.width, self.window = width, window

    def init_state(self, n, device):
        return []  # history buffer of x embeddings

    def step(self, state, x, edge_index, edge_attr):
        state = state + [self.inp(x)]
        return state[-self.window:]

    def readout(self, state):
        hist = torch.stack(state, dim=1)  # [n, t, w]
        t = hist.shape[1]
        h = hist + self.pos[:t].unsqueeze(0)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=h.device), diagonal=1)
        return self.enc(h, mask=mask)[:, -1]


_CORES = {"lstm": _LSTMCore, "tgn": _TGNCore, "transformer": _TransformerCore}


class NeuralForecaster:
    """cfg: same keys as LTNForecaster plus width; core fixed per subclass name."""

    core_name = "lstm"

    # build dims that must match between a checkpoint and its warm-started child,
    # or the state_dict shapes will not align (mirrors ltn.exp.train:92-101)
    BUILD_KEYS = ("width", "n_heads", "n_layers", "window")

    def __init__(self) -> None:
        self.model: nn.Module | None = None
        self.seqs: dict[str, DeviceSequence] = {}
        self._cfg_cache: dict = {}

    @property
    def name(self) -> str:
        return self.core_name

    def save(self, path) -> None:
        torch.save({"state_dict": self.model.state_dict(), "cfg": self._cfg_cache}, path)

    def load_weights(self, path, strict: bool = True) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["state_dict"], strict=strict)

    def _build(self, in_dim, n_obs, cfg):
        width = int(cfg.get("width", 64))
        if self.core_name == "tgn":
            cfg["width"] = width
            return _TGNCore(in_dim, width, n_obs)
        if self.core_name == "transformer":
            n_heads = int(cfg.get("n_heads", 4))
            width = n_heads * max(1, round(width / n_heads))  # embed_dim % n_heads == 0
            # persist the EFFECTIVE width so a warm start rebuilds the same shapes
            cfg["width"], cfg["n_heads"] = width, n_heads
            return _TransformerCore(in_dim, width, n_obs,
                                    n_layers=int(cfg.get("n_layers", 2)),
                                    n_heads=n_heads,
                                    window=int(cfg.get("window", 64)))
        cfg["width"] = width
        return _LSTMCore(in_dim, width, n_obs)

    def fit(self, panel: Panel, train_mask: np.ndarray, val_mask: np.ndarray, cfg: dict) -> None:
        torch.manual_seed(int(cfg.get("seed", 0)))
        uts, observables = cfg["uts"], cfg["observables"]
        horizons = list(cfg.get("horizons", [1, 3, 9, 21]))
        dev = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        devices = cfg.get("devices") or sorted(panel.keys["device"].unique())
        self.seqs = {
            d: build_device_sequence(uts, panel, d, observables, cfg["norm_stats"],
                                     time_unit_h=float(cfg.get("time_unit_h", 8.0)))
            for d in devices
        }
        n_obs = len(observables)
        in_dim = self.seqs[devices[0]].x.shape[-1]
        if cfg.get("init_from"):
            # adopt the checkpoint's architecture dims so state_dicts align across
            # the E1a/E1b observable spaces (LTN does the same at train.py:92-101)
            ck = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
            for k in self.BUILD_KEYS:
                if k in ck.get("cfg", {}) and k not in cfg:
                    cfg[k] = ck["cfg"][k]
        self.model = self._build(in_dim, n_obs, cfg).to(dev)
        if cfg.get("init_from"):  # warm start from a pretrained checkpoint (E4)
            self.load_weights(cfg["init_from"], strict=False)
            self.model.to(dev)
        self._cfg_cache = {k: v for k, v in cfg.items()
                           if k not in ("uts", "norm_stats", "observables")}
        self._cfg_cache["in_dim"], self._cfg_cache["n_obs"] = int(in_dim), int(n_obs)
        self._torch_device = dev
        opt = torch.optim.Adam(self.model.parameters(), lr=float(cfg.get("lr", 1e-3)))
        epochs = int(cfg.get("epochs", 60))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        bptt = int(cfg.get("bptt", 64))
        patience = int(cfg.get("patience", 8))
        ts_to_row = {t: i for i, t in enumerate(panel.ts)}
        best_val, best_state, bad = float("inf"), None, 0
        for _ in range(epochs):
            self.model.train()
            for d, seq in self.seqs.items():
                self._run_device(seq, train_mask, ts_to_row, horizons, opt, bptt, dev)
            sched.step()
            val = self._eval(val_mask, ts_to_row, horizons, dev)
            if val < best_val - 1e-6:
                best_val, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

    def _detach_state(self, state):
        if isinstance(state, tuple):
            return tuple(s.detach() for s in state)
        if isinstance(state, list):
            return [s.detach() for s in state]
        return state.detach()

    def _run_device(self, seq, t_mask, ts_to_row, horizons, opt, bptt, dev):
        model = self.model
        rows = np.array([ts_to_row[t] for t in seq.ts])
        x, y = seq.x.to(dev), seq.y.to(dev)
        ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
        state = model.init_state(x.shape[1], dev)
        opt.zero_grad()
        loss_acc, n_acc = None, 0
        T = len(seq.ts)
        n_obs = len(seq.observables)
        pretrain = bool(self._cfg_cache.get("pretrain", False))
        mask_frac = float(self._cfg_cache.get("mask_frac", 0.3))
        gen = torch.Generator(device="cpu").manual_seed(int(self._cfg_cache.get("seed", 0)))
        if pretrain:
            # masked-telemetry reconstruction — byte-for-byte the same objective the
            # LTN pretrains with (ltn/exp/train.py:153-182), so the E4 pretraining
            # stage is identical across model families.
            for t in range(T):
                x_in = x[t].clone()
                observed = x_in[:, n_obs:2 * n_obs] > 0.5
                drop = (torch.rand(observed.shape, generator=gen) < mask_frac).to(dev) & observed
                x_in[:, :n_obs] = torch.where(drop, torch.zeros_like(x_in[:, :n_obs]), x_in[:, :n_obs])
                x_in[:, n_obs:2 * n_obs] = torch.where(
                    drop, torch.zeros_like(drop, dtype=x_in.dtype), x_in[:, n_obs:2 * n_obs])
                state = model.step(state, x_in, ei, ea)
                if drop.any() and t_mask[rows[t]]:
                    mu, sigma = model.head(
                        model.readout(state),
                        torch.zeros(x.shape[1], dtype=RD, device=dev))
                    target = y[t]
                    sel = drop & ~torch.isnan(target)
                    if sel.any():
                        l = _nll(target[sel], mu[sel], sigma[sel]).mean()
                        loss_acc = l if loss_acc is None else loss_acc + l
                        n_acc += 1
                if (t + 1) % bptt == 0 or t == T - 1:
                    if loss_acc is not None:
                        (loss_acc / max(n_acc, 1)).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                    opt.zero_grad()
                    loss_acc, n_acc = None, 0
                    state = self._detach_state(state)
            return
        for t in range(T):
            state = model.step(state, x[t], ei, ea)
            ro = None
            for h in horizons:
                if t + h >= T or not t_mask[rows[t + h]]:
                    continue
                target = y[t + h]
                obs = ~torch.isnan(target)
                if not obs.any():
                    continue
                ro = model.readout(state) if ro is None else ro
                mu, sigma = model.head(ro, torch.full((x.shape[1],), float(h), dtype=RD, device=dev))
                l = _nll(target[obs], mu[obs], sigma[obs]).mean()
                loss_acc = l if loss_acc is None else loss_acc + l
                n_acc += 1
            if (t + 1) % bptt == 0 or t == T - 1:
                if loss_acc is not None:
                    (loss_acc / max(n_acc, 1)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad()
                loss_acc, n_acc = None, 0
                state = self._detach_state(state)

    @torch.no_grad()
    def _eval(self, t_mask, ts_to_row, horizons, dev) -> float:
        from ltn.eval.metrics import crps_gaussian
        model = self.model
        ys, mus, sigmas = [], [], []
        for d, seq in self.seqs.items():
            rows = np.array([ts_to_row[t] for t in seq.ts])
            x, y = seq.x.to(dev), seq.y
            ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
            state = model.init_state(x.shape[1], dev)
            T = len(seq.ts)
            for t in range(T):
                state = model.step(state, x[t], ei, ea)
                ro = None
                for h in horizons:
                    if t + h >= T or not t_mask[rows[t + h]]:
                        continue
                    target = y[t + h]
                    obs = ~torch.isnan(target)
                    if not obs.any():
                        continue
                    ro = model.readout(state) if ro is None else ro
                    mu, sigma = model.head(ro, torch.full((x.shape[1],), float(h), dtype=RD, device=dev))
                    ys.append(target[obs].numpy())
                    mus.append(mu[obs].cpu().numpy())
                    sigmas.append(sigma[obs].cpu().numpy())
        if not ys:
            return float("inf")
        return float(crps_gaussian(np.concatenate(ys), np.concatenate(mus), np.concatenate(sigmas)))

    @torch.no_grad()
    def predict(self, pairs: pd.DataFrame, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
        dev = self._torch_device
        model = self.model
        col_info = {}
        for d, seq in self.seqs.items():
            for (ent, lbl), gj in seq.panel_col.items():
                col_info[gj] = (d, ent, lbl)
        readouts: dict[str, list] = {}
        for d, seq in self.seqs.items():
            x = seq.x.to(dev)
            ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
            st = model.init_state(x.shape[1], dev)
            per_t = []
            for t in range(len(seq.ts)):
                st = model.step(st, x[t], ei, ea)
                per_t.append(model.readout(st))
            readouts[d] = per_t
        loc = {
            d: {int(np.nonzero(panel.ts == t)[0][0]): i for i, t in enumerate(seq.ts)}
            for d, seq in self.seqs.items()
        }
        s_idx = pairs["s_idx"].to_numpy()
        dev_arr = np.array([col_info[j][0] for j in s_idx])
        mu_out, sig_out = np.empty(len(pairs)), np.empty(len(pairs))
        for d, seq in self.seqs.items():
            rows_d = np.nonzero(dev_arr == d)[0]
            if not len(rows_d):
                continue
            e_index = {e: i for i, e in enumerate(seq.entity_ids)}
            f_index = {l: i for i, l in enumerate(seq.observables)}
            ne = np.array([e_index[col_info[j][1]] for j in s_idx[rows_d]])
            fi = np.array([f_index[col_info[j][2]] for j in s_idx[rows_d]])
            ms = np.array([seq.norm[col_info[j][2]] for j in s_idx[rows_d]])
            t_loc = np.array([loc[d][int(t)] for t in pairs["t_idx"].to_numpy()[rows_d]])
            hor = pairs["horizon"].to_numpy()[rows_d].astype(int)
            order = np.argsort(t_loc * 10000 + hor, kind="stable")
            key_sorted = (t_loc * 10000 + hor)[order]
            bounds = np.nonzero(np.diff(key_sorted))[0] + 1
            for chunk in np.split(order, bounds):
                tl, h = int(t_loc[chunk[0]]), int(hor[chunk[0]])
                mu, sigma = model.head(
                    readouts[d][tl],
                    torch.full((len(seq.entity_ids),), float(h), dtype=RD, device=dev))
                mu_np, sig_np = mu.cpu().numpy(), sigma.cpu().numpy()
                sel = rows_d[chunk]
                mu_out[sel] = mu_np[ne[chunk], fi[chunk]] * ms[chunk, 1] + ms[chunk, 0]
                sig_out[sel] = np.maximum(sig_np[ne[chunk], fi[chunk]] * ms[chunk, 1], 1e-12)
        return mu_out, sig_out

    def n_params(self) -> int:
        from ltn.models.param_match import count_trainable
        return count_trainable(self.model)


class LSTMBaseline(NeuralForecaster):
    core_name = "lstm"


class TGNBaseline(NeuralForecaster):
    core_name = "tgn"


class TransformerBaseline(NeuralForecaster):
    core_name = "transformer"
