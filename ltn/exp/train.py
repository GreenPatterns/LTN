"""LTN training + the Forecaster adapter (same evaluation pairs as every baseline).

Training: truncated BPTT over each device timeline; loss = Gaussian NLL of z-scored
targets at the config horizons, evaluated at every train-region origin whose target is
also in the train region. Early stopping on val CRPS (z-space). Optimizer per §2.2:
Adam 1e-3 cosine, clip 1.0.

Fairness invariants:
  * identical (t, series, horizon) pairs as baselines at predict time, raw units;
  * norm stats come in via cfg (train-only, computed by the split builder);
  * no forward-time information: sequences are rolled causally, forecasts are
    evidence-free rolls from the origin state.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from ltn.data.series import Panel
from ltn.exp.dataset import DeviceSequence, build_device_sequence
from ltn.models.ltn import LTNModel


def _nll(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return 0.5 * (((y - mu) / sigma) ** 2 + 2 * sigma.log() + math.log(2 * math.pi))


def _detach_state(state):
    """BPTT truncation for any dataclass state (LTNStep, GaussStep, ...)."""
    import dataclasses
    return dataclasses.replace(state, **{
        f.name: getattr(state, f.name).detach()
        for f in dataclasses.fields(state)
        if torch.is_tensor(getattr(state, f.name))
    })


class LTNForecaster:
    """cfg keys: uts (DataFrame), observables [(obs, channel)...], norm_stats,
    horizons [steps], devices [names] | None=all, model overrides (d, carry_dim,
    hidden, use_graph, use_lindblad, physics_gamma), epochs, lr, bptt, patience,
    device ('cuda:0'|'cpu'), seed."""

    name = "ltn"

    def __init__(self) -> None:
        self.model: LTNModel | None = None
        self.seqs: dict[str, DeviceSequence] = {}
        self._cfg_cache: dict = {}

    # hook for E2 variants (gaussian_state / node_ode): subclass and override
    def _make_model(self, n_obs: int, cfg: dict) -> LTNModel:
        return LTNModel(
            n_obs=n_obs,
            platform_dim=7,
            d=int(cfg.get("d", 4)),
            carry_dim=int(cfg.get("carry_dim", 32)),
            hidden=int(cfg.get("hidden", 64)),
            use_graph=bool(cfg.get("use_graph", True)),
            use_lindblad=bool(cfg.get("use_lindblad", True)),
            physics_gamma=bool(cfg.get("physics_gamma", True)),
            use_hamiltonian=bool(cfg.get("use_hamiltonian", True)),
            time_unit_h=float(cfg.get("time_unit_h", 8.0)),
        )

    def save(self, path) -> None:
        torch.save({"state_dict": self.model.state_dict(), "cfg": self._cfg_cache}, path)

    def load_weights(self, path, strict: bool = True) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["state_dict"], strict=strict)

    # ------------------------------------------------------------------ fit --
    def fit(self, panel: Panel, train_mask: np.ndarray, val_mask: np.ndarray, cfg: dict) -> None:
        torch.manual_seed(int(cfg.get("seed", 0)))
        uts = cfg["uts"]
        observables = cfg["observables"]
        horizons = list(cfg.get("horizons", [1, 3, 9, 21]))
        dev = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        devices = cfg.get("devices") or sorted(
            panel.keys["device"].unique()
        )
        self.seqs = {
            d: build_device_sequence(uts, panel, d, observables, cfg["norm_stats"],
                                     time_unit_h=float(cfg.get("time_unit_h", 8.0)))
            for d in devices
        }
        n_obs = len(observables)
        if cfg.get("init_from"):
            # adopt the checkpoint's architecture dims so state_dicts align
            ck = torch.load(cfg["init_from"], map_location="cpu", weights_only=False)
            # purity_eps is v2-only and is a plain float attribute, absent from
            # the state_dict, so a warm start would silently rebuild with the
            # default rather than the checkpoint's floor unless it is adopted here
            for k in ("d", "carry_dim", "hidden", "evidence_rank", "rounds",
                      "use_graph", "use_lindblad", "physics_gamma", "purity_eps"):
                if k in ck.get("cfg", {}) and k not in cfg:
                    cfg[k] = ck["cfg"][k]
        self._cfg_cache = {k: v for k, v in cfg.items() if k != "uts"}
        self.model = self._make_model(n_obs, cfg).to(dev)
        if cfg.get("init_from"):  # warm start from a pretrained/parent checkpoint (E4)
            self.load_weights(cfg["init_from"], strict=False)
            self.model.to(dev)
        self._torch_device = dev
        # epochs=0 + init_from = pure zero-shot evaluation of the loaded checkpoint
        # (the loop below simply doesn't execute; loaded weights are kept)
        opt = torch.optim.Adam(self.model.parameters(), lr=float(cfg.get("lr", 1e-3)))
        epochs = int(cfg.get("epochs", 60))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        bptt = int(cfg.get("bptt", 64))
        patience = int(cfg.get("patience", 8))

        # global-ts -> panel row map for split masks on device-local grids
        ts_to_row = {t: i for i, t in enumerate(panel.ts)}
        best_val, best_state, bad = float("inf"), None, 0
        for epoch in range(epochs):
            self.model.train()
            total = 0.0
            for d, seq in self.seqs.items():
                total += self._run_device(seq, train_mask, ts_to_row, horizons, opt, bptt, dev)
            sched.step()
            val = self._eval_masked(val_mask, ts_to_row, horizons, dev)
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

    def _run_device(self, seq, t_mask, ts_to_row, horizons, opt, bptt, dev) -> float:
        model = self.model
        T = len(seq.ts)
        n_obs = len(seq.observables)
        rows = np.array([ts_to_row[t] for t in seq.ts])
        x = seq.x.to(dev)
        cal = seq.cal_age_h.to(dev)
        y = seq.y.to(dev)
        ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
        state = model.init_state(len(seq.entity_ids), device=dev)
        total = 0.0
        opt.zero_grad()
        loss_acc, n_acc = None, 0
        pretrain = bool(self._cfg_cache.get("pretrain", False))
        mask_frac = float(self._cfg_cache.get("mask_frac", 0.3))
        gen = torch.Generator(device="cpu").manual_seed(int(self._cfg_cache.get("seed", 0)))
        for t in range(T):
            if pretrain:
                # masked-telemetry reconstruction (lambda3, §2.2 pretraining objective):
                # hide a random subset of observed values; the model must recover them
                # from state + graph at horizon 0 in z-space.
                x_in = x[t].clone()
                observed = x_in[:, n_obs:2 * n_obs] > 0.5
                drop = (torch.rand(observed.shape, generator=gen) < mask_frac).to(dev) & observed
                x_in[:, :n_obs] = torch.where(drop, torch.zeros_like(x_in[:, :n_obs]), x_in[:, :n_obs])
                x_in[:, n_obs:2 * n_obs] = torch.where(drop, torch.zeros_like(drop, dtype=x_in.dtype), x_in[:, n_obs:2 * n_obs])
                state = model.step(state, x_in, cal[t], seq.dt_steps[t].expand(x.shape[1]).to(dev), ei, ea)
                if drop.any() and t_mask[rows[t]]:
                    mu, sigma, _ = model.forecast(
                        state, x_in, cal[t], torch.zeros(x.shape[1], dtype=torch.float64, device=dev))
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
                        total += float(loss_acc.detach()) / max(n_acc, 1)
                    opt.zero_grad()
                    loss_acc, n_acc = None, 0
                    state = _detach_state(state)
                continue
            state = model.step(state, x[t], cal[t], seq.dt_steps[t].expand(x.shape[1]).to(dev), ei, ea)
            for h in horizons:
                if t + h >= T:
                    continue
                # train on pairs whose TARGET row is inside the mask (split owns target)
                if not t_mask[rows[t + h]]:
                    continue
                target = y[t + h]
                obs = ~torch.isnan(target)
                if not obs.any():
                    continue
                mu, sigma, _ = model.forecast(
                    state, x[t], cal[t], torch.full((x.shape[1],), float(h), dtype=torch.float64, device=dev)
                )
                l = _nll(target[obs], mu[obs], sigma[obs]).mean()
                loss_acc = l if loss_acc is None else loss_acc + l
                n_acc += 1
            if (t + 1) % bptt == 0 or t == T - 1:
                if loss_acc is not None:
                    (loss_acc / max(n_acc, 1)).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    total += float(loss_acc.detach()) / max(n_acc, 1)
                opt.zero_grad()
                loss_acc, n_acc = None, 0
                state = _detach_state(state)
        return total

    @torch.no_grad()
    def _eval_masked(self, t_mask, ts_to_row, horizons, dev) -> float:
        """Mean z-space CRPS over targets in the mask (early-stop criterion)."""
        from ltn.eval.metrics import crps_gaussian
        model = self.model
        model.eval()
        ys, mus, sigmas = [], [], []
        for d, seq in self.seqs.items():
            rows = np.array([ts_to_row[t] for t in seq.ts])
            x, cal, y = seq.x.to(dev), seq.cal_age_h.to(dev), seq.y
            ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
            state = model.init_state(len(seq.entity_ids), device=dev)
            T = len(seq.ts)
            for t in range(T):
                state = model.step(state, x[t], cal[t], seq.dt_steps[t].expand(x.shape[1]).to(dev), ei, ea)
                for h in horizons:
                    if t + h >= T or not t_mask[rows[t + h]]:
                        continue
                    target = y[t + h]
                    obs = ~torch.isnan(target)
                    if not obs.any():
                        continue
                    mu, sigma, _ = model.forecast(
                        state, x[t], cal[t], torch.full((x.shape[1],), float(h), dtype=torch.float64, device=dev)
                    )
                    ys.append(target[obs].numpy())
                    mus.append(mu[obs].cpu().numpy())
                    sigmas.append(sigma[obs].cpu().numpy())
        model.train()
        if not ys:
            return float("inf")
        return float(crps_gaussian(np.concatenate(ys), np.concatenate(mus), np.concatenate(sigmas)))

    # -------------------------------------------------------------- predict --
    @torch.no_grad()
    def predict(self, pairs: pd.DataFrame, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
        """Raw-unit (mu, sigma) for exactly the baseline pairs. Rolls each device
        timeline once, caches per-(device, t) states, reads out per pair."""
        dev = self._torch_device
        model = self.model
        model.eval()
        # panel col -> (device, entity, obs_label) and reverse into sequences
        col_info = {}
        for d, seq in self.seqs.items():
            for (ent, lbl), gj in seq.panel_col.items():
                col_info[gj] = (d, ent, lbl)
        # roll every device once, store states per grid position
        states: dict[str, list] = {}
        for d, seq in self.seqs.items():
            x, cal = seq.x.to(dev), seq.cal_age_h.to(dev)
            ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
            st = model.init_state(len(seq.entity_ids), device=dev)
            per_t = []
            for t in range(len(seq.ts)):
                st = model.step(st, x[t], cal[t], seq.dt_steps[t].expand(x.shape[1]).to(dev), ei, ea)
                per_t.append(st)
            states[d] = per_t
        # panel row index of each device-local grid position
        loc = {
            d: {int(np.nonzero(panel.ts == t)[0][0]): i for i, t in enumerate(seq.ts)}
            for d, seq in self.seqs.items()
        }
        # vectorized per-pair assembly: annotate rows, then one forecast per
        # (device, origin, horizon) group with fancy-indexed scatter
        s_idx = pairs["s_idx"].to_numpy()
        dev_arr = np.array([col_info[j][0] for j in s_idx])
        mu_out = np.empty(len(pairs))
        sig_out = np.empty(len(pairs))
        for d, seq in self.seqs.items():
            rows_d = np.nonzero(dev_arr == d)[0]
            if not len(rows_d):
                continue
            e_index = {e: i for i, e in enumerate(seq.entity_ids)}
            f_index = {l: i for i, l in enumerate(seq.observables)}
            ne = np.array([e_index[col_info[j][1]] for j in s_idx[rows_d]])
            fi = np.array([f_index[col_info[j][2]] for j in s_idx[rows_d]])
            ms = np.array([seq.norm[col_info[j][2]] for j in s_idx[rows_d]])  # [k, 2]
            t_loc = np.array([loc[d][int(t)] for t in pairs["t_idx"].to_numpy()[rows_d]])
            hor = pairs["horizon"].to_numpy()[rows_d].astype(int)
            order = np.argsort(t_loc * 10000 + hor, kind="stable")
            key_sorted = (t_loc * 10000 + hor)[order]
            bounds = np.nonzero(np.diff(key_sorted))[0] + 1
            for chunk in np.split(order, bounds):
                tl, h = int(t_loc[chunk[0]]), int(hor[chunk[0]])
                st = states[d][tl]
                mu, sigma, _ = model.forecast(
                    st, seq.x[tl].to(dev), seq.cal_age_h[tl].to(dev),
                    torch.full((len(seq.entity_ids),), float(h), dtype=torch.float64, device=dev),
                )
                mu_np, sig_np = mu.cpu().numpy(), sigma.cpu().numpy()
                sel = rows_d[chunk]
                mu_out[sel] = mu_np[ne[chunk], fi[chunk]] * ms[chunk, 1] + ms[chunk, 0]
                sig_out[sel] = np.maximum(sig_np[ne[chunk], fi[chunk]] * ms[chunk, 1], 1e-12)
        return mu_out, sig_out

    def n_params(self) -> int:
        from ltn.models.param_match import count_trainable
        return count_trainable(self.model)
