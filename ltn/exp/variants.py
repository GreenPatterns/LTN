"""E2 ablation-variant forecasters, driven by the unchanged LTNForecaster trainer.

Equal-capacity discipline (prereg): each variant's free width knob is binary-searched
so trainable parameters land within ±10% of the LTN model built from the SAME config.
"""
from __future__ import annotations

from ltn.exp.train import LTNForecaster
from ltn.models.gaussian_state import GaussianModel
from ltn.models.ltn import LTNModel
from ltn.models.ltn_v2 import LTNv2Model
from ltn.models.node_ode import NodeODEModel
from ltn.models.param_match import count_trainable, match_width


def _ltn_target(n_obs: int, cfg: dict) -> int:
    ref = LTNModel(
        n_obs=n_obs, platform_dim=7, d=int(cfg.get("d", 4)),
        carry_dim=int(cfg.get("carry_dim", 32)), hidden=int(cfg.get("hidden", 64)),
        time_unit_h=float(cfg.get("time_unit_h", 8.0)),
    )
    return count_trainable(ref)


class GaussianForecaster(LTNForecaster):
    name = "ltn_gaussian"

    def _make_model(self, n_obs: int, cfg: dict):
        target = _ltn_target(n_obs, cfg)

        def builder(h: int):
            return GaussianModel(
                n_obs=n_obs, platform_dim=7, d=int(cfg.get("d", 4)),
                carry_dim=int(cfg.get("carry_dim", 32)), hidden=h,
                use_graph=bool(cfg.get("use_graph", True)),
                physics_gamma=bool(cfg.get("physics_gamma", True)),
                time_unit_h=float(cfg.get("time_unit_h", 8.0)),
            )

        model, _, _ = match_width(target, builder, lo=8, hi=4096)
        return model


class NodeODEForecaster(LTNForecaster):
    name = "ltn_ode"

    def _make_model(self, n_obs: int, cfg: dict):
        target = _ltn_target(n_obs, cfg)

        def builder(h: int):
            return NodeODEModel(
                n_obs=n_obs, platform_dim=7, d=int(cfg.get("d", 4)),
                carry_dim=int(cfg.get("carry_dim", 32)), hidden=int(cfg.get("hidden", 64)),
                use_graph=bool(cfg.get("use_graph", True)),
                physics_gamma=bool(cfg.get("physics_gamma", True)),
                time_unit_h=float(cfg.get("time_unit_h", 8.0)),
                ode_hidden=h,
            )

        model, _, _ = match_width(target, builder, lo=4, hi=2048)
        return model


class LTNv2Forecaster(LTNForecaster):
    """Tier-1 v2 (ltn_v2): Born readout, unital dissipative-only saturating
    drift, observable-aligned d = n_obs + 1 (overridable via cfg d for the
    pre-registered d-sweep falsifier).

    NOT width-matched to v1, by design: v2 is a competing model, not an E2
    ablation of v1, so it follows the paper's baseline protocol -- the same
    50-trial Optuna budget over the same search space, with width a tuned
    dimension -- rather than the equal-capacity discipline above. Its tuned
    parameter counts must be reported alongside v1's like any other model.

    evidence_rank/rounds are forwarded here (v1 silently drops them), but they
    are absent from ltn_v2's tuning search space, so both models sit at the same
    defaults (4, 2); see the comment in ltn/exp/tune.py:search_space.
    """

    name = "ltn_v2"

    def _make_model(self, n_obs: int, cfg: dict):
        return LTNv2Model(
            n_obs=n_obs, platform_dim=7,
            d=int(cfg["d"]) if cfg.get("d") else None,
            carry_dim=int(cfg.get("carry_dim", 32)),
            hidden=int(cfg.get("hidden", 64)),
            evidence_rank=int(cfg.get("evidence_rank", 4)),
            rounds=int(cfg.get("rounds", 2)),
            use_graph=bool(cfg.get("use_graph", True)),
            use_lindblad=bool(cfg.get("use_lindblad", True)),
            physics_gamma=bool(cfg.get("physics_gamma", True)),
            time_unit_h=float(cfg.get("time_unit_h", 8.0)),
            purity_eps=float(cfg.get("purity_eps", 1e-3)),
        )


VARIANT_MODELS = {
    "ltn_gaussian": GaussianForecaster,
    "ltn_ode": NodeODEForecaster,
    "ltn_v2": LTNv2Forecaster,
}
