"""Neural-ODE variant (the -Lindblad ablation, E2): identical to LTNModel except the
exact CPTP superoperator exponential is replaced by an unconstrained neural ODE on
the real-vectorized state, RK4-integrated, followed by projection back onto valid
density matrices (hermitize, clamp eigenvalues, renormalize). Matched capacity, no
physical structure in the dynamics — isolates the value of the Lindblad form."""
from __future__ import annotations

import torch
import torch.nn as nn

from ltn.models.dmstate import CDTYPE, RDTYPE
from ltn.models.heads import rho_to_real
from ltn.models.ltn import LTNModel, LTNStep
from ltn.models.dmstate import dissonance, vacuity, von_neumann_entropy


def _project_to_state(rho: torch.Tensor) -> torch.Tensor:
    rho = (rho + rho.conj().transpose(-2, -1)) / 2
    evals, vecs = torch.linalg.eigh(rho)
    evals = evals.clamp_min(1e-10)
    rho = vecs @ torch.diag_embed(evals.to(CDTYPE)) @ vecs.conj().transpose(-2, -1)
    tr = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real
    return rho / tr.to(CDTYPE).view(-1, 1, 1)


def _real_to_rho(z: torch.Tensor, d: int) -> torch.Tensor:
    """Inverse of heads.rho_to_real (diag + scaled upper re/im)."""
    b = z.shape[0]
    iu = torch.triu_indices(d, d, offset=1, device=z.device)
    n_off = iu.shape[1]
    diag = z[:, :d].to(CDTYPE)
    re = z[:, d:d + n_off] / 2 ** 0.5
    im = z[:, d + n_off:d + 2 * n_off] / 2 ** 0.5
    rho = torch.zeros(b, d, d, dtype=CDTYPE, device=z.device)
    rho[:, torch.arange(d), torch.arange(d)] = diag
    rho[:, iu[0], iu[1]] = torch.complex(re, im)
    rho[:, iu[1], iu[0]] = torch.complex(re, -im)
    return rho


class NodeODEModel(LTNModel):
    def __init__(self, *args, ode_hidden: int = 32, **kw) -> None:
        super().__init__(*args, **kw)
        # the Lindblad machinery is replaced by the ODE — drop its (unused)
        # parameters so capacity accounting compares like for like
        self.gamma_net = None
        self.jumps = None
        self.hermitian = None
        d = self.d
        gdim = 4 + self.platform_dim
        hidden = ode_hidden
        self.ode_f = nn.Sequential(
            nn.Linear(d * d + gdim, hidden, dtype=RDTYPE), nn.SiLU(),
            nn.Linear(hidden, hidden, dtype=RDTYPE), nn.SiLU(),
            nn.Linear(hidden, d * d, dtype=RDTYPE),
        )
        with torch.no_grad():  # start near the identity flow, like exp(L*0)
            self.ode_f[-1].weight.mul_(1e-2)
            self.ode_f[-1].bias.mul_(0.0)

    def _ode_evolve(self, rho: torch.Tensor, gfeat: torch.Tensor, dt_steps: torch.Tensor) -> torch.Tensor:
        z = rho_to_real(rho)
        remaining = dt_steps.to(RDTYPE).clamp_min(0)
        n_sub = int(min(32, max(4, float(remaining.max().item()) * 4))) if remaining.numel() else 4
        h = (remaining / n_sub).unsqueeze(-1)

        def f(zz):
            return self.ode_f(torch.cat([zz, gfeat], dim=-1))

        for _ in range(n_sub):
            k1 = f(z)
            k2 = f(z + 0.5 * h * k1)
            k3 = f(z + 0.5 * h * k2)
            k4 = f(z + h * k3)
            z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return _project_to_state(_real_to_rho(z, self.d))

    def drift(self, state: LTNStep, x_prev, cal_age, dt_steps) -> torch.Tensor:
        gfeat = self._gamma_features(x_prev, cal_age, dt_steps)
        return self._ode_evolve(state.rho, gfeat, dt_steps)

    def forecast(self, state: LTNStep, x_now, cal_age_now, horizon_steps):
        gfeat = self._gamma_features(x_now, cal_age_now + horizon_steps * self.time_unit_h, horizon_steps)
        rho_h = self._ode_evolve(state.rho, gfeat, horizon_steps)
        mu, sigma = self.forecast_head(rho_h, state.carry, horizon_steps)
        return mu, sigma, von_neumann_entropy(rho_h)
