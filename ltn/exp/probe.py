"""Roll a fitted LTN over its device sequences and export per-(device, entity, ts)
trust-state diagnostics — the substrate for E3 (divergence prediction) and E7
(event studies). No gradients, no training: this is a read-out of the state."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def extract_states(forecaster) -> pd.DataFrame:
    """forecaster: a fitted ltn.exp.train.LTNForecaster (or API-compatible variant).

    Returns rows [device, entity_id, ts, entropy, vacuity, dissonance, p_diverge]."""
    model = forecaster.model
    model.eval()
    dev = forecaster._torch_device
    rows = []
    for d, seq in forecaster.seqs.items():
        x, cal = seq.x.to(dev), seq.cal_age_h.to(dev)
        ei, ea = seq.edge_index.to(dev), seq.edge_attr.to(dev)
        st = model.init_state(len(seq.entity_ids), device=dev)
        for t in range(len(seq.ts)):
            st = model.step(st, x[t], cal[t], seq.dt_steps[t].expand(x.shape[1]).to(dev), ei, ea)
            p_div = model.divergence(st).cpu().numpy()
            ent = st.entropy.cpu().numpy()
            vac = st.vacuity.cpu().numpy()
            dis = st.dissonance.cpu().numpy()
            for i, e in enumerate(seq.entity_ids):
                rows.append((d, e, seq.ts[t], float(ent[i]), float(vac[i]), float(dis[i]), float(p_div[i])))
    return pd.DataFrame(rows, columns=["device", "entity_id", "ts", "entropy", "vacuity", "dissonance", "p_diverge"])
