"""E6 regret replay worlds (H3 operational value).

Pre-registered job-success model (SimWorld):
  p_success(S, t) = prod_{q in S} [(1 - readout_q(t)) * (1 - epc1q_q(t))^G1]
                  * prod_{(u,v) in E(S)} (1 - gate2q_uv(t))^G2
with G1 = 10 single-qubit and G2 = 5 two-qubit gate layers. Reward = p_success;
regret(t) = best candidate reward - chosen reward. SimWorld evaluates on the TRUTH
channel (model-invisible latents); policies see only claimed values or model
forecasts. LoggedWorld (E6b, secondary [DEV-5]) replays DAQEC logged arms with
delivered logical error rates (reward = 1 - LER), partial feedback.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

G1_LAYERS = 10
G2_LAYERS = 5


def connected_candidates(edges: list[tuple[int, int]], width: int, k: int = 20,
                         seed: int = 0) -> list[frozenset[int]]:
    """K frozen connected qubit subsets of the given width (BFS from random roots)."""
    rng = np.random.default_rng([seed, width])
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    nodes = sorted(adj)
    out, seen = [], set()
    attempts = 0
    while len(out) < k and attempts < 50 * k:
        attempts += 1
        root = int(rng.choice(nodes))
        subset, frontier = {root}, [root]
        while frontier and len(subset) < width:
            cur = frontier.pop(0)
            nbrs = sorted(adj[cur] - subset)
            rng.shuffle(nbrs)
            for nb in nbrs:
                if len(subset) >= width:
                    break
                subset.add(nb)
                frontier.append(nb)
        fs = frozenset(subset)
        if len(fs) == width and fs not in seen:
            seen.add(fs)
            out.append(fs)
    return out


def success_prob(subset: frozenset[int], edges: list[tuple[int, int]],
                 readout: dict[int, float], epc1q: dict[int, float],
                 gate2q: dict[tuple[int, int], float]) -> float:
    p = 1.0
    for q in subset:
        p *= (1.0 - min(readout.get(q, 0.05), 0.999))
        p *= (1.0 - min(epc1q.get(q, 0.001), 0.999)) ** G1_LAYERS
    for (u, v) in edges:
        if u in subset and v in subset:
            p *= (1.0 - min(gate2q.get((u, v), gate2q.get((v, u), 0.01)), 0.999)) ** G2_LAYERS
    return float(p)


@dataclass
class Arm:
    device: str
    subset: frozenset
    features: dict          # whatever the policy scores on (claimed / predicted values)
    truth_reward: float     # world-side outcome


def replay(arms_by_t: dict, policy, seed: int = 0) -> pd.DataFrame:
    """arms_by_t: {t -> [Arm...]}; policy(arms, rng) -> chosen index."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in sorted(arms_by_t):
        arms = arms_by_t[t]
        if not arms:
            continue
        j = policy(arms, rng)
        rewards = np.array([a.truth_reward for a in arms])
        best = float(rewards.max())
        rows.append({"t": t, "chosen": int(j), "reward": float(rewards[j]),
                     "best_reward": best, "regret": best - float(rewards[j])})
    df = pd.DataFrame(rows)
    df["cum_regret"] = df["regret"].cumsum()
    return df


# ----------------------------------------------------------------------------- policies
def policy_random(arms, rng):
    return int(rng.integers(len(arms)))


def policy_oracle(arms, rng):
    return int(np.argmax([a.truth_reward for a in arms]))


def _score_from_errors(a: Arm, key: str, kappa: float = 0.0) -> float:
    """Score an arm from (mu, sigma) error estimates: pessimism = mu + kappa*sigma."""
    f = a.features
    readout = {q: f[key]["readout_error"][q][0] + kappa * f[key]["readout_error"][q][1]
               for q in a.subset if q in f[key]["readout_error"]}
    epc = {q: f[key]["epc_1q"][q][0] + kappa * f[key]["epc_1q"][q][1]
           for q in a.subset if q in f[key]["epc_1q"]}
    g2 = {e: v[0] + kappa * v[1] for e, v in f[key]["gate_error_2q"].items()}
    return success_prob(a.subset, list(g2.keys()), readout, epc, g2)


def policy_greedy_claimed(arms, rng):
    return int(np.argmax([_score_from_errors(a, "claimed", 0.0) for a in arms]))


def policy_ltn_mean(arms, rng):
    return int(np.argmax([_score_from_errors(a, "pred", 0.0) for a in arms]))


def make_policy_ltn_lcb(kappa: float):
    def policy_ltn_lcb(arms, rng):
        return int(np.argmax([_score_from_errors(a, "pred", kappa) for a in arms]))
    return policy_ltn_lcb


POLICIES = {
    "random": policy_random,
    "greedy_claimed": policy_greedy_claimed,
    "ltn_mean": policy_ltn_mean,
    "oracle": policy_oracle,
}
