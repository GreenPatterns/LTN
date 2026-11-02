"""Equal-capacity accounting (prereg-binding): trainable parameter counts of every
model variant matched to LTN's within ±10% by width search. Per-node state tensors
are excluded symmetrically for all stateful models (LTN rho/theta, LSTM/TGN hidden)."""
from __future__ import annotations

from typing import Callable

import torch.nn as nn


def count_trainable(model: nn.Module, exclude_substrings: tuple[str, ...] = ()) -> int:
    n = 0
    for name, p in model.named_parameters():
        if p.requires_grad and not any(s in name for s in exclude_substrings):
            n += p.numel()
    return n


def match_width(
    target_n: int,
    builder: Callable[[int], nn.Module],
    lo: int = 4,
    hi: int = 4096,
    tol: float = 0.10,
    exclude_substrings: tuple[str, ...] = (),
) -> tuple[nn.Module, int, int]:
    """Binary-search the width w so count_trainable(builder(w)) is within tol of target.

    Returns (model, width, n_params). Raises if unreachable within [lo, hi]."""
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_trainable(builder(mid), exclude_substrings)
        if best is None or abs(n - target_n) < abs(best[2] - target_n):
            best = (mid, None, n)
        if n < target_n:
            lo = mid + 1
        else:
            hi = mid - 1
    w, _, n = best
    if abs(n - target_n) > tol * target_n:
        raise ValueError(
            f"cannot match {target_n} params within ±{tol:.0%}: closest width {w} gives {n}"
        )
    return builder(w), w, n
