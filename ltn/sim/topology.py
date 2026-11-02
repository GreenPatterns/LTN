"""Coupling-map topologies for the synthetic device fleet [DEV-2].

Guarantee: ``make_topology(kind, **kw)`` returns ``(n_qubits, edges)`` where
``edges`` is a sorted list of undirected qubit pairs ``(u, v)`` with ``u < v``,
no duplicates, and every index in ``[0, n_qubits)``. Supported kinds:

* ``heavy_hex(distance)`` — IBM heavy-hex lattice via
  :meth:`qiskit.transpiler.CouplingMap.from_heavy_hex`.
* ``grid(rows, cols)`` — rectangular nearest-neighbour lattice.
* ``linear(n)`` — open chain.
* ``ring(n)`` — closed chain.

Deterministic: no randomness anywhere in this module.
"""
from __future__ import annotations

Edge = tuple[int, int]


def _dedup(edges) -> list[Edge]:
    seen = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges}
    return sorted(seen)


def heavy_hex(distance: int) -> tuple[int, list[Edge]]:
    """Heavy-hex lattice of the given (odd) code distance."""
    from qiskit.transpiler import CouplingMap

    cmap = CouplingMap.from_heavy_hex(int(distance), bidirectional=False)
    return int(cmap.size()), _dedup(cmap.get_edges())


def grid(rows: int, cols: int) -> tuple[int, list[Edge]]:
    """rows x cols nearest-neighbour grid; qubit index = r * cols + c."""
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1:
        raise ValueError(f"grid needs rows, cols >= 1, got ({rows}, {cols})")
    edges: list[Edge] = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                edges.append((i, i + 1))
            if r + 1 < rows:
                edges.append((i, i + cols))
    return rows * cols, _dedup(edges)


def linear(n: int) -> tuple[int, list[Edge]]:
    """Open chain of n qubits."""
    n = int(n)
    if n < 2:
        raise ValueError(f"linear needs n >= 2, got {n}")
    return n, [(i, i + 1) for i in range(n - 1)]


def ring(n: int) -> tuple[int, list[Edge]]:
    """Closed chain of n qubits."""
    n = int(n)
    if n < 3:
        raise ValueError(f"ring needs n >= 3, got {n}")
    return n, _dedup([(i, (i + 1) % n) for i in range(n)])


_KINDS = {"heavy_hex": heavy_hex, "grid": grid, "linear": linear, "ring": ring}


def make_topology(kind: str, **kw) -> tuple[int, list[Edge]]:
    """Dispatch on `kind` in {'heavy_hex', 'grid', 'linear', 'ring'}."""
    try:
        fn = _KINDS[kind]
    except KeyError:
        raise ValueError(f"unknown topology kind {kind!r}; choose from {sorted(_KINDS)}") from None
    return fn(**kw)
