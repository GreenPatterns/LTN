"""Tests for the generator-spectrum diagnostics (lane A7-spectra).

These pin the MATH of scripts/spectra.py against cases with known closed-form
answers, so the numbers reported in paper/fixnotes/A7-spectra.md are auditable:

  * departure from normality is 0 exactly on normal matrices and positive on a
    Jordan block, where its value is known;
  * kappa(V) = 1 on normal matrices, and diverges on a near-defective matrix;
  * numerical abscissa >= spectral abscissa always, with equality iff normal;
  * the central claim of the fixnote -- for HERMITIAN jump operators the Lindblad
    kernel equals the commutant of the jumps and is therefore independent of the
    (positive) rates -- is verified on cases whose commutant dimension is known
    analytically: the full Pauli set (dim 1, primitive) vs a purely diagonal
    dephasing set (dim d, NOT primitive, which is exactly the counterexample
    raised against Proposition 3).
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from ltn.models.dmstate import CDTYPE
from ltn.models.lindblad import lindblad_generator

_spec = importlib.util.spec_from_file_location(
    "spectra", Path(__file__).resolve().parents[1] / "scripts" / "spectra.py")
spectra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spectra)


# ---------------------------------------------------------------- primitives --
def test_departure_zero_for_normal():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    herm = (A + A.conj().T) / 2          # Hermitian => normal
    unit = np.linalg.qr(A)[0]            # unitary => normal
    assert spectra.departure_from_normality(herm) == pytest.approx(0.0, abs=1e-9)
    assert spectra.departure_from_normality(unit) == pytest.approx(0.0, abs=1e-9)


def test_departure_jordan_block_known_value():
    # J = [[0,1],[0,0]]: ||J||_F^2 = 1, all eigenvalues 0 => dep = 1 exactly.
    J = np.array([[0.0, 1.0], [0.0, 0.0]])
    assert spectra.departure_from_normality(J) == pytest.approx(1.0, abs=1e-12)
    # scaling by c scales the departure by |c|
    assert spectra.departure_from_normality(7.5 * J) == pytest.approx(7.5, abs=1e-10)


def test_kappa_V_one_for_normal_and_large_for_near_defective():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    herm = (A + A.T) / 2
    assert spectra.eigvec_condition(herm) == pytest.approx(1.0, rel=1e-8)
    near_defective = np.array([[1.0, 1e6], [0.0, 1.0 + 1e-9]])
    assert spectra.eigvec_condition(near_defective) > 1e9


def test_numerical_abscissa_dominates_spectral_abscissa():
    rng = np.random.default_rng(2)
    for _ in range(20):
        A = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
        num = spectra.numerical_abscissa(A)
        spec = float(np.max(np.linalg.eigvals(A).real))
        assert num >= spec - 1e-9
    # equality for a normal (Hermitian) matrix
    herm = np.diag([-3.0, -1.0, 0.0])
    assert spectra.numerical_abscissa(herm) == pytest.approx(0.0, abs=1e-12)


def test_spectral_stats_on_a_normal_generator():
    A = np.diag([0.0, -1.0, -2.5, -2.5])
    st = spectra.spectral_stats(A)
    assert st["abscissa"] == pytest.approx(0.0, abs=1e-12)
    assert st["gap"] == pytest.approx(1.0, abs=1e-9)
    assert st["n_zero_eig"] == 1
    assert st["departure"] == pytest.approx(0.0, abs=1e-9)
    assert st["kappa_V"] == pytest.approx(1.0, rel=1e-8)


# ------------------------------------------------------- kernel vs commutant --
def _paulis() -> np.ndarray:
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return np.stack([X, Y, Z]) / math.sqrt(2)


def _gen(Ls: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    d = Ls.shape[-1]
    Lt = [torch.tensor(L, dtype=CDTYPE) for L in Ls]
    H0 = torch.zeros(1, d, d, dtype=CDTYPE)
    G = lindblad_generator(H0, Lt, torch.tensor([gammas], dtype=torch.float64))
    return G[0].numpy()


def test_pauli_dephasing_is_primitive_kernel_dim_one():
    """Full Pauli set generates M_2 => commutant is scalars => unique fixed point."""
    Ls = _paulis()
    assert spectra.commutant_dim(Ls) == 1
    kd, _ = spectra.kernel_dim(_gen(Ls, np.ones(3)))
    assert kd == 1


@pytest.mark.parametrize("d", [2, 3, 4, 5])
def test_diagonal_hermitian_jumps_have_d_dimensional_kernel(d):
    """The counterexample to Proposition 3: Hermitian DEPHASING jumps
    are unital and entropy-nondecreasing, yet every diagonal state is fixed, so
    the kernel has dimension d, not 1. Hermiticity alone buys no uniqueness."""
    Ls = np.stack([np.diag(np.arange(d) - (d - 1) / 2).astype(complex)])
    assert spectra.commutant_dim(Ls) == d
    kd, _ = spectra.kernel_dim(_gen(Ls, np.ones(1)))
    assert kd == d
    # I/d is in the kernel (unitality holds) even though it is NOT unique
    assert spectra.identity_in_kernel(_gen(Ls, np.ones(1)), d) < 1e-12


@pytest.mark.parametrize("gammas", [
    [1.0, 1.0, 1.0], [1e-3, 1e3, 1.0], [1e3, 1e-3, 1e-3], [0.7, 0.02, 55.0],
])
def test_kernel_dim_is_independent_of_positive_rates(gammas):
    """For Hermitian jumps the generator is -1/2 sum_k gamma_k ad_{L_k}^2, which is
    negative semidefinite and self-adjoint in Hilbert-Schmidt, so its kernel is the
    intersection of the ad-kernels for ANY strictly positive rates."""
    Ls = _paulis()
    kd, _ = spectra.kernel_dim(_gen(Ls, np.array(gammas)))
    assert kd == spectra.commutant_dim(Ls) == 1


def test_kernel_equals_commutant_on_random_hermitian_jumps():
    rng = np.random.default_rng(3)
    for d in (2, 3, 4):
        for _ in range(5):
            A = rng.normal(size=(2, d, d)) + 1j * rng.normal(size=(2, d, d))
            Ls = A + np.conj(np.swapaxes(A, -2, -1))  # Hermitian
            g = 10.0 ** rng.uniform(-2, 2, size=2)
            kd, _ = spectra.kernel_dim(_gen(Ls, g))
            assert kd == spectra.commutant_dim(Ls)


def test_identity_is_in_the_kernel_for_hermitian_jumps_but_not_for_a_ladder():
    """Unitality is structural for Hermitian jumps (I is fixed) and FAILS for the
    non-normal lowering ladder that v1 carries."""
    d = 4
    rng = np.random.default_rng(4)
    A = rng.normal(size=(3, d, d)) + 1j * rng.normal(size=(3, d, d))
    herm = A + np.conj(np.swapaxes(A, -2, -1))
    assert spectra.identity_in_kernel(_gen(herm, np.ones(3)), d) < 1e-12
    ladder = np.zeros((1, d, d), dtype=complex)
    ladder[0, np.arange(d - 1), np.arange(1, d)] = 1.0
    assert spectra.identity_in_kernel(_gen(ladder, np.ones(1)), d) > 1e-3


def test_v2_checkpoint_jumps_are_exactly_hermitian():
    """The Hermiticity that Proposition 3 rests on is a construction guarantee, so
    it must hold to machine precision on real fitted weights (not approximately)."""
    ck = Path(__file__).resolve().parents[1] / "runs/E1a/ltn_v2/0/model.pt"
    if not ck.exists():
        pytest.skip("no fitted v2 checkpoint available")
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    Ls = spectra.v2_jump_ops(sd)
    for L in Ls:
        assert np.linalg.norm(L - L.conj().T) < 1e-15
