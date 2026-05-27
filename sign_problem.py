"""
sign_problem.py

Sign problem of a (possibly non-unitary) gate U:

    s(U) = |tr(U) / tr(|U|)|

where |U| is the element-wise absolute value (NOT the operator |·|), so
s(U) is basis-dependent.

Translationally-invariant local basis search:
given U on n qubits, conjugate by R^⊗n with
    R(nx,ny,nz) = exp(i*pi*(nx*X + ny*Y + nz*Z))
and minimise s(R^⊗n^† U R^⊗n) over the 3 real parameters (nx,ny,nz)
via gradient descent.

NB: literally enforcing nx^2+ny^2+nz^2 = 1 would give R = -I (trivial
conjugation, since exp(iπ n·σ) = -I for unit n).  We therefore leave
(nx,ny,nz) unconstrained — |n| then plays the role of the rotation
angle through R = cos(π|n|)I + i sin(π|n|) (n̂·σ).  Pass
normalize=True to enforce the unit constraint explicitly (which
collapses the search and is included only for completeness).
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ── sign problem ─────────────────────────────────────────────────────────────
def sign_problem(U: np.ndarray) -> float:
    """|tr(U) / tr(|U|)|, with |U| the element-wise absolute value."""
    denom = np.trace(np.abs(U))
    if denom == 0:
        return float('inf')
    return float(np.abs(np.trace(U) / denom))


# ── single-qubit rotation and its n-fold Kronecker power ─────────────────────
def single_qubit_rotation(n_vec: np.ndarray, *, normalize: bool = False) -> np.ndarray:
    """R = exp(i*pi*(nx*X + ny*Y + nz*Z))."""
    n_vec = np.asarray(n_vec, dtype=float).ravel()
    if normalize:
        norm = np.linalg.norm(n_vec)
        if norm < 1e-14:
            return np.eye(2, dtype=complex)
        n_vec = n_vec / norm
    H = n_vec[0] * _X + n_vec[1] * _Y + n_vec[2] * _Z
    return expm(1j * np.pi * H)


def basis_rotation(n_vec: np.ndarray, n_qubits: int,
                   *, normalize: bool = False) -> np.ndarray:
    """R^⊗n_qubits, with the same single-qubit R on every site."""
    R = single_qubit_rotation(n_vec, normalize=normalize)
    R_full = R
    for _ in range(n_qubits - 1):
        R_full = np.kron(R_full, R)
    return R_full


def rotated_gate(U: np.ndarray, n_vec: np.ndarray,
                 *, normalize: bool = False) -> np.ndarray:
    """R^⊗n^† U R^⊗n.  Number of qubits inferred from U.shape[0]."""
    n_qubits = int(round(np.log2(U.shape[0])))
    if 2 ** n_qubits != U.shape[0]:
        raise ValueError(f'U shape {U.shape} is not a power-of-2 square matrix')
    R = basis_rotation(n_vec, n_qubits, normalize=normalize)
    return R.conj().T @ U @ R


# ── gradient-descent search ──────────────────────────────────────────────────
def minimize_sign_problem(U: np.ndarray,
                          n_restarts: int = 20,
                          method: str = 'BFGS',
                          seed: int = 0,
                          normalize: bool = False,
                          verbose: bool = False
                          ) -> tuple[float, np.ndarray]:
    """Gradient descent over (nx,ny,nz) for the basis that minimises s(U).

    Returns (best_value, best_n).  best_n is normalised iff normalize=True.
    """
    rng = np.random.default_rng(seed)

    def obj(params: np.ndarray) -> float:
        return sign_problem(rotated_gate(U, params, normalize=normalize))

    best_val = sign_problem(U)
    best_n = np.zeros(3)
    if verbose:
        print(f'[init] s(U) = {best_val:.6f}  (no rotation)')

    for r in range(n_restarts):
        x0 = rng.standard_normal(3)
        x0 /= max(np.linalg.norm(x0), 1e-14)
        res = minimize(obj, x0, method=method)
        f_cand = float(res.fun)
        if f_cand < best_val:
            best_val = f_cand
            best_n = res.x.copy()
            if normalize and np.linalg.norm(best_n) > 1e-14:
                best_n = best_n / np.linalg.norm(best_n)
        if verbose:
            print(f'  restart {r + 1:2d}/{n_restarts}:  '
                  f'f={f_cand:.6f}   best={best_val:.6f}')

    return best_val, best_n


# ── demo ─────────────────────────────────────────────────────────────────────
def _demo(n_qubits: int = 2, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    d = 2 ** n_qubits
    G = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    U = expm(0.4 * G)
    print(f'random gate on {n_qubits} qubits, shape {U.shape}')
    print(f's(U) before optimisation: {sign_problem(U):.6f}')

    best_val, best_n = minimize_sign_problem(U, n_restarts=30, verbose=True)
    print(f'\nbest n      = {best_n}')
    print(f'|best n|    = {np.linalg.norm(best_n):.4f}')
    print(f's(U) after  = {best_val:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true',
                        help='Run a self-demo on a random non-unitary gate.')
    parser.add_argument('--n_qubits', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.demo:
        _demo(n_qubits=args.n_qubits, seed=args.seed)
    else:
        parser.print_help()
