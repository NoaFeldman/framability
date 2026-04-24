"""
Worker script: product-state Schrödinger framability (chi=30) for
depolarising-channel compositions.

Gates:
  gate_idx 0  depol(p) ∘ H           single-qubit Hadamard
  gate_idx 1  depol(p) ∘ T           single-qubit T gate
  gate_idx 2  depol(p)⊗² ∘ CNOT     two-qubit CNOT

task_id = gate_idx * N_P + p_idx   (total = 3 × 5 = 15 tasks)

Depolarising convention (one qubit, matching paper Eq. 38):
    N_p(ρ) = (1-3p) ρ + p (X ρ X + Y ρ Y + Z ρ Z)
  → traceless Paulis are suppressed by (1-4p), identity unchanged.
  For two qubits the channel is N_p ⊗ N_p applied after the gate.

Output
------
    <out_dir>/depol_fra_<gate_idx>_<p_idx:02d>.npy   shape (1,) = [framability]

Skips silently if the output file already exists.

Usage
-----
    python depol_fra_worker.py --task_id 7 --out_dir results_depol
"""

import argparse
import os
import sys

import numpy as np
import scipy.linalg as la
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from framability import haar_measure

# ── experiment parameters ────────────────────────────────────────────────────
P_VALUES   = [0.05, 0.07, 0.09, 0.11, 0.13]
GATE_NAMES = ['H', 'T', 'CNOT']
CHI        = 30
SEED       = 42          # fixed seed → same random frame for every task

# ── Pauli matrices ────────────────────────────────────────────────────────────
_I = np.eye(2, dtype=complex)
_X = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0 ]], dtype=complex)
_Z = np.array([[1,  0 ], [0, -1 ]], dtype=complex)
PAULIS = [_I, _X, _Y, _Z]


# ── frame construction ────────────────────────────────────────────────────────

def _build_D1(chi, seed=SEED):
    """Single-qubit frame matrix, shape (4, chi).

    Column j = (Tr(σ_a ρ_j) / 2)_a for a Haar-random pure state ρ_j.
    Uses the same convention and seed as make_product_state_D in framability.py.
    """
    np.random.seed(seed)          # must come before any haar_measure call
    zero_dm = np.array([[1, 0], [0, 0]], dtype=complex)
    D1 = np.zeros((4, chi), dtype=float)
    for i in range(chi):
        u = haar_measure(2)
        rho = u @ zero_dm @ u.T.conj()
        for a, s in enumerate(PAULIS):
            D1[a, i] = (np.trace(s @ rho) / 2).real
    return D1


# ── superoperator helpers ─────────────────────────────────────────────────────

def _superop_1q(U):
    """4×4 real superoperator for conjugation ρ → U ρ U†, single-qubit Pauli basis."""
    L = np.zeros((4, 4), dtype=float)
    for j, Bj in enumerate(PAULIS):
        img = U @ Bj @ U.conj().T
        for i, Bi in enumerate(PAULIS):
            L[i, j] = (np.trace(Bi.conj().T @ img) / 2).real
    return L


def _superop_2q(U):
    """16×16 real superoperator for conjugation ρ → U ρ U†, two-qubit Pauli basis.

    Basis ordering matches two_qubit_lindbladian.py: (I,I),(I,X),...,(Z,Z).
    """
    basis = [np.kron(a, b) for a in PAULIS for b in PAULIS]
    n = len(basis)
    L = np.zeros((n, n), dtype=float)
    for j, Bj in enumerate(basis):
        img = U @ Bj @ U.conj().T
        for i, Bi in enumerate(basis):
            L[i, j] = (np.trace(Bi.conj().T @ img) / 4).real
    return L


def _depol_1q(p):
    """4×4 superoperator for single-qubit depolarising channel N_p (paper Eq. 38).

    N_p(ρ) = (1-3p)ρ + p(XρX + YρY + ZρZ)
    In the Pauli basis: identity component preserved, traceless Paulis ×(1-4p).
    """
    return np.diag([1.0, 1.0 - 4*p, 1.0 - 4*p, 1.0 - 4*p])


def _depol_2q(p):
    """16×16 superoperator for N_p ⊗ N_p on two qubits (paper Eq. 38).

    Basis element (σ_a ⊗ σ_b) at index 4a+b is suppressed by (1-4p)^k where
    k = #{a≠0} + #{b≠0} is the number of non-trivial single-qubit factors.
    """
    diag = np.array(
        [(1.0 - 4*p) ** ((a != 0) + (b != 0))
         for a in range(4) for b in range(4)],
        dtype=float,
    )
    return np.diag(diag)


# ── framability via dual LP ───────────────────────────────────────────────────

def _fra_schroedinger(D, channel):
    """Schrödinger framability via the dual LP.

    For each frame column d_j, the primal problem is
        min ||v||_1   s.t.  D v = channel @ d_j.
    By LP duality this equals
        max  b^T y    s.t.  ||D^T y||_inf <= 1,
    with b = channel @ d_j and y free.

    Parameters
    ----------
    D       : (d, chi_total) real frame matrix.
    channel : (d, d) real superoperator matrix.

    Returns
    -------
    float : maximum primal 1-norm over all frame columns.
    """
    n_cols = D.shape[1]
    A_ub  = csc_matrix(np.vstack([D.T, -D.T]))   # (2*chi_total, d)
    b_ub  = np.ones(2 * n_cols)
    bounds = [(None, None)] * D.shape[0]

    best = -np.inf
    for j in range(n_cols):
        b = channel @ D[:, j]
        res = linprog(-b, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            best = max(best, -res.fun)
    return float(best)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Product-state framability (chi=30) of a depolarised gate.'
    )
    parser.add_argument('--task_id', type=int, required=True,
                        help='task_id = gate_idx * 5 + p_idx  (0..14)')
    parser.add_argument('--out_dir', type=str, default='results_depol',
                        help='Output directory (default: results_depol).')
    args = parser.parse_args()

    n_p      = len(P_VALUES)
    gate_idx = args.task_id // n_p
    p_idx    = args.task_id %  n_p

    if gate_idx >= len(GATE_NAMES):
        print(f'ERROR: task_id {args.task_id} out of range '
              f'(max {len(GATE_NAMES) * n_p - 1})', file=sys.stderr)
        sys.exit(1)

    p_val = P_VALUES[p_idx]
    print(f'[task {args.task_id}] gate={GATE_NAMES[gate_idx]}  p={p_val}',
          flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir,
                            f'depol_fra_{gate_idx}_{p_idx:02d}.npy')
    if os.path.exists(out_path):
        print(f'Skip: {out_path} already exists', flush=True)
        return

    # ── build frame ───────────────────────────────────────────────────────────
    D1 = _build_D1(CHI)     # shape (4, 30); seed fixed inside

    # ── build channel superoperator ───────────────────────────────────────────
    if gate_idx == 0:
        # depol(p) ∘ H  — single-qubit
        H       = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
        channel = _depol_1q(p_val) @ _superop_1q(H)
        D       = D1                           # (4, 30)

    elif gate_idx == 1:
        # depol(p) ∘ T  — single-qubit
        T       = np.diag([1.0, np.exp(1j * np.pi / 4)]).astype(complex)
        channel = _depol_1q(p_val) @ _superop_1q(T)
        D       = D1                           # (4, 30)

    else:
        # depol(p)⊗² ∘ CNOT  — two-qubit
        CNOT    = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=float)
        channel = _depol_2q(p_val) @ _superop_2q(CNOT)
        D       = np.kron(D1, D1)              # (16, 900)

    # ── compute framability ───────────────────────────────────────────────────
    fra = _fra_schroedinger(D, channel)

    np.save(out_path, np.array([fra]))
    print(f'[task {args.task_id}] fra={fra:.6f}  -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
