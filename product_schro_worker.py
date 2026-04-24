"""
Per-point SLURM worker: product-state Schrödinger framability (chi=30) at a
single (gamma, gamma') grid point.

Uses the same D_1 base matrix as all other scan workers (seed=42), building
    D = kron(D_1, D_1)   shape (16, 900)

LP solved via duality: 16 dual variables independent of chi=30² frame size,
iterated once per column of D (900 iterations per point).

For each frame column d_j the Schrödinger LP is

    min ||v||_1   s.t.  D v = gate @ d_j          [primal]

whose dual is

    max  b^T y    s.t.  ||D^T y||_inf <= 1,  b = gate @ d_j

Output
------
    <out_dir>/prod_schro_pt_XXXX_XXXX.npy   shape (1,) = [framability]

Skips silently if the output file already exists.

Usage
-----
    python product_schro_worker.py --task_id 42 --n_pts 41 --J 1.0 \\
                                   --gamma_step 0.2 --out_dir results
"""

import argparse
import os
import sys

import numpy as np
import scipy.linalg as _la
from scipy.linalg import expm
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

from two_qubit_lindbladian import numeric_two_qubit_lindbladian

CHI = 30   # single-qubit frame size; two-qubit frame has CHI² = 900 columns


def _build_D1(chi):
    """Single-qubit frame matrix, shape (4, chi).

    Identical construction to make_product_state_D in framability.py:
    D_1[a, i] = Tr(σ_a ρ_i) / 2  for Haar-random single-qubit pure states.
    Seed fixed to 42 so the frame is shared across all workers.
    """
    paulis = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    zero_dm = np.array([[1, 0], [0, 0]], dtype=complex)

    def _haar(n):
        z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
        q, r = _la.qr(z)
        d = np.diagonal(r)
        ph = d / np.abs(d)
        return np.multiply(q, ph, q)

    np.random.seed(42)
    D1 = np.zeros((4, chi), dtype=float)
    for i in range(chi):
        u = _haar(2)
        rho = u @ zero_dm @ u.T.conj()
        for a, s in enumerate(paulis):
            D1[a, i] = (np.trace(s @ rho) / 2).real
    return D1


def _fra_schroedinger_dual(D, gate):
    """Schrödinger framability via the dual LP.

    For each frame column d_j solves the dual of
        min ||v||_1   s.t.  D v = gate @ d_j
    which is
        max  b^T y    s.t.  ||D^T y||_inf <= 1,  b = gate @ d_j.

    Pre-computes the sparse constraint matrix once; only the RHS b changes
    per column.  This keeps memory small (16 dual variables) regardless of
    the frame size.
    """
    n_cols = D.shape[1]
    A_ub   = csc_matrix(np.vstack([D.T, -D.T]))   # (2*n_cols, 16)
    b_ub   = np.ones(2 * n_cols)
    bounds = [(None, None)] * 16                   # y free

    best = -np.inf
    for j in range(n_cols):
        b = gate @ D[:, j]
        r = linprog(-b, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if r.success:
            best = max(best, -r.fun)
    return float(best)


def main():
    p = argparse.ArgumentParser(
        description='Product-state Schrödinger framability (chi=30) for one grid point.'
    )
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Flat grid index: ig = task_id // n_pts, igp = task_id %% n_pts.')
    p.add_argument('--n_pts',      type=int,   default=41)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str,   default='results')
    args = p.parse_args()

    n   = args.n_pts
    ig  = args.task_id // n
    igp = args.task_id % n

    if ig >= n or igp >= n:
        print(f'ERROR: task_id {args.task_id} out of range for n_pts={n}',
              file=sys.stderr)
        sys.exit(1)

    out = os.path.join(args.out_dir, f'prod_schro_pt_{ig:04d}_{igp:04d}.npy')
    if os.path.exists(out):
        print(f'Skip: {out} already exists', flush=True)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    gamma = args.gamma_step * ig
    gp    = args.gamma_step * igp
    print(f'Task {args.task_id}: ig={ig} igp={igp}  '
          f'gamma={gamma:.4f}  gamma\'={gp:.4f}', flush=True)

    # Build frame (seed fixed; same states every run)
    D1 = _build_D1(CHI)
    D  = np.kron(D1, D1)   # (16, 900)

    L    = numeric_two_qubit_lindbladian(J=args.J, gamma=gamma, gamma_p=gp)
    gate = expm(0.01 * args.gamma_step * L).real

    fra = _fra_schroedinger_dual(D, gate)

    np.save(out, np.array([fra]))
    print(f'Task {args.task_id}: fra={fra:.6f}  -> {out}', flush=True)


if __name__ == '__main__':
    main()
