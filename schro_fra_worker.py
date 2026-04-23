"""
Per-point SLURM worker: product-state Schrödinger framability for
chi=20 and chi=40 at a single (gamma, gamma') grid point.

Uses the same D_1 base matrix as scan_worker_extra.py (seed=42),
building D_chi = kron(D_1[:, :chi], D_1[:, :chi]).

LP solved via duality: 16 dual variables independent of chi,
iterated once per column of D (chi^2 iterations per chi value).

Output
------
    <out_dir>/schro_fra_pt_XXXX_XXXX.npy   shape (2,) = [val_chi20, val_chi40]

Skips if the output file already exists.

Usage
-----
    python schro_fra_worker.py --task_id 42 --n_pts 41 --J 1.0 \\
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

# Chi values to compute (chi=10 and chi=30 are already done; chi=20 was skipped)
CHIS = [20, 40]


def _build_D1_base(chi_max):
    """Build the single-qubit frame matrix D_1 of shape (4, chi_max).

    Uses the same Haar-random construction and seed as scan_worker_extra.py
    so that columns 0..29 match the existing chi=30 data exactly.
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
    D1 = np.zeros((4, chi_max), dtype=float)
    for i in range(chi_max):
        u = _haar(2)
        rho = u @ zero_dm @ u.T.conj()
        for a, s in enumerate(paulis):
            D1[a, i] = (np.trace(s @ rho) / 2).real
    return D1


def _build_dual(D):
    """Pre-compute dual LP ingredients for a fixed frame matrix D."""
    A_ub = csc_matrix(np.vstack([D.T, -D.T]))   # (2*d_ext, 16)
    b_ub = np.ones(2 * D.shape[1])
    bounds = [(None, None)] * 16
    return D, A_ub, b_ub, bounds


def _fra_dual(D, A_ub, b_ub, bounds, gate):
    """Schrödinger framability via dual LP (max over all frame columns)."""
    best = -1e30
    for j in range(D.shape[1]):
        b = gate @ D[:, j]
        r = linprog(-b, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if r.success and -r.fun > best:
            best = -r.fun
    return float(best)


def main():
    p = argparse.ArgumentParser(
        description='Compute Schrödinger framability (chi=20,40) for one grid point.'
    )
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Flat grid index: task_id = ig * n_pts + igp')
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

    out = os.path.join(args.out_dir, f'schro_fra_pt_{ig:04d}_{igp:04d}.npy')
    if os.path.exists(out):
        print(f'Skip: {out} already exists', flush=True)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    gamma = args.gamma_step * ig
    gp    = args.gamma_step * igp
    print(f'Task {args.task_id}: ig={ig} igp={igp}  '
          f'gamma={gamma:.4f}  gamma\'={gp:.4f}', flush=True)

    # Build D_1 with enough columns for the largest chi
    D1base = _build_D1_base(max(CHIS))

    L    = numeric_two_qubit_lindbladian(J=args.J, gamma=gamma, gamma_p=gp)
    gate = expm(0.01 * args.gamma_step * L).real

    vals = np.zeros(len(CHIS))
    for k, chi in enumerate(CHIS):
        D     = np.kron(D1base[:, :chi], D1base[:, :chi])
        setup = _build_dual(D)
        vals[k] = _fra_dual(*setup, gate)
        print(f'  chi={chi:2d}: fra={vals[k]:.6f}', flush=True)

    np.save(out, vals)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
