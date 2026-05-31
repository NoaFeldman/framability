"""
Worker: optimized framability with d_ext_single=4 and NO fixed columns.

Each SLURM array task processes one row (fixed gamma index).
All 4 columns of S (shape 4×4) are free parameters, projected via the
Bloch constraint |c_I| + ||(c_X,c_Y,c_Z)||_2 <= 1.

Output:  <out_dir>/d4row_<task_id:04d>.npy   shape (n_pts,)
         One optimized framability value per gamma' point.

Usage:
    python d4_scan_worker.py --task_id 5 --n_pts 41 --gamma_step 0.2 --out_dir results_d4
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from optimize_framability import (
    _get_framability_fast,
    _kron_power,
    _project_columns_bloch,
    DEFAULT_METHOD,
)
from analysis import compute_steady_state

N_S_ROWS = 4       # qubit_d^2
N_QUBITS = 2
D_EXT_SINGLE = 4   # all columns free
N_PARAMS = N_S_ROWS * D_EXT_SINGLE  # 16


def _params_to_D_free(params):
    """Decode flat params (16,) into D = kron(S, S) with S = 4×4, all columns free."""
    S = _project_columns_bloch(params.reshape(N_S_ROWS, D_EXT_SINGLE))
    return _kron_power(S, N_QUBITS), S


def _objective(params, gate):
    D, _ = _params_to_D_free(params)
    return _get_framability_fast(D, gate)


def _build_inits(n_restarts, rng):
    """Build initial points: identity S, Pauli-like, and random."""
    inits = []

    # Init 1: identity S = I_4
    inits.append(np.eye(N_S_ROWS, D_EXT_SINGLE).ravel())

    # Init 2: Pauli basis columns [I, X, Y, Z] (each as unit vectors)
    S_pauli = np.eye(4)
    inits.append(S_pauli.ravel())

    # Init 3: columns = I, Z, X, Y (matching old fixed + free structure)
    S_izxy = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
    ], dtype=float)
    inits.append(S_izxy.ravel())

    # Random inits
    while len(inits) < n_restarts:
        M = rng.standard_normal((N_S_ROWS, D_EXT_SINGLE))
        inits.append(_project_columns_bloch(M).ravel())

    return inits


def optimize_d4(gate, n_restarts=10, seed=None, maxfev=2000, extra_init_xs=None):
    """Minimize framability over D = kron(S,S), S 4×4, all columns free."""
    rng = np.random.default_rng(seed)
    gate = np.asarray(gate, dtype=float)

    inits = _build_inits(n_restarts, rng)
    if extra_init_xs:
        for x in extra_init_xs:
            inits.append(np.asarray(x, dtype=float))

    best_val = np.inf
    best_x = None

    for x0 in inits:
        f0 = _objective(x0, gate)
        if f0 < best_val:
            best_val = f0
            best_x = x0.copy()

        res = minimize(_objective, x0, args=(gate,),
                       method=DEFAULT_METHOD,
                       options={'maxfev': maxfev})
        f_cand = _objective(res.x, gate)
        if f_cand < best_val:
            best_val = f_cand
            best_x = res.x.copy()

    return best_val, best_x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',    type=int, required=True)
    p.add_argument('--n_pts',      type=int, default=41)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str, default='results_d4')
    p.add_argument('--n_restarts', type=int, default=10)
    p.add_argument('--maxfev',     type=int, default=2000)
    args = p.parse_args()

    ig = args.task_id
    if ig < 0 or ig >= args.n_pts:
        print(f'ERROR: task_id {ig} out of range', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'd4row_{ig:04d}.npy')
    if os.path.exists(out_path):
        print(f'Skip: {out_path} exists', flush=True)
        return

    gamma = args.gamma_step * ig
    n = args.n_pts
    row = np.full(n, np.nan)
    dt = 0.01 * args.gamma_step

    print(f'[task {ig}] gamma={gamma:.4f}, {n} points, d_ext_single=4 (free)',
          flush=True)

    for igp in range(n):
        gp = args.gamma_step * igp
        _, L = compute_steady_state(args.J, gamma, gp)
        gate = expm(dt * L).real

        fra, _ = optimize_d4(gate, n_restarts=args.n_restarts,
                             seed=ig * 10000 + igp, maxfev=args.maxfev)
        row[igp] = fra
        print(f'[task {ig}] col {igp+1}/{n}  gp={gp:.4f}  fra={fra:.6f}',
              flush=True)

    np.save(out_path, row)
    print(f'[task {ig}] saved {out_path}', flush=True)


if __name__ == '__main__':
    main()
