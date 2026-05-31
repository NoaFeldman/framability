"""
Worker: optimized framability with d_ext_single=6 and ALL columns free.

Each SLURM array task processes one row (fixed gamma index).
All 6 columns of S (shape 4×6) are free parameters, projected via the
Bloch constraint |c_I| + ||(c_X,c_Y,c_Z)||_2 <= 1.

Additional constraint: diag(S @ S.T) >= 1  (enforced via penalty).

Output:  <out_dir>/free6row_<task_id:04d>.npy   shape (n_pts,)

Usage:
    python free_6_scan_worker.py --task_id 5 --n_pts 41 --gamma_step 0.2 --out_dir results_free6
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

N_S_ROWS = 4        # qubit_d^2
N_QUBITS = 2
D_EXT_SINGLE = 6    # all 6 columns free
N_PARAMS = N_S_ROWS * D_EXT_SINGLE  # 24

PENALTY_WEIGHT = 100.0  # penalty for violating diag(S @ S.T) >= 1


def _params_to_D_free6(params):
    """Decode flat params (24,) into D = kron(S, S) with S = 4×6, all columns free."""
    S = _project_columns_bloch(params.reshape(N_S_ROWS, D_EXT_SINGLE))
    return _kron_power(S, N_QUBITS), S


def _diag_penalty(S):
    """Penalty for violating diag(S @ S.T) >= 1."""
    diag = np.sum(S ** 2, axis=1)  # shape (4,)
    violations = np.maximum(1.0 - diag, 0.0)
    return PENALTY_WEIGHT * np.sum(violations ** 2)


def _objective(params, gate):
    D, S = _params_to_D_free6(params)
    fra = _get_framability_fast(D, gate)
    return fra + _diag_penalty(S)


def _objective_no_penalty(params, gate):
    """Framability only (no penalty), for final evaluation."""
    D, _ = _params_to_D_free6(params)
    return _get_framability_fast(D, gate)


def _build_inits(n_restarts, rng):
    """Build initial points: structured seeds + random."""
    inits = []

    # Init 1: extended Pauli S (the default d_ext=6 frame)
    a = 1.0
    S_pauli = np.array([
        [1, 0, 0, 0, 0,            0],
        [0, 1, 0, 0, a/np.sqrt(2), a/np.sqrt(2)],
        [0, 0, 1, 0, 0,            0],
        [0, 0, 0, 1, a/np.sqrt(2), -a/np.sqrt(2)],
    ], dtype=float)
    inits.append(_project_columns_bloch(S_pauli).ravel())

    # Init 2: identity-like (first 4 cols = I_4, last 2 random)
    S_id = np.zeros((N_S_ROWS, D_EXT_SINGLE))
    S_id[:4, :4] = np.eye(4)
    S_id[:, 4:] = rng.standard_normal((N_S_ROWS, 2))
    inits.append(_project_columns_bloch(S_id).ravel())

    # Init 3: columns cycling through basis vectors
    S_cyc = np.zeros((N_S_ROWS, D_EXT_SINGLE))
    for j in range(D_EXT_SINGLE):
        S_cyc[j % N_S_ROWS, j] = 1.0
    inits.append(_project_columns_bloch(S_cyc).ravel())

    # Init 4: I and Z in columns 0,1 + free (like the fixed-col d6 init)
    S_izxy = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=float)
    S_izxy[:, 4:] = rng.standard_normal((N_S_ROWS, 2))
    inits.append(_project_columns_bloch(S_izxy).ravel())

    # Random inits
    while len(inits) < n_restarts:
        M = rng.standard_normal((N_S_ROWS, D_EXT_SINGLE))
        inits.append(_project_columns_bloch(M).ravel())

    return inits


def optimize_free6(gate, n_restarts=10, seed=None, maxfev=2000, extra_init_xs=None):
    """Minimize framability over D = kron(S,S), S 4×6, all columns free,
    with penalty for diag(S @ S.T) < 1."""
    rng = np.random.default_rng(seed)
    gate = np.asarray(gate, dtype=float)

    inits = _build_inits(n_restarts, rng)
    if extra_init_xs:
        for x in extra_init_xs:
            inits.append(np.asarray(x, dtype=float))

    best_val = np.inf
    best_x = None

    for x0 in inits:
        # Evaluate with penalty during optimization
        f0 = _objective(x0, gate)
        fra0 = _objective_no_penalty(x0, gate)
        if fra0 < best_val:
            _, S0 = _params_to_D_free6(x0)
            if np.all(np.sum(S0 ** 2, axis=1) >= 1.0 - 1e-8):
                best_val = fra0
                best_x = x0.copy()

        res = minimize(_objective, x0, args=(gate,),
                       method=DEFAULT_METHOD,
                       options={'maxfev': maxfev})

        # Final evaluation: check constraint satisfaction
        D_cand, S_cand = _params_to_D_free6(res.x)
        diag_ok = np.all(np.sum(S_cand ** 2, axis=1) >= 1.0 - 1e-8)
        f_cand = _get_framability_fast(D_cand, gate)

        if diag_ok and f_cand < best_val:
            best_val = f_cand
            best_x = res.x.copy()

    # If no feasible solution found, take the best penalized one
    if best_x is None:
        best_val = np.inf
        for x0 in inits:
            res = minimize(_objective, x0, args=(gate,),
                           method=DEFAULT_METHOD,
                           options={'maxfev': maxfev})
            f_cand = _objective_no_penalty(res.x, gate)
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
    p.add_argument('--out_dir',    type=str, default='results_free6')
    p.add_argument('--n_restarts', type=int, default=10)
    p.add_argument('--maxfev',     type=int, default=2000)
    args = p.parse_args()

    ig = args.task_id
    if ig < 0 or ig >= args.n_pts:
        print(f'ERROR: task_id {ig} out of range', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'free6row_{ig:04d}.npy')
    if os.path.exists(out_path):
        print(f'Skip: {out_path} exists', flush=True)
        return

    gamma = args.gamma_step * ig
    n = args.n_pts
    row = np.full(n, np.nan)
    dt = 0.01 * args.gamma_step

    print(f'[task {ig}] gamma={gamma:.4f}, {n} points, d_ext_single=6 (free)',
          flush=True)

    for igp in range(n):
        gp = args.gamma_step * igp
        _, L = compute_steady_state(args.J, gamma, gp)
        gate = expm(dt * L).real

        fra, _ = optimize_free6(gate, n_restarts=args.n_restarts,
                                seed=ig * 10000 + igp, maxfev=args.maxfev)
        row[igp] = fra
        print(f'[task {ig}] col {igp+1}/{n}  gp={gp:.4f}  fra={fra:.6f}',
              flush=True)

    np.save(out_path, row)
    print(f'[task {ig}] saved {out_path}', flush=True)


if __name__ == '__main__':
    main()
