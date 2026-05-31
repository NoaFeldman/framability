"""
Unified neighbor-seeded framability refinement worker.

Supports three variants:
    d6     — d_ext_single=6, fixed columns (I, Z), uses minimize_framability
    d4     — d_ext_single=4, all columns free, uses optimize_d4
    free6  — d_ext_single=6, all columns free, uses optimize_free6

Each SLURM array task processes one grid point (ig, igp).
Task mapping: task_id = ig * n_igp + igp

For each point whose framability exceeds that of any 4-connected neighbor,
the script:
  1. Re-optimises on the neighbor's gate (1 restart) to get x_nb.
  2. Re-optimises this point seeded with x_nb.
  3. Saves the (possibly improved) value to a point file.

Output:
    <out_dir>/refine_<variant>_<round>_<ig:04d>_<igp:04d>.npy   scalar float

Usage:
    python unified_nb_refine_worker.py --variant free6 --round 1 \
        --task_id 42 --n_pts 41 --n_igp 21 --out_dir results_free6
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from analysis import compute_steady_state

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _make_gate(J, gamma, gamma_p, gamma_step):
    L = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)
    dt = 0.01 * gamma_step
    return expm(dt * L).real


def _load_grid(variant, out_dir, n_pts):
    """Load the framability grid for a given variant."""
    if variant == 'd6':
        scan_path = os.path.join(out_dir, 'scan_full.npy')
        if not os.path.exists(scan_path):
            print(f'ERROR: {scan_path} not found', file=sys.stderr)
            sys.exit(1)
        return np.load(scan_path)[:, :, 3]  # col 3 = optimised fra
    elif variant == 'd4':
        path = os.path.join(out_dir, 'd4_scan.npy')
        if not os.path.exists(path):
            print(f'ERROR: {path} not found', file=sys.stderr)
            sys.exit(1)
        return np.load(path)
    elif variant == 'free6':
        path = os.path.join(out_dir, 'free6_scan.npy')
        if not os.path.exists(path):
            print(f'ERROR: {path} not found', file=sys.stderr)
            sys.exit(1)
        return np.load(path)
    else:
        print(f'ERROR: unknown variant {variant!r}', file=sys.stderr)
        sys.exit(1)


def _optimize_variant(variant, gate, n_restarts, maxfev, seed, extra_init_xs=None):
    """Run the appropriate optimizer for the given variant.
    Returns (fra_value, x_opt)."""
    if variant == 'd6':
        from optimize_framability import minimize_framability, DEFAULT_METHOD
        _, f, x = minimize_framability(
            gate, d_ext_single=6,
            n_restarts=n_restarts, method=DEFAULT_METHOD,
            maxfev=maxfev, verbose=False, seed=seed,
            extra_init_xs=extra_init_xs, return_x=True,
        )
        return f, x
    elif variant == 'd4':
        from d4_scan_worker import optimize_d4
        return optimize_d4(gate, n_restarts=n_restarts, seed=seed,
                           maxfev=maxfev, extra_init_xs=extra_init_xs)
    elif variant == 'free6':
        from free_6_scan_worker import optimize_free6
        return optimize_free6(gate, n_restarts=n_restarts, seed=seed,
                              maxfev=maxfev, extra_init_xs=extra_init_xs)


def main():
    p = argparse.ArgumentParser(
        description='Unified neighbor-seeded framability refinement.'
    )
    p.add_argument('--variant',    type=str, required=True,
                   choices=['d6', 'd4', 'free6'])
    p.add_argument('--round',      type=int, required=True,
                   help='Refinement round number (1 or 2).')
    p.add_argument('--task_id',    type=int, required=True,
                   help='Flat index ig * n_igp + igp.')
    p.add_argument('--n_pts',      type=int, default=41,
                   help='Total number of gamma points.')
    p.add_argument('--n_igp',      type=int, default=21,
                   help='Number of gamma\' points (gamma\' up to 4 with step 0.2 = 21).')
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str, required=True,
                   help='Directory with scan data (variant-specific).')
    p.add_argument('--n_restarts', type=int, default=5)
    p.add_argument('--maxfev',     type=int, default=1000)
    args = p.parse_args()

    n = args.n_pts
    n_igp = args.n_igp
    ig  = args.task_id // n_igp
    igp = args.task_id %  n_igp

    if ig < 0 or ig >= n or igp < 0 or igp >= n_igp:
        print(f'ERROR: task_id {args.task_id} → ({ig},{igp}) out of range '
              f'for {n}×{n_igp} grid', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f'refine_{args.variant}_{args.round}_{ig:04d}_{igp:04d}'
    out_path = os.path.join(args.out_dir, f'{tag}.npy')

    if os.path.exists(out_path):
        print(f'Skip: {out_path} exists', flush=True)
        return

    # Load current grid
    grid = _load_grid(args.variant, args.out_dir, n)
    my_val = grid[ig, igp]

    # Find best 4-connected neighbor (within entire grid, not just n_igp cols)
    best_nb_val = np.inf
    best_ni, best_nj = -1, -1
    grid_rows, grid_cols = grid.shape
    for di, dj in NEIGHBORS:
        ni, nj = ig + di, igp + dj
        if 0 <= ni < grid_rows and 0 <= nj < grid_cols:
            nb_val = grid[ni, nj]
            if np.isfinite(nb_val) and nb_val < best_nb_val:
                best_nb_val = nb_val
                best_ni, best_nj = ni, nj

    # Skip if no neighbor is better
    if best_nb_val >= my_val:
        np.save(out_path, np.inf)
        print(f'[{args.variant} r{args.round}] ({ig},{igp}) '
              f'fra={my_val:.6f}  best_nb={best_nb_val:.6f}  → skip')
        return

    print(f'[{args.variant} r{args.round}] ({ig},{igp}) '
          f'fra={my_val:.6f}  best_nb=({best_ni},{best_nj}) '
          f'{best_nb_val:.6f}  → refining ...', flush=True)

    gs = args.gamma_step

    # Step 1: get neighbor's optimal parameter vector
    gamma_nb   = gs * best_ni
    gamma_p_nb = gs * best_nj
    gate_nb = _make_gate(args.J, gamma_nb, gamma_p_nb, gs)
    _, x_nb = _optimize_variant(
        args.variant, gate_nb, n_restarts=1, maxfev=args.maxfev,
        seed=best_ni * 10000 + best_nj,
    )

    # Step 2: re-optimise this point seeded with neighbor's x
    gamma   = gs * ig
    gamma_p = gs * igp
    gate_self = _make_gate(args.J, gamma, gamma_p, gs)
    f_refined, _ = _optimize_variant(
        args.variant, gate_self,
        n_restarts=args.n_restarts, maxfev=args.maxfev,
        seed=ig * 10000 + igp + 99999,
        extra_init_xs=[x_nb],
    )

    if f_refined < my_val - 1e-9:
        np.save(out_path, f_refined)
        print(f'[{args.variant} r{args.round}] ({ig},{igp}) '
              f'improved: {my_val:.6f} → {f_refined:.6f}  '
              f'(Δ = {my_val - f_refined:.6f})')
    else:
        np.save(out_path, np.inf)
        print(f'[{args.variant} r{args.round}] ({ig},{igp}) '
              f'no improvement: {my_val:.6f} → {f_refined:.6f}')


if __name__ == '__main__':
    main()
