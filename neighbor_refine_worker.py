"""
Worker script for neighbor-seeded min-framability refinement.

Each SLURM array task processes one grid point (ig, igp) identified by
a flat task_id = ig * n_pts + igp.

For each point whose min_fra exceeds that of any 4-connected neighbor,
the script:
  1. Re-runs minimize_framability on the NEIGHBOR's gate (1 restart) to
     capture its locally-optimal parameter vector x_nb.
  2. Re-runs minimize_framability on this point's gate seeded with x_nb
     as an extra restart.
  3. Saves the (possibly improved) min_fra to a point file.

Requires scan_full.npy to already exist in out_dir.

Output
------
    <out_dir>/refine_nb_<ig:04d>_<igp:04d>.npy   scalar float
    (np.inf if the point was skipped or no improvement was found)

Usage
-----
    python neighbor_refine_worker.py --task_id 42 --n_pts 20 --J 1.0 \
                                     --gamma_step 0.1 --out_dir results
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from framability import extended_pauli_D
from optimize_framability import minimize_framability, DEFAULT_METHOD

# Column indices in scan_full.npy (must match scan_worker.py / scan_collect.py)
# Row files have 5 columns: 0=entropy, 1=negativity, 2=pauli_fra, 3=min_fra, 4=dec_rate
COL_MIN_FRA = 3

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _make_gate(J, gamma, gamma_p, gamma_step):
    L = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)
    dt = 0.01 * gamma_step
    return expm(dt * L).real


def main():
    p = argparse.ArgumentParser(
        description='Neighbor-seeded refinement of a single grid point.'
    )
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Flat index ig*n_igp+igp (or ig*n_pts+igp when --n_igp is
                        omitted); maps to SLURM_ARRAY_TASK_ID.')
    p.add_argument('--n_pts',      type=int,   default=20)
    p.add_argument('--n_igp',      type=int,   default=0,
                   help='Number of gamma\' columns per row in the task-id mapping.
                        0 (default) means use --n_pts (full grid).')
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.1)
    p.add_argument('--out_dir',    type=str,   default='results')
    p.add_argument('--n_restarts', type=int,   default=5,
                   help='Standard restarts for the outlier re-optimisation.')
    p.add_argument('--maxfev',     type=int,   default=1000,
                   help='Max function evaluations per restart.')
    args = p.parse_args()

    n     = args.n_pts
    n_igp = args.n_igp if args.n_igp > 0 else n
    ig  = args.task_id // n_igp
    igp = args.task_id %  n_igp
    if ig < 0 or ig >= n or igp < 0 or igp >= n:
        print(f'ERROR: task_id {args.task_id} out of range for {n}x{n} grid '
              f'(n_igp={n_igp})',
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'refine_nb_{ig:04d}_{igp:04d}.npy')

    # Load existing scan data
    scan_path = os.path.join(args.out_dir, 'scan_full.npy')
    if not os.path.exists(scan_path):
        print(f'ERROR: {scan_path} not found', file=sys.stderr)
        sys.exit(1)
    data = np.load(scan_path)
    min_fra = data[:, :, COL_MIN_FRA]

    my_val = min_fra[ig, igp]

    # Find best (lowest) 4-connected neighbor
    best_nb_val = np.inf
    best_ni, best_nj = -1, -1
    for di, dj in NEIGHBORS:
        ni, nj = ig + di, igp + dj
        if 0 <= ni < n and 0 <= nj < n:
            nb_val = min_fra[ni, nj]
            if nb_val < best_nb_val:
                best_nb_val = nb_val
                best_ni, best_nj = ni, nj

    # Skip if no neighbor is better
    if best_nb_val >= my_val:
        np.save(out_path, np.inf)
        print(f'({ig},{igp}) min_fra={my_val:.6f}  '
              f'best_nb={best_nb_val:.6f}  → skip (no better neighbor)')
        return

    print(f'({ig},{igp}) min_fra={my_val:.6f}  '
          f'best_nb=({best_ni},{best_nj}) {best_nb_val:.6f}  → refining ...',
          flush=True)

    d_ext_single = int(round(np.sqrt(extended_pauli_D().shape[1])))  # 6
    gamma_step = args.gamma_step

    # Step 1: get the neighbor's locally-optimal parameter vector
    gamma_nb   = gamma_step * best_ni
    gamma_p_nb = gamma_step * best_nj
    gate_nb = _make_gate(args.J, gamma_nb, gamma_p_nb, gamma_step)
    _, _, x_nb = minimize_framability(
        gate_nb, d_ext_single=d_ext_single,
        n_restarts=1, method=DEFAULT_METHOD,
        maxfev=args.maxfev, verbose=False, return_x=True,
    )

    # Step 2: re-optimise this point seeded with the neighbor's x
    gamma   = gamma_step * ig
    gamma_p = gamma_step * igp
    gate_self = _make_gate(args.J, gamma, gamma_p, gamma_step)
    _, f_refined, _ = minimize_framability(
        gate_self, d_ext_single=d_ext_single,
        n_restarts=args.n_restarts, method=DEFAULT_METHOD,
        maxfev=args.maxfev, verbose=False,
        extra_init_xs=[x_nb], return_x=True,
    )

    if f_refined < my_val - 1e-9:
        np.save(out_path, f_refined)
        print(f'({ig},{igp}) improved: {my_val:.6f} → {f_refined:.6f}  '
              f'(Δ = {my_val - f_refined:.6f})')
    else:
        np.save(out_path, np.inf)
        print(f'({ig},{igp}) no improvement: {my_val:.6f} → {f_refined:.6f}')


if __name__ == '__main__':
    main()
