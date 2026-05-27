"""
Worker: optimized framability with d_ext_single=6 (ALL columns free),
seeded from the d_ext=6 fixed-cols results in results_opt/opt_S_<ig>_<igp>.npy.

Each SLURM array task processes one row (fixed gamma index).
For every (ig, igp) we load the previously-optimized 1-qubit S (4x6) from
results_opt/ and inject it as an extra initial point to the free_6 optimizer.

Output:  <out_dir>/free6row_<task_id:04d>.npy   shape (n_pts,)

Usage:
    python free_6_from_d6_worker.py --task_id 5 --n_pts 41 \
        --gamma_step 0.2 --out_dir results_free6 --opt_dir results_opt
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm

from analysis import compute_steady_state
from free_6_scan_worker import optimize_free6


def _load_opt_S(opt_dir, ig, igp):
    fp = os.path.join(opt_dir, f'opt_S_{ig:04d}_{igp:04d}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',    type=int, required=True)
    p.add_argument('--n_pts',      type=int, default=41)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str, default='results_free6')
    p.add_argument('--opt_dir',    type=str, default='results_opt',
                   help='Directory containing opt_S_<ig>_<igp>.npy seeds.')
    p.add_argument('--n_restarts', type=int, default=10)
    p.add_argument('--maxfev',     type=int, default=2000)
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing output row file.')
    args = p.parse_args()

    ig = args.task_id
    if ig < 0 or ig >= args.n_pts:
        print(f'ERROR: task_id {ig} out of range', file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'free6row_{ig:04d}.npy')
    if os.path.exists(out_path) and not args.force:
        print(f'Skip: {out_path} exists (use --force to overwrite)', flush=True)
        return

    gamma = args.gamma_step * ig
    n = args.n_pts
    row = np.full(n, np.nan)
    dt = 0.01 * args.gamma_step

    print(f'[task {ig}] gamma={gamma:.4f}, {n} points, d_ext_single=6 (free), '
          f'seeded from {args.opt_dir}', flush=True)

    for igp in range(n):
        gp = args.gamma_step * igp
        _, L = compute_steady_state(args.J, gamma, gp)
        gate = expm(dt * L).real

        seed_S = _load_opt_S(args.opt_dir, ig, igp)
        extra_init_xs = [seed_S.ravel()] if seed_S is not None else None
        seed_tag = 'd6-seeded' if seed_S is not None else 'no-seed'

        fra, _ = optimize_free6(
            gate,
            n_restarts=args.n_restarts,
            seed=ig * 10000 + igp,
            maxfev=args.maxfev,
            extra_init_xs=extra_init_xs,
        )
        row[igp] = fra
        print(f'[task {ig}] col {igp+1}/{n}  gp={gp:.4f}  fra={fra:.6f}  [{seed_tag}]',
              flush=True)

    np.save(out_path, row)
    print(f'[task {ig}] saved {out_path}', flush=True)


if __name__ == '__main__':
    main()
