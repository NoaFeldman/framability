"""
Worker script: recompute max LPDO bond entropy with a higher fidelity threshold.

Each SLURM array task processes one restricted grid point (ig, igp) where
igp < n_igp (i.e. gamma' < n_igp * gamma_step, typically gamma' < 4).

Task-id mapping:
    ig  = task_id // n_igp      (full gamma row, 0 .. n_pts-1)
    igp = task_id %  n_igp      (restricted gamma' column, 0 .. n_igp-1)

Output
------
    <out_dir>/refine_maxbond_<ig:04d>_<igp:04d>.npy   scalar float
        Maximum LPDO bond entropy along the trajectory from the initial
        state to the steady state, with fidelity_threshold=0.99.

Usage
-----
    python bond_entropy_refine_worker.py \
        --task_id 42 --n_pts 41 --n_igp 20 \
        --J 1.0 --gamma_step 0.2 --out_dir results
"""

import argparse
import os
import sys

import numpy as np

from analysis import compute_steady_state, compute_max_bond_dim

FIDELITY_THRESHOLD = 0.99


def main():
    p = argparse.ArgumentParser(
        description='Recompute max LPDO bond entropy at fidelity_threshold=0.99 '
                    'for a restricted (ig, igp) grid point.'
    )
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Flat index ig*n_igp + igp; maps to SLURM_ARRAY_TASK_ID.')
    p.add_argument('--n_pts',      type=int,   default=41,
                   help='Full grid size (default: 41).')
    p.add_argument('--n_igp',      type=int,   default=20,
                   help='Number of restricted gamma\' columns (default: 20, '
                        'i.e. gamma\' < n_igp * gamma_step = 4.0 for step=0.2).')
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str,   default='results')
    p.add_argument('--N',          type=int,   default=2,
                   help='Number of qubits (default: 2).')
    args = p.parse_args()

    n     = args.n_pts
    n_igp = args.n_igp
    total = n * n_igp
    tid   = args.task_id

    if tid < 0 or tid >= total:
        print(f'ERROR: task_id {tid} out of range [0, {total - 1}]',
              file=sys.stderr)
        sys.exit(1)

    ig  = tid // n_igp
    igp = tid %  n_igp

    if ig >= n or igp >= n:
        print(f'ERROR: (ig={ig}, igp={igp}) out of bounds for {n}x{n} grid',
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir,
                            f'refine_maxbond_{ig:04d}_{igp:04d}.npy')

    gamma = args.gamma_step * ig
    gp    = args.gamma_step * igp

    print(f'[task {tid}] ig={ig} igp={igp}  gamma={gamma:.4f}  '
          f"gamma'={gp:.4f}  fidelity_threshold={FIDELITY_THRESHOLD}",
          flush=True)

    rho_ss, L = compute_steady_state(args.J, gamma, gp, N=args.N)

    _, max_bond_entropy = compute_max_bond_dim(
        L, rho_ss, args.gamma_step,
        N=args.N,
        fidelity_threshold=FIDELITY_THRESHOLD,
    )

    np.save(out_path, np.float64(max_bond_entropy))
    print(f'[task {tid}] max_bond_entropy={max_bond_entropy:.6f}  -> {out_path}',
          flush=True)


if __name__ == '__main__':
    main()
