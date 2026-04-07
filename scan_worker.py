"""
Worker script for the embarrassingly-parallel 2-D parameter scan.

Each SLURM array task processes one row of the (gamma, gamma') grid
(fixed gamma index = SLURM_ARRAY_TASK_ID) and saves results to

    <out_dir>/row_<task_id:04d>.npy   shape (n_pts, 8)

Column order (axis-1):
    0  entropy          Von Neumann entropy
    1  negativity       Negativity from partial transpose
    2  magic            Weighted-average stabilizer Renyi entropy
    3  pauli_fra        Max-row 1-norm of expm(dt*L)
    4  ext_fra          Extended Pauli framability
    5  min_fra          Minimal framability (Kronecker optimisation)
    6  dec_rate         Spectral gap of the Lindbladian
    7  chi              Minimal LPDO bond dimension

Usage
-----
    python scan_worker.py --task_id 5 --n_pts 20 --J 1.0 \\
                          --gamma_step 0.1 --out_dir results
"""

import argparse
import os
import sys

import numpy as np

from analysis import analyze_steady_state


def main():
    p = argparse.ArgumentParser(
        description='Compute one row of the (gamma, gamma\') scan grid.'
    )
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Row index (= gamma index) to process; '
                        'maps to SLURM_ARRAY_TASK_ID.')
    p.add_argument('--n_pts',      type=int,   default=20,
                   help='Number of grid points along each axis (default: 20).')
    p.add_argument('--J',          type=float, default=1.0,
                   help='Coupling constant J (default: 1.0).')
    p.add_argument('--gamma_step', type=float, default=0.1,
                   help='Grid spacing for gamma and gamma\' (default: 0.1).')
    p.add_argument('--out_dir',    type=str,   default='results',
                   help='Directory to write output files (default: results/).')
    args = p.parse_args()

    ig = args.task_id
    if ig < 0 or ig >= args.n_pts:
        print(f'ERROR: task_id {ig} out of range [0, {args.n_pts - 1}]',
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    gamma    = args.gamma_step * ig
    n        = args.n_pts
    row_data = np.zeros((n, 8))

    print(f'[task {ig}] gamma={gamma:.4f}, processing {n} gamma\' values',
          flush=True)

    for igp in range(n):
        gp = args.gamma_step * igp
        _, ent, neg, mag, pfra, efra, mfra, dr, chi = analyze_steady_state(
            args.J, gamma, gp, args.gamma_step
        )
        row_data[igp] = [ent, neg, mag, pfra, efra, mfra, dr, chi]
        print(f'[task {ig}] col {igp + 1}/{n}  gamma\'={gp:.4f}  done',
              flush=True)

    out_path = os.path.join(args.out_dir, f'row_{ig:04d}.npy')
    np.save(out_path, row_data)
    print(f'[task {ig}] wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
