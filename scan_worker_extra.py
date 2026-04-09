"""
Worker script for extra steady-state properties (max LPDO bond dim, magnetization).

Each SLURM array task processes one row of the (gamma, gamma') grid
(fixed gamma index = SLURM_ARRAY_TASK_ID) and saves results to

    <out_dir>/row_extra_<task_id:04d>.npy   shape (n_pts, 2)

Column order (axis-1):
    0  max_bond_dim     Maximum LPDO bond dimension during time evolution
    1  magnetization    Tr(rho_ss @ Z^{otimes N})

Usage
-----
    python scan_worker_extra.py --task_id 5 --n_pts 20 --J 1.0 \\
                                --gamma_step 0.1 --out_dir results
"""

import argparse
import os
import sys

import numpy as np

from analysis import compute_steady_state, compute_max_bond_dim, compute_magnetization


def main():
    p = argparse.ArgumentParser(
        description='Compute extra properties for one row of the scan grid.'
    )
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Row index (= gamma index); maps to SLURM_ARRAY_TASK_ID.')
    p.add_argument('--n_pts',      type=int,   default=20,
                   help='Number of grid points along each axis (default: 20).')
    p.add_argument('--J',          type=float, default=1.0,
                   help='Coupling constant J (default: 1.0).')
    p.add_argument('--gamma_step', type=float, default=0.1,
                   help='Grid spacing for gamma and gamma\' (default: 0.1).')
    p.add_argument('--out_dir',    type=str,   default='results',
                   help='Directory to write output files (default: results/).')
    p.add_argument('--N',          type=int,   default=2,
                   help='Number of qubits (default: 2).')
    args = p.parse_args()

    ig = args.task_id
    if ig < 0 or ig >= args.n_pts:
        print(f'ERROR: task_id {ig} out of range [0, {args.n_pts - 1}]',
              file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    gamma    = args.gamma_step * ig
    n        = args.n_pts
    N_qubits = args.N
    row_data = np.zeros((n, 2))

    print(f'[task {ig}] gamma={gamma:.4f}, processing {n} gamma\' values '
          f'(N={N_qubits})', flush=True)

    for igp in range(n):
        gp = args.gamma_step * igp

        rho_ss, L = compute_steady_state(args.J, gamma, gp, N=N_qubits)

        max_chi = compute_max_bond_dim(L, rho_ss, args.gamma_step,
                                       N=N_qubits)
        mag = compute_magnetization(rho_ss, N=N_qubits)

        row_data[igp] = [max_chi, mag]
        print(f'[task {ig}] col {igp + 1}/{n}  gamma\'={gp:.4f}  '
              f'max_chi={max_chi}  mag={mag:.6f}', flush=True)

    out_path = os.path.join(args.out_dir, f'row_extra_{ig:04d}.npy')
    np.save(out_path, row_data)
    print(f'[task {ig}] wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
