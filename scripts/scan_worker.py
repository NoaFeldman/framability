"""
Worker script for the embarrassingly-parallel 2-D parameter scan.

Each SLURM array task processes one row of the (gamma, gamma') grid
(fixed gamma index = SLURM_ARRAY_TASK_ID) and saves results to

    <out_dir>/row_<task_id:04d>.npy   shape (n_pts, 5)

Column order (axis-1):
    0  entropy          Von Neumann entropy
    1  negativity       Negativity from partial transpose
    2  pauli_fra        Max-row 1-norm of expm(dt*L)
    3  min_fra          Optimised framability (Kronecker frame)
    4  dec_rate         Spectral gap of the Lindbladian

Usage
-----
    python scan_worker.py --task_id 5 --n_pts 41 --J 1.0 \\
                          --gamma_step 0.2 --out_dir results
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from framability import extended_pauli_D, heisenberg_framability
from optimize_framability import minimize_framability, DEFAULT_METHOD
from analysis import compute_steady_state, decay_rate


def main():
    p = argparse.ArgumentParser(
        description='Compute one row of the (gamma, gamma\') scan grid.'
    )
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Row index (= gamma index) to process; '
                        'maps to SLURM_ARRAY_TASK_ID.')
    p.add_argument('--n_pts',      type=int,   default=41,
                   help='Number of grid points along each axis (default: 41).')
    p.add_argument('--J',          type=float, default=1.0,
                   help='Coupling constant J (default: 1.0).')
    p.add_argument('--gamma_step', type=float, default=0.2,
                   help='Grid spacing for gamma and gamma\' (default: 0.2).')
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
    row_data = np.zeros((n, 5))

    print(f'[task {ig}] gamma={gamma:.4f}, processing {n} gamma\' values',
          flush=True)

    D_ext = extended_pauli_D()
    d_ext_single = int(round(np.sqrt(D_ext.shape[1])))

    for igp in range(n):
        gp = args.gamma_step * igp

        rho_ss, L = compute_steady_state(args.J, gamma, gp)

        # Von Neumann entropy
        evals = np.linalg.eigvalsh(rho_ss)
        evals_pos = evals[evals > 1e-15]
        entropy = float(-np.sum(evals_pos * np.log(evals_pos)))

        # Negativity
        rho_pt = (rho_ss.reshape([2, 2, 2, 2])
                         .transpose([0, 3, 2, 1])
                         .reshape([4, 4]))
        evals_pt = np.linalg.eigvalsh(rho_pt)
        negativity = float(np.sum(np.abs(evals_pt[evals_pt < -1e-15])))

        # Gate
        dt   = 0.01 * args.gamma_step
        gate = expm(dt * L).real

        # Pauli framability (max row 1-norm)
        pauli_fra = float(np.max(np.sum(np.abs(gate), axis=1)))

        # Optimised framability
        _, min_fra = minimize_framability(
            gate, d_ext_single=d_ext_single, n_restarts=5,
            method=DEFAULT_METHOD, max_iter=200, maxfev=1000,
            verbose=False,
        )
        ext_fra = heisenberg_framability(D_ext, gate)
        min_fra = float(min(min_fra, ext_fra))

        # Decay rate
        dr = float(decay_rate(L))

        row_data[igp] = [entropy, negativity, pauli_fra, min_fra, dr]
        print(f'[task {ig}] col {igp + 1}/{n}  gamma\'={gp:.4f}  done',
              flush=True)

    out_path = os.path.join(args.out_dir, f'row_{ig:04d}.npy')
    np.save(out_path, row_data)
    print(f'[task {ig}] wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
