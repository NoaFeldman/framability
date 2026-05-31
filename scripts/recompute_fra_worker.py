"""
Worker: recompute optimised framability for one (gamma, gamma') grid point
using the updated frame structure:
  - First 2 columns of S fixed to identity (1,0,0,0)^T and Z (0,0,0,1)^T
  - Column normalisation: |c_I| + ||(c_X,c_Y,c_Z)||_2 = 1

Output
------
    <out_dir>/opt_fra_<ig:04d>_<igp:04d>.npy   shape (2,): [fra, fra_pauli]
    where fra_pauli = max row-L1-norm of gate (unchanged from original scan).

Usage
-----
    python recompute_fra_worker.py --task_id 42 --n_pts 41 \
                                   --J 1.0 --gamma_step 0.2 --out_dir results_opt
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from framability import extended_pauli_D, heisenberg_framability
from optimize_framability import minimize_framability, DEFAULT_METHOD


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Flat index ig*n_pts+igp; maps to SLURM_ARRAY_TASK_ID.')
    p.add_argument('--n_pts',      type=int,   default=41)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str,   default='results_opt')
    p.add_argument('--n_restarts', type=int,   default=5)
    p.add_argument('--maxfev',     type=int,   default=1000)
    args = p.parse_args()

    n   = args.n_pts
    ig  = args.task_id // n
    igp = args.task_id %  n

    if ig >= n:
        print(f'ERROR: task_id {args.task_id} out of range for {n}x{n} grid',
              file=sys.stderr)
        sys.exit(1)

    gamma  = args.gamma_step * ig
    gp     = args.gamma_step * igp
    dt     = 0.01 * args.gamma_step

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'opt_fra_{ig:04d}_{igp:04d}.npy')

    L    = numeric_two_qubit_lindbladian(J=args.J, gamma=gamma, gamma_p=gp)
    gate = expm(dt * L).real

    pauli_fra = float(np.max(np.sum(np.abs(gate), axis=1)))

    D_ext = extended_pauli_D()
    d_ext_single = int(round(np.sqrt(D_ext.shape[1])))  # 6

    _, min_fra = minimize_framability(
        gate, d_ext_single=d_ext_single,
        n_restarts=args.n_restarts,
        method=DEFAULT_METHOD,
        max_iter=200, maxfev=args.maxfev,
        verbose=False,
    )
    # also evaluate extended-Pauli D as a sanity floor
    ext_fra = heisenberg_framability(D_ext, gate)
    min_fra = float(min(min_fra, ext_fra))

    np.save(out_path, np.array([min_fra, pauli_fra]))
    print(f'({ig},{igp}) gamma={gamma:.3f} gp={gp:.3f}  '
          f'min_fra={min_fra:.6f}  pauli_fra={pauli_fra:.6f}', flush=True)


if __name__ == '__main__':
    main()
