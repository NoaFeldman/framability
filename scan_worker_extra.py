"""
Worker script for extra steady-state properties per grid point.

Each SLURM array task processes ONE (gamma, gamma') grid point:
    ig  = task_id // n_pts   (gamma index)
    igp = task_id %  n_pts   (gamma' index)

Result saved to:
    <out_dir>/point_extra_<ig:04d>_<igp:04d>.npy   shape (4,)

    [0]  ss_bond_entropy  Bond entropy of the steady-state LPDO
    [1]  mag_x            sum_i Tr(rho_ss @ X_i)
    [2]  stabilizer_fra   Schrödinger framability w.r.t. the dyadic stabilizer frame
    [3]  product_fra      Schrödinger framability w.r.t. a random product-state frame

Total tasks to submit: n_pts * n_pts  (e.g. 1681 for a 41x41 grid).

Usage
-----
    python scan_worker_extra.py --task_id 42 --n_pts 41 --J 1.0 \\
                                --gamma_step 0.2 --out_dir results
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm

from analysis import (compute_steady_state, compute_magnetization_x,
                      compute_ss_bond_entropy)
from framability import dyadic_stabilizer_framability, product_state_framability

PRODUCT_CHI = 6   # number of random single-qubit states for product-state frame


def main():
    p = argparse.ArgumentParser(
        description='Compute extra properties for one (gamma, gamma\') grid point.'
    )
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Flat grid index: ig = task_id // n_pts, '
                        'igp = task_id %% n_pts.  Maps to SLURM_ARRAY_TASK_ID.')
    p.add_argument('--n_pts',      type=int,   default=41,
                   help='Number of grid points along each axis (default: 41).')
    p.add_argument('--J',          type=float, default=1.0,
                   help='Coupling constant J (default: 1.0).')
    p.add_argument('--gamma_step', type=float, default=0.2,
                   help='Grid spacing for gamma and gamma\' (default: 0.2).')
    p.add_argument('--out_dir',    type=str,   default='results',
                   help='Directory to write output files (default: results/).')
    p.add_argument('--N',          type=int,   default=2,
                   help='Number of qubits (default: 2).')
    args = p.parse_args()

    n = args.n_pts
    total = n * n
    tid = args.task_id
    if tid < 0 or tid >= total:
        print(f'ERROR: task_id {tid} out of range [0, {total - 1}]',
              file=sys.stderr)
        sys.exit(1)

    ig  = tid // n
    igp = tid %  n

    os.makedirs(args.out_dir, exist_ok=True)

    gamma    = args.gamma_step * ig
    gp       = args.gamma_step * igp
    N_qubits = args.N

    print(f'[task {tid}] ig={ig} igp={igp}  gamma={gamma:.4f}  '
          f'gamma\'={gp:.4f}  (N={N_qubits})', flush=True)

    rho_ss, L = compute_steady_state(args.J, gamma, gp, N=N_qubits)

    ss_ent = compute_ss_bond_entropy(rho_ss, N=N_qubits)
    mag_x  = compute_magnetization_x(rho_ss, N=N_qubits)

    dt   = 0.01 * args.gamma_step
    gate = expm(dt * L).real

    stab_fra    = dyadic_stabilizer_framability(gate, n_qubits=N_qubits)
    product_fra = product_state_framability(PRODUCT_CHI, gate)

    out_path = os.path.join(args.out_dir, f'point_extra_{ig:04d}_{igp:04d}.npy')
    np.save(out_path, np.array([ss_ent, mag_x, stab_fra, product_fra]))
    print(f'[task {tid}] ss_ent={ss_ent:.6f}  mag_x={mag_x:.6f}  '
          f'stab_fra={stab_fra:.6f}  product_fra={product_fra:.6f}  -> {out_path}',
          flush=True)


if __name__ == '__main__':
    main()
