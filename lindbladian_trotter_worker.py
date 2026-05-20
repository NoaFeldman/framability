"""
Worker: minimize framability of the two-qubit Lindbladian Trotter step
exp(L*dt) for one (d_ext_single, gamma, gamma_p) grid point.

task_id = d_idx * N_GAMMA * N_GP + ig * N_GP + igp
  d_idx  in 0..N_D-1    D_EXT_SINGLES = [4, 6]
  ig     in 0..N_GAMMA-1 gamma   = GAMMA_STEP * ig,  up to GAMMA_MAX=8
  igp    in 0..N_GP-1    gamma_p = GAMMA_STEP * igp, up to GP_MAX=4

Total tasks: 2 * 41 * 21 = 1722

Output: <out_dir>/trotter_<d>_<ig:03d>_<igp:03d>.npz
  keys: framability, D (16, d_ext), x, gamma, gamma_p, d_ext_single, J, dt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from optimize_framability import minimize_framability, DEFAULT_METHOD

D_EXT_SINGLES = [4, 6]
N_D           = len(D_EXT_SINGLES)

GAMMA_MAX  = 8.0
GP_MAX     = 4.0
GAMMA_STEP = 0.2

N_GAMMA = int(round(GAMMA_MAX / GAMMA_STEP)) + 1   # 41
N_GP    = int(round(GP_MAX   / GAMMA_STEP)) + 1    # 21


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',    type=int,   required=True,
                        help=f'0..{N_D * N_GAMMA * N_GP - 1}')
    parser.add_argument('--out_dir',    type=str,   default='results_trotter')
    parser.add_argument('--n_restarts', type=int,   default=10)
    parser.add_argument('--maxfev',     type=int,   default=2000)
    parser.add_argument('--max_iter',   type=int,   default=500)
    parser.add_argument('--J',          type=float, default=1.0)
    parser.add_argument('--dt',         type=float, default=0.01)
    parser.add_argument('--method',     type=str,   default=DEFAULT_METHOD)
    parser.add_argument('--seed',       type=int,   default=0)
    args = parser.parse_args()

    total = N_D * N_GAMMA * N_GP
    if not (0 <= args.task_id < total):
        print(f'ERROR: task_id {args.task_id} out of range (0..{total - 1})',
              file=sys.stderr)
        sys.exit(1)

    d_idx = args.task_id // (N_GAMMA * N_GP)
    rem   = args.task_id  % (N_GAMMA * N_GP)
    ig    = rem // N_GP
    igp   = rem  % N_GP

    d_ext_single = D_EXT_SINGLES[d_idx]
    gamma   = GAMMA_STEP * ig
    gamma_p = GAMMA_STEP * igp

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f'trotter_{d_ext_single}_{ig:03d}_{igp:03d}.npz'
    )
    if os.path.exists(out_path):
        print(f'Skip: {out_path} already exists', flush=True)
        return

    print(f'[task {args.task_id}] d_ext_single={d_ext_single}  '
          f'gamma={gamma:.2f}  gamma_p={gamma_p:.2f}', flush=True)

    L    = numeric_two_qubit_lindbladian(J=args.J, gamma=gamma, gamma_p=gamma_p)
    gate = expm(L * args.dt).real

    t0 = time.perf_counter()
    D_opt, f_opt, x_opt = minimize_framability(
        gate, d_ext_single=d_ext_single,
        n_restarts=args.n_restarts,
        method=args.method,
        max_iter=args.max_iter,
        maxfev=args.maxfev,
        seed=args.seed + args.task_id,
        verbose=True,
        return_x=True,
    )
    elapsed = time.perf_counter() - t0

    np.savez(out_path,
             framability=np.array(f_opt),
             D=D_opt,
             x=x_opt,
             d_ext_single=np.array(d_ext_single),
             gamma=np.array(gamma),
             gamma_p=np.array(gamma_p),
             J=np.array(args.J),
             dt=np.array(args.dt))
    print(f'[task {args.task_id}] saved {out_path}  '
          f'fra={f_opt:.6f}  elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
