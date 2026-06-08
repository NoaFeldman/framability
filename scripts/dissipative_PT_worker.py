"""
Per-point cluster worker for the 2D dissipative phase transition scan.

Parameter grid (J=1 fixed):
    h      in [0.1*i for i in range(-10, 10)]  ->  -1.0 … 0.9, N_H=20
    gamma  in [0.1*i for i in range(10)]        ->   0.0 … 0.9, N_G=10

task_id = ih * N_G + ig   (0 … N_H*N_G-1 = 199)

Output: <out_dir>/dpt_<ih:02d>_<ig:02d>.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dissipative_PT import (compute_point,
                             H_LIST, GAMMA_LIST, N_H, N_G, N_TOTAL,
                             J_DEFAULT, DT_DEFAULT)

J  = J_DEFAULT
DT = DT_DEFAULT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',      type=int,   required=True,
                   help=f'0..{N_TOTAL - 1}')
    p.add_argument('--out_dir',      type=str,   default='results_dpt')
    p.add_argument('--dt',           type=float, default=DT)
    p.add_argument('--fra_restarts', type=int,   default=5)
    p.add_argument('--fra_maxfev_4', type=int,   default=1000)
    p.add_argument('--fra_maxfev_6', type=int,   default=500)
    p.add_argument('--sign_restarts',type=int,   default=10)
    p.add_argument('--seed',         type=int,   default=0)
    args = p.parse_args()

    if not (0 <= args.task_id < N_TOTAL):
        print(f'ERROR: task_id must be in [0, {N_TOTAL})', file=sys.stderr)
        sys.exit(1)

    ih    = args.task_id // N_G
    ig    = args.task_id %  N_G
    h     = H_LIST[ih]
    gamma = GAMMA_LIST[ig]
    out   = Path(args.out_dir) / f'dpt_{ih:02d}_{ig:02d}.npz'

    if out.exists():
        print(f'[skip] {out.name} already exists', flush=True)
        return

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print(f'[task {args.task_id}/{N_TOTAL}] ih={ih} ig={ig} '
          f'h={h:.2f} gamma={gamma:.2f} J={J} dt={args.dt}', flush=True)

    res = compute_point(
        h=h, J=J, gamma=gamma, dt=args.dt,
        fra_restarts=args.fra_restarts,
        fra_maxfev_4=args.fra_maxfev_4,
        fra_maxfev_6=args.fra_maxfev_6,
        sign_restarts=args.sign_restarts,
        seed=args.seed + args.task_id,
        Lx=2, Ly=2, verbose=True,
    )

    np.savez(out,
             h=np.array(h), J=np.array(J), gamma=np.array(gamma), dt=np.array(args.dt),
             pauli_fra  = np.array(res['pauli_fra']),
             opt_fra_4  = np.array(res['opt_fra_4']),
             opt_fra_6  = np.array(res['opt_fra_6']),
             sign_init  = np.array(res['sign_init']),
             sign_opt   = np.array(res['sign_opt']),
             chan_stab  = np.array(res['chan_stab']),
             decay_rate = np.array(res['decay_rate']),
             ss_vn      = np.array(res['ss_vn']),
             mean_mag   = np.array(res['mean_mag']),
             neg_half   = np.array(res['neg_half']),
             max_lpdo   = np.array(res['max_lpdo']),
             ih=np.array(ih), ig=np.array(ig))

    elapsed = time.perf_counter() - t_start
    print(f'  saved {out.name}  ({elapsed:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
