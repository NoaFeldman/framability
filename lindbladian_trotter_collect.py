"""
Collect per-task .npz files from lindbladian_trotter_worker.py and
produce a summary array.

Output: <out_dir>/trotter_summary.npz
  keys: framability (N_D, N_GAMMA, N_GP),
        gamma_values (N_GAMMA,), gp_values (N_GP,),
        d_ext_singles (N_D,), J, dt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

D_EXT_SINGLES = [4, 6]
N_D           = len(D_EXT_SINGLES)

GAMMA_MAX  = 8.0
GP_MAX     = 4.0
GAMMA_STEP = 0.2

N_GAMMA = int(round(GAMMA_MAX / GAMMA_STEP)) + 1   # 41
N_GP    = int(round(GP_MAX   / GAMMA_STEP)) + 1    # 21


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_trotter')
    parser.add_argument('--out_dir', type=str, default='results_trotter')
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fra = np.full((N_D, N_GAMMA, N_GP), np.nan)
    J_val  = np.nan
    dt_val = np.nan

    for di, d in enumerate(D_EXT_SINGLES):
        for ig in range(N_GAMMA):
            for igp in range(N_GP):
                f = in_dir / f'trotter_{d}_{ig:03d}_{igp:03d}.npz'
                if not f.exists():
                    print(f'[missing] {f}')
                    continue
                data = np.load(f, allow_pickle=True)
                fra[di, ig, igp] = float(data['framability'])
                if np.isnan(J_val):
                    J_val  = float(data['J'])
                    dt_val = float(data['dt'])

    gamma_values = GAMMA_STEP * np.arange(N_GAMMA)
    gp_values    = GAMMA_STEP * np.arange(N_GP)

    out_path = out_dir / 'trotter_summary.npz'
    np.savez(out_path,
             framability=fra,
             gamma_values=gamma_values,
             gp_values=gp_values,
             d_ext_singles=np.array(D_EXT_SINGLES),
             J=np.array(J_val),
             dt=np.array(dt_val))
    print(f'[saved] {out_path}')
    for di, d in enumerate(D_EXT_SINGLES):
        n_ok = int(np.sum(np.isfinite(fra[di])))
        print(f'  d_ext_single={d}: {n_ok}/{N_GAMMA * N_GP} tasks done')


if __name__ == '__main__':
    main()
