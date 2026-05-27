"""
Collect minimax_toffoli_<d>_<pi:02d>.npz files written by
minimax_toffoli_worker.py into a single summary array.

Output: <out_dir>/minimax_toffoli_summary.npz
  keys: worst (N_D, N_P), framability (N_D, N_P, N_GATES),
        p_values (N_P,), d_ext_singles (N_D,), gates (N_GATES,)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

D_EXT_SINGLES = [4, 6]
P_VALUES      = [0.01 * i for i in range(11)]
N_D, N_P      = len(D_EXT_SINGLES), len(P_VALUES)
GATES         = ['H', 'Toffoli']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_minimax_toffoli')
    parser.add_argument('--out_dir', type=str, default='results_minimax_toffoli')
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    worst = np.full((N_D, N_P), np.nan)
    fra   = np.full((N_D, N_P, len(GATES)), np.nan)

    for di, d in enumerate(D_EXT_SINGLES):
        for pi in range(N_P):
            f = in_dir / f'minimax_toffoli_{d}_{pi:02d}.npz'
            if not f.exists():
                print(f'[missing] {f}')
                continue
            data = np.load(f, allow_pickle=True)
            worst[di, pi] = float(data['worst'])
            fra[di, pi]   = data['framability']

    out_path = out_dir / 'minimax_toffoli_summary.npz'
    np.savez(out_path,
             worst=worst,
             framability=fra,
             p_values=np.array(P_VALUES),
             d_ext_singles=np.array(D_EXT_SINGLES),
             gates=np.array(GATES))
    print(f'[saved] {out_path}')
    for di, d in enumerate(D_EXT_SINGLES):
        n_ok = int(np.sum(np.isfinite(worst[di])))
        print(f'  d_ext_single={d}: {n_ok}/{N_P} tasks done')


if __name__ == '__main__':
    main()
