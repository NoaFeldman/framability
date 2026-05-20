"""
Collect trotter_nb_<d>_<ig:03d>_<igp:03d>.npz files produced by
trotter_nb_refine_worker.py, take the element-wise minimum of the
original framability and the refined values, update the per-task npz
files where improved, and rewrite trotter_summary.npz.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

D_EXT_SINGLES = [4, 6]
N_D           = len(D_EXT_SINGLES)
GAMMA_MAX     = 8.0
GP_MAX        = 4.0
GAMMA_STEP    = 0.2
N_GAMMA       = int(round(GAMMA_MAX  / GAMMA_STEP)) + 1
N_GP          = int(round(GP_MAX     / GAMMA_STEP)) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_trotter')
    parser.add_argument('--out_dir', type=str, default='results_trotter')
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load original summary
    summary_path = in_dir / 'trotter_summary.npz'
    if not summary_path.exists():
        print(f'ERROR: {summary_path} not found — run lindbladian_trotter_collect.py first',
              flush=True)
        raise SystemExit(1)
    summary = np.load(summary_path)
    fra = summary['framability'].copy()   # (N_D, N_GAMMA, N_GP)

    n_improved = 0
    n_missing  = 0

    for di, d in enumerate(D_EXT_SINGLES):
        for ig in range(N_GAMMA):
            for igp in range(N_GP):
                nb_path = in_dir / f'trotter_nb_{d}_{ig:03d}_{igp:03d}.npz'
                if not nb_path.exists():
                    n_missing += 1
                    continue
                data     = np.load(nb_path)
                f_nb     = float(data['framability'])
                improved = bool(data.get('improved', False))
                if not improved:
                    continue
                old = fra[di, ig, igp]
                if f_nb < old - 1e-9:
                    fra[di, ig, igp] = f_nb
                    n_improved += 1
                    # Overwrite the original per-task file so next refinement
                    # round seeds from the best-known x.
                    orig_path = out_dir / f'trotter_{d}_{ig:03d}_{igp:03d}.npz'
                    if orig_path.exists():
                        orig = np.load(orig_path)
                        np.savez(orig_path,
                                 framability=np.array(f_nb),
                                 D=data['D'],
                                 x=data['x'],
                                 d_ext_single=orig['d_ext_single'],
                                 gamma=orig['gamma'],
                                 gamma_p=orig['gamma_p'],
                                 J=orig['J'],
                                 dt=orig['dt'])
                    print(f'  ({d},{ig:03d},{igp:03d}) {old:.6f} → {f_nb:.6f}  '
                          f'(Δ={old-f_nb:.6f})')

    # Rewrite summary
    out_path = out_dir / 'trotter_summary.npz'
    np.savez(out_path,
             framability=fra,
             gamma_values=summary['gamma_values'],
             gp_values=summary['gp_values'],
             d_ext_singles=summary['d_ext_singles'],
             J=summary['J'],
             dt=summary['dt'])

    print(f'\n[saved] {out_path}')
    print(f'Improved {n_improved} point(s).')
    if n_missing:
        print(f'({n_missing} refinement files missing — those points were not updated)')

    for di, d in enumerate(D_EXT_SINGLES):
        arr = fra[di][np.isfinite(fra[di])]
        if arr.size:
            print(f'  d_ext_single={d}: min={arr.min():.4f}  '
                  f'max={arr.max():.4f}  median={np.median(arr):.4f}')


if __name__ == '__main__':
    main()
