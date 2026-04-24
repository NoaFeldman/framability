"""
Collect per-point prod_schro_pt results into product_fra_schro_chi030.npy
and regenerate the two_qubit_scan figure.

Per-point files written by product_schro_worker.py:
    <out_dir>/prod_schro_pt_XXXX_XXXX.npy   shape (1,) = [framability]

Output:
    <out_dir>/product_fra_schro_chi030.npy   shape (n_pts, n_pts)

Then calls scan_collect.py to regenerate <out_dir>/two_qubit_scan.png.

Usage
-----
    python product_schro_collect.py [--n_pts N] [--J J] [--gamma_step S]
                                    [--out_dir DIR] [--no_plot]
"""

import argparse
import os
import subprocess
import sys

import numpy as np

CHI = 30


def main():
    p = argparse.ArgumentParser(
        description='Assemble prod_schro_pt files and regenerate figure.'
    )
    p.add_argument('--n_pts',      type=int,   default=41)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str,   default='results')
    p.add_argument('--no_plot',    action='store_true',
                   help='Skip the scan_collect.py step.')
    args = p.parse_args()

    n       = args.n_pts
    total   = n * n
    grid    = np.full((n, n), np.nan)
    missing = 0

    for tid in range(total):
        ig  = tid // n
        igp = tid % n
        fpath = os.path.join(args.out_dir,
                             f'prod_schro_pt_{ig:04d}_{igp:04d}.npy')
        if os.path.exists(fpath):
            grid[ig, igp] = float(np.load(fpath)[0])
        else:
            missing += 1

    if missing:
        print(f'WARNING: {missing}/{total} point files missing '
              f'(those entries filled with NaN).', file=sys.stderr)
    else:
        print(f'All {total} point files found.')

    out_path = os.path.join(args.out_dir, f'product_fra_schro_chi{CHI:03d}.npy')
    np.save(out_path, grid)
    valid = int(np.sum(~np.isnan(grid)))
    print(f'Saved {out_path}  '
          f'({valid}/{total} valid,  '
          f'min={np.nanmin(grid):.4f}  max={np.nanmax(grid):.4f})')

    if not args.no_plot:
        cmd = [
            sys.executable, 'scan_collect.py',
            '--n_pts',      str(args.n_pts),
            '--J',          str(args.J),
            '--gamma_step', str(args.gamma_step),
            '--out_dir',    args.out_dir,
        ]
        print(f'\nRunning: {" ".join(cmd)}', flush=True)
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
