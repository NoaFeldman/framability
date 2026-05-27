"""
Collect unified neighbor-refine results and update the scan data files.

For each variant, reads refine_<variant>_<round>_<ig>_<igp>.npy point files,
updates the corresponding scan data, and optionally cleans up point files.

For d6:  updates results/row_XXXX.npy (col 3) and rebuilds scan_full.npy
For d4:  updates results_d4/d4row_XXXX.npy and rebuilds d4_scan.npy
For free6: updates results_free6/free6row_XXXX.npy and rebuilds free6_scan.npy

Usage:
    python unified_nb_refine_collect.py --variant free6 --round 1 \
        --n_pts 41 --n_igp 21 --out_dir results_free6
"""

import argparse
import os
import sys

import numpy as np


COL_MIN_FRA = 3  # column index in scan_full.npy row files


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant',    type=str, required=True,
                   choices=['d6', 'd4', 'free6'])
    p.add_argument('--round',      type=int, required=True)
    p.add_argument('--n_pts',      type=int, default=41)
    p.add_argument('--n_igp',      type=int, default=21)
    p.add_argument('--out_dir',    type=str, required=True)
    p.add_argument('--cleanup',    action='store_true',
                   help='Delete refine point files after collecting.')
    args = p.parse_args()

    n = args.n_pts
    n_igp = args.n_igp
    variant = args.variant
    rnd = args.round

    n_improved = 0
    n_missing = 0
    rows_touched = set()

    for ig in range(n):
        # Load the current row/grid data
        if variant == 'd6':
            row_path = os.path.join(args.out_dir, f'row_{ig:04d}.npy')
            if not os.path.exists(row_path):
                continue
            row = np.load(row_path)  # shape (n_pts, 5+)
        elif variant == 'd4':
            row_path = os.path.join(args.out_dir, f'd4row_{ig:04d}.npy')
            if not os.path.exists(row_path):
                continue
            row = np.load(row_path)  # shape (n_pts,)
        elif variant == 'free6':
            row_path = os.path.join(args.out_dir, f'free6row_{ig:04d}.npy')
            if not os.path.exists(row_path):
                continue
            row = np.load(row_path)  # shape (n_pts,)

        for igp in range(n_igp):
            tag = f'refine_{variant}_{rnd}_{ig:04d}_{igp:04d}'
            refine_path = os.path.join(args.out_dir, f'{tag}.npy')
            if not os.path.exists(refine_path):
                n_missing += 1
                continue

            val = float(np.load(refine_path))

            if variant == 'd6':
                old_val = row[igp, COL_MIN_FRA]
            else:
                old_val = row[igp]

            if np.isfinite(val) and val < old_val - 1e-9:
                if variant == 'd6':
                    row[igp, COL_MIN_FRA] = val
                else:
                    row[igp] = val
                rows_touched.add(ig)
                n_improved += 1
                print(f'  ({ig:2d},{igp:2d}) {old_val:.6f} → {val:.6f}  '
                      f'(Δ = {old_val - val:.6f})')

            if args.cleanup:
                os.remove(refine_path)

        if ig in rows_touched:
            np.save(row_path, row)

    print(f'\n[{variant} round {rnd}] Updated {n_improved} point(s) '
          f'across {len(rows_touched)} row(s).')
    if n_missing:
        print(f'({n_missing} refine point files missing — those points were skipped)')

    # Rebuild the aggregated scan file
    if variant == 'd6':
        print('\nRebuilding scan_full.npy via scan_collect.py ...')
        import subprocess
        subprocess.check_call([
            sys.executable, 'scan_collect.py',
            '--n_pts', str(n), '--out_dir', args.out_dir,
        ])
    elif variant == 'd4':
        print('\nRebuilding d4_scan.npy ...')
        grid = np.full((n, n), np.nan)
        for ig in range(n):
            f = os.path.join(args.out_dir, f'd4row_{ig:04d}.npy')
            if os.path.exists(f):
                grid[ig] = np.load(f)
        np.save(os.path.join(args.out_dir, 'd4_scan.npy'), grid)
        print(f'Saved {args.out_dir}/d4_scan.npy  shape={grid.shape}')
    elif variant == 'free6':
        print('\nRebuilding free6_scan.npy ...')
        grid = np.full((n, n), np.nan)
        for ig in range(n):
            f = os.path.join(args.out_dir, f'free6row_{ig:04d}.npy')
            if os.path.exists(f):
                grid[ig] = np.load(f)
        np.save(os.path.join(args.out_dir, 'free6_scan.npy'), grid)
        print(f'Saved {args.out_dir}/free6_scan.npy  shape={grid.shape}')

    print('Done.')


if __name__ == '__main__':
    main()
