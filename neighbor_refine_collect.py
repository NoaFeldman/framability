"""
Collect neighbor-refine results and regenerate the scan figure.

Reads refine_nb_<ig:04d>_<igp:04d>.npy point files produced by
neighbor_refine_worker.py, updates min_fra (column 5) in the row files
where an improvement was found, then re-runs scan_collect.py to rebuild
scan_full.npy and two_qubit_scan.png.

Usage
-----
    python neighbor_refine_collect.py --n_pts 20 --J 1.0 \
                                      --gamma_step 0.1 --out_dir results
"""

import argparse
import os
import sys
import subprocess

import numpy as np

COL_MIN_FRA = 5


def main():
    p = argparse.ArgumentParser(
        description='Collect neighbor-refine results and update row files.'
    )
    p.add_argument('--n_pts',      type=int,   default=20)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.1)
    p.add_argument('--out_dir',    type=str,   default='results')
    args = p.parse_args()

    n = args.n_pts
    n_improved = 0
    n_missing  = 0
    rows_touched = set()

    for ig in range(n):
        row_path = os.path.join(args.out_dir, f'row_{ig:04d}.npy')
        if not os.path.exists(row_path):
            print(f'WARNING: {row_path} not found, skipping row {ig}',
                  file=sys.stderr)
            continue
        row = np.load(row_path)  # shape (n, 8)

        for igp in range(n):
            refine_path = os.path.join(
                args.out_dir, f'refine_nb_{ig:04d}_{igp:04d}.npy')
            if not os.path.exists(refine_path):
                n_missing += 1
                continue

            val = float(np.load(refine_path))
            if np.isfinite(val) and val < row[igp, COL_MIN_FRA] - 1e-9:
                old = row[igp, COL_MIN_FRA]
                row[igp, COL_MIN_FRA] = val
                rows_touched.add(ig)
                n_improved += 1
                print(f'  ({ig:2d},{igp:2d}) {old:.6f} → {val:.6f}  '
                      f'(Δ = {old - val:.6f})')

        if ig in rows_touched:
            np.save(row_path, row)

    print(f'\nUpdated {n_improved} point(s) across {len(rows_touched)} row(s).')
    if n_missing:
        print(f'({n_missing} refine point files missing — those points were skipped)')

    # Re-run scan_collect.py to rebuild scan_full.npy and the figure
    print('\nRegenerating scan_full.npy and two_qubit_scan.png ...')
    subprocess.check_call([
        sys.executable, 'scan_collect.py',
        '--n_pts',      str(args.n_pts),
        '--J',          str(args.J),
        '--gamma_step', str(args.gamma_step),
        '--out_dir',    args.out_dir,
    ])
    print('Done.')


if __name__ == '__main__':
    main()
