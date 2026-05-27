"""
Collect bond-entropy refinement results and update point_extra files.

Reads refine_maxbond_<ig:04d>_<igp:04d>.npy files produced by
bond_entropy_refine_worker.py (fidelity_threshold=0.99) for all points
with igp < n_igp (gamma' < n_igp * gamma_step).

For each point, if the refined value exceeds the stored value in
point_extra_<ig:04d>_<igp:04d>.npy[6] (max_bond_entropy), it is
replaced by the larger value, then scan_collect.py is re-run to
rebuild scan_full.npy.

Usage
-----
    python bond_entropy_refine_collect.py \
        --n_pts 41 --n_igp 20 --J 1.0 --gamma_step 0.2 --out_dir results
"""

import argparse
import os
import subprocess
import sys

import numpy as np

# Index of max_bond_entropy inside point_extra files
COL_MAX_BOND = 6


def main():
    p = argparse.ArgumentParser(
        description='Collect bond-entropy refinement outputs and update '
                    'point_extra files.'
    )
    p.add_argument('--n_pts',      type=int,   default=41)
    p.add_argument('--n_igp',      type=int,   default=20,
                   help='Number of restricted gamma\' columns (igp < n_igp).')
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str,   default='results')
    args = p.parse_args()

    n       = args.n_pts
    n_igp   = args.n_igp
    n_improved = 0
    n_missing  = 0
    pts_touched = set()

    for ig in range(n):
        for igp in range(n_igp):
            refine_path = os.path.join(
                args.out_dir, f'refine_maxbond_{ig:04d}_{igp:04d}.npy')
            if not os.path.exists(refine_path):
                n_missing += 1
                continue

            new_val = float(np.load(refine_path))
            if not np.isfinite(new_val):
                continue

            extra_path = os.path.join(
                args.out_dir, f'point_extra_{ig:04d}_{igp:04d}.npy')
            if not os.path.exists(extra_path):
                print(f'WARNING: {extra_path} not found – skipping ({ig},{igp})',
                      file=sys.stderr)
                continue

            extra = np.load(extra_path)
            old_val = float(extra[COL_MAX_BOND])

            if new_val > old_val + 1e-12:
                extra[COL_MAX_BOND] = new_val
                np.save(extra_path, extra)
                pts_touched.add((ig, igp))
                n_improved += 1
                print(f'  ({ig:2d},{igp:2d}) max_bond_entropy: '
                      f'{old_val:.6f} -> {new_val:.6f}  '
                      f'(+{new_val - old_val:.6f})')

    print(f'\nUpdated {n_improved} point(s).')
    if n_missing:
        print(f'({n_missing} refinement files missing – those points skipped)')

    # Re-run scan_collect.py to rebuild scan_full.npy with updated values
    print('\nRegenerating scan_full.npy ...')
    subprocess.check_call([
        sys.executable, 'scan_collect.py',
        '--n_pts',      str(args.n_pts),
        '--J',          str(args.J),
        '--gamma_step', str(args.gamma_step),
        '--out_dir',    args.out_dir,
    ])


if __name__ == '__main__':
    main()
