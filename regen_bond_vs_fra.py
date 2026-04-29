"""
Regenerate results/two_qubit_scan_full_bond_vs_fra.png from the current
scan_full.npy without recomputing any scan data.

Usage
-----
    python regen_bond_vs_fra.py [--out_dir results] [--n_pts 41]
                                [--J 1.0] [--gamma_step 0.2]
                                [--out_name two_qubit_scan_full.png]
"""

import argparse
from pathlib import Path

import numpy as np

from build_two_qubit_scan_full import plot_bond_entropy_vs_framability


def main():
    p = argparse.ArgumentParser(
        description='Regenerate the bond-entropy vs framability figure '
                    'from the existing scan_full.npy.'
    )
    p.add_argument('--out_dir',    type=str,   default='results')
    p.add_argument('--n_pts',      type=int,   default=41)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_name',   type=str,   default='two_qubit_scan_full.png')
    args = p.parse_args()

    scan_path = Path(args.out_dir) / 'scan_full.npy'
    if not scan_path.exists():
        raise FileNotFoundError(
            f'{scan_path} not found. Run scan_collect.py first.'
        )

    scan_full = np.load(scan_path)
    print(f'Loaded {scan_path}  shape={scan_full.shape}')

    plot_bond_entropy_vs_framability(args, scan_full)


if __name__ == '__main__':
    main()
