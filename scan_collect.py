"""
Collect per-row results produced by scan_worker.py and generate colourmap plots.

Expects files  <out_dir>/row_0000.npy … row_<n_pts-1:04d>.npy,
each of shape (n_pts, 8) with columns:
    0 entropy  1 negativity  2 magic  3 pauli_fra
    4 ext_fra  5 min_fra     6 dec_rate  7 chi

Usage
-----
    python scan_collect.py --n_pts 20 --J 1.0 --gamma_step 0.1 --out_dir results
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


METRICS = [
    (0, 'Von Neumann entropy'),
    (1, 'Negativity'),
    (2, 'Magic (avg SRE)'),
    (3, 'Pauli framability'),
    (4, 'Extended Pauli framability'),
    (5, 'Min. framability'),
    (6, 'Decay rate'),
    (7, r'LPDO bond dim $\chi$'),
]


def main():
    p = argparse.ArgumentParser(
        description='Aggregate scan_worker outputs and produce colourmap figure.'
    )
    p.add_argument('--n_pts',      type=int,   default=20)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.1)
    p.add_argument('--out_dir',    type=str,   default='results')
    args = p.parse_args()

    n = args.n_pts

    # Load all row files ---------------------------------------------------
    missing = [
        os.path.join(args.out_dir, f'row_{ig:04d}.npy')
        for ig in range(n)
        if not os.path.exists(os.path.join(args.out_dir, f'row_{ig:04d}.npy'))
    ]
    if missing:
        print('ERROR: missing result files:', file=sys.stderr)
        for f in missing:
            print(f'  {f}', file=sys.stderr)
        sys.exit(1)

    # data[ig, igp, metric]
    data = np.stack([
        np.load(os.path.join(args.out_dir, f'row_{ig:04d}.npy'))
        for ig in range(n)
    ])  # shape (n, n, 8)

    # Save full array for later inspection
    combined_path = os.path.join(args.out_dir, 'scan_full.npy')
    np.save(combined_path, data)
    print(f'Saved combined array to {combined_path}')

    # Plot -----------------------------------------------------------------
    gs      = args.gamma_step
    gammas   = [gs * i for i in range(n)]
    gamma_ps = [gs * i for i in range(n)]
    extent   = [gamma_ps[0], gamma_ps[-1], gammas[0], gammas[-1]]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))
    axes = axes.ravel()

    for ax, (k, title) in zip(axes, METRICS):
        im = ax.imshow(data[:, :, k], origin='lower', aspect='auto',
                       extent=extent)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    for ax in axes[len(METRICS):]:
        ax.set_visible(False)

    fig.suptitle(f'Steady-state properties  (J = {args.J})', fontsize=14)
    plt.tight_layout()

    out_fig = os.path.join(args.out_dir, 'two_qubit_scan.png')
    plt.savefig(out_fig, dpi=150)
    print(f'Saved figure to {out_fig}')


if __name__ == '__main__':
    main()
