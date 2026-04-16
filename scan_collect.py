"""
Collect per-row and per-point results and generate a 2-row colourmap figure.

Base data      : <out_dir>/row_XXXX.npy        shape (n_pts, 5)
Extra data     : <out_dir>/point_extra_XXXX_XXXX.npy  shape (4,)

Combined array shape: (n_pts, n_pts, 9)
  Base columns (0-4):
      0  entropy          Von Neumann entropy
      1  negativity       Negativity
      2  pauli_fra        Pauli framability
      3  min_fra          Optimised framability (Kronecker frame)
      4  dec_rate         Decay rate

  Extra columns (5-8):
      5  ss_bond_entropy  Steady-state LPDO bond entropy
      6  mag_x            X magnetisation
      7  stabilizer_fra   Dyadic stabilizer framability
      8  product_fra      Product-state framability

Plot layout (2 rows × 5 cols):
  Row 0 (top)   : entropy | negativity | ss_bond_entropy | mag_x | dec_rate
  Row 1 (bottom): pauli_fra | min_fra | stabilizer_fra | product_fra | (empty)

Usage
-----
    python scan_collect.py --n_pts 41 --J 1.0 --gamma_step 0.2 --out_dir results
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# (merged_col, title)
TOP_ROW = [
    (0, 'Von Neumann entropy'),
    (1, 'Negativity'),
    (5, 'Steady-state LPDO bond entropy'),
    (6, r'$X$ magnetisation'),
    (4, 'Decay rate'),
]
BOTTOM_ROW = [
    (2, 'Pauli framability'),
    (3, 'Optimised framability'),
    (7, 'Stabilizer framability'),
    (8, 'Product-state framability'),
]


def main():
    p = argparse.ArgumentParser(
        description='Aggregate scan outputs and produce colourmap figure.'
    )
    p.add_argument('--n_pts',      type=int,   default=41)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str,   default='results')
    args = p.parse_args()

    n = args.n_pts

    # ── Load base row files ──────────────────────────────────────────────
    missing = [
        os.path.join(args.out_dir, f'row_{ig:04d}.npy')
        for ig in range(n)
        if not os.path.exists(os.path.join(args.out_dir, f'row_{ig:04d}.npy'))
    ]
    if missing:
        print('ERROR: missing base result files:', file=sys.stderr)
        for f in missing:
            print(f'  {f}', file=sys.stderr)
        sys.exit(1)

    # data[ig, igp, metric],  shape (n, n, 5)
    data = np.stack([
        np.load(os.path.join(args.out_dir, f'row_{ig:04d}.npy'))
        for ig in range(n)
    ])

    # ── Load extra point files ───────────────────────────────────────────
    has_extra = all(
        os.path.exists(os.path.join(args.out_dir,
                                    f'point_extra_{ig:04d}_{igp:04d}.npy'))
        for ig in range(n) for igp in range(n)
    )

    if has_extra:
        extra = np.stack([
            np.stack([
                np.load(os.path.join(args.out_dir,
                                     f'point_extra_{ig:04d}_{igp:04d}.npy'))
                for igp in range(n)
            ])
            for ig in range(n)
        ])  # shape (n, n, 4)
        data = np.concatenate([data, extra], axis=2)  # shape (n, n, 9)
        print('Merged extra properties (ss_bond_entropy, mag_x, '
              'stabilizer_fra, product_fra)')
    else:
        print('WARNING: extra point files not found – '
              'bottom-row framability panels will be empty.',
              file=sys.stderr)

    # ── Save combined array ──────────────────────────────────────────────
    combined_path = os.path.join(args.out_dir, 'scan_full.npy')
    np.save(combined_path, data)
    print(f'Saved combined array to {combined_path}  shape={data.shape}')

    # ── Plot ─────────────────────────────────────────────────────────────
    gs       = args.gamma_step
    gammas   = [gs * i for i in range(n)]
    gamma_ps = [gs * i for i in range(n)]
    extent   = [gamma_ps[0], gamma_ps[-1], gammas[0], gammas[-1]]

    ncols = 5
    fig, axes = plt.subplots(2, ncols, figsize=(7 * ncols, 10))

    for col, (k, title) in enumerate(TOP_ROW):
        ax = axes[0, col]
        if k < data.shape[2]:
            im = ax.imshow(data[:, :, k], origin='lower', aspect='auto',
                           extent=extent)
            fig.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title)

    for col, (k, title) in enumerate(BOTTOM_ROW):
        ax = axes[1, col]
        if k < data.shape[2]:
            im = ax.imshow(data[:, :, k], origin='lower', aspect='auto',
                           extent=extent)
            fig.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title)

    # Hide the unused 5th panel in the bottom row
    axes[1, len(BOTTOM_ROW)].set_visible(False)

    fig.suptitle(f'Steady-state properties  (J = {args.J})', fontsize=14)
    plt.tight_layout()

    out_fig = os.path.join(args.out_dir, 'two_qubit_scan.png')
    plt.savefig(out_fig, dpi=150)
    print(f'Saved figure to {out_fig}')


if __name__ == '__main__':
    main()
