"""
Collect per-row and per-point results and generate a 2-row colourmap figure.

Base data      : <out_dir>/row_XXXX.npy        shape (n_pts, 5)
Extra data     : <out_dir>/point_extra_XXXX_XXXX.npy  shape (6,)

Combined array shape: (n_pts, n_pts, 11)
  Base columns (0-4):
      0  entropy               Von Neumann entropy
      1  negativity            Negativity
      2  pauli_fra             Pauli framability
      3  min_fra               Optimised framability (Kronecker frame)
      4  dec_rate              Decay rate

  Extra columns (5-11):
      5  ss_bond_entropy       Steady-state LPDO bond entropy
      6  mag_x                 X magnetisation
      7  stab_dyadic_fra       Dyadic stabilizer framability
      8  projector_stab_fra    Projector stabilizer framability
      9  product_fra_heis      Product-state framability (Heisenberg)
     10  product_fra_schro     Product-state framability (Schrödinger)
     11  max_bond_entropy      Maximum LPDO bond entropy during time evolution

Plot layout (2 rows × 6 cols):
  Row 0 (top)   : entropy | negativity | ss_bond_entropy | mag_x | dec_rate | max_bond_entropy
  Row 1 (bottom): pauli_fra | min_fra | stab_dyadic | projector_stab | product_heis | product_schro

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
# Columns 0-11 come from base+extra data.
# Columns 12+ are appended from external files:
#   12  operator_bond_entropy   (results/operator_bond_entropy.npy)
#   13  product_fra_schro_chi10 (results/product_fra_schro_chi010.npy)
#   14  product_fra_schro_chi20 (results/product_fra_schro_chi020.npy)
#   15  product_fra_schro_chi40 (results/product_fra_schro_chi040.npy)
#   16  product_fra_schro_chi30 (results/product_fra_schro_chi030.npy)
TOP_ROW = [
    (0, 'Von Neumann entropy'),
    (1, 'Negativity'),
    (6, r'$X$ magnetisation'),
    (4, 'Decay rate'),
    (5, 'Steady-state LPDO bond entropy'),
    (11, 'Max LPDO bond entropy'),
    (12, 'Operator bond entropy'),
]
# Column indices that share a common colour scale
SHARED_CLIM_GROUPS = [(5, 11), (13, 14, 15, 16)]  # bond entropies | chi groups
BOTTOM_ROW = [
    (2, 'Pauli framability'),
    (3, 'Optimised framability'),
    (7, 'Dyadic stabilizer framability'),
    (8, 'Projector stabilizer framability'),
    (13, r'Product-state fra. Schrödinger $\chi=10$'),
    (14, r'Product-state fra. Schrödinger $\chi=20$'),
    (16, r'Product-state fra. Schrödinger $\chi=30$'),
    (15, r'Product-state fra. Schrödinger $\chi=40$'),
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
              'stab_dyadic_fra, projector_stab_fra, product_fra_heis, product_fra_schro, '
              'max_bond_entropy)')
    else:
        print('WARNING: extra point files not found – '
              'bottom-row framability panels will be empty.',
              file=sys.stderr)

    # ── Load external computed grids and append as extra columns ─────────
    _ext_files = [
        ('operator_bond_entropy',   'operator_bond_entropy.npy'),
        ('product_fra_schro_chi10', 'product_fra_schro_chi010.npy'),
        ('product_fra_schro_chi20', 'product_fra_schro_chi020.npy'),
        ('product_fra_schro_chi40', 'product_fra_schro_chi040.npy'),
        ('product_fra_schro_chi30', 'product_fra_schro_chi030.npy'),
    ]
    for key, fname in _ext_files:
        fpath = os.path.join(args.out_dir, fname)
        if os.path.exists(fpath):
            arr = np.load(fpath)  # shape (n, n)
            data = np.concatenate([data, arr[:, :, None]], axis=2)
            print(f'Appended {key} from {fpath}  (col {data.shape[2]-1})')
        else:
            data = np.concatenate([data, np.full((n, n, 1), np.nan)], axis=2)
            print(f'WARNING: {fpath} not found – column filled with NaN',
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

    ncols = max(len(TOP_ROW), len(BOTTOM_ROW))
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10))

    # Build shared colour-limit lookup: col_index -> (vmin, vmax)
    shared_clim = {}
    for group in SHARED_CLIM_GROUPS:
        cols_present = [k for k in group if k < data.shape[2]]
        if len(cols_present) > 1:
            vmin = min(data[:, :, k].min() for k in cols_present)
            vmax = max(data[:, :, k].max() for k in cols_present)
            for k in cols_present:
                shared_clim[k] = (vmin, vmax)

    for col, (k, title) in enumerate(TOP_ROW):
        ax = axes[0, col]
        if k < data.shape[2]:
            kwargs = {}
            if k in shared_clim:
                kwargs['vmin'], kwargs['vmax'] = shared_clim[k]
            im = ax.imshow(data[:, :, k], origin='lower', aspect='auto',
                           extent=extent, **kwargs)
            fig.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title)

    for col, (k, title) in enumerate(BOTTOM_ROW):
        ax = axes[1, col]
        if k < data.shape[2]:
            im = ax.imshow(data[:, :, k], origin='lower', aspect='auto',
                           extent=extent, vmin=1.0)
            ax.contour(data[:, :, k], levels=[1.0 + 1e-6], colors='white',
                       linewidths=0.8, extent=extent, origin='lower')
            fig.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title)

    fig.suptitle(f'Steady-state properties  (J = {args.J})', fontsize=14)
    plt.tight_layout()

    out_fig = os.path.join(args.out_dir, 'two_qubit_scan.png')
    plt.savefig(out_fig, dpi=150)
    print(f'Saved figure to {out_fig}')


if __name__ == '__main__':
    main()
