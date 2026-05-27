"""
Collect per-point outputs from recompute_fra_worker and build a
results_opt/scan_opt.npy array with the same (41, 41, 3) layout:
    col 0: min_fra     optimised framability (new frame structure)
    col 1: pauli_fra   max row-L1-norm of gate
    col 2: min_both    min(min_fra, pauli_fra)

Also regenerates results_opt/two_qubit_scan_opt_bond_vs_fra.png
using the existing bond-entropy data from results/scan_full.npy.

Usage
-----
    python recompute_fra_collect.py [--out_dir results_opt]
                                    [--src_dir results]
                                    [--n_pts 41] [--gamma_step 0.2]
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',    type=str,   default='results_opt')
    p.add_argument('--src_dir',    type=str,   default='results',
                   help='Directory containing scan_full.npy for bond-entropy data.')
    p.add_argument('--n_pts',      type=int,   default=41)
    p.add_argument('--gamma_step', type=float, default=0.2)
    args = p.parse_args()

    n = args.n_pts
    n_tasks = n * n

    # ── load per-point files ──────────────────────────────────────────────
    missing = []
    min_fra   = np.full((n, n), np.nan)
    pauli_fra = np.full((n, n), np.nan)

    for task_id in range(n_tasks):
        ig  = task_id // n
        igp = task_id %  n
        fp = os.path.join(args.out_dir, f'opt_fra_{ig:04d}_{igp:04d}.npy')
        if not os.path.exists(fp):
            missing.append((ig, igp))
        else:
            arr = np.load(fp)
            min_fra[ig, igp]   = arr[0]
            pauli_fra[ig, igp] = arr[1]

    if missing:
        print(f'WARNING: {len(missing)} point files missing:', file=sys.stderr)
        for ig, igp in missing[:10]:
            print(f'  opt_fra_{ig:04d}_{igp:04d}.npy', file=sys.stderr)
        if len(missing) > 10:
            print(f'  ... and {len(missing)-10} more', file=sys.stderr)

    min_both = np.minimum(min_fra, pauli_fra)

    scan_opt = np.stack([min_fra, pauli_fra, min_both], axis=-1)  # (n, n, 3)
    out_npy = os.path.join(args.out_dir, 'scan_opt.npy')
    np.save(out_npy, scan_opt)
    print(f'Saved {out_npy}  shape={scan_opt.shape}')

    # ── load bond-entropy from original scan_full ──────────────────────────
    scan_full_path = os.path.join(args.src_dir, 'scan_full.npy')
    if not os.path.exists(scan_full_path):
        print(f'WARNING: {scan_full_path} not found; skipping figure.', file=sys.stderr)
        return

    scan_full = np.load(scan_full_path)
    ngp = min(n, scan_full.shape[1])
    bond_entropy = scan_full[:n, :ngp, 11]  # col 11 = max_bond_entropy

    # ── side-by-side heatmaps ─────────────────────────────────────────────
    gamma_step = args.gamma_step
    half = gamma_step / 2
    extent = [
        0 - half, (ngp - 1) * gamma_step + half,
        0 - half, (n   - 1) * gamma_step + half,
    ]

    fra = min_both[:, :ngp]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: max LPDO bond entropy
    ax = axes[0]
    vmin0 = min(np.nanmin(bond_entropy), 0.0)
    vmax0 = max(np.nanmax(bond_entropy), 0.0)
    im0 = ax.imshow(bond_entropy, origin='lower', extent=extent, aspect='auto',
                    cmap='viridis', vmin=vmin0, vmax=vmax0)
    ax.set_title('Max LPDO bond entropy')
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r'$\gamma$')
    cb0 = fig.colorbar(im0, ax=ax)
    cb0.set_ticks(sorted(set(list(cb0.get_ticks()) + [0.0])))
    if np.nanmin(bond_entropy) < 0.0 < np.nanmax(bond_entropy):
        try:
            ax.contour(bond_entropy, levels=[0.0], colors='white', linewidths=0.8,
                       extent=extent, origin='lower')
        except Exception:
            pass

    # Right: optimised framability (min of optimised and Pauli)
    ax = axes[1]
    im1 = ax.imshow(fra, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    ax.set_title('Optimised framability\n(min of optimised and Pauli)')
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r'$\gamma$')
    fig.colorbar(im1, ax=ax)
    if np.nanmin(fra) < 1.0 < np.nanmax(fra):
        ax.contour(fra, levels=[1.0], colors='white', linewidths=0.8,
                   extent=extent, origin='lower')

    fig.suptitle(f'Max LPDO bond entropy vs optimised framability  '
                 f'(step={gamma_step}, updated frame structure)')
    fig.tight_layout()

    out_png = os.path.join(args.out_dir, 'two_qubit_scan_opt_bond_vs_fra.png')
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()
