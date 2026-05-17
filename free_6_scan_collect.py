"""
Collect free_6 scan results and plot 3-way comparison:
  d_ext=4 (free)  vs  d_ext=6 (fixed cols)  vs  d_ext=6 (free, "free_6").

Reads:
    <free6_dir>/free6row_XXXX.npy      shape (n_pts,)   — d_ext=6 all free
    <d4_dir>/d4_scan.npy               shape (n_pts, n_pts) — d_ext=4 free
    results/scan_full.npy              shape (n_pts, n_pts, >=5) — col 3 = d_ext=6 opt (fixed)

Produces:
    <free6_dir>/free6_scan.npy         shape (n_pts, n_pts) — collected free_6 grid
    <free6_dir>/free6_comparison.png   5-panel figure

Usage:
    python free_6_scan_collect.py --free6_dir results_free6 --d4_dir results_d4 --d6_dir results
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--free6_dir',  type=str, default='results_free6')
    p.add_argument('--d4_dir',     type=str, default='results_d4')
    p.add_argument('--d6_dir',     type=str, default='results')
    p.add_argument('--n_pts',      type=int, default=41)
    p.add_argument('--gamma_step', type=float, default=0.2)
    args = p.parse_args()

    n = args.n_pts
    gs = args.gamma_step

    # ── Load free_6 rows ─────────────────────────────────────────────────
    free6 = np.full((n, n), np.nan)
    missing = []
    for ig in range(n):
        f = os.path.join(args.free6_dir, f'free6row_{ig:04d}.npy')
        if os.path.exists(f):
            free6[ig] = np.load(f)
        else:
            missing.append(f)

    if missing:
        print(f'[warn] {len(missing)} missing free_6 row files', file=sys.stderr)
        for m in missing[:5]:
            print(f'  {m}', file=sys.stderr)

    np.save(os.path.join(args.free6_dir, 'free6_scan.npy'), free6)
    print(f'[saved] {args.free6_dir}/free6_scan.npy  shape={free6.shape}')

    # ── Load d4 data ─────────────────────────────────────────────────────
    d4_path = os.path.join(args.d4_dir, 'd4_scan.npy')
    if os.path.exists(d4_path):
        d4 = np.load(d4_path)
        print(f'Loaded d4 from {d4_path}  shape={d4.shape}')
    else:
        print(f'[warn] {d4_path} not found; d4 panels will be blank',
              file=sys.stderr)
        d4 = np.full((n, n), np.nan)

    # ── Load d6 (fixed cols) data ────────────────────────────────────────
    scan_full_path = os.path.join(args.d6_dir, 'scan_full.npy')
    if not os.path.exists(scan_full_path):
        print(f'ERROR: {scan_full_path} not found', file=sys.stderr)
        sys.exit(1)

    scan_full = np.load(scan_full_path)
    d6_fixed = scan_full[:, :, 3]  # col 3 = optimized framability (d_ext=6, fixed cols)
    print(f'Loaded d6_fixed from {scan_full_path}  shape={scan_full.shape}')

    # ── Plot ─────────────────────────────────────────────────────────────
    extent = [0, gs * (n - 1), 0, gs * (n - 1)]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # Row 1: the three scans
    panels = [
        (axes[0, 0], d4,      r'Opt. framability ($d_{\rm ext}=4$, free)'),
        (axes[0, 1], d6_fixed, r'Opt. framability ($d_{\rm ext}=6$, fixed cols)'),
        (axes[0, 2], free6,   r'Opt. framability ($d_{\rm ext}=6$, free)'),
    ]
    for ax, data, title in panels:
        im = ax.imshow(data, origin='lower', aspect='auto', extent=extent, vmin=1.0)
        ax.contour(data, levels=[1.0 + 1e-6], colors='white',
                   linewidths=0.8, extent=extent, origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title)

    # Row 2: differences
    diff_panels = [
        (axes[1, 0], free6 - d6_fixed,
         r'free\_6 $-$ d6\_fixed'),
        (axes[1, 1], free6 - d4,
         r'free\_6 $-$ d4\_free'),
        (axes[1, 2], d4 - d6_fixed,
         r'd4\_free $-$ d6\_fixed'),
    ]
    for ax, diff, title in diff_panels:
        vmax = np.nanmax(np.abs(diff))
        if vmax == 0:
            vmax = 1e-6
        im = ax.imshow(diff, origin='lower', aspect='auto', extent=extent,
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.contour(diff, levels=[0.0], colors='black',
                   linewidths=0.8, extent=extent, origin='lower')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title)

    fig.suptitle(
        r'Optimized framability comparison: $d_{\rm ext}=4$ (free) vs '
        r'$d_{\rm ext}=6$ (fixed) vs $d_{\rm ext}=6$ (free)',
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_fig = os.path.join(args.free6_dir, 'free6_comparison.png')
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)
    print(f'[saved] {out_fig}')


if __name__ == '__main__':
    main()
