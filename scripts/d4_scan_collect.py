"""
Collect d4 scan results and plot comparison with d_ext=6 optimized framability.

Reads:
    <d4_dir>/d4row_XXXX.npy          shape (n_pts,)  — d_ext=4 free
    results/scan_full.npy            shape (n_pts, n_pts, >=5)  — col 3 = d_ext=6 opt fra

Produces:
    <d4_dir>/d4_vs_d6.png            3-panel figure
    <d4_dir>/d4_scan.npy             shape (n_pts, n_pts)  — collected d4 grid

Usage:
    python d4_scan_collect.py --d4_dir results_d4 --d6_dir results
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--d4_dir',     type=str, default='results_d4')
    p.add_argument('--d6_dir',     type=str, default='results')
    p.add_argument('--n_pts',      type=int, default=41)
    p.add_argument('--gamma_step', type=float, default=0.2)
    args = p.parse_args()

    n = args.n_pts
    gs = args.gamma_step

    # ── Load d4 rows ──────────────────────────────────────────────────────
    d4 = np.full((n, n), np.nan)
    missing = []
    for ig in range(n):
        f = os.path.join(args.d4_dir, f'd4row_{ig:04d}.npy')
        if os.path.exists(f):
            d4[ig] = np.load(f)
        else:
            missing.append(f)

    if missing:
        print(f'[warn] {len(missing)} missing d4 row files', file=sys.stderr)
        for m in missing[:5]:
            print(f'  {m}', file=sys.stderr)

    np.save(os.path.join(args.d4_dir, 'd4_scan.npy'), d4)
    print(f'[saved] {args.d4_dir}/d4_scan.npy  shape={d4.shape}')

    # ── Load d6 data ──────────────────────────────────────────────────────
    scan_full_path = os.path.join(args.d6_dir, 'scan_full.npy')
    if not os.path.exists(scan_full_path):
        print(f'ERROR: {scan_full_path} not found', file=sys.stderr)
        sys.exit(1)

    scan_full = np.load(scan_full_path)
    d6 = scan_full[:, :, 3]  # col 3 = optimized framability (d_ext=6)
    print(f'Loaded d6 from {scan_full_path}  shape={scan_full.shape}')

    # ── Plot ──────────────────────────────────────────────────────────────
    extent = [0, gs * (n - 1), 0, gs * (n - 1)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: d_ext=4
    ax = axes[0]
    im = ax.imshow(d4, origin='lower', aspect='auto', extent=extent, vmin=1.0)
    ax.contour(d4, levels=[1.0 + 1e-6], colors='white',
               linewidths=0.8, extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(r'Optimized framability ($d_{\rm ext}=4$, free)')

    # Panel 2: d_ext=6
    ax = axes[1]
    im = ax.imshow(d6, origin='lower', aspect='auto', extent=extent, vmin=1.0)
    ax.contour(d6, levels=[1.0 + 1e-6], colors='white',
               linewidths=0.8, extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(r'Optimized framability ($d_{\rm ext}=6$, fixed cols)')

    # Panel 3: difference d4 - d6
    ax = axes[2]
    diff = d4 - d6
    vmax = np.nanmax(np.abs(diff))
    im = ax.imshow(diff, origin='lower', aspect='auto', extent=extent,
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.contour(diff, levels=[0.0], colors='black',
               linewidths=0.8, extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(r'Difference: $d_{\rm ext}=4$ $-$ $d_{\rm ext}=6$')

    fig.suptitle(r'Optimized framability: $d_{\rm ext}=4$ (all free) vs $d_{\rm ext}=6$ (2 fixed cols)',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_fig = os.path.join(args.d4_dir, 'd4_vs_d6.png')
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)
    print(f'[saved] {out_fig}')


if __name__ == '__main__':
    main()
