"""
Aggregate per-point sign-problem results for the 6-qubit star+plaquette
Lindbladian Trotter step into a summary npz and a 1x2 colormap figure.

Grid: 20 x 20 over (gamma_s, gamma_p), step 0.2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

GAMMA_STEP = 0.2
N_GRID     = int(round(4.0 / GAMMA_STEP))   # 20


def _load(in_dir: Path):
    s_init = np.full((N_GRID, N_GRID), np.nan)
    s_opt  = np.full((N_GRID, N_GRID), np.nan)
    n_opt  = np.full((N_GRID, N_GRID, 3), np.nan)
    missing = 0
    for ig in range(N_GRID):
        for igp in range(N_GRID):
            f = in_dir / f'sign_six_{ig:03d}_{igp:03d}.npz'
            if not f.exists():
                missing += 1
                continue
            d = np.load(f)
            s_init[ig, igp]    = float(d['sign_init'])
            s_opt [ig, igp]    = float(d['sign_opt'])
            n_opt [ig, igp, :] = np.asarray(d['n_opt'], dtype=float)
    if missing:
        print(f'Warning: {missing}/{N_GRID * N_GRID} files missing')
    return s_init, s_opt, n_opt


def _panel(ax, data, vals, title, *, vmin, vmax):
    edges = np.append(vals - GAMMA_STEP / 2, vals[-1] + GAMMA_STEP / 2)
    im = ax.pcolormesh(edges, edges, data, cmap='viridis',
                       vmin=vmin, vmax=vmax, shading='flat')
    if np.any(np.isfinite(data)):
        finite = data[np.isfinite(data)]
        if finite.min() < 1.0 < finite.max():
            try:
                ax.contour(vals, vals, data,
                           levels=[1.0], colors='red', linewidths=1.2)
            except Exception:
                pass
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r'$\gamma_p$')
    ax.set_ylabel(r'$\gamma_s$')
    ax.set_title(title, fontsize=11)
    n_pts = int(np.sum(np.isfinite(data)))
    ax.text(0.98, 0.98, f'{n_pts}/{N_GRID * N_GRID}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='white')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',  default='results_sign_six')
    p.add_argument('--out',     default='results_plots/sign_six.png')
    p.add_argument('--summary_out', default=None,
                   help='Summary npz path (default: <in_dir>/sign_six_summary.npz).')
    p.add_argument('--title',   default=None,
                   help='Optional figure title prefix.')
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    s_init, s_opt, n_opt = _load(in_dir)
    vals = GAMMA_STEP * np.arange(N_GRID)

    summary_path = (Path(args.summary_out) if args.summary_out
                    else in_dir / 'sign_six_summary.npz')
    np.savez(summary_path,
             gamma_values = vals,
             sign_init    = s_init,
             sign_opt     = s_opt,
             n_opt        = n_opt)
    print(f'Saved {summary_path}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    base_title = args.title or (
        r'6-qubit star+plaquette Lindbladian Trotter step  $e^{L\,dt}$  '
        r'($dt=0.02$):  sign problem  $s = |\sum U|/\sum|U|$'
    )
    fig.suptitle(base_title, fontsize=11)

    init_finite = s_init[np.isfinite(s_init)]
    opt_finite  = s_opt [np.isfinite(s_opt )]
    vmin_i, vmax_i = ((float(init_finite.min()), float(init_finite.max()))
                      if len(init_finite) else (0.0, 1.0))
    vmin_o, vmax_o = ((float(opt_finite.min()),  float(opt_finite.max()))
                      if len(opt_finite ) else (0.0, 1.0))

    _panel(axes[0], s_init, vals, r's in Pauli basis  (no rotation)',
           vmin=vmin_i, vmax=vmax_i)
    _panel(axes[1], s_opt,  vals, r'optimised s  (max over local $R^{\otimes 6}$)',
           vmin=vmin_o, vmax=vmax_o)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    print(f'Saved {args.out}')

    for name, arr in [('s_init', s_init), ('s_opt', s_opt)]:
        v = arr[np.isfinite(arr)]
        if len(v):
            print(f'  {name:8s}  min={v.min():.4f}  max={v.max():.4f}  '
                  f'median={np.median(v):.4f}  n={len(v)}')


if __name__ == '__main__':
    main()
