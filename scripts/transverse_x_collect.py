"""
Collect results from transverse_x_worker.py and produce a 4-panel colormap:
  1. Pauli framability
  2. Optimised framability d_ext=4
  3. Optimised framability d_ext=6
  4. Max LPDO bond entropy

All panels are colormaps over the (gamma', gamma) grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

GAMMA_MAX  = 8.0
GP_MAX     = 4.0
GAMMA_STEP = 0.2
N_GAMMA    = int(round(GAMMA_MAX  / GAMMA_STEP)) + 1   # 41
N_GP       = int(round(GP_MAX     / GAMMA_STEP)) + 1   # 21


def _load(in_dir: Path):
    pauli   = np.full((N_GAMMA, N_GP), np.nan)
    opt4    = np.full((N_GAMMA, N_GP), np.nan)
    opt6    = np.full((N_GAMMA, N_GP), np.nan)
    entropy = np.full((N_GAMMA, N_GP), np.nan)
    n_missing = 0
    for ig in range(N_GAMMA):
        for igp in range(N_GP):
            f = in_dir / f'transverse_x_{ig:03d}_{igp:03d}.npz'
            if not f.exists():
                n_missing += 1
                continue
            d = np.load(f)
            pauli[ig, igp]   = float(d['pauli_fra'])
            opt4[ig, igp]    = float(d['opt_fra_4'])
            opt6[ig, igp]    = float(d['opt_fra_6'])
            entropy[ig, igp] = float(d['max_lpdo_entropy'])
    if n_missing:
        print(f'Warning: {n_missing}/{N_GAMMA * N_GP} files missing')
    return pauli, opt4, opt6, entropy


def _colormap_panel(ax, data, gamma_vals, gp_vals, title, *,
                    cmap='viridis', vmin=None, vmax=None, contour_at=None):
    # data[ig, igp]: ig=gamma axis, igp=gamma' axis
    # pcolormesh expects (gamma', gamma) with gamma' on x-axis
    gp_edges = np.append(gp_vals - GAMMA_STEP/2, gp_vals[-1] + GAMMA_STEP/2)
    g_edges  = np.append(gamma_vals - GAMMA_STEP/2, gamma_vals[-1] + GAMMA_STEP/2)
    im = ax.pcolormesh(gp_edges, g_edges, data,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    if contour_at is not None:
        try:
            ax.contour(gp_vals, gamma_vals, data,
                       levels=[contour_at], colors='white', linewidths=1.2)
        except Exception:
            pass
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\gamma'$", fontsize=11)
    ax.set_ylabel(r'$\gamma$', fontsize=11)
    ax.set_title(title, fontsize=11)
    n_pts = int(np.sum(~np.isnan(data)))
    ax.text(0.98, 0.98, f'{n_pts}/{N_GAMMA * N_GP} pts',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='white')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  default='results_transverse_x')
    parser.add_argument('--out',     default='results_plots/transverse_x.png')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    pauli, opt4, opt6, entropy = _load(in_dir)

    gamma_vals = GAMMA_STEP * np.arange(N_GAMMA)
    gp_vals    = GAMMA_STEP * np.arange(N_GP)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        r'Two-qubit Lindbladian $H = J\,Z\!\otimes\!Z + h\,(X\!\otimes\!I + I\!\otimes\!X)$'
        '\n'
        r'Trotter step $e^{L\,dt}$  —  $J=h=1$,  $dt=0.01$',
        fontsize=12,
    )

    _colormap_panel(axes[0, 0], pauli, gamma_vals, gp_vals,
                    'Pauli framability', cmap='viridis', contour_at=1.0)

    _colormap_panel(axes[0, 1], opt4, gamma_vals, gp_vals,
                    r'Optimised framability  $d_{\rm ext}=4$',
                    cmap='viridis', contour_at=1.0)

    _colormap_panel(axes[1, 0], opt6, gamma_vals, gp_vals,
                    r'Optimised framability  $d_{\rm ext}=6$',
                    cmap='viridis', contour_at=1.0)

    _colormap_panel(axes[1, 1], entropy, gamma_vals, gp_vals,
                    'Max LPDO bond entropy', cmap='plasma')

    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    print(f'Saved {args.out}')

    # summary statistics
    for label, arr in [('pauli_fra', pauli), ('opt_fra_4', opt4),
                       ('opt_fra_6', opt6), ('max_lpdo_entropy', entropy)]:
        valid = arr[~np.isnan(arr)]
        if len(valid):
            print(f'  {label}: min={valid.min():.4f}  max={valid.max():.4f}  '
                  f'median={np.median(valid):.4f}')

    # Save summary npz
    np.savez(in_dir / 'transverse_x_summary.npz',
             pauli_fra       = pauli,
             opt_fra_4       = opt4,
             opt_fra_6       = opt6,
             max_lpdo_entropy= entropy,
             gamma_values    = gamma_vals,
             gp_values       = gp_vals)
    print(f'Saved {in_dir / "transverse_x_summary.npz"}')


if __name__ == '__main__':
    main()
