"""
Plot the optimised framability of the two-qubit Lindbladian Trotter step
exp(L*dt) as a (gamma, gamma') colormap for d_ext_single in {4, 6}.

Reads results_trotter/trotter_summary.npz (built by
lindbladian_trotter_collect.py).  Falls back to aggregating the per-task
npz files directly if the summary is not present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

D_EXT_SINGLES = [4, 6]
GAMMA_MAX, GP_MAX, GAMMA_STEP = 8.0, 4.0, 0.2
N_GAMMA = int(round(GAMMA_MAX / GAMMA_STEP)) + 1  # 41
N_GP    = int(round(GP_MAX   / GAMMA_STEP)) + 1   # 21


def _aggregate(in_dir: Path) -> np.ndarray:
    fra = np.full((len(D_EXT_SINGLES), N_GAMMA, N_GP), np.nan)
    for di, d in enumerate(D_EXT_SINGLES):
        for ig in range(N_GAMMA):
            for igp in range(N_GP):
                f = in_dir / f'trotter_{d}_{ig:03d}_{igp:03d}.npz'
                if not f.exists():
                    continue
                fra[di, ig, igp] = float(np.load(f)['framability'])
    return fra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_trotter')
    parser.add_argument('--out_dir', type=str, default='results_trotter')
    parser.add_argument('--out_name', type=str, default='trotter_framability.png')
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = in_dir / 'trotter_summary.npz'
    if summary.exists():
        print(f'[load] {summary}')
        d = np.load(summary)
        fra = d['framability']
    else:
        print(f'[aggregate] {in_dir} (summary not found)')
        fra = _aggregate(in_dir)

    gammas = GAMMA_STEP * np.arange(N_GAMMA)
    gps    = GAMMA_STEP * np.arange(N_GP)
    half   = GAMMA_STEP / 2
    extent = [gps[0] - half, gps[-1] + half, gammas[0] - half, gammas[-1] + half]

    vmin = float(np.nanmin(fra))
    vmax = float(np.nanmax(fra))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for k, d in enumerate(D_EXT_SINGLES):
        ax = axes[k]
        im = ax.imshow(fra[k], origin='lower', extent=extent, aspect='auto',
                       cmap='viridis', vmin=vmin, vmax=vmax)
        if np.nanmin(fra[k]) < 1.0 < np.nanmax(fra[k]):
            ax.contour(fra[k], levels=[1.0], colors='white',
                       linewidths=0.8, extent=extent, origin='lower')
        n_ok = int(np.sum(np.isfinite(fra[k])))
        ax.set_title(f'Optimised framability  '
                     f'$d_\\mathrm{{ext,1q}}={d}$  ({n_ok}/{N_GAMMA*N_GP} pts)')
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r'$\gamma$')
        fig.colorbar(im, ax=ax)

    fig.suptitle(r'Two-qubit Lindbladian Trotter step $\exp(L\,dt)$ — '
                 r'minimal framability over Kronecker frames $D=S\otimes S$')
    fig.tight_layout()

    out_png = out_dir / args.out_name
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f'[saved] {out_png}')

    # Quick summary stats
    for k, d in enumerate(D_EXT_SINGLES):
        arr = fra[k][np.isfinite(fra[k])]
        if arr.size:
            print(f'  d_ext_single={d}:  min={arr.min():.4f}  '
                  f'max={arr.max():.4f}  median={np.median(arr):.4f}')


if __name__ == '__main__':
    main()
