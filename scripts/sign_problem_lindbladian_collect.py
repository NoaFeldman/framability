"""
Collect optimised-sign-problem results for the two-qubit Lindbladian Trotter
step, both with and without the transverse field, and plot a 2x2 colormap:

    row 0: no transverse field   (h = 0)
    row 1: with transverse field (h ≠ 0)
    col 0: s_init (sign problem in the original Pauli basis)
    col 1: s_opt  (after gradient-descent rotation)

All panels are colormaps over the (gamma', gamma) grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

GAMMA_MAX  = 8.0
GP_MAX     = 4.0
GAMMA_STEP = 0.2
N_GAMMA    = int(round(GAMMA_MAX  / GAMMA_STEP)) + 1
N_GP       = int(round(GP_MAX     / GAMMA_STEP)) + 1


def _load(in_dir: Path, tag: str):
    s_init = np.full((N_GAMMA, N_GP), np.nan)
    s_opt  = np.full((N_GAMMA, N_GP), np.nan)
    n_opt  = np.full((N_GAMMA, N_GP, 3), np.nan)
    n_missing = 0
    for ig in range(N_GAMMA):
        for igp in range(N_GP):
            f = in_dir / f'sign_{tag}_{ig:03d}_{igp:03d}.npz'
            if not f.exists():
                n_missing += 1
                continue
            d = np.load(f)
            s_init[ig, igp]    = float(d['sign_init'])
            s_opt [ig, igp]    = float(d['sign_opt'])
            n_opt [ig, igp, :] = np.asarray(d['n_opt'], dtype=float)
    if n_missing:
        print(f'[{tag}] Warning: {n_missing}/{N_GAMMA * N_GP} files missing')
    return s_init, s_opt, n_opt


def _panel(ax, data, gamma_vals, gp_vals, title, *,
           cmap='viridis', vmin=None, vmax=None):
    gp_edges = np.append(gp_vals - GAMMA_STEP / 2, gp_vals[-1] + GAMMA_STEP / 2)
    g_edges  = np.append(gamma_vals - GAMMA_STEP / 2, gamma_vals[-1] + GAMMA_STEP / 2)
    im = ax.pcolormesh(gp_edges, g_edges, data,
                       cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
    # Contour marking the sign-problem-free locus s = 1
    if np.any(np.isfinite(data)):
        finite = data[np.isfinite(data)]
        if finite.min() < 1.0 < finite.max():
            try:
                ax.contour(gp_vals, gamma_vals, data,
                           levels=[1.0], colors='red', linewidths=1.2)
            except Exception:
                pass
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\gamma'$", fontsize=11)
    ax.set_ylabel(r'$\gamma$',  fontsize=11)
    ax.set_title(title, fontsize=11)
    n_pts = int(np.sum(~np.isnan(data)))
    ax.text(0.98, 0.98, f'{n_pts}/{N_GAMMA * N_GP} pts',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='white')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--nofield_dir',    default='results_sign_problem_nofield')
    parser.add_argument('--transverse_dir', default='results_sign_problem_transverse')
    parser.add_argument('--nofield_tag',    default='nofield')
    parser.add_argument('--transverse_tag', default='transverse')
    parser.add_argument('--out',            default='sign_problem_lindbladian.png')
    args = parser.parse_args()

    nf_init, nf_opt, nf_n = _load(Path(args.nofield_dir),    args.nofield_tag)
    tx_init, tx_opt, tx_n = _load(Path(args.transverse_dir), args.transverse_tag)

    gamma_vals = GAMMA_STEP * np.arange(N_GAMMA)
    gp_vals    = GAMMA_STEP * np.arange(N_GP)

    # Shared colour scale across all four panels for fair comparison.
    all_vals = np.concatenate([nf_init.ravel(), nf_opt.ravel(),
                               tx_init.ravel(), tx_opt.ravel()])
    finite   = all_vals[np.isfinite(all_vals)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        r'Optimised sign problem  $s = |\,\mathrm{tr}\,U / \mathrm{tr}\,|U|\,|$'
        '\n'
        r'Trotter step $e^{L\,dt}$  —  $J=1$, $dt=0.01$',
        fontsize=12,
    )

    _panel(axes[0, 0], nf_init, gamma_vals, gp_vals,
           'No field  —  s in Pauli basis',          vmin=vmin, vmax=vmax)
    _panel(axes[0, 1], nf_opt,  gamma_vals, gp_vals,
           'No field  —  optimised s',               vmin=vmin, vmax=vmax)
    _panel(axes[1, 0], tx_init, gamma_vals, gp_vals,
           r'Transverse $h=1$  —  s in Pauli basis', vmin=vmin, vmax=vmax)
    _panel(axes[1, 1], tx_opt,  gamma_vals, gp_vals,
           r'Transverse $h=1$  —  optimised s',      vmin=vmin, vmax=vmax)

    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    print(f'Saved {args.out}')

    for label, arr in [('nofield/s_init',    nf_init),
                       ('nofield/s_opt',     nf_opt),
                       ('transverse/s_init', tx_init),
                       ('transverse/s_opt',  tx_opt)]:
        valid = arr[~np.isnan(arr)]
        if len(valid):
            print(f'  {label}: min={valid.min():.4f}  max={valid.max():.4f}  '
                  f'median={np.median(valid):.4f}')

    summary_path = Path(args.out).with_suffix('.npz')
    np.savez(summary_path,
             nofield_s_init    = nf_init,
             nofield_s_opt     = nf_opt,
             nofield_n_opt     = nf_n,
             transverse_s_init = tx_init,
             transverse_s_opt  = tx_opt,
             transverse_n_opt  = tx_n,
             gamma_values      = gamma_vals,
             gp_values         = gp_vals)
    print(f'Saved {summary_path}')


if __name__ == '__main__':
    main()
