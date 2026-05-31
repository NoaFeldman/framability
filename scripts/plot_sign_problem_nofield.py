"""
1x2 colormap of the optimised sign problem for the no-field two-qubit
Lindbladian Trotter step, over the (gamma, gamma') grid.

  s = |sum(U) / sum(|U|)|   -- s=1 means no sign problem (best),
                                s->0 is severe cancellation.

Reads results_sign_problem_nofield/sign_nofield_<ig>_<igp>.npz and writes
results_plots/sign_problem_nofield.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

GAMMA_MAX  = 8.0
GP_MAX     = 4.0
GAMMA_STEP = 0.2
N_GAMMA    = int(round(GAMMA_MAX / GAMMA_STEP)) + 1   # 41
N_GP       = int(round(GP_MAX    / GAMMA_STEP)) + 1   # 21


def _load(in_dir: Path, tag: str):
    s_init = np.full((N_GAMMA, N_GP), np.nan)
    s_opt  = np.full((N_GAMMA, N_GP), np.nan)
    missing = 0
    for ig in range(N_GAMMA):
        for igp in range(N_GP):
            f = in_dir / f'sign_{tag}_{ig:03d}_{igp:03d}.npz'
            if not f.exists():
                missing += 1
                continue
            d = np.load(f)
            s_init[ig, igp] = float(d['sign_init'])
            s_opt [ig, igp] = float(d['sign_opt'])
    if missing:
        print(f'Warning: {missing}/{N_GAMMA * N_GP} files missing')
    return s_init, s_opt


def _panel(ax, data, gamma_vals, gp_vals, title, *, vmin, vmax):
    gp_edges = np.append(gp_vals - GAMMA_STEP / 2, gp_vals[-1] + GAMMA_STEP / 2)
    g_edges  = np.append(gamma_vals - GAMMA_STEP / 2, gamma_vals[-1] + GAMMA_STEP / 2)
    im = ax.pcolormesh(gp_edges, g_edges, data, cmap='viridis',
                       vmin=vmin, vmax=vmax, shading='flat')
    # red contour at s=1 (no sign problem)
    if np.any(np.isfinite(data)):
        finite = data[np.isfinite(data)]
        if finite.min() < 1.0 < finite.max():
            try:
                ax.contour(gp_vals, gamma_vals, data,
                           levels=[1.0], colors='red', linewidths=1.2)
            except Exception:
                pass
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r'$\gamma$')
    ax.set_title(title, fontsize=11)
    n_pts = int(np.sum(np.isfinite(data)))
    ax.text(0.98, 0.98, f'{n_pts}/{N_GAMMA * N_GP}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='white')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='results_sign_problem_nofield')
    parser.add_argument('--tag',    default='nofield')
    parser.add_argument('--out',    default='results_plots/sign_problem_nofield.png')
    parser.add_argument('--title',  default=None,
                        help='Override the figure title prefix. '
                             'Default: derived from --tag (nofield -> h=0, '
                             'transverse -> h=1).')
    args = parser.parse_args()

    s_init, s_opt = _load(Path(args.in_dir), args.tag)
    gamma_vals = GAMMA_STEP * np.arange(N_GAMMA)
    gp_vals    = GAMMA_STEP * np.arange(N_GP)

    init_finite = s_init[np.isfinite(s_init)]
    opt_finite  = s_opt [np.isfinite(s_opt )]
    vmin_init, vmax_init = ((float(init_finite.min()), float(init_finite.max()))
                            if len(init_finite) else (0.0, 1.0))
    vmin_opt,  vmax_opt  = ((float(opt_finite .min()), float(opt_finite .max()))
                            if len(opt_finite ) else (0.0, 1.0))

    if args.title is not None:
        title_prefix = args.title
    elif args.tag == 'transverse':
        title_prefix = (r'Transverse-field two-qubit Lindbladian Trotter step  '
                        r'$e^{L\,dt}$  ($J=1$, $h=1$, $dt=0.01$)')
    else:
        title_prefix = (r'No-field two-qubit Lindbladian Trotter step  '
                        r'$e^{L\,dt}$  ($J=1$, $h=0$, $dt=0.01$)')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        title_prefix + r':  sign problem  $s = |\sum U|/\sum|U|$',
        fontsize=11,
    )

    _panel(axes[0], s_init, gamma_vals, gp_vals,
           r's in Pauli basis  (no rotation)',
           vmin=vmin_init, vmax=vmax_init)
    _panel(axes[1], s_opt,  gamma_vals, gp_vals,
           r'optimised s  (max over local $R^{\otimes 2}$)',
           vmin=vmin_opt,  vmax=vmax_opt)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f'Saved {out_path}')

    for name, arr in [('s_init', s_init), ('s_opt', s_opt)]:
        v = arr[np.isfinite(arr)]
        if len(v):
            print(f'  {name}: min={v.min():.4f}  max={v.max():.4f}  '
                  f'median={np.median(v):.4f}')


if __name__ == '__main__':
    main()
