"""
Aggregate xyz_chain_worker.py results and produce a multi-panel LINE plot
(one parameter gamma = gamma_L = gamma_R, so ordinary x-y plots, not colormaps).

Sweep: gamma in [0, 10] step 0.5 (units Jz=1), 21 points.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

GAMMA_STEP = 0.5
GAMMA_MAX  = 10.0
N_POINTS   = int(round(GAMMA_MAX / GAMMA_STEP)) + 1   # 21

_KEYS = [
    'ss_vn_entropy', 'neg_12_34', 'neg_1_234', 'lpdo_12_34', 'lpdo_1_234',
    'decay_rate', 'otoc_small', 'otoc_large', 'channel_stab_purity',
    'pauli_fra', 'opt_fra_4', 'opt_fra_6', 'sign_init', 'sign_opt',
    'nsign_init', 'nsign_opt',
]


def _load(in_dir: Path):
    g = {k: np.full(N_POINTS, np.nan) for k in _KEYS}
    gammas = np.full(N_POINTS, np.nan)
    missing = 0
    for i in range(N_POINTS):
        f = in_dir / f'xyz_{i:02d}.npz'
        if not f.exists():
            missing += 1
            continue
        d = np.load(f)
        gammas[i] = float(d['gamma'])
        for k in _KEYS:
            if k in d.files:
                g[k][i] = float(d[k])
    if missing:
        print(f'Warning: {missing}/{N_POINTS} files missing')
    # fill gamma axis for any missing points
    gammas = np.where(np.isfinite(gammas), gammas, GAMMA_STEP * np.arange(N_POINTS))
    return gammas, g


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', default='results_xyz_chain')
    p.add_argument('--out', default='results_plots/xyz_chain.png')
    p.add_argument('--summary_out', default=None)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    gammas, g = _load(in_dir)

    summary = (Path(args.summary_out) if args.summary_out
               else in_dir / 'xyz_chain_summary.npz')
    np.savez(summary, gamma=gammas, **{k: g[k] for k in _KEYS})
    print(f'Saved {summary}')

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle(
        r'Boundary-driven anisotropic XYZ chain ($N=4$):  '
        r'$J_x=1.5,\ J_y=0.5,\ J_z=1.0$,  '
        r'$L_L=\sqrt{\gamma}\,\sigma^+_1$, $L_R=\sqrt{\gamma}\,\sigma^-_N$,  '
        r'$\gamma_L=\gamma_R=\gamma$',
        fontsize=13,
    )

    def line(ax, series, title, ylabel):
        for label, key, style in series:
            ax.plot(gammas, g[key], style, label=label, markersize=4)
        ax.set_xlabel(r'$\gamma\ /\ J_z$')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize=8)

    line(axes[0, 0], [('VN entropy', 'ss_vn_entropy', 'o-')],
         'NESS von Neumann entropy', r'$S(\rho_{ss})$')
    line(axes[0, 1], [('[1,2]|[3,4]', 'neg_12_34', 'o-'),
                      ('[1]|[2,3,4]', 'neg_1_234', 's-')],
         'NESS negativity', r'$\mathcal{N}$')
    line(axes[0, 2], [('[1,2]|[3,4]', 'lpdo_12_34', 'o-'),
                      ('[1]|[2,3,4]', 'lpdo_1_234', 's-')],
         'LPDO bond entropy', r'$S_\mathrm{bond}$')
    line(axes[0, 3], [('decay rate', 'decay_rate', 'o-')],
         'Liouvillian gap', r'$|\mathrm{Re}\,\lambda_1|$')
    line(axes[1, 0], [(r'$t=0.1\gamma$', 'otoc_small', 'o-'),
                      (r'$t=10\gamma$', 'otoc_large', 's-')],
         'OTOC ($X_1, X_N$)', 'OTOC')
    line(axes[1, 1], [('stab. purity', 'channel_stab_purity', 'o-')],
         'Channel stabilizer purity', r'$\log_2(\cdot)$')
    line(axes[1, 2], [('Pauli', 'pauli_fra', 'o-'),
                      (r'opt $d_\mathrm{ext}=4$', 'opt_fra_4', 's-'),
                      (r'opt $d_\mathrm{ext}=6$', 'opt_fra_6', '^-')],
         'Framability', 'framability')
    axes[1, 2].axhline(1.0, color='red', ls='--', lw=0.8)
    line(axes[1, 3], [('Pauli basis', 'nsign_init', 'o-'),
                      ('local-opt', 'nsign_opt', 's-')],
         r'Sign problem  $-\log s$', r'$-\log s$')
    axes[1, 3].axhline(0.0, color='red', ls='--', lw=0.8)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'Saved {out}')

    for k in _KEYS:
        v = g[k][np.isfinite(g[k])]
        if len(v):
            print(f'  {k:20s} min={v.min():.4f} max={v.max():.4f} '
                  f'median={np.median(v):.4f} n={len(v)}')


if __name__ == '__main__':
    main()
