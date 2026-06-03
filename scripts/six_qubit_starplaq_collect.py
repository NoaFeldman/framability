"""
Aggregate per-point results from six_qubit_starplaq_worker.py into a single
summary .npz, and produce a multi-panel colormap figure analogous to
results/two_qubit_scan_full.png.

Grid: 10 x 10 over (gamma_s, gamma_p), step 0.4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

GAMMA_STEP = 0.4
N_GRID     = int(round(4.0 / GAMMA_STEP))   # 10

# Output keys we care about
_KEYS = [
    'ss_vn_entropy',
    'neg_urdl_urru', 'neg_dl_urur',
    'lpdo_urdl_urru', 'lpdo_dl_urur',
    'decay_rate',
    'otoc_small', 'otoc_large',
    'channel_stab_purity',
    'pauli_fra', 'opt_fra_4', 'opt_fra_6',
]

_PANEL_TITLES = {
    'ss_vn_entropy':       r'Von Neumann entropy of $\rho_{ss}$',
    'neg_urdl_urru':       r'Negativity  $[u,r,d,l]|[ur,ru]$',
    'neg_dl_urur':         r'Negativity  $[d,l]|[u,r,ur,ru]$',
    'lpdo_urdl_urru':      r'LPDO bond entropy  $[u,r,d,l]|[ur,ru]$',
    'lpdo_dl_urur':        r'LPDO bond entropy  $[d,l]|[u,r,ur,ru]$',
    'decay_rate':          r'Lindbladian decay rate',
    'otoc_small':          r'OTOC at $t=0.1\min(\gamma_s,\gamma_p)$',
    'otoc_large':          r'OTOC at $t=10\max(\gamma_s,\gamma_p)$',
    'channel_stab_purity': r'Channel stabilizer purity',
    'pauli_fra':           r'Pauli framability',
    'opt_fra_4':           r'Optimised framability ($d_\mathrm{ext}=4$)',
    'opt_fra_6':           r'Optimised framability ($d_\mathrm{ext}=6$)',
}


def _load(in_dir: Path):
    grids = {k: np.full((N_GRID, N_GRID), np.nan) for k in _KEYS}
    n_missing = 0
    for ig in range(N_GRID):
        for igp in range(N_GRID):
            f = in_dir / f'starplaq_{ig:03d}_{igp:03d}.npz'
            if not f.exists():
                n_missing += 1
                continue
            d = np.load(f)
            for k in _KEYS:
                if k in d.files:
                    grids[k][ig, igp] = float(d[k])
    if n_missing:
        print(f'Warning: {n_missing}/{N_GRID * N_GRID} files missing')
    return grids


def _panel(ax, data, gamma_vals, title, *, cmap='viridis',
           contour_at=None, contour_color='white'):
    edges = np.append(gamma_vals - GAMMA_STEP / 2,
                      gamma_vals[-1] + GAMMA_STEP / 2)
    im = ax.pcolormesh(edges, edges, data, cmap=cmap, shading='flat')
    if contour_at is not None and np.any(np.isfinite(data)):
        finite = data[np.isfinite(data)]
        if finite.min() < contour_at < finite.max():
            try:
                ax.contour(gamma_vals, gamma_vals, data,
                           levels=[contour_at], colors=contour_color,
                           linewidths=0.9)
            except Exception:
                pass
    plt.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\gamma_p$")
    ax.set_ylabel(r"$\gamma_s$")
    ax.set_title(title, fontsize=10)
    n_pts = int(np.sum(np.isfinite(data)))
    ax.text(0.98, 0.98, f'{n_pts}/{N_GRID*N_GRID}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, color='white')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='results_six_starplaq')
    parser.add_argument('--out',    default='results_plots/six_starplaq.png')
    parser.add_argument('--summary_out', default=None,
                        help='Path for the summary npz (default: <in_dir>/summary.npz).')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    grids = _load(in_dir)
    gamma_vals = GAMMA_STEP * np.arange(N_GRID)

    summary_path = (Path(args.summary_out) if args.summary_out
                    else in_dir / 'six_starplaq_summary.npz')
    np.savez(summary_path,
             gamma_values=gamma_vals,
             **{k: grids[k] for k in _KEYS})
    print(f'Saved {summary_path}')

    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    fig.suptitle(
        r'6-qubit star+plaquette Lindbladian'
        '\n'
        r'$H=h(X_u+X_r)+\lambda(Z_u+Z_r)$,  '
        r'$L_s=\sqrt{\gamma_s}\,X_uX_rX_dX_l$,  '
        r'$L_p=\sqrt{\gamma_p}\,Z_uZ_rZ_{ur}Z_{ru}$;  '
        r'$h=\lambda=1$,  $dt=0.04$',
        fontsize=12,
    )

    layout = [
        ['ss_vn_entropy',   'neg_urdl_urru',   'neg_dl_urur'],
        ['lpdo_urdl_urru',  'lpdo_dl_urur',    'decay_rate'],
        ['otoc_small',      'otoc_large',      'channel_stab_purity'],
        ['pauli_fra',       'opt_fra_4',       'opt_fra_6'],
    ]

    for r, row in enumerate(layout):
        for c, key in enumerate(row):
            data = grids[key]
            contour = 1.0 if key.startswith(('pauli_fra', 'opt_fra')) else None
            _panel(axes[r, c], data, gamma_vals, _PANEL_TITLES[key],
                   contour_at=contour)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f'Saved {args.out}')

    for k in _KEYS:
        finite = grids[k][np.isfinite(grids[k])]
        if len(finite):
            print(f'  {k:24s} min={finite.min():.4f}  max={finite.max():.4f}  '
                  f'median={np.median(finite):.4f}  n={len(finite)}')


if __name__ == '__main__':
    main()
