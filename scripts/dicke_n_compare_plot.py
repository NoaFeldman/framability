"""
Cross-N comparison of the framability-rate panels of the atom-only open Dicke
model (trotter_lindbladian_scan._make_model_dicke, dicke_lindbladian.md), from
the per-N figure data of the seeded rate pipeline
(scripts/submit_dicke_rate.sh -> scripts/submit_rate_seeded.sh):

    results_dicke_N<N>_seeded/dicke_N<N>_seeded_rate_panels.npz

Figure 1 (<out>_panels.png):
    rows | N = N_1 | N_2 | ... | N_1 - N_last (difference)
    cols | Pauli | opt Heisenberg d_ext = 4 | 8 | 12
  one colour scale per column across the N rows; white contour = mu* = 0
  (within CONTOUR_TOL), dashed line = mean-field transition lam/lam_c = 1.
Figure 2 (<out>_cuts.png): mu* against lam/lam_c at the gamma_phi values
  --cuts, one curve per N, one panel per (column, cut).

Missing N files are skipped (with a message), so the script also draws a
partial comparison while some runs are still going.

    python scripts/dicke_n_compare_plot.py --ns 10 100 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (MODELS, MODEL_DICKE_OMEGA,          # noqa: E402
                                      MODEL_DICKE_OMEGA0, MODEL_DICKE_KAPPA)

CONTOUR_TOL = 1e-6          # as scripts/model10_panels_common.py
COLS = [('rate_pauli', 'Pauli'),
        ('panel_rate_heis_4', r'opt $d_{\rm ext}=4$'),
        ('panel_rate_heis_8', r'opt $d_{\rm ext}=8$'),
        ('panel_rate_heis_12', r'opt $d_{\rm ext}=12$')]


def load(n: int, root: Path) -> dict | None:
    model = f'dicke_N{n}'
    f = root / f'results_{model}_seeded' / f'{model}_seeded_rate_panels.npz'
    if not f.exists():
        print(f'[dicke compare] N={n}: {f} missing -- skipped', flush=True)
        return None
    d = np.load(f, allow_pickle=True)
    spec = MODELS[model]
    out = {'x': np.asarray(d[f'{spec.p1_name}_vals'], float),
           'y': np.asarray(d[f'{spec.p2_name}_vals'], float)}
    for k, _ in COLS:
        out[k] = np.asarray(d[k], float) if k in d.files else None
    return out


def draw_panel(ax, x, y, Z, vmin, vmax, cmap, zero_contour=True):
    im = ax.pcolormesh(x, y, Z.T, shading='nearest', cmap=cmap,
                       vmin=vmin, vmax=vmax)
    if zero_contour and np.isfinite(Z).any() and np.nanmin(Z) <= CONTOUR_TOL:
        ax.contour(x, y, (Z <= CONTOUR_TOL).astype(float).T, levels=[0.5],
                   colors='w', linewidths=1.2)
    ax.axvline(1.0, color='k', ls='--', lw=0.8)
    return im


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ns', type=int, nargs='+', default=[10, 100, 1000])
    p.add_argument('--root', default='.', help='repo root holding results_dicke_N*_seeded')
    p.add_argument('--out', default='results_model4_rate/dicke_n_compare')
    p.add_argument('--cuts', type=float, nargs='+', default=[0.0, 0.5],
                   help='gamma_phi values of the line-cut figure')
    args = p.parse_args()

    root = Path(args.root)
    data = {n: load(n, root) for n in args.ns}
    data = {n: d for n, d in data.items() if d is not None}
    if not data:
        sys.exit('[dicke compare] no data')
    ns = sorted(data)
    spec = MODELS[f'dicke_N{ns[0]}']
    xlab, ylab = spec.p1_label, spec.p2_label
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- figure 1: panels ----------------------------------------------------
    with_diff = len(ns) >= 2
    nrow = len(ns) + int(with_diff)
    fig, axes = plt.subplots(nrow, len(COLS), figsize=(4.0 * len(COLS), 3.2 * nrow),
                             squeeze=False, constrained_layout=True)
    for c, (key, label) in enumerate(COLS):
        grids = [data[n][key] for n in ns if data[n][key] is not None]
        if grids:
            vmax = max(float(np.nanmax(g)) for g in grids)
            vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        else:
            vmax = 1.0
        im = None
        for r, n in enumerate(ns):
            ax = axes[r, c]
            Z = data[n][key]
            if Z is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
            else:
                im = draw_panel(ax, data[n]['x'], data[n]['y'], Z, 0.0, vmax, 'viridis')
            ax.set_title(f'{label},  $N={n}$', fontsize=10)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
        if im is not None:
            fig.colorbar(im, ax=axes[:len(ns), c], shrink=0.8, label=r'$\mu^*$')
        if with_diff:
            ax = axes[-1, c]
            a, b = data[ns[0]], data[ns[-1]]
            if a[key] is not None and b[key] is not None and a[key].shape == b[key].shape:
                D = a[key] - b[key]
                lim = float(np.nanmax(np.abs(D))) if np.isfinite(D).any() else 0.0
                lim = lim if lim > 0 else 1e-12
                imd = draw_panel(ax, a['x'], a['y'], D, -lim, lim, 'RdBu_r',
                                 zero_contour=False)
                fig.colorbar(imd, ax=ax, shrink=0.8,
                             label=rf'$\mu^*_{{N={ns[0]}}}-\mu^*_{{N={ns[-1]}}}$')
            else:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
            ax.set_title(f'{label}:  $N={ns[0]}$ minus $N={ns[-1]}$', fontsize=10)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
    fig.suptitle('Open Dicke model (cavity eliminated), framability rate of the '
                 r'dimer patch $(N-1)\,\mathcal{L}_{ij}$;  '
                 rf'$\omega={MODEL_DICKE_OMEGA:g}$, $\kappa={MODEL_DICKE_KAPPA:g}$, '
                 rf'$\omega_0={MODEL_DICKE_OMEGA0:g}$', fontsize=11)
    f1 = out.with_name(out.name + '_panels.png')
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print(f'[dicke compare] wrote {f1}', flush=True)

    # ---- figure 2: line cuts at fixed gamma_phi ------------------------------
    fig, axes = plt.subplots(len(args.cuts), len(COLS),
                             figsize=(4.0 * len(COLS), 3.0 * len(args.cuts)),
                             squeeze=False, constrained_layout=True)
    for r, g in enumerate(args.cuts):
        for c, (key, label) in enumerate(COLS):
            ax = axes[r, c]
            for n in ns:
                d = data[n]
                if d[key] is None:
                    continue
                iy = int(np.argmin(np.abs(d['y'] - g)))
                ax.plot(d['x'], d[key][:, iy], 'o-', ms=3, label=f'$N={n}$')
            ax.axvline(1.0, color='k', ls='--', lw=0.8)
            ax.set_title(rf'{label},  $\gamma_\phi={g:g}$', fontsize=10)
            ax.set_xlabel(xlab)
            ax.set_ylabel(r'$\mu^*$')
            ax.legend(fontsize=8)
    f2 = out.with_name(out.name + '_cuts.png')
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f'[dicke compare] wrote {f2}', flush=True)

    # ---- numbers for the log ---------------------------------------------------
    for key, label in COLS:
        for n in ns:
            Z = data[n][key]
            if Z is None:
                continue
            nz = int(np.sum(Z <= CONTOUR_TOL))
            print(f'[dicke compare] {label:>22s} N={n:<5d} min {np.nanmin(Z):.4g} '
                  f'max {np.nanmax(Z):.4g}  rate-0 points {nz}/{Z.size}', flush=True)


if __name__ == '__main__':
    main()
