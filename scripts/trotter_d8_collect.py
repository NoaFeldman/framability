"""
Collect and plot the d_ext_single=8 Heisenberg optimised framability (opt_fra_8)
backfilled by scripts/trotter_d8_worker.py.

Two panels over the model's (p1, p2) grid, in the same style as the main scan
figure's framability row:

    (d) opt_fra_8                          -- the optimised framability itself,
                                              viridis, colour scale spanning 1.0,
                                              a thin white contour at framability=1
                                              separating framable (=1) from >1.
    (d*) opt_fra_8 ^ (1/(dt*Delta_L))      -- normalised per equilibration time
                                              (Delta_L = Liouvillian gap = lind_rate),
                                              removing the per-point dt so the value
                                              measures the framability of the
                                              propagator over one equilibration time.

Usage:
    python scripts/trotter_d8_collect.py --model model7a \
        --in_dir results_trotter_v3 --out_png results_plots/trotter_model7a_d8.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS

FRA_ONE_TOL = 1e-6


def load(in_dir: Path, model):
    """(N_X, N_Y) arrays: opt_fra_8, dt, lind_rate; NaN where missing."""
    fra = np.full((model.N_X, model.N_Y), np.nan)
    dt = np.full((model.N_X, model.N_Y), np.nan)
    rate = np.full((model.N_X, model.N_Y), np.nan)
    n = 0
    mdir = in_dir / model.name
    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f)
                if 'opt_fra_8' in d:
                    fra[ix, iy] = float(d['opt_fra_8'])
                    n += 1
                if 'dt' in d:
                    dt[ix, iy] = float(d['dt'])
                if 'lind_rate' in d:
                    rate[ix, iy] = float(d['lind_rate'])
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    print(f'Loaded opt_fra_8 for {n}/{model.N_TOTAL} points of {model.name}.',
          flush=True)
    return fra, dt, rate


def _edges(vals):
    if len(vals) == 1:
        return np.array([vals[0] - 0.05, vals[0] + 0.05])
    step = np.diff(vals).mean()
    return np.concatenate([[vals[0] - step / 2],
                           (vals[:-1] + vals[1:]) / 2, [vals[-1] + step / 2]])


def _fra_limits(data):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 2.0
    vmin, vmax = min(float(finite.min()), 1.0), max(float(finite.max()), 1.0)
    return (vmin - 0.05, vmax + 0.05) if vmin == vmax else (vmin, vmax)


def _derived_limits(p):
    finite = p[np.isfinite(p)]
    if finite.size == 0:
        return 0.0, 2.0
    vmin = min(float(np.nanpercentile(finite, 2)), 1.0)
    vmax = max(float(np.nanpercentile(finite, 98)), 1.0 + 1e-3)
    return (vmin - 0.05, vmax + 0.05) if vmin == vmax else (vmin, vmax)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',   type=str, required=True, choices=list(MODELS))
    p.add_argument('--in_dir',  type=str, default='results_trotter_v3')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_<model>_d8.png')
    args = p.parse_args()

    model = MODELS[args.model]
    in_dir = Path(args.in_dir)
    out_png = Path(args.out_png) if args.out_png else \
        Path('results_plots') / f'trotter_{model.name}_d8.png'

    fra, dt, rate = load(in_dir, model)
    x_vals, y_vals = np.array(model.p1_vals), np.array(model.p2_vals)
    x_edges, y_edges = _edges(x_vals), _edges(y_vals)

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        derived = np.power(fra, 1.0 / (dt * rate))         # per equilibration time

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    # (d) opt_fra_8
    ax = axes[0]
    vmin, vmax = _fra_limits(fra)
    im = ax.pcolormesh(x_edges, y_edges, fra.T, cmap='viridis',
                       vmin=vmin, vmax=vmax, shading='flat')
    title = '(d) Opt framability H (d=8)'
    finite = fra[np.isfinite(fra)]
    if finite.size and float(finite.min()) > 1.0 + FRA_ONE_TOL:
        title += f'\nmin$-1$ = {float(finite.min()) - 1.0:.3g}'
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(model.p1_label); ax.set_ylabel(model.p2_label)
    try:
        ax.contour(x_vals, y_vals, fra.T, levels=[1.0], colors='white',
                   linewidths=1.0)
    except Exception:
        pass
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_ticks(sorted(set(np.linspace(vmin, vmax, 5).tolist() + [1.0])))
    cbar.set_label('framability')

    # (d*) opt_fra_8 ^ (1/(dt*Delta_L))
    ax = axes[1]
    dvmin, dvmax = _derived_limits(derived)
    im = ax.pcolormesh(x_edges, y_edges, derived.T, cmap='magma',
                       vmin=dvmin, vmax=dvmax, shading='flat')
    title = (r'(d$^\star$) Opt framability H (d=8)'
             r'$^{1/(\Delta t\,\Delta_{\mathrm{L}})}$')
    finite = derived[np.isfinite(derived)]
    if finite.size and float(finite.min()) > 1.0 + FRA_ONE_TOL:
        title += f'\nmin$-1$ = {float(finite.min()) - 1.0:.3g}'
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(model.p1_label); ax.set_ylabel(model.p2_label)
    try:
        ax.contour(x_vals, y_vals, derived.T, levels=[1.0 + FRA_ONE_TOL],
                   colors='white', linewidths=1.0)
    except Exception:
        pass
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r'framability$^{1/(\Delta t\,\Delta_{\mathrm{L}})}$')

    fig.suptitle(f'{model.name}:  {model.title}', fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_png}', flush=True)
    plt.close(fig)


if __name__ == '__main__':
    main()
