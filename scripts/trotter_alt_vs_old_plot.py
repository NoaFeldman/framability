"""
Heatmaps comparing the new alternating-scheme Heisenberg framabilities against
the old (Powell) scheme for one trotter scan model.

Reads the per-point keys written by scripts/trotter_alt_opt_worker.py:
    prev_fra_4/6  old-scheme value (captured before the in-place update)
    alt_fra_4/6   raw new-scheme (alternating + Polyak polish) value

Figure: one row per d_ext_single in {4, 6}, three columns --
    old scheme | new scheme          (shared framability colour scale incl. 1)
    old - new                        (diverging map; blue = new scheme better)
The improvement panel's title reports the fraction of points the new scheme
improves / worsens (beyond a 1e-6 tolerance) and the median/max improvement.

Usage:
    python scripts/trotter_alt_vs_old_plot.py --model model1 \
        --in_dir results_trotter_v3 \
        --out_png results_plots/trotter_alt_vs_old_model1.png
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

D_EXTS = (4, 6)
TOL = 1e-6          # |old - new| below this counts as a tie


def load_pairs(in_dir: Path, model) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{d_ext: (old, new)} grids of shape (N_X, N_Y); NaN where missing."""
    out = {de: (np.full((model.N_X, model.N_Y), np.nan),
                np.full((model.N_X, model.N_Y), np.nan)) for de in D_EXTS}
    n_loaded = 0
    mdir = in_dir / model.name
    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f)
                got = False
                for de in D_EXTS:
                    if f'prev_fra_{de}' in d and f'alt_fra_{de}' in d:
                        out[de][0][ix, iy] = float(d[f'prev_fra_{de}'])
                        out[de][1][ix, iy] = float(d[f'alt_fra_{de}'])
                        got = True
                n_loaded += got
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    print(f'Loaded {n_loaded}/{model.N_TOTAL} re-optimised points for '
          f'{model.name}.', flush=True)
    return out


def _edges(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 1:
        return np.array([vals[0] - 0.05, vals[0] + 0.05])
    step = np.diff(vals).mean()
    return np.concatenate([[vals[0] - step / 2],
                           (vals[:-1] + vals[1:]) / 2,
                           [vals[-1] + step / 2]])


def plot(pairs, model, out_png: Path) -> None:
    x_vals, y_vals = np.array(model.p1_vals), np.array(model.p2_vals)
    x_edges, y_edges = _edges(x_vals), _edges(y_vals)

    # Shared framability scale across all old/new panels, always spanning 1.
    finite = np.concatenate([g[np.isfinite(g)].ravel()
                             for de in D_EXTS for g in pairs[de]] or [np.zeros(1)])
    if finite.size == 0:
        print('No re-optimised points found -- nothing to plot.', flush=True)
        return
    vmin = min(float(finite.min()), 1.0)
    vmax = max(float(finite.max()), 1.0)
    if vmin == vmax:
        vmin, vmax = vmin - 0.05, vmax + 0.05

    fig, axes = plt.subplots(len(D_EXTS), 3, figsize=(14.5, 7.4),
                             constrained_layout=True, squeeze=False)
    fra_im = None
    fra_axes = []

    for r, de in enumerate(D_EXTS):
        old, new = pairs[de]
        imp = old - new                       # > 0: new scheme is better
        for c, (data, label) in enumerate(
                [(old, f'old scheme (Powell), d={de}'),
                 (new, f'new scheme (alternating), d={de}'),
                 (imp, f'old $-$ new, d={de}')]):
            ax = axes[r][c]
            if c < 2:
                im = ax.pcolormesh(x_edges, y_edges, data.T, cmap='viridis',
                                   vmin=vmin, vmax=vmax, shading='flat')
                fra_im = im
                fra_axes.append(ax)
                try:
                    ax.contour(x_vals, y_vals, data.T, levels=[1.0],
                               colors='white', linewidths=1.0)
                except Exception:
                    pass
                ax.set_title(label, fontsize=10)
            else:
                fin = imp[np.isfinite(imp)]
                lim = float(np.nanpercentile(np.abs(fin), 98)) if fin.size else 1.0
                lim = max(lim, 1e-12)
                im = ax.pcolormesh(x_edges, y_edges, data.T, cmap='RdBu',
                                   vmin=-lim, vmax=lim, shading='flat')
                fig.colorbar(im, ax=ax, pad=0.02, label='old $-$ new')
                if fin.size:
                    frac_up = float(np.mean(fin > TOL))
                    frac_dn = float(np.mean(fin < -TOL))
                    ax.set_title(f'{label}\nbetter {frac_up:.0%} | worse '
                                 f'{frac_dn:.0%} | median {np.median(fin):.2e} '
                                 f'| max {fin.max():.2e}', fontsize=9)
                else:
                    ax.set_title(label, fontsize=10)
            ax.set_xlabel(model.p1_label)
            ax.set_ylabel(model.p2_label)

    if fra_im is not None:
        cbar = fig.colorbar(fra_im, ax=fra_axes, pad=0.02)
        ticks = sorted(set(np.linspace(vmin, vmax, 5).tolist() + [1.0]))
        cbar.set_ticks(ticks)
        cbar.set_label('framability')

    fig.suptitle(f'{model.name}: alternating vs Powell Heisenberg framability'
                 f'\n{model.title}', fontsize=12)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_png}', flush=True)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',   type=str, required=True, choices=list(MODELS))
    p.add_argument('--in_dir',  type=str, default='results_trotter_v3')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_alt_vs_old_<model>.png')
    args = p.parse_args()

    model = MODELS[args.model]
    out_png = Path(args.out_png) if args.out_png else \
        Path('results_plots') / f'trotter_alt_vs_old_{model.name}.png'
    plot(load_pairs(Path(args.in_dir), model), model, out_png)


if __name__ == '__main__':
    main()
