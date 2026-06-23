"""
Collect one model's Trotter-scan worker results into a summary npz and a
colormap figure.

The figure has one row per group letter (a, b, c, d), so quantities that share
a letter sit on the same line:
    a : sign problem (min) | NESS LPDO bond entropy | Liouvillian gap
    b : <Z> | <X> | NESS VN entropy | NESS negativity
    c : stabilizer framability | Pauli framability
    d : opt framability d=4 | opt framability d=6 | gamma_CH1 (product)

Framability panels (c1, c2, d1, d2, d3) share one colour scale that always
includes 1.0, and each draws a thin white contour at framability = 1 separating
the framable (=1) region from >1.

Usage:
    python scripts/trotter_scan_collect.py --model model1 \
        --in_dir results_trotter --out_png results_plots/trotter_model1.png
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
from trotter_lindbladian_scan import MODELS, QUANTITIES

GROUPS = ['a', 'b', 'c', 'd']
N_Q = len(QUANTITIES)


def load_results(in_dir: Path, model) -> np.ndarray:
    """Return (N_X, N_Y, N_Q) array; NaN where data is missing."""
    arr = np.full((model.N_X, model.N_Y, N_Q), np.nan)
    n_loaded = 0
    mdir = in_dir / model.name
    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f)
                for qi, (key, _, _, _) in enumerate(QUANTITIES):
                    if key in d:
                        arr[ix, iy, qi] = float(d[key])
                n_loaded += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    print(f'Loaded {n_loaded}/{model.N_TOTAL} points for {model.name}.', flush=True)
    return arr


def _edges(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 1:
        return np.array([vals[0] - 0.05, vals[0] + 0.05])
    step = np.diff(vals).mean()
    return np.concatenate([[vals[0] - step / 2],
                           (vals[:-1] + vals[1:]) / 2,
                           [vals[-1] + step / 2]])


def _robust_limits(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))
    if lo == hi:
        lo, hi = lo - 0.05, hi + 0.05
    return lo, hi


def _shared_fra_limits(arr: np.ndarray, fra_idx: list) -> tuple[float, float]:
    """Common (vmin, vmax) across all framability panels, always spanning 1.0."""
    finite = arr[:, :, fra_idx]
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 2.0
    vmin = min(float(finite.min()), 1.0)
    vmax = max(float(finite.max()), 1.0)
    if vmin == vmax:
        vmin, vmax = vmin - 0.05, vmax + 0.05
    return vmin, vmax


def plot_colormaps(arr: np.ndarray, model, out_png: Path) -> None:
    x_vals, y_vals = np.array(model.p1_vals), np.array(model.p2_vals)
    x_edges, y_edges = _edges(x_vals), _edges(y_vals)

    fra_idx = [qi for qi, (_, _, _, is_fra) in enumerate(QUANTITIES) if is_fra]
    fra_vmin, fra_vmax = _shared_fra_limits(arr, fra_idx)

    # Group quantities by letter; one figure row per group.
    by_group: dict[str, list[int]] = {g: [] for g in GROUPS}
    for qi, (_, _, group, _) in enumerate(QUANTITIES):
        by_group[group].append(qi)
    n_cols = max(len(v) for v in by_group.values())
    n_rows = len(GROUPS)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.6 * n_rows),
                             constrained_layout=True, squeeze=False)

    fra_im = None
    fra_axes = []

    for r, g in enumerate(GROUPS):
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(by_group[g]):
                ax.set_visible(False)
                continue
            qi = by_group[g][c]
            key, label, _, is_fra = QUANTITIES[qi]
            data = arr[:, :, qi].T          # (N_Y, N_X): y = p2, x = p1

            if is_fra:
                vmin, vmax = fra_vmin, fra_vmax
            else:
                vmin, vmax = _robust_limits(data)

            im = ax.pcolormesh(x_edges, y_edges, data, cmap='viridis',
                               vmin=vmin, vmax=vmax, shading='flat')
            ax.set_xlabel(model.p1_label)
            ax.set_ylabel(model.p2_label)
            ax.set_title(f'({g}) {label}', fontsize=10)

            if is_fra:
                fra_im = im
                fra_axes.append(ax)
                # thin white separator between framability = 1 and > 1
                try:
                    ax.contour(x_vals, y_vals, data, levels=[1.0],
                               colors='white', linewidths=1.0)
                except Exception:
                    pass
            else:
                fig.colorbar(im, ax=ax, pad=0.02)

    if fra_im is not None:
        cbar = fig.colorbar(fra_im, ax=fra_axes, pad=0.02)
        ticks = sorted(set(np.linspace(fra_vmin, fra_vmax, 5).tolist() + [1.0]))
        cbar.set_ticks(ticks)
        cbar.set_label('framability')

    fig.suptitle(f'{model.name}:  {model.title}', fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_png}', flush=True)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',   type=str, required=True, choices=list(MODELS))
    p.add_argument('--in_dir',  type=str, default='results_trotter')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_<model>.png')
    p.add_argument('--save_npz', action='store_true',
                   help='also save <in_dir>/<model>/scan_summary.npz')
    args = p.parse_args()

    model = MODELS[args.model]
    in_dir = Path(args.in_dir)
    out_png = Path(args.out_png) if args.out_png else \
        Path('results_plots') / f'trotter_{model.name}.png'

    arr = load_results(in_dir, model)
    if args.save_npz:
        npz = in_dir / model.name / 'scan_summary.npz'
        np.savez(npz, arr=arr, p1=model.p1_vals, p2=model.p2_vals,
                 quantities=[k for k, _, _, _ in QUANTITIES])
        print(f'Saved {npz}', flush=True)

    plot_colormaps(arr, model, out_png)


if __name__ == '__main__':
    main()
