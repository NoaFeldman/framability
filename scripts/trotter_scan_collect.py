"""
Collect one model's Trotter-scan worker results into a summary npz and a
colormap figure.

The figure has one row per group letter (derived from QUANTITIES), so quantities
that share a letter sit on the same line:
    a : sign problem (min) | NESS LPDO | max LPDO (path) | Liouvillian gap
    b : <Z> | <X> | NESS VN entropy | NESS negativity
    c : stabilizer-3 framability | Pauli framability
    d : opt framability d=4 | opt framability d=6 | gamma_CH1 (product)
    e : product-state framability chi=10 | chi=20 | chi=40
    f : opt Schrodinger framability d=4 | d=6 | d=8
    g : dephasing opt framability H (d=4) | S (d=4)
    d*: opt framability d=4 ^(1/dt) | d=6 ^(1/dt)   [bottom row]

All framability panels share one colour scale that always includes 1.0, and each
draws a thin white contour at framability = 1 separating the framable (=1) region
from >1.

The extra bottom row (d*) rescales the optimised framability per unit time:
G = expm(L*dt) is one Trotter step, so a small (per-point adaptive) dt trivially
pushes G toward the identity and its framability toward 1.  Raising the
framability to 1/dt removes the dt-dependence and measures it per unit time, i.e.
the framability of the propagator evolved over one time unit expm(L) =
G^(1/dt).  The white contour is drawn only where fra^(1/dt) is still
<= 1 + FRA_ONE_TOL.

Usage:
    python scripts/trotter_scan_collect.py --model model1 \
        --in_dir results_trotter_v3 --out_png results_plots/trotter_model1.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, QUANTITIES

# One figure row per group letter, in order; derived from QUANTITIES so new
# groups (e, f, g, ...) can't drift out of sync with the stored quantities.
GROUPS = sorted({group for _, _, group, _ in QUANTITIES})
N_Q = len(QUANTITIES)

# A framability panel counts as having a framable point if its minimum reaches 1
# to within this tolerance (optimised values sit at 1 +/- ~1e-9 when framable).
FRA_ONE_TOL = 1e-6

# LPDO bond-entropy panels (a2 NESS, a4 max-along-path) get a thin white contour
# tracing the boundary of the vanishing-entropy region.  The entropy is >= 0 and
# sits at ~0 (to numerical noise) where the state is a product across the cut, so
# the contour is drawn at a small level rather than exactly 0.
LPDO_ZERO_LEVEL = 1e-6
LPDO_KEYS = ('lpdo', 'lpdo_max')


def load_results(in_dir: Path, model) -> tuple[np.ndarray, np.ndarray]:
    """Return ((N_X, N_Y, N_Q) quantity array, (N_X, N_Y) Trotter-step array);
    NaN where data is missing.  dt is loaded alongside the quantities because it
    varies per point and is needed for the per-unit-time framability panels."""
    arr = np.full((model.N_X, model.N_Y, N_Q), np.nan)
    dt_arr = np.full((model.N_X, model.N_Y), np.nan)
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
                if 'dt' in d:
                    dt_arr[ix, iy] = float(d['dt'])
                n_loaded += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    print(f'Loaded {n_loaded}/{model.N_TOTAL} points for {model.name}.', flush=True)
    return arr, dt_arr


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


def _derived_limits(panels: list) -> tuple[float, float]:
    """Shared (vmin, vmax) for the per-unit-time framability panels.
    These explode for non-framable points (fra>1 raised to 1/dt), so the
    upper limit is a robust percentile rather than the raw max; the scale spans 1."""
    finite = np.concatenate([p[np.isfinite(p)].ravel() for p in panels]) \
        if panels else np.array([])
    if finite.size == 0:
        return 0.0, 2.0
    vmin = min(float(np.nanpercentile(finite, 2)), 1.0)
    vmax = max(float(np.nanpercentile(finite, 98)), 1.0 + 1e-3)
    if vmin == vmax:
        vmin, vmax = vmin - 0.05, vmax + 0.05
    return vmin, vmax


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


def plot_colormaps(arr: np.ndarray, dt_arr: np.ndarray, model,
                   out_png: Path) -> None:
    x_vals, y_vals = np.array(model.p1_vals), np.array(model.p2_vals)
    x_edges, y_edges = _edges(x_vals), _edges(y_vals)

    fra_idx = [qi for qi, (_, _, _, is_fra) in enumerate(QUANTITIES) if is_fra]
    fra_vmin, fra_vmax = _shared_fra_limits(arr, fra_idx)

    # Group quantities by letter; one figure row per group.
    by_group: dict[str, list[int]] = {g: [] for g in GROUPS}
    for qi, (_, _, group, _) in enumerate(QUANTITIES):
        by_group[group].append(qi)

    # Extra bottom row: optimised framability normalised per unit time.
    # One Trotter step is G = expm(L*dt); a small dt trivially pushes G toward the
    # identity (hence framability -> 1), so the raw opt framability is not
    # comparable across points with different adaptive dt.  Raising the framability
    # to 1/dt removes the dt-dependence and estimates the framability of the
    # propagator over one time unit (submultiplicativity: fra(expm(L)) <=
    # fra(expm(L*dt))^(1/dt)).  A point is only framable under this stricter test
    # if fra^(1/dt) is still <= 1 + FRA_ONE_TOL, so the white contour is drawn at
    # that level.
    key_index = {key: qi for qi, (key, _, _, _) in enumerate(QUANTITIES)}
    DERIVED = [('opt_fra_4', r'Opt framability H (d=4)$^{1/\Delta t}$'),
               ('opt_fra_6', r'Opt framability H (d=6)$^{1/\Delta t}$')]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        inv_dt = 1.0 / dt_arr
        derived = [(label, np.power(arr[:, :, key_index[key]], inv_dt).T)
                   for key, label in DERIVED]
    dvmin, dvmax = _derived_limits([d for _, d in derived])

    n_cols = max(max(len(v) for v in by_group.values()), len(DERIVED))
    n_rows = len(GROUPS) + 1

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

            title = f'({g}) {label}'
            if is_fra:
                # A framability panel that never reaches 1 has no framable point;
                # annotate the title with how far the best point is from framable,
                # i.e. (min framability) - 1.
                finite = data[np.isfinite(data)]
                if finite.size and float(finite.min()) > 1.0 + FRA_ONE_TOL:
                    title += f'\nmin$-1$ = {float(finite.min()) - 1.0:.3g}'
            ax.set_title(title, fontsize=10)

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
                # white contour tracing where the LPDO bond entropy vanishes
                if key in LPDO_KEYS:
                    try:
                        ax.contour(x_vals, y_vals, data,
                                   levels=[LPDO_ZERO_LEVEL],
                                   colors='white', linewidths=1.0)
                    except Exception:
                        pass

    if fra_im is not None:
        cbar = fig.colorbar(fra_im, ax=fra_axes, pad=0.02)
        ticks = sorted(set(np.linspace(fra_vmin, fra_vmax, 5).tolist() + [1.0]))
        cbar.set_ticks(ticks)
        cbar.set_label('framability')

    # Bottom row: per-unit-time (fra^(1/dt)) opt-framability panels.
    r = len(GROUPS)
    der_im, der_axes = None, []
    for c in range(n_cols):
        ax = axes[r][c]
        if c >= len(derived):
            ax.set_visible(False)
            continue
        label, pdata = derived[c]
        im = ax.pcolormesh(x_edges, y_edges, pdata, cmap='magma',
                           vmin=dvmin, vmax=dvmax, shading='flat')
        ax.set_xlabel(model.p1_label)
        ax.set_ylabel(model.p2_label)

        title = f'(d$^\\star$) {label}'
        finite = pdata[np.isfinite(pdata)]
        if finite.size and float(finite.min()) > 1.0 + FRA_ONE_TOL:
            title += f'\nmin$-1$ = {float(finite.min()) - 1.0:.3g}'
        ax.set_title(title, fontsize=10)

        der_im = im
        der_axes.append(ax)
        # white contour only where fra^(1/dt) is still <= 1 + FRA_ONE_TOL
        try:
            ax.contour(x_vals, y_vals, pdata, levels=[1.0 + FRA_ONE_TOL],
                       colors='white', linewidths=1.0)
        except Exception:
            pass

    if der_im is not None:
        cbar = fig.colorbar(der_im, ax=der_axes, pad=0.02)
        cbar.set_label(r'framability$^{1/\Delta t}$')

    fig.suptitle(f'{model.name}:  {model.title}', fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_png}', flush=True)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',   type=str, required=True, choices=list(MODELS))
    p.add_argument('--in_dir',  type=str, default='results_trotter_v3')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_<model>.png')
    p.add_argument('--save_npz', action='store_true',
                   help='also save <in_dir>/<model>/scan_summary.npz')
    args = p.parse_args()

    model = MODELS[args.model]
    in_dir = Path(args.in_dir)
    out_png = Path(args.out_png) if args.out_png else \
        Path('results_plots') / f'trotter_{model.name}.png'

    arr, dt_arr = load_results(in_dir, model)
    if args.save_npz:
        npz = in_dir / model.name / 'scan_summary.npz'
        np.savez(npz, arr=arr, dt=dt_arr, p1=model.p1_vals, p2=model.p2_vals,
                 quantities=[k for k, _, _, _ in QUANTITIES])
        print(f'Saved {npz}', flush=True)

    plot_colormaps(arr, dt_arr, model, out_png)


if __name__ == '__main__':
    main()
