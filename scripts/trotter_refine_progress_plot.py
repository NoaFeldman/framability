"""
Plot the *mid-run* progress of the island-fix / floor-hunt refinement of the
Trotter-scan optimised framabilities (opt_fra_4 / opt_fra_6).

The new pipeline stages write loose per-point files that no existing figure
picks up until the final refine-collect merges them:

    pt_xeval_r<NN>_*   cross-eval sweep      (trotter_scan_cross_eval.py)
    pt_polish_r<NN>_*  island polish         (trotter_scan_island_polish_worker.py)
    pt_fhunt_r<NN>_*   floor hunt            (trotter_scan_floor_hunt_worker.py)

This script is read-only and merges purely in memory.  For each key it builds

    BEFORE = min over base + pt_refine_r* + pt_qrefine_r*      (pre-run state)
    AFTER  = min over BEFORE + pt_xeval_r* + pt_polish_r* + pt_fhunt_r*

and draws, per key, a three-panel row: BEFORE map, AFTER map (shared colour
scale spanning 1, white contour at framability = 1, red x on the remaining
'islands' — points > 1 whose finite neighbours all sit at the floor), and the
improvement BEFORE - AFTER (log scale).  Panel titles carry the framable-point
count, the island count, and — when a panel has no framable point at all (the
floor-hunt models) — how far the grid minimum is from the floor (min - 1),
before and after.  A per-flavour file count is printed so you can see how far
the run has progressed.

Usage:
    # all models -> results_plots/trotter_<model>_refine_progress.png
    python scripts/trotter_refine_progress_plot.py

    # one model, custom output
    python scripts/trotter_refine_progress_plot.py --model model3 \
        --out_png results_plots/model3_progress.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS

KEYS = ('opt_fra_4', 'opt_fra_6')
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_BASE_RE = re.compile(r'^pt_(\d{3})_(\d{3})\.npz$')
# stage name -> (regex, counts toward BEFORE?)
_STAGES = {
    'refine':  (re.compile(r'^pt_refine_r\d{2}_(\d{3})_(\d{3})\.npz$'),  True),
    'qrefine': (re.compile(r'^pt_qrefine_r\d{2}_(\d{3})_(\d{3})\.npz$'), True),
    'xeval':   (re.compile(r'^pt_xeval_r\d{2}_(\d{3})_(\d{3})\.npz$'),   False),
    'polish':  (re.compile(r'^pt_polish_r\d{2}_(\d{3})_(\d{3})\.npz$'),  False),
    'fhunt':   (re.compile(r'^pt_fhunt_r\d{2}_(\d{3})_(\d{3})\.npz$'),   False),
    'verdict': (re.compile(r'^pt_verdict_r\d{2}_(\d{3})_(\d{3})\.npz$'), False),
}


def _edges(v: np.ndarray) -> np.ndarray:
    """Bin edges for pcolormesh from grid centres (uniform spacing assumed)."""
    v = np.asarray(v, dtype=float)
    if len(v) == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    d = np.diff(v) / 2.0
    return np.concatenate([[v[0] - d[0]], v[:-1] + d, [v[-1] + d[-1]]])


def load_before_after(in_dir: Path, model, fra_tol: float):
    """(before, after) dicts of (N_X, N_Y) arrays per key, plus file counts."""
    nx, ny = model.N_X, model.N_Y
    before = {k: np.full((nx, ny), np.nan) for k in KEYS}
    after  = {k: np.full((nx, ny), np.nan) for k in KEYS}
    counts = {'base': 0, **{s: 0 for s in _STAGES}}

    mdir = in_dir / model.name
    if not mdir.is_dir():
        print(f'  (no directory {mdir})', flush=True)
        return before, after, counts

    def _lower(arr_dict, ix, iy, d):
        for k in KEYS:
            if k not in d:
                continue
            v = float(d[k])
            if not np.isfinite(v):
                continue
            cur = arr_dict[k][ix, iy]
            if not np.isfinite(cur) or v < cur:
                arr_dict[k][ix, iy] = v

    # Base files first so refinements only lower existing points.
    entries = sorted(mdir.iterdir())
    for f in entries:
        m = _BASE_RE.match(f.name)
        if not m:
            continue
        ix, iy = int(m.group(1)), int(m.group(2))
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        try:
            d = np.load(f)
        except Exception as e:
            print(f'  warning: {f.name}: {e}', flush=True)
            continue
        counts['base'] += 1
        _lower(before, ix, iy, d)

    for f in entries:
        for stage, (rex, is_before) in _STAGES.items():
            m = rex.match(f.name)
            if not m:
                continue
            ix, iy = int(m.group(1)), int(m.group(2))
            if not (0 <= ix < nx and 0 <= iy < ny):
                break
            try:
                d = np.load(f)
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
                break
            counts[stage] += 1
            if is_before:
                _lower(before, ix, iy, d)
            else:
                _lower(after, ix, iy, d)
            break

    for k in KEYS:
        after[k] = np.fmin(before[k], after[k])   # fmin: NaN-ignoring minimum

    return before, after, counts


def islands(vals: np.ndarray, fra_tol: float) -> list[tuple[int, int]]:
    """Points > 1 whose finite 4-connected neighbours ALL sit at the floor."""
    nx, ny = vals.shape
    out = []
    for ix in range(nx):
        for iy in range(ny):
            v = vals[ix, iy]
            if not np.isfinite(v) or v <= 1.0 + fra_tol:
                continue
            nbs = [vals[ix + dx, iy + dy] for dx, dy in NEIGHBORS
                   if 0 <= ix + dx < nx and 0 <= iy + dy < ny]
            nbs = [w for w in nbs if np.isfinite(w)]
            if nbs and all(w <= 1.0 + fra_tol for w in nbs):
                out.append((ix, iy))
    return out


def _panel_title(tag: str, vals: np.ndarray, isl, fra_tol: float) -> str:
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return f'{tag}\n(no data)'
    n_floor = int(np.sum(finite <= 1.0 + fra_tol))
    t = f'{tag}\nframable: {n_floor}/{finite.size}   islands: {len(isl)}'
    if n_floor == 0:
        t += f'   min$-1$ = {float(finite.min()) - 1.0:.3g}'
    return t


def _draw_map(ax, vals, model, x_edges, y_edges, vmin, vmax, isl, fra_tol):
    im = ax.pcolormesh(x_edges, y_edges, vals.T, cmap='viridis',
                       vmin=vmin, vmax=vmax, shading='flat')
    # thin white separator between framability = 1 and > 1 (as in collect)
    finite = vals[np.isfinite(vals)]
    if finite.size and finite.min() <= 1.0 < finite.max():
        ax.contour(model.p1_vals, model.p2_vals, vals.T,
                   levels=[1.0], colors='w', linewidths=0.7)
    if isl:
        xs = [model.p1_vals[ix] for ix, _ in isl]
        ys = [model.p2_vals[iy] for _, iy in isl]
        ax.plot(xs, ys, 'x', color='red', ms=6, mew=1.5)
    ax.set_xlabel(model.p1_label)
    ax.set_ylabel(model.p2_label)
    return im


def plot_model(model, before, after, counts, fra_tol, out_png: Path) -> None:
    x_edges = _edges(np.array(model.p1_vals))
    y_edges = _edges(np.array(model.p2_vals))

    # one colour scale across all four maps, always spanning 1
    finite = np.concatenate([a[np.isfinite(a)].ravel()
                             for d in (before, after) for a in d.values()]
                            or [np.array([1.0])])
    vmin = min(float(finite.min()), 1.0) if finite.size else 0.0
    vmax = max(float(finite.max()), 1.0) if finite.size else 2.0
    if vmin == vmax:
        vmin, vmax = vmin - 0.05, vmax + 0.05

    fig, axes = plt.subplots(len(KEYS), 3, figsize=(15, 4.2 * len(KEYS)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    fra_im, fra_axes = None, []

    for r, key in enumerate(KEYS):
        b, a = before[key], after[key]
        isl_b = islands(b, fra_tol)
        isl_a = islands(a, fra_tol)

        im = _draw_map(axes[r, 0], b, model, x_edges, y_edges, vmin, vmax,
                       isl_b, fra_tol)
        axes[r, 0].set_title(_panel_title(f'{key} — before this run', b,
                                          isl_b, fra_tol), fontsize=10)
        fra_im = im
        fra_axes.append(axes[r, 0])

        im = _draw_map(axes[r, 1], a, model, x_edges, y_edges, vmin, vmax,
                       isl_a, fra_tol)
        axes[r, 1].set_title(_panel_title(f'{key} — with xeval/polish/fhunt',
                                          a, isl_a, fra_tol), fontsize=10)
        fra_im = im
        fra_axes.append(axes[r, 1])

        # improvement panel (log scale; masked where nothing changed)
        delta = np.where(np.isfinite(b) & np.isfinite(a), b - a, np.nan)
        delta = np.where(delta > 1e-12, delta, np.nan)
        ax = axes[r, 2]
        n_impr = int(np.isfinite(delta).sum())
        n_new_floor = int(np.sum(np.isfinite(b) & np.isfinite(a)
                                 & (b > 1.0 + fra_tol) & (a <= 1.0 + fra_tol)))
        if n_impr:
            dmin, dmax = np.nanmin(delta), np.nanmax(delta)
            imd = ax.pcolormesh(x_edges, y_edges, delta.T, cmap='magma',
                                norm=LogNorm(vmin=dmin, vmax=max(dmax, dmin * 1.01)),
                                shading='flat')
            fig.colorbar(imd, ax=ax, pad=0.02, label='improvement')
        else:
            ax.text(0.5, 0.5, 'no improvement yet', ha='center', va='center',
                    transform=ax.transAxes)
        ax.set_title(f'{key} — improvement (before $-$ after)\n'
                     f'{n_impr} point(s) lowered, {n_new_floor} newly framable',
                     fontsize=10)
        ax.set_xlabel(model.p1_label)
        ax.set_ylabel(model.p2_label)

        print(f'  {key}: islands {len(isl_b)} -> {len(isl_a)},  '
              f'{n_impr} improved,  {n_new_floor} newly framable', flush=True)

    if fra_im is not None:
        cbar = fig.colorbar(fra_im, ax=fra_axes, pad=0.02)
        ticks = sorted(set(np.linspace(vmin, vmax, 5).tolist() + [1.0]))
        cbar.set_ticks(ticks)
        cbar.set_label('framability')

    stage_str = '  '.join(f'{s}: {counts[s]}' for s in
                          ('base', 'refine', 'qrefine', 'xeval', 'polish',
                           'fhunt', 'verdict'))
    fig.suptitle(f'{model.name}  —  {model.title}\nfiles on disk:  {stage_str}',
                 fontsize=12)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f'  wrote {out_png}', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',   type=str, default=None, choices=list(MODELS),
                   help='default: all models')
    p.add_argument('--in_dir',  type=str, default='results_trotter_v3')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_<model>_refine_progress.png '
                        '(only valid with a single --model)')
    p.add_argument('--fra_tol', type=float, default=1e-6,
                   help='tolerance for framability == 1')
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    names = [args.model] if args.model else list(MODELS)
    if args.out_png and len(names) > 1:
        p.error('--out_png requires a single --model')

    for name in names:
        model = MODELS[name]
        out_png = Path(args.out_png) if args.out_png else \
            Path('results_plots') / f'trotter_{model.name}_refine_progress.png'
        print(f'[{model.name}] loading {in_dir / model.name} ...', flush=True)
        before, after, counts = load_before_after(in_dir, model, args.fra_tol)
        if not any(np.isfinite(a).any() for a in before.values()):
            print(f'  no data for {model.name}; skipping figure.', flush=True)
            continue
        plot_model(model, before, after, counts, args.fra_tol, out_png)


if __name__ == '__main__':
    main()
