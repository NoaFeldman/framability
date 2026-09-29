"""
Collect scripts/model10_seeded_worker.py, cross-evaluate neighbouring frames,
and draw the seeded-analysis figure (the main model10 figure is drawn by
scripts/model10_seeded_panels_collect.py).

1. Load the per-point files <out_dir>/model10_seeded[_s<stride>]/pt_*.npz
   (plus the _xeval files of earlier collects).
2. Neighbour cross-evaluation (in parallel, --n_proc processes): every point's
   frame is evaluated on the generators of the points within --radius4 /
   --radius8 grid steps (at d_ext = 8 also the neighbours' and the point's own
   d_ext = 4 frames, padded).  A candidate replaces the incumbent only if the
   independent per-column LP confirms it is lower, so every value stays a
   certified upper bound.  Jacobi sweeps repeat over the neighbourhoods of the
   changed points until nothing changes (or --max_sweeps).  Improved points
   are written to pt_<ix>_<iy>_xeval.npz (read back by the next collect).
3. Draw two rows:
      row A | seeded d_ext=4 | seeded d_ext=8 | gain d=4 | gain d=8
      row B | closed-form/structured d=4 | d=8 | origin d=4 | origin d=8
   The seeded-rate panels carry the analytic rate-zero curves of frame B
   (red) and of the continuous YZ-circle frame C (yellow dashed).
   Gains: previous optimised d_ext=4 minus seeded d_ext=4, and
   min(previous optimised d_ext=4, 6) minus seeded d_ext=8, the previous
   values read by scripts/model10_panels_common.load_base (first of
   --base_in_dirs holding a model10/ subdirectory, else --base_npz).

Outputs:
    --out_png (default <out_dir>/model10_seeded_extended.png)
    <out_dir>/model10_seeded_rates.npz      the new grids

Usage:
    python scripts/model10_seeded_collect.py
    python scripts/model10_seeded_collect.py --n_proc 8 --no_xeval
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS, MODEL10_J, MODEL10_H         # noqa: E402
from framability_rate_global import frame_rate_value, fit_columns        # noqa: E402
from framability_rate_families import b_boundary, c_boundary             # noqa: E402
from model10_seeded_worker import (MODEL, TOL, grid_vals, pt_dir_name,   # noqa: E402
                                   generator)
import model10_panels_common as common                                   # noqa: E402

D_EXTS = (4, 8)
XEVAL_SUFFIX = '_xeval'

# coarse origin categories for the label panels: (label prefix, category)
_CATEGORIES = [
    ('analytic B', 'B: rescaled Pauli (closed form)'),
    ('analytic C', 'C: YZ polygon (closed form)'),
    ('analytic P', 'P: Z proj + XY polygon (closed form)'),
    ('family yzxy', 'YZ polygon + XY pair'),
    ('family yz', 'YZ affine polygon'),
    ('family xy', 'Z pair + XY affine polygon'),
    ('family xp', 'X pair + YZ affine polygon'),
    ('stored', 'earlier optimiser frame'),
    ('library', 'gopt seed library'),
]


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------
def load_seeded(pt: Path, nx: int, ny: int, d_exts=D_EXTS) -> dict:
    """Grids of the seeded pipeline: rate_<m> / S_<m> / label_<m> take the
    minimum over the worker file and the _xeval file; fam_<m>, famlabel_<m>,
    prev_best_<m> come from the worker file."""
    g = {}
    for m in d_exts:
        for k in (f'rate_{m}', f'fam_{m}', f'prev_best_{m}'):
            g[k] = np.full((nx, ny), np.nan)
        for k in (f'S_{m}', f'label_{m}', f'famlabel_{m}'):
            g[k] = np.full((nx, ny), None, dtype=object)
    found = 0
    for ix in range(nx):
        for iy in range(ny):
            files = [pt / f'pt_{ix:03d}_{iy:03d}.npz',
                     pt / f'pt_{ix:03d}_{iy:03d}{XEVAL_SUFFIX}.npz']
            for i, f in enumerate(files):
                if not f.exists():
                    continue
                try:
                    d = np.load(f, allow_pickle=True)
                except Exception as e:                      # noqa: BLE001
                    print(f'  warning: {f.name}: {e}', flush=True)
                    continue
                found += (i == 0)
                for m in d_exts:
                    if f'rate_{m}' not in d.files:
                        continue
                    v = float(d[f'rate_{m}'])
                    cur = g[f'rate_{m}'][ix, iy]
                    if np.isfinite(v) and (not np.isfinite(cur) or v < cur):
                        g[f'rate_{m}'][ix, iy] = v
                        g[f'S_{m}'][ix, iy] = np.asarray(d[f'S_{m}'], float)
                        g[f'label_{m}'][ix, iy] = str(d[f'label_{m}'])
                    if i == 0:
                        for k in (f'fam_{m}', f'prev_best_{m}'):
                            if k in d.files:
                                g[k][ix, iy] = float(d[k])
                        if f'famlabel_{m}' in d.files:
                            g[f'famlabel_{m}'][ix, iy] = str(d[f'famlabel_{m}'])
    print(f'[{MODEL} seeded] {found}/{nx * ny} points loaded from {pt}', flush=True)
    return dict(n_points=found, **g)


# ---------------------------------------------------------------------------
#  Neighbour cross-evaluation
# ---------------------------------------------------------------------------
def _xeval_task(task):
    """Best confirmed candidate per d_ext for one point (runs in a worker)."""
    ix, iy, d1, d2, dim, own, cands = task
    A = generator(d1, d2, dim)
    res = {}
    for m, lst in cands.items():
        best_v, best = np.inf, None
        for label, S in lst:
            v = frame_rate_value(S, A)
            if np.isfinite(v) and v < best_v:
                best_v, best = v, (S, label)
        if best is None or best_v >= own[m] - TOL:
            continue
        v_ref = frame_rate_value(best[0], A, reference=True)
        if np.isfinite(v_ref) and v_ref < own[m] - TOL:
            res[m] = (max(float(v_ref), 0.0) if v_ref > -TOL else float(v_ref),
                      best[0], best[1])
    return ix, iy, res


def _build_task(ix, iy, g, p1, p2, dim, radius, d_exts):
    nx, ny = len(p1), len(p2)
    own, cands = {}, {}
    m_small = min(d_exts)
    for m in d_exts:
        v = g[f'rate_{m}'][ix, iy]
        own[m] = float(v) if np.isfinite(v) else np.inf
        if own[m] <= TOL:
            continue
        r = radius[m]
        lst = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < nx and 0 <= jy < ny):
                    continue
                if (dx, dy) != (0, 0):
                    S = g[f'S_{m}'][jx, jy]
                    if S is not None:
                        lst.append((g[f'label_{m}'][jx, jy], S))
                # smaller frames padded (the point's own included); nesting
                # makes the padded value an upper bound for the larger size
                if m > m_small and max(abs(dx), abs(dy)) <= radius[m_small]:
                    S = g[f'S_{m_small}'][jx, jy]
                    if S is not None:
                        lab = str(g[f'label_{m_small}'][jx, jy])
                        lst.append((lab if lab.startswith('d4 ') else f'd4 {lab}',
                                    fit_columns(S, m)))
        if lst:
            cands[m] = lst
    if not cands:
        return None
    return (ix, iy, float(p1[ix]), float(p2[iy]), dim, own, cands)


def xeval(g, p1, p2, pt: Path, *, dim, radius, n_proc, max_sweeps,
          d_exts=D_EXTS) -> int:
    """Jacobi neighbour sweeps; returns the number of improved points."""
    nx, ny = len(p1), len(p2)
    active = {(ix, iy) for ix in range(nx) for iy in range(ny)}
    improved = set()
    rmax = max(radius.values())
    ctx = mp.get_context('fork') if 'fork' in mp.get_all_start_methods() \
        else mp.get_context()
    for sweep in range(1, max_sweeps + 1):
        t0 = time.perf_counter()
        tasks = [t for t in (_build_task(ix, iy, g, p1, p2, dim, radius, d_exts)
                             for ix, iy in sorted(active)) if t is not None]
        if not tasks:
            break
        if n_proc > 1:
            with ctx.Pool(n_proc) as pool:
                results = pool.map(_xeval_task, tasks,
                                   chunksize=max(1, len(tasks) // (8 * n_proc)))
        else:
            results = [_xeval_task(t) for t in tasks]
        changed = set()
        n_upd = {m: 0 for m in d_exts}
        for ix, iy, res in results:
            for m, (v, S, label) in res.items():
                g[f'rate_{m}'][ix, iy] = v
                g[f'S_{m}'][ix, iy] = S
                g[f'label_{m}'][ix, iy] = label
                n_upd[m] += 1
                changed.add((ix, iy))
        improved |= changed
        print(f'  [xeval] sweep {sweep}: {len(tasks)} points evaluated, '
              + ', '.join(f'd{m}: {n_upd[m]} improved' for m in d_exts)
              + f' ({time.perf_counter() - t0:.0f}s)', flush=True)
        if not changed:
            break
        active = {(ix + dx, iy + dy) for ix, iy in changed
                  for dx in range(-rmax, rmax + 1) for dy in range(-rmax, rmax + 1)
                  if 0 <= ix + dx < nx and 0 <= iy + dy < ny}
    for ix, iy in sorted(improved):
        f = pt / f'pt_{ix:03d}_{iy:03d}{XEVAL_SUFFIX}.npz'
        tmp = f.with_suffix('.tmp.npz')
        payload = {}
        for m in d_exts:
            if g[f'S_{m}'][ix, iy] is None:
                continue
            payload[f'rate_{m}'] = g[f'rate_{m}'][ix, iy]
            payload[f'S_{m}'] = g[f'S_{m}'][ix, iy]
            payload[f'label_{m}'] = g[f'label_{m}'][ix, iy]
        np.savez(tmp, model=MODEL, ix=ix, iy=iy, **payload)
        os.replace(tmp, f)
    return len(improved)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------
def category(label, m: int) -> str:
    """Coarse origin of a frame label (see _CATEGORIES)."""
    if label is None:
        return ''
    s = str(label)
    opt = s.endswith(' +opt')
    if opt:
        s = s[:-len(' +opt')]
    if s.startswith('d4 '):
        return 'd=4 frame (no gain from d=8)' if m > 4 else s[3:]
    cat = next((c for p, c in _CATEGORIES if s.startswith(p)), s)
    return cat + (' + global opt' if opt else '')


def draw_category_panel(fig, ax, xv, yv, cats, title, *, xlabel, ylabel):
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    C = np.asarray(cats, dtype=object).T
    names = sorted({c for c in C.ravel() if c})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not names:
        ax.set_title(f'{title}\n(no data)', fontsize=10)
        return
    code = {c: i for i, c in enumerate(names)}
    Z = np.full(C.shape, np.nan)
    for idx, c in np.ndenumerate(C):
        if c:
            Z[idx] = code[c]
    colors = [colormaps['tab20'](i % 20) for i in range(len(names))]
    ax.pcolormesh(common.edges(xv), common.edges(yv), Z,
                  cmap=ListedColormap(colors), vmin=-0.5,
                  vmax=len(names) - 0.5, shading='flat')
    ax.legend(handles=[Patch(color=colors[i], label=c)
                       for i, c in enumerate(names)],
              fontsize=6, loc='upper right', framealpha=0.85)
    ax.set_title(title, fontsize=10)


def draw_analytic(ax, xv, yv, J, h) -> None:
    """Rate-zero curves: frame B (red) and continuous frame C (yellow)."""
    xmax, ymax = float(np.max(xv)), float(np.max(yv))
    xs = np.linspace(h, xmax, 300)
    yb = b_boundary(xs, J, h)
    ok = yb <= ymax
    ax.plot(xs[ok], yb[ok], color='red', lw=1.2, label='B: rescaled Pauli = 0')
    ys = np.linspace(0.2, ymax, 600)
    xc = c_boundary(ys, J)
    ok = xc <= xmax
    ax.plot(xc[ok], ys[ok], color='yellow', lw=1.2, ls='--',
            label='C: YZ circle = 0 (sufficient)')
    ax.legend(fontsize=6, loc='upper right', framealpha=0.6)


def plot_extended(new: dict, png: Path, *, floor: float) -> None:
    """The eight seeded-analysis panels (two rows of four)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model = MODEL
    m = MODELS[model]
    lab = dict(xlabel=m.p1_label, ylabel=m.p2_label)
    dl = r'$d_{\rm ext}'
    new_panels = [
        ('rate', 'rate_4', rf'Seeded Heisenberg rate ({dl}=4$)' '\n'
                           r'(analytic B/C seeds + structured families + global)'),
        ('rate', 'rate_8', rf'Seeded Heisenberg rate ({dl}=8$)' '\n'
                           r'(analytic B/C/P seeds + structured families + global)'),
        ('gain', 'gain_4', rf'gain: previous opt {dl}=4$ $-$ seeded {dl}=4$'),
        ('gain', 'gain_8', rf'gain: min(previous opt {dl}=4,6$) $-$ seeded {dl}=8$'),
        ('rate', 'fam_4', rf'Closed-form / structured frames only ({dl}=4$)'),
        ('rate', 'fam_8', rf'Closed-form / structured frames only ({dl}=8$)'),
        ('cat', 'cat_4', rf'origin of the best {dl}=4$ frame'),
        ('cat', 'cat_8', rf'origin of the best {dl}=8$ frame'),
    ]
    nrow = int(np.ceil(len(new_panels) / 4))
    fig, axes = plt.subplots(nrow, 4, figsize=(22, 5 * nrow),
                             constrained_layout=True)
    fig.suptitle(
        f'{model}:  {m.title}'
        '\n'
        r'seeded optimisation from the analytic rate-zero frames of the bond '
        r'generator (white: $\mu^*=0$; red: frame B boundary, yellow dashed: '
        r'continuous YZ-circle frame C, sufficient)',
        fontsize=13)
    flat = list(np.atleast_1d(axes).flat)
    J, h = MODEL10_J, MODEL10_H
    xv, yv = new['p1_vals'], new['p2_vals']
    for ax, (kind, key, title) in zip(flat, new_panels):
        if kind == 'rate':
            common.draw_panel(fig, ax, xv, yv, new[key], title, common.FRA_CMAP,
                              floor_contour=floor, **lab)
            draw_analytic(ax, xv, yv, J, h)
        elif kind == 'gain':
            common.draw_panel(fig, ax, xv, yv, new[key], title, common.MB_CMAP,
                              **lab)
        else:
            draw_category_panel(fig, ax, xv, yv, new[key], title, **lab)
    for ax in flat[len(new_panels):]:
        ax.axis('off')

    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[{model} seeded] wrote {png}', flush=True)


# ---------------------------------------------------------------------------
def _old_on_new_grid(bd, key, stride: int, nx: int, ny: int) -> np.ndarray:
    """A base-grid quantity sampled on the seeded grid (stride ratio)."""
    out = np.full((nx, ny), np.nan)
    if bd is None:
        return out
    Z = np.asarray(bd['rates'][key], float)
    r = stride // max(1, bd['stride'])
    if r < 1 or stride % max(1, bd['stride']):
        return out
    for ix in range(nx):
        for iy in range(ny):
            jx, jy = ix * r, iy * r
            if jx < Z.shape[0] and jy < Z.shape[1]:
                out[ix, iy] = Z[jx, jy]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default='results_model10_rate',
                    help='worker output root (holds model10_seeded/)')
    ap.add_argument('--stride', type=int, default=1,
                    help='stride the seeded worker ran with')
    ap.add_argument('--out_png', type=str, default=None,
                    help='default <out_dir>/model10_seeded_extended.png')
    ap.add_argument('--base_in_dirs', type=str, nargs='*',
                    default=['results_model10_rate', 'results_model4_rate'],
                    help='original rate pipeline roots, first with a model10/ '
                         'subdirectory wins (previous values for the gains)')
    ap.add_argument('--base_npz', type=str, nargs='*',
                    default=['results_model4_rate/model10_rate_panels.npz',
                             'results_model10_rate/model10_rate_panels.npz'],
                    help='fallback: stored data of the existing figure')
    ap.add_argument('--base_stride', type=int, default=1)
    ap.add_argument('--mb_stride', type=int, default=5)
    ap.add_argument('--floor', type=float, default=0.0)
    ap.add_argument('--no_xeval', action='store_true')
    ap.add_argument('--radius4', type=int, default=2)
    ap.add_argument('--radius8', type=int, default=1)
    ap.add_argument('--max_sweeps', type=int, default=6)
    ap.add_argument('--n_proc', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    ap.add_argument('--dim', type=int, default=None)
    args = ap.parse_args()
    dim = args.dim if args.dim is not None else MODELS[MODEL].dim

    p1, p2 = grid_vals(args.stride)
    nx, ny = len(p1), len(p2)
    pt = Path(args.out_dir) / pt_dir_name(args.stride)
    g = load_seeded(pt, nx, ny)
    if g['n_points'] == 0:
        sys.exit(f'no seeded worker output under {pt}')

    if not args.no_xeval:
        n_imp = xeval(g, p1, p2, pt, dim=dim,
                      radius={4: args.radius4, 8: args.radius8},
                      n_proc=args.n_proc, max_sweeps=args.max_sweeps)
        print(f'[{MODEL} seeded] neighbour cross-evaluation improved '
              f'{n_imp} point(s)', flush=True)

    bd = common.load_base(args.base_in_dirs, args.base_npz,
                          base_stride=args.base_stride, mb_stride=args.mb_stride)
    if bd is None:
        print(f'[{MODEL} seeded] WARNING: no base data (dirs {args.base_in_dirs}, '
              f'npz {args.base_npz}); the gain panels stay empty', flush=True)
    old4 = _old_on_new_grid(bd, 'rate_heis_4', args.stride, nx, ny)
    old6 = _old_on_new_grid(bd, 'rate_heis_6', args.stride, nx, ny)
    old46 = np.fmin(old4, old6)
    new = dict(p1_vals=p1, p2_vals=p2,
               rate_4=g['rate_4'], rate_8=g['rate_8'],
               fam_4=g['fam_4'], fam_8=g['fam_8'],
               gain_4=old4 - g['rate_4'], gain_8=old46 - g['rate_8'],
               cat_4=np.vectorize(lambda s: category(s, 4), otypes=[object])(
                   g['label_4']),
               cat_8=np.vectorize(lambda s: category(s, 8), otypes=[object])(
                   g['label_8']))

    # ---- summary -------------------------------------------------------
    for m, old in ((4, old4), (8, old46)):
        r = g[f'rate_{m}']
        ok = np.isfinite(r) & np.isfinite(old)
        print(f'  d{m}: {int(np.isfinite(r).sum())} pts, rate 0 at '
              f'{int((r <= 1e-6).sum())} (previous {int((old[ok] <= 1e-6).sum())} '
              f'of the same pts); improved by >1e-6 at '
              f'{int(((old - r)[ok] > 1e-6).sum())}, max gain '
              f'{np.nanmax(np.where(ok, old - r, np.nan)) if ok.any() else np.nan:.4g}',
              flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f'{MODEL}_seeded_rates.npz'
    x, y = MODELS[MODEL].p1_name, MODELS[MODEL].p2_name
    np.savez(npz, model=MODEL, stride=args.stride,
             **{f'{x}_vals': p1, f'{y}_vals': p2},
             **{k: new[k] for k in ('rate_4', 'rate_8', 'fam_4', 'fam_8',
                                    'gain_4', 'gain_8')},
             **{f'label_{m}': np.vectorize(lambda s: '' if s is None else str(s),
                                           otypes=['U64'])(g[f'label_{m}'])
                for m in D_EXTS},
             **{f'prev_best_{m}': g[f'prev_best_{m}'] for m in D_EXTS})
    print(f'[{MODEL} seeded] wrote {npz}', flush=True)

    png = Path(args.out_png or out_dir / f'{MODEL}_seeded_extended.png')
    plot_extended(new, png, floor=args.floor)


if __name__ == '__main__':
    main()
