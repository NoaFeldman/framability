"""
Loaders and panel drawing shared by the model10 figure scripts
(scripts/model10_seeded_panels_collect.py, scripts/model10_seeded_collect.py).

Self-contained on purpose: the same per-point loading rule and panel style as
scripts/model4_rate_panels_collect.py (min over base scan, refine rounds and
gopt files for the optimised rates; viridis rate panels with a white contour
on the mu* = 0 floor; magma for the many-body panels), without importing that
module, so the model10 figure has no dependency on the quality-factor code.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from model4_manybody_worker import mb_tag, N_QUBITS, mb_geometry          # noqa: E402
from model4_product_rate_worker import PROD_RATE_KEYS, prod_tag           # noqa: E402

MODEL = 'model10'

# base-scan rate keys used by the figure (the Schrodinger rates are replaced
# by the product-state rates) and the keys the refine pipelines improve
BASE_RATE_KEYS = [('rate_stab3', 'Stabilizer-3 framability rate'),
                  ('rate_pauli', 'Pauli framability rate'),
                  ('rate_heis_4', r'Opt Heisenberg rate ($d_{\rm ext}=4$)'),
                  ('rate_heis_6', r'Opt Heisenberg rate ($d_{\rm ext}=6$)')]
RATE_REFINE_KEYS = frozenset({'rate_heis_4', 'rate_heis_6'})
MB_KEYS = [('osc_rate', rf'{N_QUBITS}q osc rate  $\max_k|{{\rm Im}}\lambda_k/'
                        rf'{{\rm Re}}\lambda_k|$'),
           ('gap', f'{N_QUBITS}q Lindbladian gap')]

FRA_CMAP = 'viridis'
MB_CMAP = 'magma'
CONTOUR_LW = 1.3
CONTOUR_TOL = 1e-6          # rate within this of 0 = on the floor (white contour)
_LOG_DECADES = 4.0


def grid_vals(stride: int, model: str = MODEL):
    m = MODELS[model]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def load_group(pt_dir: Path, keys, stride: int, label: str, *,
               refine_keys=frozenset(), model: str = MODEL) -> dict:
    """Every key on the strided grid from per-point files pt_<ix>_<iy>.npz;
    keys in refine_keys take the minimum over the base file, the refine rounds
    (*refine_r*.npz) and the gopt files (*_gopt*.npz) -- each a certified
    upper bound."""
    p1_vals, p2_vals = grid_vals(stride, model)
    nx, ny = len(p1_vals), len(p2_vals)
    grids = {k: np.full((nx, ny), np.nan) for k in keys}
    found = 0
    for ix in range(nx):
        for iy in range(ny):
            base = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not base.exists():
                continue
            files = [base]
            if refine_keys:
                files += sorted(pt_dir.glob(f'pt_{ix:03d}_{iy:03d}_*refine_r*.npz'))
                files += sorted(pt_dir.glob(f'pt_{ix:03d}_{iy:03d}_gopt*.npz'))
            for f in files:
                try:
                    d = np.load(f, allow_pickle=True)
                except Exception as e:                      # noqa: BLE001
                    print(f'  warning: {f.name}: {e}', flush=True)
                    continue
                for k in keys:
                    if k not in d.files or not (f == base or k in refine_keys):
                        continue
                    v = float(d[k])
                    if not np.isfinite(grids[k][ix, iy]) or v < grids[k][ix, iy]:
                        grids[k][ix, iy] = v
            found += 1
    print(f'[{label}] {found}/{nx * ny} grid points loaded from {pt_dir}',
          flush=True)
    return dict(p1_vals=p1_vals, p2_vals=p2_vals, n_points=found, **grids)


def load_base(base_in_dirs, base_npz, *, base_stride: int = 1,
              mb_stride: int = 5) -> dict | None:
    """Base-scan rates and many-body panels of model10: the first of
    base_in_dirs holding a model10/ subdirectory, else the stored figure data
    (first existing base_npz).  None if neither exists."""
    for d in base_in_dirs:
        root = Path(d)
        if not (root / MODEL).is_dir():
            continue
        rates = load_group(root / MODEL, [k for k, _ in BASE_RATE_KEYS],
                           base_stride, f'{MODEL}-rates',
                           refine_keys=RATE_REFINE_KEYS)
        if rates['n_points'] == 0:
            continue
        mb = load_group(root / mb_tag(MODEL), [k for k, _ in MB_KEYS],
                        mb_stride, f'{MODEL}-{N_QUBITS}q')
        return dict(rates=rates, mb=mb, stride=base_stride,
                    source=str(root / MODEL))
    for f in base_npz:
        f = Path(f)
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        x, y = MODELS[MODEL].p1_name, MODELS[MODEL].p2_name
        rates = dict(p1_vals=d[f'{x}_vals'], p2_vals=d[f'{y}_vals'],
                     n_points=int(np.isfinite(d['rate_pauli']).sum()),
                     **{k: d[k] for k, _ in BASE_RATE_KEYS})
        mb = dict(p1_vals=d[f'mb_{x}_vals'], p2_vals=d[f'mb_{y}_vals'],
                  **{k: d[k] for k, _ in MB_KEYS})
        print(f'[{MODEL}] base panels from {f}', flush=True)
        return dict(rates=rates, mb=mb,
                    stride=int(d['stride']) if 'stride' in d.files else 1,
                    source=str(f))
    return None


def load_prod(prod_dirs, stride: int = 1) -> dict | None:
    """Product-state rates (scripts/model4_product_rate_worker.py) from the
    first of prod_dirs holding a model10_product/ subdirectory with data."""
    for d in prod_dirs:
        pt = Path(d) / prod_tag(MODEL)
        if not pt.is_dir():
            continue
        g = load_group(pt, [k for k, _ in PROD_RATE_KEYS], stride,
                       f'{MODEL}-product')
        if g['n_points'] > 0:
            return g
    return None


def edges(v):
    """Cell edges of a (possibly non-uniform) axis, for pcolormesh."""
    v = np.asarray(v, float)
    if v.size == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])


def draw_panel(fig, ax, xv, yv, Z, title, cmap, *, floor_contour=None,
               xlabel, ylabel) -> None:
    """One pcolormesh panel of Z[ix, iy] spanning exactly its finite data
    range (log scale beyond _LOG_DECADES decades); with floor_contour, a white
    outline of the region where Z is within CONTOUR_TOL of that value."""
    Zt = np.asarray(Z, float).T
    finite = np.isfinite(Zt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not finite.any():
        ax.set_title(f'{title}\n(no data)', fontsize=10)
        return
    vals = Zt[finite]
    lo, hi = float(vals.min()), float(vals.max())
    kw = dict(cmap=cmap, shading='flat')
    if lo > 0 and hi / lo > 10.0 ** _LOG_DECADES:
        from matplotlib.colors import LogNorm
        kw['norm'] = LogNorm(vmin=lo, vmax=hi)
        title = f'{title}  [log scale]'
    else:
        kw.update(vmin=lo, vmax=hi if hi > lo else lo + 1e-12)
    pcm = ax.pcolormesh(edges(xv), edges(yv), Zt, **kw)
    if floor_contour is not None:
        mask = np.where(finite, np.abs(Zt - floor_contour) <= CONTOUR_TOL,
                        False).astype(float)
        if 0.0 < mask.sum() < mask.size:
            ax.contour(np.asarray(xv, float), np.asarray(yv, float), mask,
                       levels=[0.5], colors='white', linewidths=CONTOUR_LW)
    ax.set_title(title, fontsize=10)
    fig.colorbar(pcm, ax=ax)
