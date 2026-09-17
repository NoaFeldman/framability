"""
dt -> 0 extrapolation of the random-frame DT_BASE lines
(scripts/dtbase_randframe_worker.py) into framability panels for
results_dtbase_line/<model>_dtbase_extrap.png.

Same construction as scripts/trotter_dtbase_line_extrap.py (it calls its
extrapolate(): fit of ln(framability)/dt over the fit_n smallest dt with
DT_BASE <= max_dt_base, plotted value exp(rate0)), on the full model grid;
points that were not swept (gamma' > 4.2) stay NaN and are drawn empty.

Not a standalone figure: trotter_dtbase_line_extrap.py and
collect_and_plot_all.py call extrapolate_model() + panels() and append the
six panels (kind='fra', framability styling) right after their own measures,
and store the grids in <model>_dtbase_extrap.npz.  Without data in rf_dir both
return empty, so the figure is unchanged.

Stand-alone use prints coverage only:
    python scripts/dtbase_randframe_collect.py --model model3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                              # noqa: E402
from trotter_dtbase_line_worker import base_grid, point_tag, N_BASE      # noqa: E402
from dtbase_randframe_worker import MEASURES, OUT_DIR_DEFAULT            # noqa: E402


def load_point(model: str, p1: float, p2: float, rf_dir: Path):
    """(dt_vals, {key: values}) over the base grid, or None without data."""
    pt_dir = rf_dir / point_tag(model, p1, p2)
    if not pt_dir.is_dir():
        return None
    dt_vals = np.full(N_BASE, np.nan)
    arrs = {k: np.full(N_BASE, np.nan) for k, _ in MEASURES}
    for idx in range(N_BASE):
        f = pt_dir / f'base_{idx:03d}.npz'
        if not f.exists():
            continue
        try:
            d = np.load(f, allow_pickle=True)
            dt_vals[idx] = float(d['dt'])
            for k, _ in MEASURES:
                if k in d.files:
                    arrs[k][idx] = float(d[k])
        except Exception as e:
            print(f'  warning: {f}: {e}', flush=True)
    if not np.isfinite(dt_vals).any():
        return None
    return dt_vals, arrs


def extrapolate_model(model: str, rf_dir: Path, *, fit_n: int, deg: int,
                      raw: bool, stride: int = 1,
                      max_dt_base: float = 0.10) -> dict | None:
    """{key: (nx, ny) dt=0 grid} on the same grid as the dtbase-line
    extrapolation, or None when rf_dir holds no data for `model`."""
    import trotter_dtbase_line_extrap as extrap     # local: avoids an import cycle

    rf_dir = Path(rf_dir)
    if not rf_dir.is_dir():
        return None
    m = MODELS[model]
    p1_vals = np.asarray(m.p1_vals[::stride], float)
    p2_vals = np.asarray(m.p2_vals[::stride], float)
    base_mask = base_grid() <= max_dt_base
    grids = {k: np.full((len(p1_vals), len(p2_vals)), np.nan) for k, _ in MEASURES}
    found = complete = 0
    for ix, p1 in enumerate(p1_vals):
        for iy, p2 in enumerate(p2_vals):
            pt = load_point(model, float(p1), float(p2), rf_dir)
            if pt is None:
                continue
            found += 1
            dt_vals, arrs = pt
            if all(np.isfinite(arrs[k]).all() for k, _ in MEASURES):
                complete += 1
            for k, _ in MEASURES:
                grids[k][ix, iy] = extrap.extrapolate(
                    dt_vals[base_mask], arrs[k][base_mask],
                    fit_n=fit_n, deg=deg, raw=raw)
    print(f'[randframe] {model}: {found} grid points with data '
          f'({complete} complete over all {N_BASE} DT_BASE x {len(MEASURES)} keys)',
          flush=True)
    if found == 0:
        return None
    return grids


def panels(model: str, grids: dict | None, stride: int = 1) -> list:
    """Extra-panel specs (kind='fra') for trotter_dtbase_line_extrap.plot_model."""
    if grids is None:
        return []
    m = MODELS[model]
    p1_vals = np.asarray(m.p1_vals[::stride], float)
    p2_vals = np.asarray(m.p2_vals[::stride], float)
    return [dict(kind='fra', p1_vals=p1_vals, p2_vals=p2_vals, Z=grids[k], label=label)
            for k, label in MEASURES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='model3', choices=list(MODELS))
    ap.add_argument('--rf_dir', type=str, default=OUT_DIR_DEFAULT)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--fit_n', type=int, default=15)
    ap.add_argument('--deg', type=int, default=1)
    ap.add_argument('--max_dt_base', type=float, default=0.10)
    args = ap.parse_args()
    grids = extrapolate_model(args.model, Path(args.rf_dir), fit_n=args.fit_n,
                              deg=args.deg, raw=False, stride=args.stride,
                              max_dt_base=args.max_dt_base)
    for k, _ in MEASURES:
        if grids is not None:
            Z = grids[k]
            print(f'  {k:16s} finite {int(np.isfinite(Z).sum()):5d}  '
                  f'range [{np.nanmin(Z) if np.isfinite(Z).any() else np.nan:.4g}, '
                  f'{np.nanmax(Z) if np.isfinite(Z).any() else np.nan:.4g}]')


if __name__ == '__main__':
    main()
