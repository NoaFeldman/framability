"""
Redraw the model10 rate figure (the layout of
scripts/model4_rate_panels_collect.py, no extra rows) with panels 3-4
replaced by the seeded, quick-refined Heisenberg rates:

    panel 3  'Opt Heisenberg rate (d_ext=4)'  <- seeded d_ext = 4
    panel 4  'Opt Heisenberg rate (d_ext=6)'  <- seeded d_ext = 8

Per point the value is the minimum over every certified upper bound on that
d_ext's optimum:
    d_ext = 4 :  seeded worker, collect cross-evaluation, qrefine rounds,
                 and the earlier optimiser's rate_heis_4
    d_ext = 8 :  the same seeded / refined files, plus the d_ext = 4 values
                 and the earlier rate_heis_6 (a 4- or 6-column frame padded
                 by repeated columns is an 8-column frame with the same rate)
By construction of the seeded pipeline (earlier frames are among its seeds)
the fallbacks only fill points the seeded run has not reached; the log says
how many.

Base panels: same loaders as scripts/model4_rate_panels_collect.py (the first
of --base_in_dirs holding a model10/ subdirectory, --q_dir, --obs_dir), else
the stored figure data --base_npz (scripts/model10_seeded_collect.load_base).
The figure itself is drawn by model4_rate_panels_collect.plot with the two
panel titles changed; nothing in that module is modified on disk.

Outputs:
    --out_png   (default results_model4_rate/model10_rate_panels.png)
    <out_dir>/model10_seeded_panels.npz

Usage:
    python scripts/model10_seeded_panels_collect.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from model10_seeded_worker import MODEL, grid_vals, pt_dir_name           # noqa: E402
from model10_seeded_qrefine_worker import (KEYS, ROUND_TAG,               # noqa: E402
                                           best_known)
from model10_seeded_collect import load_base                              # noqa: E402
import model4_rate_panels_collect as base                                 # noqa: E402

qcollect = base.qcollect


def load_seeded_best(pt: Path, nx: int, ny: int) -> dict:
    """rate_4 / rate_8 grids: min over worker, xeval and qrefine files."""
    g = {k: np.full((nx, ny), np.nan) for k in KEYS}
    found = 0
    for ix in range(nx):
        for iy in range(ny):
            if not (pt / f'pt_{ix:03d}_{iy:03d}.npz').exists():
                continue
            found += 1
            for key, (s_key, _) in KEYS.items():
                v, _, _ = best_known(pt, ix, iy, key, s_key)
                if np.isfinite(v):
                    g[key][ix, iy] = v
    rounds = sorted({int(m.group(1)) for f in pt.glob(f'pt_*{ROUND_TAG}*.npz')
                     if (m := re.search(rf'{ROUND_TAG}(\d\d)\.npz$', f.name))})
    print(f'[{MODEL} seeded panels] {found}/{nx * ny} seeded points in {pt}; '
          f'qrefine rounds on disk: {rounds or "none"}', flush=True)
    return dict(n_points=found, rounds=rounds, **g)


def on_base_grid(Z, stride: int, base_stride: int, shape) -> np.ndarray:
    """A seeded-grid quantity placed on the base grid (NaN where unsampled)."""
    out = np.full(shape, np.nan)
    if stride % base_stride:
        return out
    r = stride // base_stride
    nx, ny = Z.shape
    for ix in range(nx):
        for iy in range(ny):
            jx, jy = ix * r, iy * r
            if jx < shape[0] and jy < shape[1]:
                out[jx, jy] = Z[ix, iy]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default='results_model10_rate',
                    help='seeded pipeline root (holds model10_seeded/)')
    ap.add_argument('--stride', type=int, default=1,
                    help='stride the seeded worker ran with')
    ap.add_argument('--out_png', type=str,
                    default='results_model4_rate/model10_rate_panels.png')
    ap.add_argument('--base_in_dirs', type=str, nargs='*',
                    default=['results_model10_rate', 'results_model4_rate'])
    ap.add_argument('--base_npz', type=str, nargs='*',
                    default=['results_model4_rate/model10_rate_panels.npz',
                             'results_model10_rate/model10_rate_panels.npz'])
    ap.add_argument('--base_stride', type=int, default=1)
    ap.add_argument('--mb_stride', type=int, default=5)
    ap.add_argument('--prod_stride', type=int, default=1)
    ap.add_argument('--q_dir', type=str, default='results_liouvillian_q')
    ap.add_argument('--q_stride', type=int, default=1)
    ap.add_argument('--q_levels', type=float, nargs='+',
                    default=list(qcollect.Q_LEVELS_DEFAULT))
    ap.add_argument('--obs_dir', type=str, default='results_observable_q')
    ap.add_argument('--obs_stride', type=int, default=1)
    ap.add_argument('--floor', type=float, default=0.0)
    args = ap.parse_args()

    bd = load_base(args)
    if bd is None:
        sys.exit('no base figure data found (neither per-point dirs in '
                 f'{args.base_in_dirs} nor {args.base_npz}); refusing to '
                 'overwrite the figure with seeded panels alone')

    p1, p2 = grid_vals(args.stride)
    pt = Path(args.out_dir) / pt_dir_name(args.stride)
    s = load_seeded_best(pt, len(p1), len(p2))

    rates = dict(bd['rates'])
    old4 = np.asarray(rates['rate_heis_4'], float)
    old6 = np.asarray(rates['rate_heis_6'], float)
    shape = old4.shape
    new4 = on_base_grid(s['rate_4'], args.stride, bd['stride'], shape)
    new8 = on_base_grid(s['rate_8'], args.stride, bd['stride'], shape)
    panel4 = np.fmin(new4, old4)
    panel8 = np.fmin.reduce([new8, panel4, old6])

    for name, new, panel, old in (('d4', new4, panel4, old4),
                                  ('d8', new8, panel8, np.fmin(old4, old6))):
        ok = np.isfinite(new)
        fb = np.isfinite(panel) & ~ok
        both = ok & np.isfinite(old)
        print(f'  {name}: seeded at {int(ok.sum())} pts, fallback to the earlier '
              f'optimiser at {int(fb.sum())}; rate 0 at '
              f'{int((panel <= base.CONTOUR_TOL).sum())} '
              f'(earlier {int((old <= base.CONTOUR_TOL).sum())}); '
              f'improved at {int(((old - panel)[both] > 1e-6).sum())}', flush=True)

    rates['rate_heis_4'] = panel4
    rates['rate_heis_6'] = panel8
    n_r = len(s['rounds'])
    tag = f'seeded + {n_r} quick-refine round{"s" if n_r != 1 else ""}'
    base.RATE_KEYS = [
        (k, (rf'Opt Heisenberg rate ($d_{{\rm ext}}=4$), {tag}' if k == 'rate_heis_4'
             else rf'Opt Heisenberg rate ($d_{{\rm ext}}=8$), {tag}'
             if k == 'rate_heis_6' else lab))
        for k, lab in base.RATE_KEYS]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y = MODELS[MODEL].p1_name, MODELS[MODEL].p2_name
    npz = out_dir / f'{MODEL}_seeded_panels.npz'
    np.savez(npz, model=MODEL, stride=args.stride, base_stride=bd['stride'],
             qrefine_rounds=np.array(s['rounds'], int),
             **{f'{x}_vals': rates['p1_vals'], f'{y}_vals': rates['p2_vals']},
             panel_rate_heis_4=panel4, panel_rate_heis_8=panel8,
             seeded_rate_4=new4, seeded_rate_8=new8,
             prev_rate_heis_4=old4, prev_rate_heis_6=old6)
    print(f'[{MODEL} seeded panels] wrote {npz}', flush=True)

    png = Path(args.out_png)
    png.parent.mkdir(parents=True, exist_ok=True)
    base.plot(rates, bd['mb'], png, floor=args.floor, model=MODEL, q=bd['q'],
              q_levels=tuple(args.q_levels), obs=bd['obs'], prod=bd['prod'])


if __name__ == '__main__':
    main()
