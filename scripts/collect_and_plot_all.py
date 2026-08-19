"""
Unified collect-and-plot driver for the whole DT_BASE-line pipeline redesign
(items 1-6): re-collects/re-plots every existing per-point dtbase-line figure,
re-extrapolates and replots the model3/model4 dt->0 colormaps (now folding in
item 5's quick-refine merge for opt_fra_4/opt_fra_6 and the two new measures
from items 2-3), and collects/plots the two new standalone many-body figures
from items 4 and 6.  Pure aggregation -- no optimisation runs here; run the
workers/sbatch scripts first to (re)generate any missing data.

Usage:
    python scripts/collect_and_plot_all.py
    python scripts/collect_and_plot_all.py --models model3 --stride 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'scripts'))

from trotter_lindbladian_scan import MODELS
import trotter_dtbase_line_collect as collect
import trotter_dtbase_line_extrap as extrap
import eight_qubit_gap_collect as gap8_collect
import six_qubit_spectral_osc_collect as specosc6_collect


def collect_dtbase_lines(models, stride, in_dir, out_dir, fra_tol):
    """Step 1: per-point F(dt) line figures (items 1-3's 7 measures + item 5's
    refine-merge, both already wired into trotter_dtbase_line_collect.load)."""
    total = done = incomplete = missing = 0
    for model in models:
        m = MODELS[model]
        for p1 in m.p1_vals[::stride]:
            for p2 in m.p2_vals[::stride]:
                p1, p2 = float(p1), float(p2)
                total += 1
                tag = collect.point_tag(model, p1, p2)
                if not (in_dir / tag).is_dir():
                    missing += 1
                    continue
                data = collect.load(model, p1, p2, in_dir)
                found = int(np.sum(np.isfinite(data[collect.MEASURES[0][0]])))
                npz = out_dir / f'{tag}_dtbase_line.npz'
                png = out_dir / f'{tag}_dtbase_line.png'
                np.savez(npz, model=model, p1=p1, p2=p2, **data)
                collect.plot(model, p1, p2, data, png, fra_tol=fra_tol)
                if found < collect.N_BASE:
                    incomplete += 1
                else:
                    done += 1
    print(f'[collect_and_plot_all] dtbase lines: {done} complete, {incomplete} '
          f'partial, {missing} not started (of {total} points)', flush=True)


def extrapolate_and_plot(models, in_dir, out_dir, *, stride, fit_n, deg,
                         max_dt_base, fra_tol):
    """Step 2: dt->0 colormaps (model3_dtbase_extrap.png / model4_...) -- the
    replot of the optimised-Heisenberg (opt_fra_4/opt_fra_6, refine-merged)
    and every other framability, now with 7 panels instead of 5."""
    for model in models:
        data = extrap.extrapolate_model(model, in_dir, fit_n=fit_n, deg=deg,
                                        raw=False, stride=stride,
                                        max_dt_base=max_dt_base)
        npz = out_dir / f'{model}_dtbase_extrap.npz'
        png = out_dir / f'{model}_dtbase_extrap.png'
        np.savez(npz, model=model, fit_n=fit_n, deg=deg, raw=False,
                measures=[k for k, _ in extrap.MEASURES], **data)
        extrap.plot_model(model, data, png, raw=False, fra_tol=fra_tol)
        print(f'[collect_and_plot_all] extrapolated + plotted {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['model3', 'model4'],
                    choices=list(MODELS))
    ap.add_argument('--stride', type=int, default=1,
                    help='must match the STRIDE used when submitting the dtbase-'
                         'line sweep')
    ap.add_argument('--in_dir', type=str, default='results_dtbase_line')
    ap.add_argument('--out_dir', type=str, default='results_dtbase_line')
    ap.add_argument('--fra_tol', type=float, default=1e-3)
    ap.add_argument('--fit_n', type=int, default=15)
    ap.add_argument('--deg', type=int, default=1)
    ap.add_argument('--max_dt_base', type=float, default=0.10)
    ap.add_argument('--eightq_in_dir', type=str, default='results_8q')
    ap.add_argument('--eightq_stride', type=int, default=5)
    ap.add_argument('--skip_lines', action='store_true',
                    help='skip the (slow, many-file) per-point line replot and '
                         'only redo the extrapolated colormaps + item 4/6 figures')
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1+2: every existing dtbase-line figure (items 1-3 + item 5 refine-merge)
    if not args.skip_lines:
        collect_dtbase_lines(args.models, args.stride, in_dir, out_dir, args.fra_tol)
    extrapolate_and_plot(args.models, in_dir, out_dir, stride=args.stride,
                        fit_n=args.fit_n, deg=args.deg,
                        max_dt_base=args.max_dt_base, fra_tol=args.fra_tol)

    # 3: item 4 -- 8-qubit ring/lattice Lindbladian gap
    eightq_in = Path(args.eightq_in_dir)
    gap_data = {}
    for topo in ('ring', 'lattice'):
        if (eightq_in / topo).is_dir():
            d = gap8_collect.load(topo, eightq_in, args.eightq_stride)
            np.savez(eightq_in / f'eight_qubit_gap_{topo}.npz', **d)
            gap_data[topo] = d
    if gap_data:
        gap8_collect.plot(gap_data, eightq_in / 'eight_qubit_gap.png')
    else:
        print('[collect_and_plot_all] no item-4 (8-qubit gap) data found yet '
              f'under {eightq_in} -- skipped', flush=True)

    # 4: item 6 -- 6-qubit ring spectral oscillation
    specosc_in = eightq_in / 'spectral_osc_ring6'
    if specosc_in.is_dir() and any(specosc_in.glob('pt_*.npz')):
        d = specosc6_collect.load(specosc_in, args.eightq_stride)
        np.savez(eightq_in / 'six_qubit_spectral_osc_ring.npz', **d)
        specosc6_collect.plot(d, eightq_in / 'six_qubit_spectral_osc_ring.png')
    else:
        print('[collect_and_plot_all] no item-6 (spectral oscillation) data '
              f'found yet under {specosc_in} -- skipped', flush=True)

    print('[collect_and_plot_all] done.', flush=True)


if __name__ == '__main__':
    main()
