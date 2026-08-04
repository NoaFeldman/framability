"""
Batch collect: run trotter_dtbase_line_collect's per-point load()/plot() over
every (model, p1, p2) point of the same grid trotter_dtbase_flat_worker.py
computed (same --models/--stride), instead of invoking it one point at a time.

Points whose directory is missing are reported and skipped (not an error), so
it's safe to run this while a sweep is still partly finished.

Usage:
    python scripts/trotter_dtbase_line_collect_all.py --models model3 model4 --stride 4
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
from trotter_dtbase_line_worker import N_BASE, point_tag
import trotter_dtbase_line_collect as collect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['model3', 'model4'],
                     choices=list(MODELS))
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--in_dir',  type=str, default='results_dtbase_line')
    ap.add_argument('--out_dir', type=str, default='results_dtbase_line')
    ap.add_argument('--fra_tol', type=float, default=1e-3)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = done = incomplete = missing = 0
    for model in args.models:
        m = MODELS[model]
        for p1 in m.p1_vals[::args.stride]:
            for p2 in m.p2_vals[::args.stride]:
                p1, p2 = float(p1), float(p2)
                total += 1
                tag = point_tag(model, p1, p2)
                if not (in_dir / tag).is_dir():
                    missing += 1
                    continue

                data = collect.load(model, p1, p2, in_dir)
                found = int(np.sum(np.isfinite(data[collect.MEASURES[0][0]])))

                npz = out_dir / f'{tag}_dtbase_line.npz'
                png = out_dir / f'{tag}_dtbase_line.png'
                np.savez(npz, model=model, p1=p1, p2=p2, **data)
                collect.plot(model, p1, p2, data, png, fra_tol=args.fra_tol)

                if found < N_BASE:
                    incomplete += 1
                else:
                    done += 1

    print(f'[collect_all] {done} complete, {incomplete} partial, '
          f'{missing} not started (of {total} points total)')


if __name__ == '__main__':
    main()
