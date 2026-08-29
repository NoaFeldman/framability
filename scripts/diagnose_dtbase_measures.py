"""
Diagnostic: why is a DT_BASE-line measure blank in the extrapolation colormap?

Walks the three stages a measure passes through on its way to a panel and
reports where it disappears:

    stage 1  results_dtbase_line/<tag>/base_<idx>.npz   (worker output)
    stage 2  results_dtbase_line/<tag>_dtbase_line.npz  (collect output)
    stage 3  results_dtbase_line/<model>_dtbase_extrap.npz (extrap output)

For each stage and each measure it counts: key absent / present-but-NaN /
present-but-inf / finite.  A blank panel means stage 3 is all non-finite; this
tells you whether that is because the worker never wrote the key (stage 1
absent -> the backfill jobs did not run), because the value is non-finite
(stage 1 inf -> the optimiser/LP is failing, NOT a plumbing problem), or
because a later stage dropped it.

Usage (from the repo root):
    python scripts/diagnose_dtbase_measures.py
    python scripts/diagnose_dtbase_measures.py --models model3 --max_points 300
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'scripts'))

from trotter_lindbladian_scan import MODELS
from trotter_dtbase_line_worker import MEASURES, FRAME_KEYS, point_tag, N_BASE

KEYS = [k for k, _ in MEASURES]


def _classify(d, key):
    """'absent' | 'nan' | 'inf' | 'finite' for `key` in an npz-like mapping."""
    files = d.files if hasattr(d, 'files') else list(d)
    if key not in files:
        return 'absent'
    v = np.asarray(d[key], dtype=float)
    if np.all(np.isnan(v)):
        return 'nan'
    if np.any(np.isinf(v)):
        return 'inf'
    if not np.any(np.isfinite(v)):
        return 'nan'
    return 'finite'


def _report(title, counters, n):
    print(f'\n--- {title}  ({n} file(s)) ---')
    if not n:
        print('   NOTHING FOUND -- this stage produced no files at all.')
        return
    width = max(len(k) for k in KEYS + list(f for _, (f, _) in FRAME_KEYS.items()))
    for key in KEYS:
        c = counters[key]
        print(f'   {key:<{width}}  finite={c["finite"]:<6} absent={c["absent"]:<6} '
              f'nan={c["nan"]:<6} inf={c["inf"]:<6}')
    for _k, (x_key, _d) in FRAME_KEYS.items():
        c = counters[x_key]
        print(f'   {x_key:<{width}}  finite={c["finite"]:<6} absent={c["absent"]:<6} '
              f'(warm-start frames; absent => quick-refine cannot neighbour-seed)')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['model3'], choices=list(MODELS))
    ap.add_argument('--in_dir', type=str, default='results_dtbase_line')
    ap.add_argument('--max_points', type=int, default=200,
                    help='cap on how many point directories to scan (speed)')
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    print(f'scanning {in_dir.resolve()}')
    if not in_dir.is_dir():
        print('ERROR: directory does not exist from this working directory.')
        sys.exit(1)

    all_keys = KEYS + [x for _, (x, _) in FRAME_KEYS.items()]

    for model in args.models:
        print(f'\n================ {model} ================')

        # ---- stage 1: raw per-base worker output ---------------------------
        c1 = {k: Counter() for k in all_keys}
        n1 = 0
        pt_dirs = sorted(d for d in in_dir.glob(f'{model}_p1_*') if d.is_dir())
        print(f'raw point directories present: {len(pt_dirs)}')
        for pt in pt_dirs[:args.max_points]:
            for f in pt.glob('base_*.npz'):
                if '_qrefine_' in f.stem:
                    continue
                try:
                    d = np.load(f, allow_pickle=True)
                except Exception:
                    continue
                n1 += 1
                for k in all_keys:
                    c1[k][_classify(d, k)] += 1
        _report('stage 1: base_<idx>.npz (worker output)', c1, n1)

        # ---- stage 2: collected per-point summaries ------------------------
        c2 = {k: Counter() for k in all_keys}
        n2 = 0
        for f in sorted(in_dir.glob(f'{model}_p1_*_dtbase_line.npz'))[:args.max_points]:
            try:
                d = np.load(f, allow_pickle=True)
            except Exception:
                continue
            n2 += 1
            for k in all_keys:
                c2[k][_classify(d, k)] += 1
        _report('stage 2: <tag>_dtbase_line.npz (collect output)', c2, n2)

        # ---- stage 3: the extrapolation grid actually plotted --------------
        ex = in_dir / f'{model}_dtbase_extrap.npz'
        print(f'\n--- stage 3: {ex.name} (what the colormap plots) ---')
        if not ex.exists():
            print('   MISSING -- run scripts/collect_and_plot_all.py')
            continue
        d = np.load(ex, allow_pickle=True)
        for key in KEYS:
            if key not in d.files:
                print(f'   {key:<12} KEY ABSENT from the extrap npz')
                continue
            g = np.asarray(d[key], dtype=float)
            fin = int(np.sum(np.isfinite(g)))
            print(f'   {key:<12} finite={fin:<6}/{g.size:<6} '
                  f'min={np.nanmin(g) if fin else float("nan"):.6g} '
                  f'max={np.nanmax(g) if fin else float("nan"):.6g}'
                  + ('   <-- BLANK PANEL' if fin == 0 else ''))

        # ---- consistency check that motivated this script ------------------
        if 'pauli_fra' in d.files and 'opt_fra_6' in d.files:
            pf = np.asarray(d['pauli_fra'], float)
            f6 = np.asarray(d['opt_fra_6'], float)
            both = np.isfinite(pf) & np.isfinite(f6)
            bad = int(np.sum(f6[both] > pf[both] + 1e-9))
            print(f'\n   opt_fra_6 > pauli_fra at {bad}/{int(both.sum())} points '
                  f'(should be 0: the d=6 frame family contains the Pauli frame, '
                  f'so the d=6 optimum can never exceed it -- a nonzero count '
                  f'means those points are under-optimised)')


if __name__ == '__main__':
    main()
