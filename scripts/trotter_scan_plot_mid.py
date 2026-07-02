"""
Plot *mid-run* Trotter-scan results, before the refinement rounds have finished.

trotter_scan_collect.py plots only the base ``pt_<ix>_<iy>.npz`` files, and
trotter_scan_refine_collect.py folds the neighbour-refinement rounds back into
those base files -- but the latter runs on an ``afterok`` dependency, so it only
fires once *all* rounds are done.  While the scan is still in flight the refined
framability values live in loose ``pt_refine_r<NN>_<ix>_<iy>.npz`` files that no
figure has picked up yet.

This script gives an up-to-the-minute picture without touching any data: it
loads the base points and, for the two refined keys (opt_fra_4 / opt_fra_6),
lowers each value by the elementwise minimum over whatever refine rounds are
present on disk (framability is minimised, so lower = better).  It then draws the
same colormap layout as trotter_scan_collect.plot_colormaps -- the merge is done
purely in memory, so the base npz files are left untouched and the eventual
refine-collect pass still produces the identical final figure.

Usage:
    # all four models -> results_plots/trotter_<model>_mid.png
    python scripts/trotter_scan_plot_mid.py

    # one model, custom output
    python scripts/trotter_scan_plot_mid.py --model model2 \
        --out_png results_plots/trotter_model2_mid.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, QUANTITIES, FRA_REFINE_KEYS

# Reuse the exact figure layout / colour handling from the final collector.
from trotter_scan_collect import plot_colormaps, N_Q

_BASE_RE   = re.compile(r'^pt_(\d{3})_(\d{3})\.npz$')
_REFINE_RE = re.compile(r'^pt_refine_r\d{2}_(\d{3})_(\d{3})\.npz$')

# quantity index of each refined key, for writing merged values into the array.
_QI = {key: qi for qi, (key, _, _, _) in enumerate(QUANTITIES)}


def load_results_mid(in_dir: Path, model) -> np.ndarray:
    """(N_X, N_Y, N_Q) array of mid-run values; NaN where a point is missing.

    Base points supply every quantity.  For each refined framability key the
    value is the elementwise minimum of the base value and every refine round
    found on disk (missing rounds are simply skipped).  A refine value only
    lowers an existing base value -- a point with no base file stays NaN.
    """
    arr = np.full((model.N_X, model.N_Y, N_Q), np.nan)
    mdir = in_dir / model.name
    if not mdir.is_dir():
        print(f'  (no directory {mdir})', flush=True)
        return arr

    n_base = 0
    n_refine_files = 0
    n_lowered = 0

    # Single directory pass: base files first so the array is populated before
    # any refine file tries to lower it.
    entries = sorted(mdir.iterdir())
    for f in entries:
        m = _BASE_RE.match(f.name)
        if not m:
            continue
        ix, iy = int(m.group(1)), int(m.group(2))
        if not (0 <= ix < model.N_X and 0 <= iy < model.N_Y):
            continue
        try:
            d = np.load(f)
            for qi, (key, _, _, _) in enumerate(QUANTITIES):
                if key in d:
                    arr[ix, iy, qi] = float(d[key])
            n_base += 1
        except Exception as e:
            print(f'  warning: {f.name}: {e}', flush=True)

    for f in entries:
        m = _REFINE_RE.match(f.name)
        if not m:
            continue
        ix, iy = int(m.group(1)), int(m.group(2))
        if not (0 <= ix < model.N_X and 0 <= iy < model.N_Y):
            continue
        try:
            r = np.load(f)
        except Exception as e:
            print(f'  warning: {f.name}: {e}', flush=True)
            continue
        n_refine_files += 1
        for key in FRA_REFINE_KEYS:
            if key not in r:
                continue
            rv = float(r[key])
            if not np.isfinite(rv):
                continue
            qi = _QI[key]
            cur = arr[ix, iy, qi]
            if not np.isfinite(cur) or rv < cur:
                arr[ix, iy, qi] = rv
                n_lowered += 1

    print(f'Loaded {n_base}/{model.N_TOTAL} base points for {model.name}; '
          f'{n_refine_files} refine files, {n_lowered} framability value(s) '
          f'lowered.', flush=True)
    return arr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',   type=str, default=None, choices=list(MODELS),
                   help='default: all four models')
    p.add_argument('--in_dir',  type=str, default='results_trotter')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_<model>_mid.png '
                        '(only valid with a single --model)')
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    names = [args.model] if args.model else list(MODELS)
    if args.out_png and len(names) > 1:
        p.error('--out_png requires a single --model')

    for name in names:
        model = MODELS[name]
        out_png = Path(args.out_png) if args.out_png else \
            Path('results_plots') / f'trotter_{model.name}_mid.png'
        arr = load_results_mid(in_dir, model)
        if not np.isfinite(arr).any():
            print(f'  no data for {model.name}; skipping figure.', flush=True)
            continue
        plot_colormaps(arr, model, out_png)


if __name__ == '__main__':
    main()
