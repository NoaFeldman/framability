"""Report whether a model's optimised framability reaches 1 (has a framable point).

Loads the base scan (pt_<ix>_<iy>.npz) for one model and, for the optimised
framability keys that get neighbour-refined (opt_fra_4, opt_fra_6), computes the
grid minimum.  The model "qualifies" for refinement if any of those minima is
<= 1 + tol, i.e. at least one grid point is (numerically) framable -- the same
condition as the plot showing a framability-1 point (trotter_scan_collect.py's
FRA_ONE_TOL), so the gate matches what you see in the figure.

Exit code 0 if the model qualifies, 3 otherwise, so a shell driver can gate on
it (see submit_trotter_refine.sh).  Prints a one-line summary either way.

    python scripts/trotter_check_framable.py --model model1 --in_dir results_trotter_v3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, FRA_REFINE_KEYS


def grid_min(in_dir: Path, model, key: str) -> float:
    """Minimum of `key` over every existing base point file of `model`."""
    best = np.inf
    mdir = in_dir / model.name
    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f)
            except Exception:
                continue
            if key in d:
                v = float(d[key])
                if np.isfinite(v) and v < best:
                    best = v
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',  type=str, required=True, choices=list(MODELS))
    p.add_argument('--in_dir', type=str, default='results_trotter_v3')
    p.add_argument('--tol',    type=float, default=1e-6,
                   help='a point counts as framable if opt framability <= 1 + tol '
                        '(matches the figure\'s framability-1 contour)')
    args = p.parse_args()

    model = MODELS[args.model]
    in_dir = Path(args.in_dir)
    mins = {k: grid_min(in_dir, model, k) for k in FRA_REFINE_KEYS}
    qualifies = any(np.isfinite(v) and v <= 1.0 + args.tol for v in mins.values())

    summary = '  '.join(
        f'{k} min={("nan" if not np.isfinite(v) else f"{v:.6f}")}'
        for k, v in mins.items())
    verdict = ('QUALIFIES (framable point found)' if qualifies
               else 'no framable point -> skip')
    print(f'{model.name}: {summary}  ->  {verdict}', flush=True)
    sys.exit(0 if qualifies else 3)


if __name__ == '__main__':
    main()
