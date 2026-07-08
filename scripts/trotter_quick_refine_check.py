"""
Convergence check for chained quick-refinement rounds of the Trotter scan.

Decides whether quick round --round improved anything: for every
pt_qrefine_r<NN>_<ix>_<iy>.npz written in that round, the stored
opt_fra_4 / opt_fra_6 are compared against the best value over all EARLIER
files for the same point (base scan pt_*.npz, every full-refine round
pt_refine_r*, and every quick round < NN).

Exit code 0  -> the round improved at least one value by more than --tol
                (the chain driver submits the next round)
Exit code 3  -> converged: the round wrote no files, or nothing improved
                (the chain driver submits the final collect instead)
Any other exit code (e.g. 1 from an uncaught exception) is an ERROR and
stops the chain without collecting.

Usage:
    python scripts/trotter_quick_refine_check.py --model model3 --round 5 \
        --in_dir results_trotter_v3 [--tol 1e-9]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS

KEYS = ('opt_fra_4', 'opt_fra_6')


def _prev_best(mdir: Path, ix: int, iy: int, round_: int) -> dict:
    """Best (lowest) value per key over base + full refines + quick rounds < round_."""
    paths = [mdir / f'pt_{ix:03d}_{iy:03d}.npz']
    paths += sorted(mdir.glob(f'pt_refine_r*_{ix:03d}_{iy:03d}.npz'))
    paths += [mdir / f'pt_qrefine_r{r:02d}_{ix:03d}_{iy:03d}.npz'
              for r in range(1, round_)]
    best = {k: np.inf for k in KEYS}
    for p in paths:
        if not p.exists():
            continue
        try:
            d = np.load(p)
        except Exception:
            continue
        for k in KEYS:
            if k in d:
                v = float(d[k])
                if np.isfinite(v) and v < best[k]:
                    best[k] = v
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',  type=str, required=True, choices=list(MODELS))
    p.add_argument('--round',  type=int, required=True,
                   help='quick round that just finished')
    p.add_argument('--in_dir', type=str, default='results_trotter_v3')
    p.add_argument('--tol',    type=float, default=1e-9,
                   help='minimal improvement that counts as a change '
                        '(matches the collect merge threshold)')
    args = p.parse_args()

    model = MODELS[args.model]
    mdir = Path(args.in_dir) / model.name
    files = sorted(mdir.glob(f'pt_qrefine_r{args.round:02d}_*.npz'))

    if not files:
        print(f'[{model.name}] quick round {args.round}: no boundary points '
              f'were touched -> CONVERGED', flush=True)
        sys.exit(3)

    n_improved = 0
    max_gain = 0.0
    for f in files:
        parts = f.stem.split('_')          # pt, qrefine, rNN, IX, IY
        ix, iy = int(parts[3]), int(parts[4])
        try:
            d = np.load(f)
        except Exception:
            print(f'  WARNING: unreadable {f.name} — skipped', flush=True)
            continue
        prev = _prev_best(mdir, ix, iy, args.round)
        for k in KEYS:
            if k not in d:
                continue
            v = float(d[k])
            if np.isfinite(v) and v < prev[k] - args.tol:
                gain = prev[k] - v
                max_gain = max(max_gain, gain)
                n_improved += 1
                print(f'  ({ix:3d},{iy:3d}) {k}: {prev[k]:.9f} -> {v:.9f} '
                      f'(gain {gain:.3e})', flush=True)

    if n_improved:
        print(f'[{model.name}] quick round {args.round}: {len(files)} boundary '
              f'file(s), {n_improved} value(s) improved (max gain {max_gain:.3e}) '
              f'-> CHANGED', flush=True)
        sys.exit(0)

    print(f'[{model.name}] quick round {args.round}: {len(files)} boundary '
          f'file(s), no value improved beyond tol={args.tol:g} -> CONVERGED',
          flush=True)
    sys.exit(3)


if __name__ == '__main__':
    main()
