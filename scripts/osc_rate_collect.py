"""
Collect scripts/osc_rate_worker.py's per-point npz files into a (gamma, gamma')
grid.  Consumed by trotter_dtbase_line_extrap.py as the extra "oscillation rate"
panel on the framability figure, and runnable standalone to write/inspect the
grid npz on its own.

Usage:
    python scripts/osc_rate_collect.py --model model3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'scripts'))

from osc_rate_worker import grid_vals

OSC_LABEL = r'Oscillation rate  $\max_k |\mathrm{Im}\lambda_k/\mathrm{Re}\lambda_k|$'


def load(model: str, in_dir: Path, stride: int = 1) -> dict | None:
    """(gamma, gamma') grid of the oscillation rate, or None if no data yet."""
    p1_vals, p2_vals = grid_vals(model, stride)
    nx, ny = len(p1_vals), len(p2_vals)
    Z = np.full((nx, ny), np.nan)
    n_exact = found = 0
    pt_dir = in_dir / model
    if not pt_dir.is_dir():
        return None
    for ix in range(nx):
        for iy in range(ny):
            f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                Z[ix, iy] = float(d['osc_rate'])
                n_exact += bool(d['exact']) if 'exact' in d.files else 0
                found += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    if found == 0:
        return None
    print(f'[osc_rate] {model}: {found}/{nx * ny} grid points loaded '
          f'({n_exact} exact, {found - n_exact} sparse lower bounds)', flush=True)
    return dict(p1_vals=p1_vals, p2_vals=p2_vals, osc_rate=Z,
                found=found, n_exact=n_exact)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='model3')
    ap.add_argument('--in_dir', type=str, default='results_osc_rate')
    ap.add_argument('--out_dir', type=str, default='results_osc_rate')
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    data = load(args.model, Path(args.in_dir), args.stride)
    if data is None:
        print(f'[osc_rate] no data found under {args.in_dir}/{args.model}')
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f'{args.model}_osc_rate.npz'
    np.savez(npz, model=args.model, **data)
    print(f'[osc_rate] saved {npz}', flush=True)


if __name__ == '__main__':
    main()
