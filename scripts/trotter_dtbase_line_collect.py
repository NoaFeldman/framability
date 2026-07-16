"""
Collect the per-base .npz files written by trotter_dtbase_line_worker.py for one
(model, p1, p2) point and plot the five framabilities as a function of DT_BASE.

Loads every <in_dir>/<tag>/base_<idx>.npz (tag = point_tag(model, p1, p2)),
assembles the curves ordered by base, saves a summary npz and a single-panel
figure with all five framabilities vs base and the framable threshold (=1).

Usage:
    python scripts/trotter_dtbase_line_collect.py --model model3 --p1 3 --p2 3
    python scripts/trotter_dtbase_line_collect.py --model model3 --p1 3 --p2 3 \
        --in_dir results_dtbase_line --out_dir results_dtbase_line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS
from trotter_dtbase_line_worker import (  # noqa: E402  (scripts/ is on the path)
    MEASURES, base_grid, point_tag, N_BASE,
)


def load(model: str, p1: float, p2: float, in_dir: Path) -> dict:
    """Assemble the five framability curves (ordered by base) from the pt files."""
    tag = point_tag(model, p1, p2)
    pt_dir = in_dir / tag
    if not pt_dir.is_dir():
        raise SystemExit(f'no data directory {pt_dir}')

    base_vals = base_grid()
    arrs = {k: np.full(N_BASE, np.nan) for k, _ in MEASURES}
    dt_vals = np.full(N_BASE, np.nan)
    found = 0
    for f in sorted(pt_dir.glob('base_*.npz')):
        d = np.load(f, allow_pickle=True)
        idx = int(d['base_idx'])
        if not (0 <= idx < N_BASE):
            continue
        for k, _ in MEASURES:
            arrs[k][idx] = float(d[k])
        dt_vals[idx] = float(d['dt'])
        found += 1

    missing = int(np.sum(~np.isfinite(arrs[MEASURES[0][0]])))
    print(f'[collect] {tag}: {found}/{N_BASE} base points loaded '
          f'({missing} missing)', flush=True)
    return dict(base_vals=base_vals, dt_vals=dt_vals, **arrs)


def plot(model: str, p1: float, p2: float, data: dict, png: Path,
         fra_tol: float = 1e-3) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    m = MODELS[model]
    base = data['base_vals']

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for (key, label) in MEASURES:
        ax.plot(base, data[key], marker='o', ms=3, lw=1.4, label=label)

    # framable threshold: framability = 1 (drawn at 1+tol, the numerical edge)
    ax.axhline(1.0 + fra_tol, color='0.4', ls='--', lw=1.0,
               label=f'framable ($=1{"+" if fra_tol else ""}{fra_tol:g})')

    ax.set_xlabel('DT_BASE')
    ax.set_ylabel('framability')
    ax.set_title(f'{m.title}\n{m.p1_name}={p1:g}, {m.p2_name}={p2:g}   '
                 r'($dt = \mathrm{DT\_BASE}\,/\,\max(\|H\|_1,\{\gamma_k\})$)',
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='best')

    fig.savefig(png, dpi=150)
    print(f'[collect] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',   type=str, required=True, choices=list(MODELS))
    ap.add_argument('--p1',      type=float, required=True)
    ap.add_argument('--p2',      type=float, required=True)
    ap.add_argument('--in_dir',  type=str, default='results_dtbase_line')
    ap.add_argument('--out_dir', type=str, default='results_dtbase_line')
    ap.add_argument('--fra_tol', type=float, default=1e-3)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load(args.model, args.p1, args.p2, in_dir)

    tag = point_tag(args.model, args.p1, args.p2)
    npz = out_dir / f'{tag}_dtbase_line.npz'
    png = out_dir / f'{tag}_dtbase_line.png'
    np.savez(npz, model=args.model, p1=args.p1, p2=args.p2, **data)
    print(f'[collect] saved {npz}', flush=True)
    plot(args.model, args.p1, args.p2, data, png, fra_tol=args.fra_tol)


if __name__ == '__main__':
    main()
