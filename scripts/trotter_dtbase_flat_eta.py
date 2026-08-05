"""ETA check for a trotter_dtbase_flat_worker.py resubmission.

Scans results_dtbase_line/ for already-complete base_*.npz files and reports,
per SLURM array task (the same items[task_id::n_tasks] sharding the flat
worker uses), how many of its assigned items are still missing -- i.e. how
much real work resubmitting scripts/trotter_dtbase_flat.slurm.sh will
actually do, since already-current files are skipped almost instantly.

Usage (run on the cluster, where results_dtbase_line/ actually has data):
    python scripts/trotter_dtbase_flat_eta.py --models model3 model4 --stride 4 \
        --sec_per_item 271   # 4m30.959s measured for one point
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'scripts'))

from trotter_lindbladian_scan import MODELS
from trotter_dtbase_line_worker import N_BASE, point_tag, DTBASE_LINE_VERSION
import numpy as np


def is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        d = np.load(path, allow_pickle=True)
        return 'code_version' in d and str(d['code_version']) == DTBASE_LINE_VERSION
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['model3', 'model4'],
                     choices=list(MODELS))
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--n_tasks', type=int, default=200)
    ap.add_argument('--out_dir', type=str, default='results_dtbase_line')
    ap.add_argument('--sec_per_item', type=float, default=271.0,
                     help='measured wall time per item (default: the 4m30.959s sample)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    items = []
    for model in args.models:
        m = MODELS[model]
        for p1 in m.p1_vals[::args.stride]:
            for p2 in m.p2_vals[::args.stride]:
                for base_idx in range(N_BASE):
                    items.append((model, float(p1), float(p2), base_idx))

    remaining = np.zeros(args.n_tasks, dtype=int)
    total_missing = 0
    for i, (model, p1, p2, base_idx) in enumerate(items):
        task_id = i % args.n_tasks
        out = out_dir / point_tag(model, p1, p2) / f'base_{base_idx:03d}.npz'
        if not is_done(out):
            remaining[task_id] += 1
            total_missing += 1

    worst_task = int(remaining.argmax())
    print(f'total items: {len(items)}, missing: {total_missing} '
          f'({100 * total_missing / len(items):.1f}%)')
    print(f'missing per task: max={remaining.max()} (task {worst_task}), '
          f'mean={remaining.mean():.1f}')
    print(f'worst-case task wall time (all tasks run in parallel, so this '
          f'bounds the whole array): '
          f'{remaining.max() * args.sec_per_item / 60:.1f} min '
          f'= {remaining.max() * args.sec_per_item / 3600:.2f} h')


if __name__ == '__main__':
    main()
