"""
Flat, fixed-size job-array worker for the full DT_BASE line sweep.

The per-point driver (dtbase_line_submit_all.sh) needs one `sbatch` call per
(model, p1, p2) grid point -- thousands of separate submissions that a
client-side loop had to throttle by hand against squeue.  This script instead
flattens the *entire* (model, p1, p2, base_idx) grid into one job list and
shards it evenly across a single fixed-size SLURM array (default 200 tasks,
see trotter_dtbase_flat.slurm.sh).  One `sbatch` call, no external throttling,
squeue tracks it like any other array job.  Each task loops sequentially over
its shard, so tasks run much longer than the old per-point-per-base jobs.

Output layout, skip-if-current check and per-point numerics are unchanged from
trotter_dtbase_line_worker.py (this only reshapes how the grid is split across
jobs), so it's safe to point at a results_dtbase_line/ directory that already
has points computed by the old per-point driver -- already-current npz files
are skipped, so this is also safe to resubmit unchanged if a task's time limit
is hit partway through its shard.

Usage (single sbatch call, no driver script needed):
    MODELS="model3 model4" STRIDE=2 sbatch scripts/trotter_dtbase_flat.slurm.sh

Usage (direct, e.g. to sanity-check a shard locally):
    python scripts/trotter_dtbase_flat_worker.py --models model3 model4 \
        --stride 2 --task_id 0 --n_tasks 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'scripts'))

from trotter_lindbladian_scan import MODELS, DIM_DEFAULT
import trotter_dtbase_line_worker as ldw   # reuse run_point / N_BASE / point_tag


def flat_grid(models: list[str], stride: int) -> list[tuple[str, float, float, int]]:
    """All (model, p1, p2, base_idx) work items across `models`, p1/p2 taken
    every `stride`-th grid value (stride=1 -> full grid), all N_BASE base_idx."""
    items = []
    for model in models:
        m = MODELS[model]
        for p1 in m.p1_vals[::stride]:
            for p2 in m.p2_vals[::stride]:
                for base_idx in range(ldw.N_BASE):
                    items.append((model, float(p1), float(p2), base_idx))
    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['model3', 'model4'],
                    choices=list(MODELS))
    p.add_argument('--stride', type=int, default=1,
                    help='take every stride-th p1/p2 grid value (2 = half res)')
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_tasks', type=int, required=True)
    p.add_argument('--out_dir', type=str, default='results_dtbase_line')
    p.add_argument('--dim', type=int, default=DIM_DEFAULT, choices=(1, 2, 3))
    p.add_argument('--fra_restarts', type=int, default=5)
    p.add_argument('--fra_maxfev_4', type=int, default=1000)
    p.add_argument('--fra_maxfev_6', type=int, default=500)
    p.add_argument('--ch1_restarts', type=int, default=15)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    if not (0 <= args.task_id < args.n_tasks):
        print(f'ERROR: task_id must be in [0, {args.n_tasks})', file=sys.stderr)
        sys.exit(1)

    items = flat_grid(args.models, args.stride)
    shard = items[args.task_id::args.n_tasks]
    print(f'[task {args.task_id}/{args.n_tasks}] {len(shard)}/{len(items)} '
          f'work items (models={args.models} stride={args.stride})', flush=True)

    ns = argparse.Namespace(
        out_dir=args.out_dir, dim=args.dim,
        fra_restarts=args.fra_restarts, fra_maxfev_4=args.fra_maxfev_4,
        fra_maxfev_6=args.fra_maxfev_6, ch1_restarts=args.ch1_restarts,
        seed=args.seed,
    )
    for i, (model, p1, p2, base_idx) in enumerate(shard):
        ns.model, ns.p1, ns.p2 = model, p1, p2
        print(f'  [{i + 1}/{len(shard)}] {model} p1={p1:.4f} p2={p2:.4f} '
              f'base_idx={base_idx}', flush=True)
        ldw.run_point(ns, base_idx)


if __name__ == '__main__':
    main()
