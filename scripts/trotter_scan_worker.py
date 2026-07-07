"""
Per-point cluster worker for the generic two-qubit Trotter Lindbladian scan.

One model is scanned over its two varying parameters (see
trotter_lindbladian_scan.MODELS).  The grid is

    ix in 0 .. N_X-1   (p1, x-axis)
    iy in 0 .. N_Y-1   (p2, y-axis)
    point_id = ix * N_Y + iy            (0 .. N_X*N_Y - 1)

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz

Usage (single point):
    python scripts/trotter_scan_worker.py --model model1 --task_id 0

Usage (strided across a 200-task array):
    python scripts/trotter_scan_worker.py --model model1 \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

# Note: dt is chosen per point by trotter_lindbladian_scan.choose_dt,
#       dt = DT_BASE / max(||H||_1, max_k gamma_k), unless the model pins a
#       fixed dt (model5) or --dt overrides it; the resolved value is stored
#       in the pt file.
# Note: The steady state results are for a two-qubit system - we can get a 6-qubit system but not much more
# Note: the two-qubit lindbladian divides the one-qubit terms by 2D, and this is the lindbladian for which we get the steady-state results, maybe this should be changed 


from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (
    MODELS, QUANTITIES, compute_point, TLS_VERSION,
)


def _is_current(out: Path) -> bool:
    """True iff `out` exists and carries the current code version."""
    if not out.exists():
        return False
    try:
        d = np.load(out, allow_pickle=True)
        return 'code_version' in d and str(d['code_version']) == TLS_VERSION
    except Exception:
        return False


def run_point(model, point_id: int, args) -> None:
    """Compute and save one grid point (skips if already current)."""
    ix = point_id // model.N_Y
    iy = point_id %  model.N_Y
    p1 = float(model.p1_vals[ix])
    p2 = float(model.p2_vals[iy])
    out_dir = Path(args.out_dir) / model.name
    out = out_dir / f'pt_{ix:03d}_{iy:03d}.npz'

    if _is_current(out):
        print(f'[skip] {model.name}/{out.name} already at version {TLS_VERSION}',
              flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} '
          f'{model.p1_name}={p1:.3f} {model.p2_name}={p2:.3f} '
          f'dim={args.dim} dt={"auto" if args.dt is None else args.dt}',
          flush=True)

    res = compute_point(
        model, p1, p2, dim=args.dim, dt=args.dt,
        fra_restarts=args.fra_restarts,
        fra_maxfev_4=args.fra_maxfev_4,
        fra_maxfev_6=args.fra_maxfev_6,
        sign_restarts=args.sign_restarts,
        ch1_restarts=args.ch1_restarts,
        seed=args.seed + point_id, verbose=True)

    save = {k: np.array(res[k]) for k, _, _, _ in QUANTITIES}
    save.update(
        sign_init=np.array(res['sign_init']),
        floor=np.array(res['floor']),
        opt_S_4=np.asarray(res['opt_S_4']),
        opt_S_6=np.asarray(res['opt_S_6']),
        p1=np.array(p1), p2=np.array(p2),
        ix=np.array(ix), iy=np.array(iy),
        dim=np.array(args.dim), dt=np.array(res['dt']),
        model=np.array(model.name),
        code_version=np.array(TLS_VERSION),
    )
    np.savez(out, **save)
    print(f'  saved {model.name}/{out.name}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',        type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id',      type=int, required=True,
                   help='point id when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks',     type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--out_dir',      type=str, default='results_trotter_v3')
    p.add_argument('--dim',          type=int, default=None, choices=(1, 2, 3),
                   help='default: the model\'s own dim (all models: 2)')
    p.add_argument('--dt',           type=float, default=None,
                   help='pin a fixed dt; default: per-point adaptive choose_dt '
                        '(no model pins a fixed dt any more)')
    p.add_argument('--fra_restarts', type=int, default=5)
    p.add_argument('--fra_maxfev_4', type=int, default=1000)
    p.add_argument('--fra_maxfev_6', type=int, default=500)
    p.add_argument('--sign_restarts', type=int, default=10)
    p.add_argument('--ch1_restarts', type=int, default=15)
    p.add_argument('--seed',         type=int, default=0)
    args = p.parse_args()

    model = MODELS[args.model]
    N = model.N_TOTAL

    # dim defaults to the model's own convention (all models: dim=2).  dt=None is
    # passed through to compute_point, which resolves it to the per-point adaptive
    # choose_dt (no model pins a fixed dt any more); the resolved value is saved.
    if args.dim is None:
        args.dim = model.dim

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        run_point(model, args.task_id, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name}: '
          f'{len(point_ids)} points', flush=True)
    for pid in point_ids:
        run_point(model, pid, args)


if __name__ == '__main__':
    main()
