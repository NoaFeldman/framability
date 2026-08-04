"""
Per-point cluster worker for the NESS RoM (trotter_ness_rom): the Robustness of
Magic of the steady state of the 2x2 lattice Lindbladian.

Runs on the SAME decimated grid as the DT_BASE sweep (trotter_rom_dtbase, every
`stride`-th value of each model axis) and uses the SAME full-grid file naming,
so the two pipelines line up point for point and the collect step can read them
side by side.

    ix in 0 .. N_X-1     (p1, x-axis, full-grid index = irx * stride)
    iy in 0 .. N_Y-1     (p2, y-axis, full-grid index = iry * stride)
    point_id = irx * n2 + iry           (over the DECIMATED grid)

This is far cheaper than the DT_BASE sweep -- one steady state and one LP per
point, no base loop -- so the array finishes quickly.

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz

Usage (single point):
    python scripts/trotter_ness_rom_worker.py --model model1 --task_id 0

Usage (strided across a 200-task array):
    python scripts/trotter_ness_rom_worker.py --model model1 \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

from __future__ import annotations

import os
# Single-thread the BLAS/LP backend so the SLURM array owns the parallelism.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS
from trotter_rom_state import STATE_ROM_MODELS
from trotter_rom_dtbase import GRID_STRIDE_DEFAULT, dtbase_grid
from trotter_ness_rom import NESS_ROM_VERSION, compute_ness_rom_point

DEFAULT_OUT = 'results_trotter_ness_rom'

# npz keys written per point.
RESULT_KEYS = ['ness_rom', 'log2_ness_rom', 'ness_purity', 'ness_ok',
               'lind_rate', 'ness_time_s']


def _is_current(out: Path) -> bool:
    """True iff `out` exists and carries the current NESS_ROM_VERSION.

    A NaN ness_rom is a legitimate result (no unique steady state), so unlike
    the DT_BASE worker this does NOT require finite values.
    """
    if not out.exists():
        return False
    try:
        d = np.load(out, allow_pickle=True)
        return 'code_version' in d and str(d['code_version']) == NESS_ROM_VERSION
    except Exception:
        return False


def run_point(model, p1_vals, p2_vals, point_id: int, args) -> None:
    """Compute and save one grid point (skips if already current)."""
    n2 = len(p2_vals)
    irx, iry = point_id // n2, point_id % n2
    p1, p2 = float(p1_vals[irx]), float(p2_vals[iry])
    ix, iy = irx * args.stride, iry * args.stride      # full-grid indices
    out_dir = Path(args.out_dir) / model.name
    out = out_dir / f'pt_{ix:03d}_{iy:03d}.npz'

    if _is_current(out):
        print(f'[skip] {model.name}/{out.name} already at version '
              f'{NESS_ROM_VERSION}', flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'[point {point_id}/{len(p1_vals) * n2}] {model.name} '
          f'{model.p1_name}={p1:.4f} {model.p2_name}={p2:.4f}', flush=True)

    res = compute_ness_rom_point(model, p1, p2, verbose=args.verbose)

    save = {k: np.array(res[k]) for k in RESULT_KEYS}
    save.update(
        p1=np.array(p1), p2=np.array(p2),
        ix=np.array(ix), iy=np.array(iy),
        stride=np.array(args.stride),
        model=np.array(model.name),
        code_version=np.array(NESS_ROM_VERSION),
    )
    np.savez(out, **save)
    print(f'  saved {model.name}/{out.name}  ness_rom={res["ness_rom"]:.8f} '
          f'purity={res["ness_purity"]:.6f}  '
          f'({time.perf_counter() - t0:.1f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, required=True,
                   choices=list(STATE_ROM_MODELS))
    p.add_argument('--task_id',  type=int, required=True,
                   help='point id when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--out_dir',  type=str, default=DEFAULT_OUT)
    p.add_argument('--stride',   type=int, default=GRID_STRIDE_DEFAULT,
                   help='grid decimation; must match the DT_BASE sweep so the '
                        'two pipelines share grid points (default 2)')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    model = MODELS[args.model]
    p1_vals, p2_vals = dtbase_grid(args.model, args.stride)
    N = len(p1_vals) * len(p2_vals)

    # Fail fast on a broken environment before burning a task on per-point errors.
    from trotter_rom_state import load_amat, N_LATTICE
    try:
        load_amat(N_LATTICE)
    except Exception as e:
        print(f'ERROR: cannot load the {N_LATTICE}-qubit stabilizer matrix '
              f'({e!r}).', file=sys.stderr)
        sys.exit(1)

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        run_point(model, p1_vals, p2_vals, args.task_id, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name}: '
          f'{len(point_ids)} points (stride {args.stride}, '
          f'{len(p1_vals)}x{len(p2_vals)} grid)', flush=True)
    n_fail = 0
    for pid in point_ids:
        # One bad point must not kill the rest of the chunk.
        try:
            run_point(model, p1_vals, p2_vals, pid, args)
        except Exception:
            n_fail += 1
            import traceback
            traceback.print_exc()
            print(f'[fail] point {pid} failed; continuing', flush=True)
    if n_fail:
        print(f'ERROR: {n_fail}/{len(point_ids)} points failed', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
