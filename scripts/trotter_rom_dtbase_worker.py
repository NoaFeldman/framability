"""
Per-point cluster worker for the DT_BASE sweep of trotter_rom_dtbase: the whole
DT_BASE line (stabilizer-3 framability + state RoM at every base) for one grid
point of one model.

The unit of work is a GRID POINT, not a (point, base) pair: L_bond and L_full do
not depend on DT_BASE, so sweeping all bases inside one task reuses them and
keeps the output at one npz per point (12066 files over the six models, the same
layout as results_trotter_rom_state) instead of one per base.

One model (model1..model6) is scanned over its FULL trotter_lindbladian_scan
grid:

    ix in 0 .. N_X-1     (p1, x-axis)
    iy in 0 .. N_Y-1     (p2, y-axis)
    point_id = ix * N_Y + iy           (0 .. N_X*N_Y - 1)

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz, holding the arrays
base, dt, stab_fra, rom, log2_rom over the base grid.

Usage (single point):
    python scripts/trotter_rom_dtbase_worker.py --model model1 --task_id 0

Usage (strided across a 200-task array):
    python scripts/trotter_rom_dtbase_worker.py --model model1 \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

from __future__ import annotations

import os
# Single-thread the BLAS/LP backend so the SLURM array (not nested threads) owns
# the parallelism, as in trotter_dtbase_line_worker.
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
from trotter_rom_state import STATE_ROM_MODELS, grid_of
from trotter_rom_dtbase import (
    ROM_DTBASE_VERSION, BASE_MODES, DEFAULT_BASE_MODE, base_grid,
    compute_base_line,
)

DEFAULT_OUT = 'results_trotter_rom_dtbase'

# Array keys written per point (all over the base grid).
LINE_KEYS = ['base', 'dt', 'stab_fra', 'rom', 'log2_rom']


def _is_current(out: Path, mode: str) -> bool:
    """True iff `out` exists, carries the current version, and holds the whole
    base grid of the requested mode."""
    if not out.exists():
        return False
    try:
        d = np.load(out, allow_pickle=True)
        if 'code_version' not in d or str(d['code_version']) != ROM_DTBASE_VERSION:
            return False
        if str(d.get('base_mode', '')) != mode:
            return False
        return bool(np.all(np.isfinite(d['stab_fra']))
                    and np.all(np.isfinite(d['rom']))
                    and len(d['base']) == len(base_grid(mode)))
    except Exception:
        return False


def run_point(model, p1_vals, p2_vals, point_id: int, args) -> None:
    """Compute and save one grid point's whole DT_BASE line."""
    n2 = len(p2_vals)
    ix, iy = point_id // n2, point_id % n2
    p1, p2 = float(p1_vals[ix]), float(p2_vals[iy])
    out_dir = Path(args.out_dir) / model.name
    out = out_dir / f'pt_{ix:03d}_{iy:03d}.npz'

    if _is_current(out, args.mode):
        print(f'[skip] {model.name}/{out.name} already at version '
              f'{ROM_DTBASE_VERSION} (mode {args.mode})', flush=True)
        return

    bases = base_grid(args.mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'[point {point_id}/{len(p1_vals) * n2}] {model.name} '
          f'{model.p1_name}={p1:.4f} {model.p2_name}={p2:.4f}  '
          f'{len(bases)} bases ({args.mode})', flush=True)

    res = compute_base_line(model, p1, p2, bases=bases, dim=args.dim,
                            verbose=args.verbose)

    save = {k: np.asarray(res[k]) for k in LINE_KEYS}
    save.update(
        p1=np.array(p1), p2=np.array(p2),
        ix=np.array(ix), iy=np.array(iy),
        dim=np.array(res['dim']),
        lpdo_init=np.array(res['lpdo_init']),
        base_mode=np.array(args.mode),
        model=np.array(model.name),
        code_version=np.array(ROM_DTBASE_VERSION),
    )
    np.savez(out, **save)
    print(f'  saved {model.name}/{out.name}  '
          f'stab_fra {res["stab_fra"][0]:.6f}..{res["stab_fra"][-1]:.6f}  '
          f'rom {res["rom"][0]:.6f}..{res["rom"][-1]:.6f}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, required=True,
                   choices=list(STATE_ROM_MODELS))
    p.add_argument('--task_id',  type=int, required=True,
                   help='point id when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--out_dir',  type=str, default=DEFAULT_OUT)
    p.add_argument('--mode',     type=str, default=DEFAULT_BASE_MODE,
                   choices=list(BASE_MODES),
                   help="'fit': 20 bases 0.01..0.20 (default, all the "
                        "extrapolation uses); 'full': the 99-base "
                        'results_dtbase_line grid')
    p.add_argument('--dim',      type=int, default=None, choices=(1, 2, 3),
                   help="default: the model's own dim (all models: 2)")
    p.add_argument('--verbose', action='store_true',
                   help='print every base of every point (very chatty)')
    args = p.parse_args()

    model = MODELS[args.model]
    p1_vals, p2_vals = grid_of(args.model)
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
          f'{len(point_ids)} points x {len(base_grid(args.mode))} bases',
          flush=True)
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
