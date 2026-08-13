"""Generic per-point cluster worker for the "unitary magic scan" model family
(model1_ising_tilted_magic_scan.py .. model5_xy_dm_magic_scan.py).

Unlike scripts/xxz_magic_worker.py (which is tied to xxz_magic_scan.py), this
worker takes the model file to use as a module name (`--scan_module`) and
imports its `MODELS` dict at runtime, so the same worker script serves all
five models -- only `MODELS`, `MAGIC_VERSION` and the model's own `build`
function differ between them.  Everything else (compute_point,
min_dt_over_grid, TASK_KEYS, ...) is the model-agnostic machinery of
magic_scan_common.py, shared by every scan module.

The unit of work is a GRID POINT: all seven quantities (fra_D1, fra_D2,
nc_2a .. nc_2e) are computed together, because they share the model terms,
the Trotter step choose_dt and -- for the five non-cliffordness maps -- a
single eigendecomposition of the 2x2-lattice Hamiltonian.

    ix in 0 .. N_P1-1     (p1, x-axis)
    iy in 0 .. N_P2-1     (p2, y-axis)
    point_id = ix * N_P2 + iy           (0 .. N_P1*N_P2 - 1)

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz.  The raw framability
ladders are stored next to the extrapolated values, so the dt -> 0 fit can be
changed at collect time without re-running a single LP.

Usage (single point):
    python scripts/magic_worker.py --scan_module model1_ising_tilted_magic_scan --task_id 0

Usage (strided across a 200-task array):
    python scripts/magic_worker.py --scan_module model1_ising_tilted_magic_scan \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

from __future__ import annotations

import os
# Single-thread the BLAS/LP backend so the SLURM array (not nested threads) owns
# the parallelism, as in trotter_rom_dtbase_worker.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from magic_scan_common import (                                    # noqa: E402
    TASK_KEYS, FRA_KEYS, NC_KEYS, DIMS, compute_point, min_dt_over_grid,
)
from trotter_rom_dtbase import FIT_N_DEFAULT, DEG_DEFAULT          # noqa: E402

DEFAULT_OUT = 'results_unitary_magic'
GROUPS = ('fra', 'nc', 'both')

# Scalars stored per point besides the seven task keys.
SCALAR_KEYS = ['dt_pt', 'dt_min', 'gap', 'gap_next', 'e_min', 'e_max',
               't_2a', 't_2b', 't_2c', 't_2d', 't_2e',
               'fra_rate_D1', 'fra_rate_D2',
               'fra_raw_lim_D1', 'fra_raw_lim_D2']
ARRAY_KEYS = ['fra_dts', 'fra_raw_D1', 'fra_raw_D2']


def _groups_of(name: str) -> tuple[str, ...]:
    return ('fra', 'nc') if name == 'both' else (name,)


def _keys_of(groups: tuple[str, ...]) -> list[str]:
    keys: list[str] = []
    if 'fra' in groups:
        keys += list(FRA_KEYS)
    if 'nc' in groups:
        keys += list(NC_KEYS)
    return keys


def _is_current(out: Path, groups: tuple[str, ...], magic_version: str) -> bool:
    """True iff `out` exists, carries the current version, and already holds
    finite values for every key of the requested groups."""
    if not out.exists():
        return False
    try:
        d = np.load(out, allow_pickle=True)
        if str(d.get('code_version', '')) != magic_version:
            return False
        for k in _keys_of(groups):
            if k not in d.files:
                return False
            if np.isfinite(d[k]):
                continue
            # Two non-finite values are legitimate and must not trigger a
            # recompute on every resubmission:
            #   fra_D*  exp(rate) overflowed float64 -- fra_rate_D* carries it
            #   nc_2e   T = 100/gap is undefined for a degenerate ground state
            if k in FRA_KEYS and np.isfinite(d.get(f'fra_rate_D{k[-1]}', np.nan)):
                continue
            if k == 'nc_2e' and not np.isfinite(d.get('gap', np.nan)):
                continue
            return False
        if 'fra' in groups and 'fra_dts' not in d:
            return False
        return True
    except Exception:
        return False


def _merge_previous(out: Path, save: dict, magic_version: str) -> dict:
    """Keep the values of the group that was not recomputed, if any exist."""
    if not out.exists():
        return save
    try:
        d = np.load(out, allow_pickle=True)
        if str(d.get('code_version', '')) != magic_version:
            return save
        for k in d.files:
            save.setdefault(k, d[k])
    except Exception:
        pass
    return save


def run_point(model, point_id: int, dt_min: float, magic_version: str, args) -> None:
    """Compute and save one grid point."""
    n2 = len(model.p2_vals)
    ix, iy = point_id // n2, point_id % n2
    p1, p2 = model.point(ix, iy)
    groups = _groups_of(args.group)

    out_dir = Path(args.out_dir) / model.name
    out = out_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if not args.force and _is_current(out, groups, magic_version):
        print(f'[skip] {model.name}/{out.name} already at version '
              f'{magic_version}', flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.n_points}] {model.name} '
          f'{model.p1_name}={p1:.4f} {model.p2_name}={p2:.4f}  '
          f'groups={"+".join(groups)}', flush=True)

    res = compute_point(model, p1, p2, dt_min=dt_min, groups=groups,
                        dims=DIMS, fit_n=args.fit_n, deg=args.deg,
                        verbose=args.verbose)

    save: dict = {}
    for k in _keys_of(groups):
        save[k] = np.array(float(res[k]))
        if k in NC_KEYS:
            save[f'{k}_bits'] = np.array(float(res[f'{k}_bits']))
    for k in SCALAR_KEYS:
        if k in res:
            save[k] = np.array(float(res[k]))
    for k in ARRAY_KEYS:
        if k in res:
            save[k] = np.asarray(res[k], dtype=float)
    save.update(
        p1=np.array(p1), p2=np.array(p2),
        ix=np.array(ix), iy=np.array(iy),
        model=np.array(model.name),
        p1_name=np.array(model.p1_name), p2_name=np.array(model.p2_name),
        fit_n=np.array(args.fit_n), deg=np.array(args.deg),
        code_version=np.array(magic_version),
    )
    save = _merge_previous(out, save, magic_version)
    np.savez(out, **save)

    done = '  '.join(f'{k}={float(save[k]):.6f}' for k in _keys_of(groups))
    print(f'  saved {model.name}/{out.name}  {done}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--scan_module', type=str, required=True,
                   help="model file to use, e.g. 'model1_ising_tilted_magic_scan'")
    p.add_argument('--model',    type=str, default=None,
                   help='model key within --scan_module\'s MODELS dict '
                        '(defaults to its only entry)')
    p.add_argument('--task_id',  type=int, required=True,
                   help='point id when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--out_dir',  type=str, default=DEFAULT_OUT)
    p.add_argument('--group',    type=str, default='both', choices=GROUPS,
                   help="'fra' = the two dt->0 framability maps (expensive), "
                        "'nc' = the five non-cliffordness maps (cheap), "
                        "'both' = default")
    p.add_argument('--fit_n',    type=int, default=FIT_N_DEFAULT,
                   help='points nearest dt=0 used by the extrapolation fit '
                        '(the ladder only has 10, so this caps there)')
    p.add_argument('--deg',      type=int, default=DEG_DEFAULT,
                   help='degree of the polynomial in dt of the fit')
    p.add_argument('--force',    action='store_true',
                   help='recompute even if the npz is already current')
    p.add_argument('--verbose',  action='store_true',
                   help='print every dt / every time of every point')
    args = p.parse_args()

    scan_mod = importlib.import_module(args.scan_module)
    MODELS = scan_mod.MODELS
    magic_version = scan_mod.MAGIC_VERSION
    model_key = args.model or next(iter(MODELS))
    if model_key not in MODELS:
        print(f'ERROR: model {model_key!r} not in {args.scan_module}.MODELS '
              f'({list(MODELS)})', file=sys.stderr)
        sys.exit(1)
    model = MODELS[model_key]
    N = model.n_points

    # Fail fast on a broken environment: the Choi state needs rom_of_gate, which
    # exits at import time when the RoM-handbook clone is missing.
    if 'nc' in _groups_of(args.group):
        try:
            from magic_measures import noncliffordness
            noncliffordness(np.eye(2, dtype=complex))
        except Exception as e:
            print(f'ERROR: the non-cliffordness path is unusable ({e!r}).',
                  file=sys.stderr)
            sys.exit(1)

    # Deterministic and cheap; every task derives the 2a/2c step itself.
    dt_min = min_dt_over_grid(model)
    print(f'[{args.scan_module}/{model.name}] grid {model.shape[0]}x{model.shape[1]} '
          f'= {N} points, dt_min = {dt_min:.6g}', flush=True)

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        run_point(model, args.task_id, dt_min, magic_version, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name}: '
          f'{len(point_ids)} points ({args.group})', flush=True)
    n_fail = 0
    for pid in point_ids:
        # One bad point must not kill the rest of the chunk.
        try:
            run_point(model, pid, dt_min, magic_version, args)
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
