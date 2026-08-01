"""
Per-point cluster worker for the state-RoM sub-pipeline (trotter_rom_state):
stabilizer-3 framability of the two-qubit bond gate + RoM of the 2x2-lattice
state obtained by applying the lattice Trotter step once to the lpdo_max start
state.

One model (model1..model6) is scanned over its FULL trotter_lindbladian_scan
grid:

    ix in 0 .. N_X-1     (p1, x-axis)
    iy in 0 .. N_Y-1     (p2, y-axis)
    point_id = ix * N_Y + iy           (0 .. N_X*N_Y - 1)

Grid sizes: model1/2 21x51, model3/4 51x51, model5 21x101, model6 51x51
(12066 points in total).

The stabilizer-3 framability is NOT recomputed when the main scan already holds
it: for each point the worker first looks for
<scan_dir>/<model>/pt_<ix>_<iy>.npz at the current TLS_VERSION with a matching
dt and reuses its 'stab_fra'.  Points without scan data (all of model6, or any
missing/stale file) get it computed from the bond gate.

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz

Usage (single point):
    python scripts/trotter_rom_state_worker.py --model model6 --task_id 0

Usage (strided across a 200-task array):
    python scripts/trotter_rom_state_worker.py --model model1 \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, TLS_VERSION, choose_dt
from trotter_rom_state import (
    ROM_STATE_VERSION, STATE_ROM_MODELS, compute_state_rom_point, grid_of,
)

DEFAULT_OUT = 'results_trotter_rom_state'

# npz keys written per point, all produced by compute_state_rom_point.
RESULT_KEYS = [
    'stab_fra', 'stab_fra_source', 'stab_fra_pow', 'log10_stab_fra_pow',
    'rom', 'log2_rom', 'rom_rate', 'rom_n_decomp_terms', 'rom_residual_inf',
    'rom_time_lp_s', 'lpdo_init',
]


def _is_current(out: Path) -> bool:
    """True iff `out` exists and carries the current ROM_STATE_VERSION."""
    if not out.exists():
        return False
    try:
        d = np.load(out, allow_pickle=True)
        return 'code_version' in d and str(d['code_version']) == ROM_STATE_VERSION
    except Exception:
        return False


def _scan_stab_fra(scan_dir: Path, model, ix: int, iy: int,
                   dt: float) -> float | None:
    """stab_fra from the main trotter scan, or None if absent / stale / at a
    different dt (then the worker recomputes it from the bond gate)."""
    f = scan_dir / model.name / f'pt_{ix:03d}_{iy:03d}.npz'
    if not f.exists():
        return None
    try:
        d = np.load(f, allow_pickle=True)
        if 'stab_fra' not in d or 'code_version' not in d or 'dt' not in d:
            return None
        if str(d['code_version']) != TLS_VERSION:
            return None
        if not np.isclose(float(d['dt']), dt, rtol=1e-8, atol=0.0):
            return None
        v = float(d['stab_fra'])
        return v if np.isfinite(v) else None
    except Exception:
        return None


def run_point(model, p1_vals, p2_vals, point_id: int, args) -> None:
    """Compute and save one grid point (skips if already current)."""
    n2 = len(p2_vals)
    ix, iy = point_id // n2, point_id % n2
    p1, p2 = float(p1_vals[ix]), float(p2_vals[iy])
    out_dir = Path(args.out_dir) / model.name
    out = out_dir / f'pt_{ix:03d}_{iy:03d}.npz'

    if _is_current(out):
        print(f'[skip] {model.name}/{out.name} already at version '
              f'{ROM_STATE_VERSION}', flush=True)
        return

    # Resolve dt exactly as the main scan does, so the reused stab_fra and the
    # lattice propagator refer to the same Trotter step.
    dim = model.dim if args.dim is None else args.dim
    if args.dt is not None:
        dt = args.dt
    elif model.dt is not None:
        dt = model.dt
    else:
        dt = choose_dt(*model.build(p1, p2))

    stab_fra = None
    if not args.recompute_fra:
        stab_fra = _scan_stab_fra(Path(args.scan_dir), model, ix, iy, dt)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'[point {point_id}/{len(p1_vals) * n2}] {model.name} '
          f'{model.p1_name}={p1:.4f} {model.p2_name}={p2:.4f} '
          f'dim={dim} dt={dt:.6g} '
          f'stab_fra={"scan" if stab_fra is not None else "compute"}',
          flush=True)

    res = compute_state_rom_point(model, p1, p2, dim=dim, dt=dt,
                                  stab_fra=stab_fra, verbose=args.verbose)

    save = {k: np.array(res[k]) for k in RESULT_KEYS}
    save.update(
        p1=np.array(p1), p2=np.array(p2),
        ix=np.array(ix), iy=np.array(iy),
        dim=np.array(dim), dt=np.array(res['dt']),
        model=np.array(model.name),
        code_version=np.array(ROM_STATE_VERSION),
    )
    np.savez(out, **save)
    print(f'  saved {model.name}/{out.name}  rom={res["rom"]:.8f} '
          f'stab_fra={res["stab_fra"]:.6f}  '
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
    p.add_argument('--scan_dir', type=str, default='results_trotter_v3',
                   help='main trotter-scan results holding the precomputed '
                        'stabilizer-3 framabilities (models 1-5)')
    p.add_argument('--recompute_fra', action='store_true',
                   help='always recompute stab_fra instead of reusing the scan')
    p.add_argument('--dim',      type=int, default=None, choices=(1, 2, 3),
                   help="default: the model's own dim (all models: 2)")
    p.add_argument('--dt',       type=float, default=None,
                   help='pin a fixed dt; default: per-point adaptive choose_dt')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    model = MODELS[args.model]
    p1_vals, p2_vals = grid_of(args.model)
    N = len(p1_vals) * len(p2_vals)

    # Fail fast: load the stabilizer matrix once up front, so a missing handbook
    # data file kills the task with a single clear error instead of one
    # traceback per point.
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
          f'{len(point_ids)} points', flush=True)
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
