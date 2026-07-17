"""
Per-point cluster worker for the 4-qubit Trotter RoM sub-pipeline
(trotter_rom_4q): stabilizer-3 framability of the two-qubit bond gate +
Choi-state RoM of the matching 4-qubit (2x2-lattice) gate.

One model (trotter_lindbladian_scan.MODELS, model1..model6) is scanned over
its two varying parameters on the same grid as the main trotter scan:

    ix in 0 .. N_X-1   (p1, x-axis)
    iy in 0 .. N_Y-1   (p2, y-axis)
    point_id = ix * N_Y + iy            (0 .. N_X*N_Y - 1)

The stabilizer-3 framability is NOT recomputed when the main scan already
holds it: for each point the worker first looks for
<scan_dir>/<model>/pt_<ix>_<iy>.npz at the current TLS_VERSION with a matching
dt and reuses its 'stab_fra'.  Points without scan data (all of model6, or any
missing/stale file) get it computed from the bond gate directly.

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz

Usage (single point):
    python scripts/trotter_rom_worker.py --model model6 --task_id 0

Usage (strided across a 200-task array):
    python scripts/trotter_rom_worker.py --model model1 \
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
from trotter_rom_4q import ROM_VERSION, compute_rom_point

# npz keys written per point (beyond the compute_rom_point result keys).
RESULT_KEYS = [
    'stab_fra', 'stab_fra_source', 'rom', 'log2_rom', 'rom_method',
    'rom_solver', 'rom_K', 'rom_certified', 'rom_cert_value',
    'rom_cg_iterations', 'rom_residual_inf', 'rom_n_decomp_terms',
    'rom_time_lp_s',
]


def _is_current(out: Path) -> bool:
    """True iff `out` exists and carries the current ROM_VERSION."""
    if not out.exists():
        return False
    try:
        d = np.load(out, allow_pickle=True)
        return 'code_version' in d and str(d['code_version']) == ROM_VERSION
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


def run_point(model, point_id: int, args) -> None:
    """Compute and save one grid point (skips if already current)."""
    ix = point_id // model.N_Y
    iy = point_id %  model.N_Y
    p1 = float(model.p1_vals[ix])
    p2 = float(model.p2_vals[iy])
    out_dir = Path(args.out_dir) / model.name
    out = out_dir / f'pt_{ix:03d}_{iy:03d}.npz'

    if _is_current(out):
        print(f'[skip] {model.name}/{out.name} already at version {ROM_VERSION}',
              flush=True)
        return

    # Resolve dt exactly as the main scan does, so the reused stab_fra and the
    # 4-qubit gate refer to the same Trotter step.
    dim = model.dim if args.dim is None else args.dim
    if args.dt is not None:
        dt = args.dt
    elif model.dt is not None:
        dt = model.dt
    else:
        dt = choose_dt(*model.build(p1, p2))

    stab_fra = _scan_stab_fra(Path(args.scan_dir), model, ix, iy, dt)

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} '
          f'{model.p1_name}={p1:.4f} {model.p2_name}={p2:.4f} '
          f'dim={dim} dt={dt:.6g} '
          f'stab_fra={"scan" if stab_fra is not None else "compute"}',
          flush=True)

    res = compute_rom_point(
        model, p1, p2, dim=dim, dt=dt, stab_fra=stab_fra,
        method=args.method, solver=args.solver, K=args.K,
        certify=not args.no_certify, verbose=args.verbose)

    save = {k: np.array(res[k]) for k in RESULT_KEYS}
    save.update(
        p1=np.array(p1), p2=np.array(p2),
        ix=np.array(ix), iy=np.array(iy),
        dim=np.array(dim), dt=np.array(res['dt']),
        model=np.array(model.name),
        code_version=np.array(ROM_VERSION),
    )
    np.savez(out, **save)
    print(f'  saved {model.name}/{out.name}  rom={res["rom"]:.6f} '
          f'stab_fra={res["stab_fra"]:.6f}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id',  type=int, required=True,
                   help='point id when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--out_dir',  type=str, default='results_trotter_rom')
    p.add_argument('--scan_dir', type=str, default='results_trotter_v3',
                   help='main trotter-scan results holding the precomputed '
                        'stabilizer-3 framabilities (models 1-5)')
    p.add_argument('--dim',      type=int, default=None, choices=(1, 2, 3),
                   help='default: the model\'s own dim (all models: 2)')
    p.add_argument('--dt',       type=float, default=None,
                   help='pin a fixed dt; default: per-point adaptive choose_dt')
    p.add_argument('--method',   type=str, default='auto',
                   choices=['auto', 'naive', 'cg', 'approx'],
                   help='RoM method (auto -> cg at n_choi=8)')
    p.add_argument('--solver',   type=str, default='auto',
                   choices=['auto', 'gurobi', 'scipy'])
    p.add_argument('--K',        type=float, default=None,
                   help='top/bottom-K fraction for column generation '
                        '(default: handbook-recommended, 1e-8 at n_choi=8)')
    p.add_argument('--no_certify', action='store_true',
                   help='skip the final global optimality certificate')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    model = MODELS[args.model]
    N = model.N_TOTAL

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
