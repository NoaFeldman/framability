"""
Backfill the d_ext_single=8 Heisenberg optimised framability (opt_fra_8) into an
existing Trotter scan, in place, WITHOUT re-running the full scan.

The scan (trotter_lindbladian_scan.compute_point) stores the Heisenberg optimised
framability only at d_ext_single = 4 and 6 (opt_fra_4 / opt_fra_6).  This worker
adds opt_fra_8 -- and its optimised frame opt_S_8 -- at every grid point, computed
with the SAME support-enforcing optimiser as scripts/trotter_reopt_worker.py
(alternating certificate + Polyak floor-polish, per-Pauli support required, always
seeded with the full-support ixyz extended-Pauli frame), so the returned frame
spans every Pauli (never a vacuous, collapsed certificate).

The gate is rebuilt from each point's own stored (p1, p2, dim, dt); no scan
parameter is re-derived and no other stored quantity is touched.  A d8_version
stamp records provenance; a point already carrying a valid opt_fra_8 at the
current stamp is skipped, so the array is safely resubmittable.  On resubmission
the value is never degraded (kept only if a rerun finds a strictly smaller,
full-support framability).

Grid layout mirrors trotter_scan_worker: point_id = ix * N_Y + iy, strided across
--n_chunks array tasks; one read/write per pt file (no cross-task collision).

    python scripts/trotter_d8_worker.py --model model7a \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))        # repo root (top-level modules)
sys.path.insert(0, str(_HERE))               # scripts/ (reuse reopt helpers)

from trotter_lindbladian_scan import MODELS, bond_trotter_gate
from optimize_framability import spectral_floor, OPT_VERSION
# Reuse the support-enforcing Heisenberg re-optimiser and its helpers verbatim so
# opt_fra_8 is produced by exactly the same (fixed) method as opt_fra_4 / opt_fra_6.
from trotter_reopt_worker import _reopt_heis, _pauli_support_ok

DE8 = 8
D8_VERSION = f'{OPT_VERSION}-d8v1'


def process_point(in_dir: Path, name: str, ix: int, iy: int, args) -> None:
    f = in_dir / name / f'pt_{ix:03d}_{iy:03d}.npz'
    if not f.exists():
        print(f'[miss] {name}/pt_{ix:03d}_{iy:03d}.npz', flush=True)
        return
    z = np.load(f, allow_pickle=True)
    already = ('opt_fra_8' in z.files and 'd8_version' in z.files
               and str(z['d8_version']) == D8_VERSION
               and 'opt_S_8' in z.files and _pauli_support_ok(z['opt_S_8']))
    if not args.force and already:
        print(f'[skip] {name}/{f.name} already valid at {D8_VERSION}', flush=True)
        z.close()
        return
    data = {k: z[k] for k in z.files}
    z.close()

    p1, p2 = float(data['p1']), float(data['p2'])
    dt, dim = float(data['dt']), int(data['dim'])
    H1, H2, j1, j2 = MODELS[name].build(p1, p2)
    gate = bond_trotter_gate(H1, H2, j1, j2, dim, dt)
    if abs(spectral_floor(gate) - float(data.get('floor', spectral_floor(gate)))) > 1e-6:
        print(f'[WARN] {name}/{f.name}: gate/floor mismatch', flush=True)

    t0 = time.perf_counter()
    new_val, new_S = _reopt_heis(
        gate, DE8, data.get('opt_S_8'), in_dir, name, ix, iy, args)

    # Never degrade: keep a previously stored, full-support opt_fra_8 if it is
    # already smaller (a rerun only ever lowers the framability).
    prev = float(data['opt_fra_8']) if ('opt_fra_8' in data
                                        and _pauli_support_ok(data.get('opt_S_8'))) \
        else float('inf')
    if np.isfinite(new_val) and new_val < prev - 1e-12:
        data['opt_fra_8'] = np.array(new_val)
        data['opt_S_8'] = np.asarray(new_S)
        chosen = new_val
        tag = 'set' if not np.isfinite(prev) else 'IMPROVED'
    else:
        data.setdefault('opt_fra_8', np.array(new_val))
        data.setdefault('opt_S_8', np.asarray(new_S))
        chosen = float(data['opt_fra_8'])
        tag = 'kept'
    data['d8_version'] = np.array(D8_VERSION)

    tmp = f.with_name(f'{f.stem}.tmp{os.getpid()}.npz')
    np.savez(tmp, **data)
    os.replace(tmp, f)
    print(f'{name}/pt_{ix:03d}_{iy:03d} ({p1:.3f},{p2:.3f})  '
          f'opt_fra_8={chosen:.6f} [{tag}]  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id',  type=int, required=True,
                   help='point id when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--in_dir',   type=str, default='results_trotter_v3')
    p.add_argument('--n_restarts',   type=int, default=12)
    p.add_argument('--maxfev',       type=int, default=6000,
                   help='budget for the alternating method')
    p.add_argument('--polish_iters', type=int, default=300)
    p.add_argument('--seed',   type=int, default=0)
    p.add_argument('--force',  action='store_true')
    args = p.parse_args()

    model = MODELS[args.model]
    N = model.N_TOTAL
    in_dir = Path(args.in_dir)

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        ix, iy = args.task_id // model.N_Y, args.task_id % model.N_Y
        process_point(in_dir, model.name, ix, iy, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[task {args.task_id}/{args.n_chunks}] {model.name}: '
          f'{len(point_ids)} points  ({D8_VERSION})', flush=True)
    for pid in point_ids:
        ix, iy = pid // model.N_Y, pid % model.N_Y
        process_point(in_dir, model.name, ix, iy, args)


if __name__ == '__main__':
    main()
