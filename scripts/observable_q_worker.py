"""
Observable quality factor Q_obs of the bond generator over a model's scan grid
-- the Q_obs panels and Q_obs = 1 contours of
results_<model>_rate/<model>_rate_panels.png and
results_dtbase_line/<model>_dtbase_extrap.png.

For A = L_bond^T in the two-qubit Pauli basis (L_bond =
trotter_lindbladian_scan.build_bond_lindbladian at the scan's dim),

    Q_obs(P) = sum_{P' != P} |A_{P'P}| / (-A_{PP}),     Q_obs = max_P Q_obs(P)

(liouvillian_quality.observable_quality): the rate at which the generator moves
weight off the Pauli string P, in units of P's own decay rate.  Q_obs <= 1
exactly when the Pauli-frame rate vanishes, and Q_obs(P) > 1 exactly when the
Pauli-l1 weight of the Heisenberg-evolved P grows at t = 0+.  Unlike the mode
quality factor Q_max it sees rotations that the dissipation Zeno-freezes into an
overdamped spectrum (model4: the field rotating the weakly damped Z axis), which
is what shapes the kink of the mu* = 0 line.

Stored per point:
  obs_Q, obs_label, obs_cols              Pauli basis (16 column ratios)
  obs_opt_Q, obs_opt_label, obs_opt_cols  best local basis: one rotation R for
  obs_opt_rotvec, obs_opt_w               both qubits and axis lengths w
                                          (observable_quality_opt; --no_opt skips)

Bond generator only (16x16): the Pauli part is microseconds per point, the
optimised part about a second.

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz   (existing files skipped)

Usage:
    python scripts/observable_q_worker.py --model model4 --task_id 0 --n_chunks 200
    python scripts/observable_q_worker.py --model model3 --task_id 0 --n_chunks 200 --no_opt
"""

from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,     # noqa: E402
                                      DIM_DEFAULT)
from liouvillian_quality import (observable_quality,                     # noqa: E402
                                 observable_quality_opt, OBS_QUALITY_VERSION,
                                 TOL_REL_DEFAULT)


def grid_vals(model: str, stride: int):
    m = MODELS[model]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def run_point(model: str, ix: int, iy: int, args) -> None:
    m = MODELS[model]
    p1_vals, p2_vals = grid_vals(model, args.stride)
    p1, p2 = float(p1_vals[ix]), float(p2_vals[iy])

    pt_dir = Path(args.out_dir) / model
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists():
        print(f'[skip] {model}/{out_f.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    rec: dict = dict(model=model, ix=ix, iy=iy, stride=args.stride,
                     p1=p1, p2=p2, p1_name=m.p1_name, p2_name=m.p2_name,
                     dim=args.dim, tol_rel=args.tol_rel,
                     obs_quality_version=OBS_QUALITY_VERSION)
    try:
        H1, H2, jumps1, jumps2 = m.build(p1, p2)
        L = build_bond_lindbladian(H1, H2, jumps1, jumps2, args.dim).real
        ro = observable_quality(L, tol_rel=args.tol_rel)
        rec.update(obs_Q=ro.Q_obs, obs_label=ro.label, obs_cols=ro.cols)
        if not args.no_opt:
            rp = observable_quality_opt(L, n_restarts=args.n_restarts,
                                        maxfev=args.maxfev, seed=args.seed,
                                        tol_rel=args.tol_rel)
            rec.update(obs_opt_Q=rp.Q_obs, obs_opt_label=rp.label,
                       obs_opt_cols=rp.cols, obs_opt_rotvec=rp.rotvec,
                       obs_opt_w=rp.w, n_restarts=args.n_restarts,
                       maxfev=args.maxfev, seed=args.seed)
    except Exception as e:
        print(f'  ERROR ({m.p1_name}={p1:.3f} {m.p2_name}={p2:.3f}): '
              f'{type(e).__name__}: {e}', flush=True)
        return

    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_f, **rec)
    opt = (f"  opt Q_obs={rec['obs_opt_Q']:.4g} ({rec['obs_opt_label']})"
           if 'obs_opt_Q' in rec else '')
    print(f'[{model}] ({ix},{iy}) {m.p1_name}={p1:.3f} {m.p2_name}={p2:.3f}  '
          f"Q_obs={rec['obs_Q']:.4g} ({rec['obs_label']}){opt}  "
          f'({time.perf_counter() - t0:.1f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='model4', choices=list(MODELS))
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks '
                        '(n_chunks<=1: task_id is a flat grid index)')
    p.add_argument('--out_dir', type=str, default='results_observable_q')
    p.add_argument('--stride', type=int, default=1,
                   help='stride on the model grid (1 = full grid)')
    p.add_argument('--dim', type=int, default=DIM_DEFAULT,
                   help='bond Trotter convention (single-site share 1/(2 dim)); '
                        'must match the framability scans')
    p.add_argument('--no_opt', action='store_true',
                   help='Pauli basis only (skip the local-basis optimisation)')
    p.add_argument('--n_restarts', type=int, default=8)
    p.add_argument('--maxfev', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tol_rel', type=float, default=TOL_REL_DEFAULT)
    args = p.parse_args()

    p1_vals, p2_vals = grid_vals(args.model, args.stride)
    nx, ny = len(p1_vals), len(p2_vals)
    n_total = nx * ny

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
            sys.exit(1)
        run_point(args.model, args.task_id // ny, args.task_id % ny, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)
    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {args.model}: {len(ids)} of '
          f'{n_total} points ({nx}x{ny} grid)', flush=True)
    for pid in ids:
        run_point(args.model, pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
