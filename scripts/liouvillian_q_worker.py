"""
Quality factor Q_max of the Lindbladian over a model's scan grid -- the Q
panels of results_<model>_rate/<model>_rate_panels.png and of
results_dtbase_line/<model>_dtbase_extrap.png.

For each grid point two exact spectra are classified by
liouvillian_quality.quality_factor (Q_k = |Im lambda_k|/|Re lambda_k| over the
damped modes; steady and undamped modes are separated, never divided by a
numerically zero Re):

  bond_*  the 16x16 two-qubit bond generator build_bond_lindbladian(...) at the
          scan's dim -- the very generator the framability rates are computed
          for, so its Q_max is the one directly comparable with mu*
          (model3 at gamma = 0: mu*_Pauli = 0 exactly where bond Q_max = 1).
  lat_*   the full Lindbladian of an Ly x Lx open-boundary lattice
          (default 2x3 = 6 qubits, 4096x4096 dense) with the model's physics at
          full coupling strength -- the physical many-body Q.  Exact, unlike
          the sparse 8-qubit osc_rate panels, which maximise over the slowest
          64 modes only.

Models: those n_qubit_lindbladian.model_lattice_params maps onto
build_lindbladian_comp (model3, model4, model8).

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz  (existing files skipped)

Usage:
    python scripts/liouvillian_q_worker.py --model model3 --task_id 0 --n_chunks 200
    python scripts/liouvillian_q_worker.py --model model8 --task_id 5 --n_chunks 200 --no_lattice
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
from n_qubit_lindbladian import (build_lindbladian_comp,                  # noqa: E402
                                 model_lattice_params, LATTICE_MODELS)
from dissipative_PT import bonds_2d                                      # noqa: E402
from liouvillian_quality import (quality_factor, QUALITY_VERSION,        # noqa: E402
                                 TOL_REL_DEFAULT, DENSE_MAX_DIM_DEFAULT)

J = 1.0                                        # models 3, 4, 8: J = 1
N_TOP = 64                                     # most coherent lattice modes kept


def grid_vals(model: str, stride: int):
    m = MODELS[model]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def _pack(prefix: str, res, evals, *, keep_all: bool) -> dict:
    out = {
        f'{prefix}_Q_max': res.Q_max,
        f'{prefix}_lam_Q_re': np.nan if res.lam_Q is None else res.lam_Q.real,
        f'{prefix}_lam_Q_im': np.nan if res.lam_Q is None else res.lam_Q.imag,
        f'{prefix}_Q_slowest': res.Q_slowest,
        f'{prefix}_gap': res.gap,
        f'{prefix}_n_damped': res.n_damped,
        f'{prefix}_n_undamped': res.n_undamped,
        f'{prefix}_n_steady': res.n_steady,
        f'{prefix}_n_growing': res.n_growing,
        f'{prefix}_omega_undamped': res.omega_undamped,
        f'{prefix}_tol': res.tol,
    }
    if keep_all:
        out[f'{prefix}_evals'] = evals
    else:
        # the N_TOP damped modes with the largest Q (a full 4096-mode spectrum
        # per point would be ~170 MB per model grid)
        lam = np.asarray(evals, complex)
        d = lam[lam.real < -res.tol]
        q = np.abs(d.imag) / -d.real
        out[f'{prefix}_evals_topQ'] = d[np.argsort(-q)[:N_TOP]]
    return out


def run_point(model: str, ix: int, iy: int, args) -> None:
    m = MODELS[model]
    p1_vals, p2_vals = grid_vals(model, args.stride)
    p1, p2 = float(p1_vals[ix]), float(p2_vals[iy])
    params = model_lattice_params(model, p1, p2)

    pt_dir = Path(args.out_dir) / model
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists():
        print(f'[skip] {model}/{out_f.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{model}] point ({ix},{iy})  {m.p1_name}={p1:.3f} '
          f'{m.p2_name}={p2:.3f}', flush=True)
    rec: dict = dict(model=model, ix=ix, iy=iy, stride=args.stride,
                     p1=p1, p2=p2, p1_name=m.p1_name, p2_name=m.p2_name,
                     J=J, dim=args.dim, tol_rel=args.tol_rel,
                     quality_version=QUALITY_VERSION, **params)

    # ---- bond generator (16x16, the framability-rate generator) -------------
    try:
        H1, H2, jumps1, jumps2 = m.build(p1, p2)
        Lb = build_bond_lindbladian(H1, H2, jumps1, jumps2, args.dim).real
        rb, eb = quality_factor(Lb, tol_rel=args.tol_rel, return_eigenvalues=True)
        rec.update(_pack('bond', rb, eb, keep_all=True))
        for w in rb.warnings:
            print(f'  WARNING (bond): {w}', flush=True)
    except Exception as e:
        print(f'  ERROR (bond): {type(e).__name__}: {e}', flush=True)
        return

    # ---- full lattice Lindbladian (exact dense spectrum) --------------------
    rec.update(lat_Lx=args.lx, lat_Ly=args.ly, lat_N=args.lx * args.ly,
               lat_done=False)
    if not args.no_lattice:
        try:
            t1 = time.perf_counter()
            Lc = build_lindbladian_comp(J, params['gamma'], params['gamma_p'],
                                        args.lx * args.ly,
                                        bonds_2d(args.lx, args.ly),
                                        h_x=params['h_x'], h_z=params['h_z'])
            rl, el = quality_factor(Lc, tol_rel=args.tol_rel,
                                    dense_max_dim=args.dense_max_dim,
                                    return_eigenvalues=True)
            rec.update(_pack('lat', rl, el, keep_all=False), lat_done=True,
                       t_lat=time.perf_counter() - t1)
            for w in rl.warnings:
                print(f'  WARNING (lattice): {w}', flush=True)
        except Exception as e:
            print(f'  ERROR (lattice): {type(e).__name__}: {e}', flush=True)

    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_f, **rec)
    lat = (f"lat Q_max={rec['lat_Q_max']:.4g}" if rec['lat_done'] else 'lat -')
    print(f"  saved {out_f.name}  bond Q_max={rec['bond_Q_max']:.4g} "
          f"(undamped {rec['bond_n_undamped']})  {lat}  "
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, default='model3', choices=LATTICE_MODELS)
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks '
                        '(n_chunks<=1: task_id is a flat grid index)')
    p.add_argument('--out_dir', type=str, default='results_liouvillian_q')
    p.add_argument('--stride', type=int, default=1,
                   help='stride on the model grid (1 = full 51x51)')
    p.add_argument('--dim', type=int, default=DIM_DEFAULT,
                   help='bond Trotter convention (single-site share 1/(2 dim)); '
                        'must match the framability scans')
    p.add_argument('--lx', type=int, default=3, help='lattice columns')
    p.add_argument('--ly', type=int, default=2, help='lattice rows')
    p.add_argument('--no_lattice', action='store_true',
                   help='bond spectrum only (seconds for the whole grid)')
    p.add_argument('--tol_rel', type=float, default=TOL_REL_DEFAULT)
    p.add_argument('--dense_max_dim', type=int, default=DENSE_MAX_DIM_DEFAULT)
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
