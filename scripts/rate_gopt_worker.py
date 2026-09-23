"""
Global re-optimisation of the optimised Heisenberg framability RATES
(rate_heis_4 / rate_heis_6 / rate_heis_8) of a model's full (gamma, gamma')
grid with framability_rate_global.minimize_rate_global -- one point per call.

Per grid point and per d_ext the optimiser is seeded with
  * every frame already stored for the point (base scan, quick / full
    neighbour-refine rounds, earlier gopt runs), evaluated as fixed frames,
  * the deterministic seed library (exact rescaled-Pauli optimum, pump
    projectors, Z-projector + XY polygon, Pauli),
  * the optimum of the next-smaller d_ext padded (nesting),
so every stored value can only improve, and rates are monotone in d_ext.

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>_gopt.npz with keys
    rate_heis_<m>, S_heis_<m>, mu_seed_<m>, mu_de_<m>, mu_bundle_<m>,
    prev_best_<m> (best stored value before this run, nan if none), t_<m>
The model4 collect (scripts/model4_rate_panels_collect.py) and the quick
refine worker read these files alongside the base scan and refine rounds.

Usage:
    python scripts/rate_gopt_worker.py --model model4 --task_id 0 --n_chunks 200
    python scripts/rate_gopt_worker.py --model model3 --task_id 1300     # one point
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
from trotter_lindbladian_scan import MODELS, build_bond_lindbladian     # noqa: E402
from framability_rate import RATE_VERSION, spectral_abscissa           # noqa: E402
from framability_rate_global import (minimize_rate_global,             # noqa: E402
                                     RATE_GLOBAL_VERSION)

D_EXTS_DEFAULT = (4, 6, 8)
SUFFIX = '_gopt'


def grid_vals(model: str, stride: int):
    m = MODELS[model]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def stored_frames(pt_dir: Path, ix: int, iy: int, d_exts):
    """{m: [(value, frame), ...]} over every file stored for the point."""
    files = [pt_dir / f'pt_{ix:03d}_{iy:03d}.npz']
    files += sorted(pt_dir.glob(f'pt_{ix:03d}_{iy:03d}_*refine_r*.npz'))
    files += sorted(pt_dir.glob(f'pt_{ix:03d}_{iy:03d}{SUFFIX}*.npz'))
    out = {m: [] for m in d_exts}
    for f in files:
        if not f.exists():
            continue
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:                                   # noqa: BLE001
            continue
        for m in d_exts:
            if f'S_heis_{m}' in d.files:
                v = float(d[f'rate_heis_{m}']) if f'rate_heis_{m}' in d.files else np.nan
                out[m].append((v, np.asarray(d[f'S_heis_{m}'], float)))
    return out


def compute_point(model: str, p1: float, p2: float, pt_dir: Path, ix: int,
                  iy: int, args) -> dict:
    spec = MODELS[model]
    H1, H2, j1, j2 = spec.build(p1, p2)
    L = build_bond_lindbladian(H1, H2, j1, j2, args.dim).real
    A = L.T
    out: dict = {spec.p1_name: p1, spec.p2_name: p2, 'dim': args.dim,
                 'floor': spectral_abscissa(A)}
    stored = stored_frames(pt_dir, ix, iy, args.d_exts)
    prev_S = None
    for m in sorted(args.d_exts):
        t0 = time.perf_counter()
        seeds = [S for _, S in stored[m]]
        seeds += [S for k in args.d_exts if k < m for _, S in stored[k]]
        if prev_S is not None:
            seeds.append(prev_S)
        basis = (m == 4)
        S, mu, info = minimize_rate_global(
            A, m, seeds=seeds, seed=args.seed + m,
            de_popsize=args.de_popsize4 if basis else args.de_popsize,
            de_maxiter=args.de_maxiter4 if basis else args.de_maxiter,
            bundle_iters=args.bundle_iters,
            nm_maxfev=args.nm_maxfev4 if basis else args.nm_maxfev)
        prev_vals = [v for v, _ in stored[m] if np.isfinite(v)]
        out[f'rate_heis_{m}'] = mu
        out[f'S_heis_{m}'] = S
        out[f'mu_seed_{m}'] = info['mu_seed']
        out[f'mu_de_{m}'] = info['mu_de']
        out[f'mu_bundle_{m}'] = info['mu_bundle']
        out[f'seed_name_{m}'] = str(info['seed_name'])
        out[f'prev_best_{m}'] = min(prev_vals) if prev_vals else np.nan
        out[f't_{m}'] = time.perf_counter() - t0
        prev_S = S
    return out


def run_point(model: str, ix: int, iy: int, args) -> None:
    p1, p2 = grid_vals(model, args.stride)
    pt_dir = Path(args.out_dir) / model
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}{SUFFIX}.npz'
    if out_f.exists() and not args.force:
        print(f'[skip] {model}/{out_f.name}', flush=True)
        return
    t0 = time.perf_counter()
    print(f'[{model}] ({ix},{iy}) {MODELS[model].p1_name}={p1[ix]:.3f} '
          f'{MODELS[model].p2_name}={p2[iy]:.3f}', flush=True)
    try:
        res = compute_point(model, float(p1[ix]), float(p2[iy]), pt_dir, ix, iy,
                            args)
    except Exception as e:                                  # noqa: BLE001
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        return
    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_f, model=model, ix=ix, iy=iy, stride=args.stride,
             d_exts=np.array(sorted(args.d_exts)), rate_version=RATE_VERSION,
             rate_global_version=RATE_GLOBAL_VERSION, **res)
    vals = '  '.join(f'd{m}={res[f"rate_heis_{m}"]:.5f}'
                     f'(prev {res[f"prev_best_{m}"]:.5f})' for m in sorted(args.d_exts))
    print(f'  saved {out_f.name}  {vals}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='<=1: task_id is a flat grid index; otherwise the grid '
                        'is strided over n_chunks array tasks')
    p.add_argument('--out_dir', type=str, default=None,
                   help='default results_<model>_rate')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--d_exts', type=int, nargs='+', default=list(D_EXTS_DEFAULT))
    p.add_argument('--dim', type=int, default=None,
                   help="bond convention dimension (default: the model's)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--de_popsize4', type=int, default=24)
    p.add_argument('--de_maxiter4', type=int, default=300)
    p.add_argument('--nm_maxfev4', type=int, default=20000)
    p.add_argument('--de_popsize', type=int, default=6,
                   help='DE population multiplier for LP-evaluated frames')
    p.add_argument('--de_maxiter', type=int, default=25,
                   help='DE generations for LP-evaluated frames (0 disables)')
    p.add_argument('--bundle_iters', type=int, default=60)
    p.add_argument('--nm_maxfev', type=int, default=400)
    p.add_argument('--force', action='store_true',
                   help='recompute points that already have a gopt file '
                        '(their stored frames still seed the run)')
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = f'results_{args.model}_rate'
    if args.dim is None:
        args.dim = MODELS[args.model].dim

    p1, p2 = grid_vals(args.model, args.stride)
    nx, ny = len(p1), len(p2)
    n_total = nx * ny
    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            sys.exit(f'task_id must be in [0, {n_total})')
        run_point(args.model, args.task_id // ny, args.task_id % ny, args)
        return
    if not (0 <= args.task_id < args.n_chunks):
        sys.exit(f'chunk id must be in [0, {args.n_chunks})')
    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {args.model}: {len(ids)} of '
          f'{n_total} points ({nx}x{ny}), d_ext={sorted(args.d_exts)}', flush=True)
    for pid in ids:
        run_point(args.model, pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
