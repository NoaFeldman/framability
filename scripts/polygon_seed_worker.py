"""
Polygon-seeded Heisenberg rate scan for model3 / model4 (half resolution).

For every (gamma, gamma') point of MODELS[model] at stride 2 (0.4 steps,
26 x 26 = 676 points) and every d_ext in --d_exts this stores three rates of
the two-qubit bond generator A = L^T:

  rate_opt_<m>    minimize_rate(A, m) seeded with the projector-polygon frame
                  (repo convention: identity column pinned, m columns total)
  rate_seed_<m>   the seed frame itself, evaluated as a fixed frame
  rate_free_<m>   the identity-FREE projector-polygon frame with m columns
                  { |0><0|, |1><1|, cos(pi k/(m-2)) X + sin(pi k/(m-2)) Y }
                  evaluated as a fixed frame (framability_rate_frames.frame_rate)

Why both: with the identity pinned, every column d_k (x) 1 exists in D = S (x) S
and costs the full conditional-rotation amplitude 2J, so the optimiser cannot
exploit the polygon (checked: n = 3 hexagon gives 0.55 without the identity
column and 2.86 with it, at J = 1, gamma = 0.3).  rate_free_<m> is the
construction of polygon_frame_gate_note.tex; rate_opt_<m> is the best the
current optimiser reaches from that seed.

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz  (existing files skipped).

Usage:
    python scripts/polygon_seed_worker.py --model model4 --task_id 0 --n_chunks 200
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
from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,   # noqa: E402
                                      DIM_DEFAULT)
from framability_rate import minimize_rate, RATE_VERSION                 # noqa: E402
from framability_rate_frames import frame_rate                           # noqa: E402
from dissipative_PT import _kron_power                                   # noqa: E402

STRIDE_DEFAULT = 2
D_EXTS_DEFAULT = (4, 6, 8)
POLY_SEED_VERSION = '1.0-polygon-seed'

_ID = np.array([1.0, 0.0, 0.0, 0.0])
_P0 = np.array([0.5, 0.0, 0.0, 0.5])
_P1 = np.array([0.5, 0.0, 0.0, -0.5])


def _polygon(n: int) -> list:
    """n transverse elements at angles pi k / n (body = regular 2n-gon)."""
    return [np.array([0.0, np.cos(np.pi * k / n), np.sin(np.pi * k / n), 0.0])
            for k in range(n)]


def seed_frame(m: int) -> np.ndarray:
    """Identity-pinned polygon seed with m columns (repo convention)."""
    if m == 4:
        cols = [_ID, _P0] + _polygon(2)                 # 1, P0, X, Y
    else:
        cols = [_ID, _P0, _P1] + _polygon(m - 3)
    return np.array(cols).T


def free_frame(m: int) -> np.ndarray:
    """Identity-free projector-polygon frame with m columns."""
    return np.array([_P0, _P1] + _polygon(m - 2)).T


def grid_vals(model: str, stride: int):
    mod = MODELS[model]
    return (np.asarray(mod.p1_vals[::stride], float),
            np.asarray(mod.p2_vals[::stride], float))


def compute_point(model: str, gamma: float, gamma_p: float, args) -> dict:
    mod = MODELS[model]
    H1, H2, j1, j2 = mod.build(gamma, gamma_p)
    L = build_bond_lindbladian(H1, H2, j1, j2, args.dim).real
    A = L.T
    out: dict = dict(gamma=gamma, gamma_p=gamma_p, dim=args.dim)
    for m in args.d_exts:
        t0 = time.perf_counter()
        S_seed, S_free = seed_frame(m), free_frame(m)
        out[f'rate_seed_{m}'] = frame_rate(_kron_power(S_seed, 2), L,
                                           picture='heisenberg')
        out[f'rate_free_{m}'] = frame_rate(_kron_power(S_free, 2), L,
                                           picture='heisenberg')
        S_opt, mu, info = minimize_rate(
            A, m, n_restarts=args.restarts, maxfev=args.maxfev,
            seed=args.seed + m, verbose=False, extra_init_S=[S_seed],
            polish_iters=args.polish, check_swap=not args.no_swap_check)
        out[f'rate_opt_{m}'] = mu
        out[f'S_opt_{m}'] = S_opt
        out[f'mu_search_{m}'] = info['mu_search']
        out[f't_{m}'] = time.perf_counter() - t0
    return out


def run_point(model: str, ix: int, iy: int, args) -> None:
    p1, p2 = grid_vals(model, args.stride)
    gamma, gamma_p = float(p1[ix]), float(p2[iy])
    pt_dir = Path(args.out_dir) / model
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists():
        print(f'[skip] {model}/{out_f.name}', flush=True)
        return
    t0 = time.perf_counter()
    print(f"[{model}] ({ix},{iy}) gamma={gamma:.2f} gamma'={gamma_p:.2f}",
          flush=True)
    try:
        res = compute_point(model, gamma, gamma_p, args)
    except Exception as e:                       # noqa: BLE001
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        return
    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_f, model=model, ix=ix, iy=iy, stride=args.stride,
             d_exts=np.array(args.d_exts), rate_version=RATE_VERSION,
             poly_seed_version=POLY_SEED_VERSION, **res)
    vals = '  '.join(f'd{m}: opt={res[f"rate_opt_{m}"]:.4f} '
                     f'free={res[f"rate_free_{m}"]:.4f}' for m in args.d_exts)
    print(f'  saved {out_f.name}  {vals}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['model3', 'model4'], required=True)
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=200)
    p.add_argument('--out_dir', type=str, default='results_polygon_seed')
    p.add_argument('--stride', type=int, default=STRIDE_DEFAULT)
    p.add_argument('--d_exts', type=int, nargs='+', default=list(D_EXTS_DEFAULT))
    p.add_argument('--dim', type=int, default=DIM_DEFAULT)
    p.add_argument('--restarts', type=int, default=4)
    p.add_argument('--maxfev', type=int, default=2000)
    p.add_argument('--polish', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_swap_check', action='store_true')
    args = p.parse_args()

    p1, p2 = grid_vals(args.model, args.stride)
    nx, ny = len(p1), len(p2)
    n_total = nx * ny
    if args.n_chunks <= 1:
        run_point(args.model, args.task_id // ny, args.task_id % ny, args)
        return
    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {args.model}: {len(ids)} of '
          f'{n_total} points ({nx}x{ny}), d_ext={args.d_exts}', flush=True)
    for pid in ids:
        run_point(args.model, pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
