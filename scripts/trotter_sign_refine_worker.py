"""
Neighbour-seeded refinement of the optimised sign problem (sign_opt) for the
generic Trotter Lindbladian scan.

Why: sign_opt from the base scan is the max of s over translation-invariant
local rotations R = exp(i pi n.sigma), found by a handful of BFGS runs from
random starts with a *different* seed at every grid point.  The objective
s(n) = |sum U| / sum|U| of the rotated gate is multimodal and non-smooth
(kinks where matrix elements cross zero), so neighbouring points routinely
converge to different local maxima -- producing the jagged, non-monotone
sign_opt maps.  The base scan also never stored the winning rotation, so
nothing propagates between points.

This worker mirrors the framability neighbour refinement:
  1. Gather the best rotation vector n found so far for this point and its
     4-connected neighbours (from every earlier sign-refine round).
  2. Re-maximise s seeded from those vectors plus fresh random restarts,
     using Powell (robust to the kinks) with a BFGS polish.
  3. Never regress: the reported sign_opt is the max over the base value and
     everything found here.

Run a few sequential rounds (--round 1..3); round 1 has no stored vectors yet
and relies on the enlarged restart budget.

Reads:  <out_dir>/<model>/pt_<ix>_<iy>.npz                (base scan)
        <out_dir>/<model>/pt_sign_r*_<ix>_<iy>.npz        (earlier rounds)
Writes: <out_dir>/<model>/pt_sign_r<NN>_<ix>_<iy>.npz
        keys: sign_opt (merged max), sign_n (best rotation vector),
              sign_n_val (s achieved by sign_n), round, ix, iy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, bond_trotter_gate
from dissipative_PT import _sq_superop, _apply_local_rot, _SX, _SY, _SZ
from sign_problem import sign_problem as _sp

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _s_of(gate: np.ndarray, n_vec: np.ndarray) -> float:
    """s of the gate conjugated by R^tensor with R = exp(i pi n.sigma)."""
    R = expm(1j * np.pi * (n_vec[0] * _SX + n_vec[1] * _SY + n_vec[2] * _SZ))
    return float(_sp(_apply_local_rot(gate, np.real(_sq_superop(R)))))


def _optimise_sign(gate: np.ndarray, seed_ns: list, n_restarts: int,
                   seed: int) -> tuple[float, np.ndarray]:
    """Max s over n, from the given seed vectors + random restarts.

    Each start runs Powell (derivative-free, handles the non-smooth kinks)
    followed by a BFGS polish from the Powell optimum.
    """
    rng = np.random.default_rng(seed)

    def neg_s(p):
        return -_s_of(gate, p)

    starts = [np.asarray(n, dtype=float).ravel() for n in seed_ns]
    for _ in range(n_restarts):
        x0 = rng.standard_normal(3)
        x0 /= max(np.linalg.norm(x0), 1e-14)
        starts.append(x0)

    best_val, best_n = -np.inf, np.zeros(3)
    for x0 in starts:
        res = minimize(neg_s, x0, method='Powell',
                       options={'maxiter': 300, 'maxfev': 600,
                                'ftol': 1e-10, 'xtol': 1e-8})
        x1, f1 = np.asarray(res.x, dtype=float), float(-res.fun)
        res2 = minimize(neg_s, x1, method='BFGS',
                        options={'maxiter': 200, 'gtol': 1e-7})
        if float(-res2.fun) > f1:
            x1, f1 = np.asarray(res2.x, dtype=float), float(-res2.fun)
        if f1 > best_val:
            best_val, best_n = f1, x1.copy()
    return best_val, best_n


def _stored_sign(out_dir: Path, model, ix: int, iy: int):
    """(best sign_opt value, best rotation vector or None) over the base file
    and every sign-refine round for this point."""
    d = out_dir / model.name
    best_val, best_n = -np.inf, None
    base = d / f'pt_{ix:03d}_{iy:03d}.npz'
    if base.exists():
        try:
            b = np.load(base)
            if 'sign_opt' in b and np.isfinite(float(b['sign_opt'])):
                best_val = float(b['sign_opt'])
        except Exception:
            pass
    best_n_val = -np.inf
    for f in sorted(d.glob(f'pt_sign_r*_{ix:03d}_{iy:03d}.npz')):
        try:
            r = np.load(f)
        except Exception:
            continue
        v = float(r['sign_n_val']) if 'sign_n_val' in r else -np.inf
        if np.isfinite(v) and v > best_n_val and 'sign_n' in r:
            n = np.asarray(r['sign_n'], dtype=float)
            if np.all(np.isfinite(n)):
                best_n_val, best_n = v, n
        if 'sign_opt' in r:
            best_val = max(best_val, float(r['sign_opt']))
    return best_val, best_n


def run_point(model, point_id: int, args) -> None:
    out_dir = Path(args.out_dir)
    ix = point_id // model.N_Y
    iy = point_id %  model.N_Y
    out = out_dir / model.name / f'pt_sign_r{args.round:02d}_{ix:03d}_{iy:03d}.npz'

    if out.exists():
        print(f'[skip] {model.name}/{out.name} already exists', flush=True)
        return

    base = out_dir / model.name / f'pt_{ix:03d}_{iy:03d}.npz'
    if not base.exists():
        print(f'ERROR: base scan file {base} not found -- run the scan first',
              file=sys.stderr)
        sys.exit(1)

    d_base = np.load(base)
    dim = int(d_base['dim']) if 'dim' in d_base else model.dim
    dt  = float(d_base['dt']) if 'dt' in d_base else model.dt
    p1 = float(model.p1_vals[ix])
    p2 = float(model.p2_vals[iy])

    t0 = time.perf_counter()
    self_val, self_n = _stored_sign(out_dir, model, ix, iy)
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} sign round {args.round} '
          f'{model.p1_name}={p1:.3f} {model.p2_name}={p2:.3f}  '
          f'startup best={self_val:.6f}', flush=True)

    seed_ns = []
    if self_n is not None:
        seed_ns.append(self_n)
    for dx, dy in NEIGHBORS:
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < model.N_X and 0 <= ny < model.N_Y:
            _, nb_n = _stored_sign(out_dir, model, nx, ny)
            if nb_n is not None:
                seed_ns.append(nb_n)

    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    gate = bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)

    val, n_best = _optimise_sign(gate, seed_ns, args.n_restarts,
                                 args.seed + point_id + 1000 * args.round)
    merged = max(val, self_val)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             round=np.array(args.round),
             sign_opt=np.array(merged),
             sign_n=np.asarray(n_best),
             sign_n_val=np.array(val),
             ix=np.array(ix), iy=np.array(iy))
    print(f'  saved {out.name}: sign_opt {self_val:.6f} -> {merged:.6f}  '
          f'({len(seed_ns)} seed(s), {time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',      type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id',    type=int, required=True)
    p.add_argument('--n_chunks',   type=int, default=1)
    p.add_argument('--round',      type=int, required=True, help='refine round (1..)')
    p.add_argument('--out_dir',    type=str, default='results_trotter')
    p.add_argument('--n_restarts', type=int, default=20)
    p.add_argument('--seed',       type=int, default=0)
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
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name} sign round '
          f'{args.round}: {len(point_ids)} points', flush=True)
    for pid in point_ids:
        run_point(model, pid, args)


if __name__ == '__main__':
    main()
