"""
Randomised refine rounds for the seeded model10 Heisenberg rates (default
d_ext = 12; d_ext = 8 also works).  Fixes three weaknesses of the quick
refine (scripts/model10_d12_worker.py --stage refine):

  * it re-optimised only BOUNDARY points (a 4-neighbour at rate 0); here
    every positive point is a target (--targets all), or only the suspect
    ones (--targets nonmono: a 4-neighbour with LESS noise, i.e. one step
    lower in Delta1 or Delta2, has a lower rate, or a 4-neighbour is at 0);
  * its re-optimisation was deterministic (no differential evolution), so
    once the neighbours stopped changing every round repeated the last one;
    here each round is a basin-hopping search with a per-round random seed:
    Gaussian perturbations of the incumbent's free columns at several
    scales, each followed by a long polish;
  * its polish was 25 bundle steps over at most 8 near-binding columns; here
    framability_rate_margin.margin_polish (--polish_iters steps, up to 32
    columns, gradients of all columns from the duals of one LP).  At a
    positive point its objective F = max over the non-identity columns IS the
    rate; if it goes below 0 the point reaches rate 0 with margin.

Per target point and round r:
  1. start = best of {own best frame, 4-neighbours' best frames, margin
     frames within one step, at d_ext = 12 also the point's d_ext = 8 frame
     and nearby d_ext = 8 margin frames, padded}, scored on this generator;
  2. polish the start, then --hops perturbation + polish steps (scales
     --scales cycled; a hop is kept only if it lowers F), within
     --max_seconds;
  3. certify with the independent per-column LP; write
     pt_<ix>_<iy>_rrefine_r<NN>.npz if it beats the best known rate.
"Exhausted" is decided by scripts/submit_model10_rrefine.sh: a randomised
round that improves no point.

Usage:
    python scripts/model10_rrefine_worker.py --round 1 --task_id 0 --n_chunks 200
    python scripts/model10_rrefine_worker.py --d_ext 8 --targets nonmono --round 1 --task_id 5
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from optimize_framability import (_FIXED_COLS, N_FIXED_COLS,              # noqa: E402
                                  _project_columns_bloch)
from framability_rate import RATE_VERSION                                 # noqa: E402
from framability_rate_global import fit_columns                          # noqa: E402
from framability_rate_margin import (excess, excess_reference,           # noqa: E402
                                     margin_polish, RATE_MARGIN_VERSION)
from model10_seeded_worker import MODEL, grid_vals, generator             # noqa: E402
from model10_seeded_qrefine_worker import best_known, RREFINE_TAG         # noqa: E402
from model10_margin_worker import point_dir, margin_frame                 # noqa: E402

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LESS_NOISE = [(-1, 0), (0, -1)]          # one step lower in Delta1 / Delta2


def perturb(S, sigma: float, rng) -> np.ndarray:
    """Gaussian kick of the free columns, projected back onto the ball."""
    free = S[:, N_FIXED_COLS:] + sigma * rng.standard_normal(
        (S.shape[0], S.shape[1] - N_FIXED_COLS))
    return np.hstack([_FIXED_COLS, _project_columns_bloch(free)])


def is_target(pt, ix, iy, nx, ny, v_own, key, s_key, args) -> tuple:
    """(target?, reason) for a positive point."""
    if args.targets == 'all':
        return True, 'all'
    for dx, dy in NEIGHBORS4:
        jx, jy = ix + dx, iy + dy
        if not (0 <= jx < nx and 0 <= jy < ny):
            continue
        v = best_known(pt, jx, jy, key, s_key)[0]
        if v <= args.tol:
            return True, 'boundary'
        if (dx, dy) in LESS_NOISE and v < v_own - args.mono_tol:
            return True, 'non-monotone'
    return False, ''


def run_point(ix, iy, nx, ny, args) -> None:
    m = args.d_ext
    key, s_key = f'rate_{m}', f'S_{m}'
    pt = point_dir(args.out_dir, args.stride, m)
    stem = f'pt_{ix:03d}_{iy:03d}'
    out = pt / f'{stem}{RREFINE_TAG}{args.round:02d}.npz'
    if out.exists() or not (pt / f'{stem}.npz').exists():
        return
    v_own, S_own, lab_own = best_known(pt, ix, iy, key, s_key)
    if S_own is None or v_own <= args.tol:
        return                                          # rate 0 already
    target, why = is_target(pt, ix, iy, nx, ny, v_own, key, s_key, args)
    if not target:
        return

    p1, p2 = grid_vals(args.stride)
    d1, d2 = float(p1[ix]), float(p2[iy])
    A = generator(d1, d2, args.dim)
    t0 = time.perf_counter()
    rng = np.random.default_rng([args.seed, args.round, ix, iy, m])

    # ---- 1. start: best of own / neighbour / margin / padded d8 frames ----
    cands = [(S_own, lab_own)]
    for dx, dy in NEIGHBORS4:
        jx, jy = ix + dx, iy + dy
        if 0 <= jx < nx and 0 <= jy < ny:
            _, S, lab = best_known(pt, jx, jy, key, s_key)
            if S is not None:
                cands.append((S, lab))
    pt8 = point_dir(args.out_dir, args.stride, 8) if m == 12 else None
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < nx and 0 <= jy < ny):
                continue
            mf = margin_frame(pt, jx, jy, m, None)
            if mf is not None and (dx, dy) != (0, 0):
                cands.append((mf[1], mf[2]))
            if pt8 is not None:
                mf8 = margin_frame(pt8, jx, jy, 8, None)
                if mf8 is not None:
                    cands.append((fit_columns(mf8[1], m), f'd8 {mf8[2]}'))
    if pt8 is not None:
        _, S8, lab8 = best_known(pt8, ix, iy, 'rate_8', 'S_8')
        if S8 is not None:
            cands.append((fit_columns(S8, m), f'd8 {lab8}'))
    scored = sorted(((excess(S, A), i) for i, (S, _) in enumerate(cands)),
                    key=lambda t: t[0])
    F_best, i0 = scored[0]
    S_best, label = cands[i0]
    start_F = F_best

    # ---- 2. polish + basin hopping ----------------------------------------
    F, S, _ = margin_polish(S_best, A, n_iter=args.polish_iters)
    if F < F_best:
        F_best, S_best = F, S
    n_hops = n_kept = 0
    for k in range(args.hops):
        if F_best <= -args.margin_goal or \
                time.perf_counter() - t0 > args.max_seconds:
            break
        sigma = args.scales[k % len(args.scales)]
        F, S, _ = margin_polish(perturb(S_best, sigma, rng), A,
                                n_iter=args.polish_iters)
        n_hops += 1
        if F < F_best - args.gain_tol:
            F_best, S_best = F, S
            n_kept += 1

    # ---- 3. certify ------------------------------------------------------------
    F_ref = excess_reference(S_best, A)
    if not np.isfinite(F_ref):
        print(f'  ({ix},{iy}) reference LP failed; nothing written', flush=True)
        return
    rate = max(F_ref, 0.0)
    dt = time.perf_counter() - t0
    if rate >= v_own - args.gain_tol:
        print(f'  ({ix},{iy}) [{why}] {v_own:.6e}: no gain (start {start_F:.6e}, '
              f'{n_hops} hops, {dt:.0f}s)', flush=True)
        return
    lab = label if label.endswith('+rref') else f'{label} +rref'
    pt.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.npz')
    np.savez(tmp, model=MODEL, ix=ix, iy=iy, delta1=d1, delta2=d2, dim=args.dim,
             d_ext=m, round=args.round, target=why, hops=n_hops, hops_kept=n_kept,
             rate_version=RATE_VERSION, rate_margin_version=RATE_MARGIN_VERSION,
             **{key: rate, s_key: S_best, f'label_{m}': lab,
                f'excess_{m}': F_ref, f'{key}_prev': v_own})
    os.replace(tmp, out)
    print(f'  saved {out.name} [{why}] {v_own:.6e} -> {rate:.6e}'
          f'{"  (rate 0, margin %+.3e)" % -F_ref if F_ref <= 0 else ""}  '
          f'start {start_F:.6e}, {n_kept}/{n_hops} hops kept, {dt:.0f}s',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--d_ext', type=int, choices=(8, 12), default=12)
    p.add_argument('--round', type=int, required=True,
                   help='randomised refine round (1..99); seeds are whatever is on '
                        'disk (all pipelines), the RNG is seeded per round')
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1)
    p.add_argument('--out_dir', type=str, default='results_model10_rate')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--targets', choices=('all', 'nonmono'), default='all')
    p.add_argument('--tol', type=float, default=1e-6,
                   help='rate counted as 0 (the figure contour tolerance)')
    p.add_argument('--mono_tol', type=float, default=1e-4,
                   help='--targets nonmono: a less-noisy neighbour must be lower '
                        'by more than this')
    p.add_argument('--gain_tol', type=float, default=1e-9)
    p.add_argument('--polish_iters', type=int, default=100)
    p.add_argument('--hops', type=int, default=6)
    p.add_argument('--scales', type=float, nargs='+', default=[0.02, 0.06, 0.15])
    p.add_argument('--margin_goal', type=float, default=1e-3,
                   help='stop hopping once the frame is at rate 0 with this margin')
    p.add_argument('--max_seconds', type=float, default=1800.0,
                   help='per-point wall-clock budget for the hops')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.dim is None:
        args.dim = MODELS[MODEL].dim

    p1, p2 = grid_vals(args.stride)
    nx, ny = len(p1), len(p2)
    n_total = nx * ny
    if args.n_chunks <= 1:
        ids = [args.task_id]
    else:
        ids = list(range(args.task_id, n_total, args.n_chunks))
        print(f'[chunk {args.task_id}/{args.n_chunks}] rrefine d_ext={args.d_ext} '
              f'round {args.round} targets={args.targets}: {len(ids)} of '
              f'{n_total} points', flush=True)
    for pid in ids:
        ix, iy = pid // ny, pid % ny
        try:
            run_point(ix, iy, nx, ny, args)
        except Exception as e:                              # noqa: BLE001
            print(f'  ERROR at ({ix},{iy}): {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
