"""
Quick neighbour refining of the SEEDED model10 Heisenberg rates (rate_4 /
rate_8 of scripts/model10_seeded_worker.py) -- one round, one point.

A mirror of scripts/model4_rate_quick_refine_worker.py (same boundary rule,
same two stages, same accept rule, same per-round seeds), pointed at the
seeded pipeline's files and at d_ext = 4 / 8 instead of 4 / 6:

  * a point is touched for key k only if its best-known rate sits ABOVE the
    floor mu* = 0 while at least one 4-connected neighbour sits AT it (both
    to within --rate_tol);
  * cross-d_ext step: if its best d=4 rate is below its best d=8 rate (an
    optimiser artifact -- eight columns contain four), the d=4 frame is
    padded to 8 columns and re-optimised;
  * stage 1, propagation: every neighbour frame is scored on this point's
    generator with the batched LP; an improvement is confirmed with the
    independent per-column LP before it is accepted;
  * stage 2, quick re-optimisation seeded with the point's own frame and all
    neighbour frames (--optimizer rate: framability_rate.minimize_rate at
    reduced restarts, as the existing quick refine; --optimizer global:
    framability_rate_global.minimize_rate_global at a small budget).  The
    incumbent is kept unless strictly beaten.

Best-known value of a point = min over
    pt_<ix>_<iy>.npz              (scripts/model10_seeded_worker.py)
    pt_<ix>_<iy>_xeval.npz        (scripts/model10_seeded_collect.py)
    pt_<ix>_<iy>_qrefine_rNN.npz  (earlier rounds of this worker)
in <out_dir>/model10_seeded[_s<stride>]/.  Rounds are sequential: round r
reads every earlier round, so the floor propagates one ring per round.

Writes pt_<ix>_<iy>_qrefine_r<NN>.npz for improved points only (keys rate_4,
S_4, label_4, rate_8, S_8, label_8, rate_<m>_prev, rate_<m>_improved).

Usage:
    python scripts/model10_seeded_qrefine_worker.py --round 1 --task_id 0 --n_chunks 200
    python scripts/model10_seeded_qrefine_worker.py --round 1 --task_id 1300   # one point
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
from optimize_framability import _kron_power                              # noqa: E402
from framability_rate import (minimize_rate, generator_log_norm,          # noqa: E402
                              generator_log_norm_reference, RATE_VERSION)
from framability_rate_global import minimize_rate_global, fit_columns     # noqa: E402
from model10_seeded_worker import (MODEL, grid_vals, pt_dir_name,         # noqa: E402
                                   generator)
from trotter_lindbladian_scan import MODELS                               # noqa: E402

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TOL = 1e-9
RATE_FLOOR = 0.0
KEYS = {'rate_4': ('S_4', 4), 'rate_8': ('S_8', 8)}   # rate key -> (frame key, d_ext)
ROUND_TAG = '_qrefine_r'
MARGIN_TAG = '_margin_r'      # scripts/model10_margin_worker.py
RREFINE_TAG = '_rrefine_r'    # scripts/model10_rrefine_worker.py


def pt_paths(pt: Path, ix: int, iy: int):
    """Every file of the point: margin rounds (newest first), the seeded
    worker file, the collect cross-evaluation file, every quick-refine and
    randomised-refine round.  best_known keeps the FIRST file attaining the
    minimum, so on the rate-0 plateau (all values 0) a margin frame -- the one
    with slack -- wins."""
    stem = f'pt_{ix:03d}_{iy:03d}'
    files = sorted(pt.glob(f'{stem}{MARGIN_TAG}[0-9][0-9].npz'), reverse=True)
    files += [pt / f'{stem}.npz', pt / f'{stem}_xeval.npz']
    files += sorted(pt.glob(f'{stem}{ROUND_TAG}[0-9][0-9].npz'))
    files += sorted(pt.glob(f'{stem}{RREFINE_TAG}[0-9][0-9].npz'))
    return [f for f in files if f.exists()]


def best_known(pt: Path, ix: int, iy: int, key: str, s_key: str):
    """Lowest (rate, frame, label) over every file of the point."""
    best = (np.inf, None, '')
    lab_key = 'label_' + key.split('_')[1]
    for f in pt_paths(pt, ix, iy):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:                                   # noqa: BLE001
            continue
        if key not in d.files or s_key not in d.files:
            continue
        v = float(d[key])
        if np.isfinite(v) and v < best[0]:
            lab = str(d[lab_key]) if lab_key in d.files else ''
            best = (v, np.asarray(d[s_key], float), lab)
    return best


def neighbor_frames(pt: Path, nx: int, ny: int, ix: int, iy: int, key, s_key):
    """(value, frame, label) of every 4-connected neighbour, ascending."""
    out = []
    for dx, dy in NEIGHBORS:
        jx, jy = ix + dx, iy + dy
        if 0 <= jx < nx and 0 <= jy < ny:
            v, S, lab = best_known(pt, jx, jy, key, s_key)
            if S is not None:
                out.append((v, S, lab))
    out.sort(key=lambda t: t[0])
    return out


def propagate(A, own_val, own_S, own_lab, nb_list):
    """Stage 1: best neighbour frame on this point's generator (two-LP rule)."""
    best_v, best_S, best_lab = own_val, own_S, own_lab
    for _, S_nb, lab in nb_list:
        v = generator_log_norm(_kron_power(S_nb, 2), A)
        if np.isfinite(v) and v < best_v - TOL:
            v_ref = generator_log_norm_reference(_kron_power(S_nb, 2), A)
            if np.isfinite(v_ref) and v_ref < best_v - TOL:
                best_v, best_S, best_lab = v_ref, S_nb.copy(), lab
    return best_v, best_S, best_lab


def reoptimize(A, d_ext, seeds, seed, args):
    """Stage 2: quick seeded re-optimisation; returns (S, mu)."""
    if args.optimizer == 'global':
        S, mu, _ = minimize_rate_global(
            A, d_ext, seeds=seeds, seed=seed,
            de_popsize=args.g_popsize4 if d_ext == 4 else args.g_popsize,
            de_maxiter=args.g_maxiter4 if d_ext == 4 else args.g_maxiter,
            bundle_iters=args.g_bundle, nm_maxfev=args.g_nm)
        return S, mu
    S, mu, _ = minimize_rate(
        A, d_ext, n_restarts=args.n_restarts,
        maxfev=args.maxfev_4 if d_ext == 4 else args.maxfev_8,
        seed=seed, verbose=False, polish_iters=args.polish,
        extra_init_S=seeds or None)
    return S, mu


def _qref(label: str) -> str:
    return label if label.endswith('+qref') else f'{label} +qref'


def run_point(ix: int, iy: int, nx: int, ny: int, args) -> None:
    pt = Path(args.out_dir) / pt_dir_name(args.stride)
    out = pt / f'pt_{ix:03d}_{iy:03d}{ROUND_TAG}{args.round:02d}.npz'
    if out.exists():
        print(f'[skip] {out.name} already exists', flush=True)
        return
    if not (pt / f'pt_{ix:03d}_{iy:03d}.npz').exists():
        return                      # point not computed yet -- nothing to refine

    info, todo = {}, []
    for key, (s_key, _) in KEYS.items():
        sv, sS, slab = best_known(pt, ix, iy, key, s_key)
        nb = neighbor_frames(pt, nx, ny, ix, iy, key, s_key)
        nb_val = nb[0][0] if nb else np.inf
        info[key] = (sv, sS, slab, nb_val, nb)
        if sv > RATE_FLOOR + args.rate_tol and nb_val <= RATE_FLOOR + args.rate_tol:
            todo.append(key)

    v4, S4 = info['rate_4'][0], info['rate_4'][1]
    v8 = info['rate_8'][0]
    cross = v4 < v8 - TOL and S4 is not None and v8 > RATE_FLOOR + args.rate_tol
    if not np.isfinite(v4) and not np.isfinite(v8):
        return
    if not todo and not cross:
        return                      # interior point -- no file written

    p1, p2 = grid_vals(args.stride)
    d1, d2 = float(p1[ix]), float(p2[iy])
    t0 = time.perf_counter()
    point_id = ix * ny + iy
    print(f'[point {point_id}] round {args.round} delta1={d1:.3f} delta2={d2:.3f} '
          f'keys={todo or "none"} cross={cross}  best d4={v4:.6e} d8={v8:.6e}',
          flush=True)
    A = generator(d1, d2, args.dim)
    seed = args.seed + point_id + 100000 * args.round
    results = {k: (info[k][0], info[k][1], info[k][2]) for k in KEYS}

    for off, key in enumerate(KEYS):
        if key not in todo:
            continue
        sv, sS, slab, nb_val, nb = info[key]
        d_ext = KEYS[key][1]
        val, S, lab = propagate(A, sv, sS, slab, nb)
        if val < sv - TOL:
            print(f'  d{d_ext}: propagated {sv:.6e} -> {val:.6e}', flush=True)
        seeds = ([S] if S is not None else []) + [f for _, f, _ in nb]
        S_new, mu_new = reoptimize(A, d_ext, seeds, seed + off, args)
        if np.isfinite(mu_new) and mu_new < val - TOL:
            val, S, lab = float(mu_new), S_new, _qref(lab)
        results[key] = (val, S, lab)
        print(f'  d{d_ext}: {sv:.6e} -> {val:.6e}  (floor neighbour {nb_val:.6e})'
              f'  [{time.perf_counter() - t0:.0f}s]', flush=True)

    if cross:
        v8_now, S8_now, lab8 = results['rate_8']
        v4_now, S4_now, lab4 = results['rate_4']
        if S4_now is not None and v4_now < v8_now - TOL:
            seed8 = fit_columns(S4_now, 8)
            v_emb = generator_log_norm_reference(_kron_power(seed8, 2), A)
            if np.isfinite(v_emb) and v_emb < v8_now - TOL:
                v8_now, S8_now = float(v_emb), seed8
                lab8 = lab4 if lab4.startswith('d4 ') else f'd4 {lab4}'
            S_new, mu_new = reoptimize(
                A, 8, [seed8] + ([S8_now] if S8_now is not None else []),
                seed + 7, args)
            if np.isfinite(mu_new) and mu_new < v8_now - TOL:
                v8_now, S8_now, lab8 = float(mu_new), S_new, _qref(lab8)
            print(f'  cross d4->d8: {results["rate_8"][0]:.6e} -> {v8_now:.6e}',
                  flush=True)
            results['rate_8'] = (v8_now, S8_now, lab8)

    improved = {k: results[k][0] < info[k][0] - TOL for k in KEYS}
    if not any(improved.values()):
        print(f'  no improvement; nothing written '
              f'({time.perf_counter() - t0:.0f}s)', flush=True)
        return
    payload = dict(model=MODEL, ix=ix, iy=iy, delta1=d1, delta2=d2, dim=args.dim,
                   round=args.round, optimizer=args.optimizer,
                   rate_version=RATE_VERSION, rate_tol=args.rate_tol)
    for key, (s_key, m) in KEYS.items():
        payload[key] = results[key][0]
        payload[f'{key}_prev'] = info[key][0]
        payload[f'{key}_improved'] = improved[key]
        if results[key][1] is not None:
            payload[s_key] = results[key][1]
            payload[f'label_{m}'] = results[key][2]
    tmp = out.with_suffix('.tmp.npz')
    np.savez(tmp, **payload)
    os.replace(tmp, out)
    print(f'  saved {out.name}  d4={results["rate_4"][0]:.6e} '
          f'd8={results["rate_8"][0]:.6e}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--round', type=int, required=True,
                   help='refine round (1..99); round r reads rounds < r')
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1)
    p.add_argument('--out_dir', type=str, default='results_model10_rate')
    p.add_argument('--stride', type=int, default=1,
                   help='stride the seeded worker ran with')
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--rate_tol', type=float, default=1e-6,
                   help='a rate within this of 0 counts as on the floor '
                        '(the collect contour tolerance)')
    p.add_argument('--optimizer', choices=('rate', 'global'), default='rate',
                   help="stage-2 optimiser: 'rate' = minimize_rate as in the "
                        "existing quick refine, 'global' = minimize_rate_global")
    p.add_argument('--n_restarts', type=int, default=3)
    p.add_argument('--maxfev_4', type=int, default=1000)
    p.add_argument('--maxfev_8', type=int, default=500)
    p.add_argument('--polish', type=int, default=150)
    p.add_argument('--g_popsize4', type=int, default=12)
    p.add_argument('--g_maxiter4', type=int, default=100)
    p.add_argument('--g_popsize', type=int, default=4)
    p.add_argument('--g_maxiter', type=int, default=8)
    p.add_argument('--g_bundle', type=int, default=40)
    p.add_argument('--g_nm', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.dim is None:
        args.dim = MODELS[MODEL].dim

    p1, p2 = grid_vals(args.stride)
    nx, ny = len(p1), len(p2)
    n_total = nx * ny
    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            sys.exit(f'task_id must be in [0, {n_total})')
        run_point(args.task_id // ny, args.task_id % ny, nx, ny, args)
        return
    if not (0 <= args.task_id < args.n_chunks):
        sys.exit(f'chunk id must be in [0, {args.n_chunks})')
    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] round {args.round}: '
          f'{len(ids)} of {n_total} points', flush=True)
    for pid in ids:
        try:
            run_point(pid // ny, pid % ny, nx, ny, args)
        except Exception as e:                              # noqa: BLE001
            print(f'  ERROR at point {pid}: {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
