"""
Margin rounds for the seeded model10 Heisenberg rates at d_ext = 8 or 12:
give rate-0 frames SLACK so they spread to their neighbours.

A rate-0 frame found by minimising the rate sits exactly at the floor (the
identity column pins mu* >= 0), so it is slightly positive one grid step
towards less noise and never transfers.  framability_rate_margin optimises
F = max over the non-identity columns instead (margin = -F); a frame with
margin keeps rate 0 on a neighbourhood of generators.

One round, per grid point (ix, iy):
  rate 0 (best known <= --tol):
    * no margin frame yet: margin_polish from the best of {own rate-0 frame,
      neighbour margin frames that are at the floor here}; write it;
    * already has one: adopt a 4-neighbour's margin frame if it gives a
      larger margin HERE, polish it, write it; otherwise nothing.
  rate > 0:
    * transfer: every margin frame within --radius grid steps (rounds < r;
      at d_ext = 12 also the d_ext = 8 margin frames, padded) is scored on
      this point's generator; one at the floor -> certified, polished for
      margin, written: the point is now rate 0 and spreads next round;
    * push (only within --radius of a rate-0 point): margin_polish from the
      best candidate, which at positive F is a rate polish over all
      near-binding columns at once; written if it beats the best known rate.
Every written value is certified with the independent per-column LP.

Reads   <out_dir>/model10_seeded[_d12][_s<stride>]/  (worker, xeval, qrefine,
        margin files -- model10_seeded_qrefine_worker.best_known)
Writes  .../pt_<ix>_<iy>_margin_r<NN>.npz  with rate_<m>, S_<m>, label_<m>,
        excess_<m> (= -margin, certified), mode

Usage:
    python scripts/model10_margin_worker.py --d_ext 8 --round 1 --task_id 0 --n_chunks 200
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
from framability_rate import RATE_VERSION                                 # noqa: E402
from framability_rate_global import fit_columns                          # noqa: E402
from framability_rate_margin import (excess, excess_reference,           # noqa: E402
                                     margin_polish, self_check,
                                     RATE_MARGIN_VERSION)
from model10_seeded_worker import MODEL, grid_vals, pt_dir_name, generator  # noqa: E402
from model10_seeded_qrefine_worker import best_known, MARGIN_TAG          # noqa: E402
from model10_d12_worker import d12_dir                                    # noqa: E402

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def point_dir(out_dir: str, stride: int, m: int) -> Path:
    if m == 8:
        return Path(out_dir) / pt_dir_name(stride)
    if m == 12:
        return d12_dir(out_dir, stride)
    raise ValueError(f'd_ext {m}: only 8 and 12 have seeded runs')


def margin_frame(pt: Path, ix: int, iy: int, m: int, before: int | None):
    """(F, frame, label) of the point's lowest-F margin file with round
    < before (all rounds if None), or None."""
    best = None
    for f in sorted(pt.glob(f'pt_{ix:03d}_{iy:03d}{MARGIN_TAG}[0-9][0-9].npz')):
        r = int(f.name[-6:-4])
        if before is not None and r >= before:
            continue
        try:
            d = np.load(f, allow_pickle=True)
            F = float(d[f'excess_{m}'])
            if best is None or F < best[0]:
                best = (F, np.asarray(d[f'S_{m}'], float), str(d[f'label_{m}']))
        except Exception:                                   # noqa: BLE001
            continue
    return best


def _plus(label: str, tag: str) -> str:
    return label if label.endswith(tag) else f'{label} {tag}'


def run_point(ix, iy, nx, ny, args) -> None:
    m = args.d_ext
    key, s_key = f'rate_{m}', f'S_{m}'
    pt = point_dir(args.out_dir, args.stride, m)
    stem = f'pt_{ix:03d}_{iy:03d}'
    out = pt / f'{stem}{MARGIN_TAG}{args.round:02d}.npz'
    if out.exists() or not (pt / f'{stem}.npz').exists():
        return
    v_best, S_best, lab_best = best_known(pt, ix, iy, key, s_key)
    if S_best is None:
        return
    zero = v_best <= args.tol
    own_mf = margin_frame(pt, ix, iy, m, args.round)

    # margin frames around the point (earlier rounds), padded d_ext = 8 ones
    R = args.radius
    near, zero_near = [], False
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < nx and 0 <= jy < ny):
                continue
            if (dx, dy) != (0, 0):
                mf = margin_frame(pt, jx, jy, m, args.round)
                if mf is not None and mf[0] < 0:
                    near.append((abs(dx) + abs(dy), mf[1], mf[2]))
                if not zero_near:
                    zero_near = best_known(pt, jx, jy, key, s_key)[0] <= args.tol
            if m == 12 and args.use_d8:
                mf = margin_frame(point_dir(args.out_dir, args.stride, 8),
                                  jx, jy, 8, None)
                if mf is not None and mf[0] < 0:
                    near.append((abs(dx) + abs(dy), fit_columns(mf[1], m),
                                 f'd8 {mf[2]}'))
    if zero and own_mf is not None:
        # gossip: only the 4-neighbours' margin frames, and only if better here
        near = [t for t in near if t[0] == 1]
    if not zero and not near and not (args.push and zero_near):
        return                                  # nothing can reach this point

    p1, p2 = grid_vals(args.stride)
    d1, d2 = float(p1[ix]), float(p2[iy])
    A = generator(d1, d2, args.dim)
    t0 = time.perf_counter()

    # score the candidates on this point's generator
    cands = [(S, lab) for _, S, lab in near]
    if not (zero and own_mf is not None):
        cands.append((S_best, lab_best))
    else:
        cands.append((own_mf[1], own_mf[2]))
    scored = sorted(((excess(S, A), i) for i, (S, _) in enumerate(cands)),
                    key=lambda t: t[0])
    F0, i0 = scored[0]
    S0, lab0 = cands[i0]
    payload = None

    if zero:
        if own_mf is not None and F0 >= own_mf[0] - args.gain_tol:
            return                              # own margin frame still best
        F1, S1, _ = margin_polish(S0, A, n_iter=args.margin_iters)
        if not (F1 <= F0):
            F1, S1 = F0, S0
        F_ref = excess_reference(S1, A)
        if not (np.isfinite(F_ref) and F_ref <= args.tol):
            print(f'  ({ix},{iy}) margin frame not certified (F_ref={F_ref:.3e})',
                  flush=True)
            return
        if own_mf is not None and F_ref >= own_mf[0] - args.gain_tol:
            return
        payload = dict(mode='gossip' if own_mf is not None else 'margin',
                       rate=0.0, S=S1, label=_plus(lab0, '+margin'), F=F_ref)
    else:
        mode = None
        if F0 <= args.tol:                      # transfer reached the floor
            mode, S1, F1 = 'transfer', S0, F0
            F1p, S1p, _ = margin_polish(S0, A, n_iter=args.margin_iters)
            if F1p <= F1:
                S1, F1 = S1p, F1p
        elif args.push and zero_near:
            mode = 'push'
            F1, S1, _ = margin_polish(S0, A, n_iter=args.push_iters)
            if not (F1 <= F0):
                F1, S1 = F0, S0
        if mode is None:
            return
        F_ref = excess_reference(S1, A)
        if not np.isfinite(F_ref):
            print(f'  ({ix},{iy}) {mode}: reference LP failed', flush=True)
            return
        rate = max(F_ref, 0.0)            # the identity column pins mu* >= 0
        if rate >= v_best - args.gain_tol:
            print(f'  ({ix},{iy}) {mode}: {v_best:.6e} -> {rate:.6e} '
                  f'(no gain, {time.perf_counter() - t0:.0f}s)', flush=True)
            return
        payload = dict(mode=mode, rate=rate, S=S1,
                       label=_plus(lab0, '+margin'), F=F_ref)

    pt.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.npz')
    np.savez(tmp, model=MODEL, ix=ix, iy=iy, delta1=d1, delta2=d2, dim=args.dim,
             d_ext=m, round=args.round, mode=payload['mode'],
             rate_version=RATE_VERSION, rate_margin_version=RATE_MARGIN_VERSION,
             **{key: payload['rate'], s_key: payload['S'],
                f'label_{m}': payload['label'], f'excess_{m}': payload['F'],
                f'{key}_prev': v_best})
    os.replace(tmp, out)
    print(f'  saved {out.name} [{payload["mode"]}] rate {v_best:.6e} -> '
          f'{payload["rate"]:.6e}, margin {-payload["F"]:+.4e}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--d_ext', type=int, choices=(8, 12), required=True)
    p.add_argument('--round', type=int, required=True,
                   help='margin round (1..99); round r reads rounds < r')
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1)
    p.add_argument('--out_dir', type=str, default='results_model10_rate')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--radius', type=int, default=2,
                   help='Chebyshev radius (grid steps) of the transferred frames')
    p.add_argument('--tol', type=float, default=1e-6,
                   help='rate counted as 0 (the figure contour tolerance)')
    p.add_argument('--gain_tol', type=float, default=1e-9)
    p.add_argument('--margin_iters', type=int, default=40)
    p.add_argument('--push_iters', type=int, default=40)
    p.add_argument('--no_push', dest='push', action='store_false')
    p.add_argument('--no_d8', dest='use_d8', action='store_false',
                   help='d_ext=12: do not transfer padded d_ext=8 margin frames')
    p.add_argument('--self_check', action='store_true',
                   help='chunk 0 logs framability_rate_margin.self_check first')
    args = p.parse_args()
    if args.dim is None:
        args.dim = MODELS[MODEL].dim
    if args.self_check and args.task_id == 0:
        try:
            print('[self-check] ' + ('passed' if self_check() else 'FAILED'),
                  flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f'[self-check] ERROR {type(e).__name__}: {e}', flush=True)

    p1, p2 = grid_vals(args.stride)
    nx, ny = len(p1), len(p2)
    n_total = nx * ny
    if args.n_chunks <= 1:
        ids = [args.task_id]
    else:
        ids = list(range(args.task_id, n_total, args.n_chunks))
        print(f'[chunk {args.task_id}/{args.n_chunks}] margin d_ext={args.d_ext} '
              f'round {args.round}: {len(ids)} of {n_total} points', flush=True)
    for pid in ids:
        ix, iy = pid // ny, pid % ny
        try:
            run_point(ix, iy, nx, ny, args)
        except Exception as e:                              # noqa: BLE001
            print(f'  ERROR at ({ix},{iy}): {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
