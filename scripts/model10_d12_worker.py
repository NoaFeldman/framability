"""
Optimised Heisenberg framability rate of model10 at d_ext = 12, seeded from
the d_ext = 8 results of the seeded pipeline.

--stage seed  (one pass over the grid).  Per point, seeds are evaluated as
fixed frames in order of cost, stopping as soon as one is at the floor:
  1. the point's best d_ext = 8 frame (min over the seeded worker, xeval and
     qrefine files), padded to 12 columns  ->  rate_12 <= rate_8 always
  2. the 8 neighbours' best d_ext = 8 frames, padded; the closed-form frames
     B, C10, P9 (framability_rate_families.model10_frames)
  3. the point's d_ext = 8 frame grown greedily by 4 columns
     (framability_rate_families.greedy_augment: at each step the best of the
     pairwise sums / differences of its columns and the Pauli axes)
  4. the families that WON at d_ext = 8 (winning_families_for(12): YZ 10-gon,
     YZ 8-gon + XY pair, YZ hexagon + XY square), multistart Nelder-Mead from
     their analytic starts and from the point's own d_ext = 8 family optima
     (transfer_params; polygon parameters do not depend on the vertex count)
  5. framability_rate_global.minimize_rate_global with every seed above and
     NO differential evolution (44 parameters at ~1 s per 144-column LP):
     its seed library, bundle polish, Nelder-Mead, per-column-LP certification.

--stage refine --round r  (quick neighbour refine, rounds sequential):
  every point above the floor tries its 4 neighbours' best d_ext = 12 frames
  and its own current best d_ext = 8 frame padded (the two-LP accept rule);
  boundary points (a 4-neighbour at the floor) are then re-optimised with
  minimize_rate_global (no DE) seeded with their own and the neighbour frames.

Reads   <out_dir>/model10_seeded[_s<stride>]/                 (d_ext = 4 / 8)
Writes  <out_dir>/model10_seeded_d12[_s<stride>]/pt_<ix>_<iy>.npz
        ... /pt_<ix>_<iy>_qrefine_r<NN>.npz   (refine: improved points only)
        keys rate_12, S_12, label_12 (+ diagnostics)

Usage:
    python scripts/model10_d12_worker.py --stage seed --task_id 0 --n_chunks 200
    python scripts/model10_d12_worker.py --stage refine --round 1 --task_id 0 --n_chunks 200
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
from trotter_lindbladian_scan import MODELS, MODEL10_J, MODEL10_H         # noqa: E402
from framability_rate import RATE_VERSION                                 # noqa: E402
from framability_rate_global import (minimize_rate_global, fit_columns,   # noqa: E402
                                     frame_rate_value, RATE_GLOBAL_VERSION)
from framability_rate_families import (model10_frames, optimize_family,   # noqa: E402
                                       model10_family_starts, greedy_augment,
                                       winning_families_for, transfer_params,
                                       RATE_FAMILIES_VERSION)
from model10_seeded_worker import MODEL, grid_vals, pt_dir_name, generator  # noqa: E402
from model10_seeded_qrefine_worker import best_known, ROUND_TAG           # noqa: E402

M = 12
M_SRC = 8
TOL = 1e-9
KEY, S_KEY, LAB_KEY = f'rate_{M}', f'S_{M}', f'label_{M}'
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def d12_dir(out_dir: str, stride: int) -> Path:
    return Path(out_dir) / (pt_dir_name(stride).replace('_seeded', '_seeded_d12', 1))


def d8_dir(out_dir: str, stride: int) -> Path:
    return Path(out_dir) / pt_dir_name(stride)


def _certify(S, A) -> float:
    v = frame_rate_value(S, A, reference=True)
    return max(float(v), 0.0) if np.isfinite(v) and v > -TOL else float(v)


def _family_params_d8(pt8: Path, ix: int, iy: int) -> dict:
    """{family name: optimised params} stored at d_ext = 8 for the point."""
    f = pt8 / f'pt_{ix:03d}_{iy:03d}.npz'
    if not f.exists():
        return {}
    d = np.load(f, allow_pickle=True)
    return {k[len('famp_'):-len(f'_{M_SRC}')]: np.asarray(d[k], float)
            for k in d.files if k.startswith('famp_') and k.endswith(f'_{M_SRC}')}


class Best:
    """Incumbent over evaluated candidates (batched-LP screening)."""

    def __init__(self, A):
        self.A, self.v, self.S, self.label, self.n = A, np.inf, None, '', 0

    def offer(self, label, S, v=None):
        if S is None:
            return
        if v is None:
            v = frame_rate_value(S, self.A)
            self.n += 1
        if np.isfinite(v) and v < self.v:
            self.v, self.S, self.label = float(v), np.asarray(S, float), label

    def at_floor(self) -> bool:
        """Screened at the floor AND confirmed by the per-column LP."""
        if self.v > TOL:
            return False
        v = _certify(self.S, self.A)
        if v <= TOL:
            self.v = v
            return True
        self.v = v                        # screening was optimistic
        return False


# ---------------------------------------------------------------------------
#  Stage: seed
# ---------------------------------------------------------------------------
def compute_seed(ix, iy, nx, ny, args) -> dict:
    p1, p2 = grid_vals(args.stride)
    d1, d2 = float(p1[ix]), float(p2[iy])
    A = generator(d1, d2, args.dim)
    J, h = MODEL10_J, MODEL10_H
    pt8 = d8_dir(args.out_dir, args.stride)
    out = dict(delta1=d1, delta2=d2, dim=args.dim)
    best = Best(A)
    t0 = time.perf_counter()

    def done(mode):
        if best.S is None:
            raise RuntimeError('no finite d_ext = 12 frame found')
        out.update({KEY: best.v, S_KEY: best.S, LAB_KEY: best.label,
                    f'mode_{M}': mode, f'n_evals_{M}': best.n,
                    f't_{M}': time.perf_counter() - t0})
        return out

    # 1. own best d_ext = 8 frame, padded
    v8, S8, lab8 = best_known(pt8, ix, iy, f'rate_{M_SRC}', f'S_{M_SRC}')
    out[f'rate_{M_SRC}_src'] = v8
    if S8 is not None:
        best.offer(f'd8 {lab8}', fit_columns(S8, M))
        if best.at_floor():
            return done('d8 at floor')

    # 2. neighbours' d_ext = 8 frames, closed forms
    for dx, dy in NEIGHBORS8:
        jx, jy = ix + dx, iy + dy
        if 0 <= jx < nx and 0 <= jy < ny:
            _, Sn, labn = best_known(pt8, jx, jy, f'rate_{M_SRC}', f'S_{M_SRC}')
            if Sn is not None:
                best.offer(f'd8 {labn}', fit_columns(Sn, M))
    for name, S in model10_frames(d1, d2, J, h, M).items():
        best.offer(f'analytic {name}', S)
    if best.at_floor():
        return done('cheap seed at floor')

    seeds = [best.S] if best.S is not None else []
    # 3. greedy augmentation of the point's own d_ext = 8 frame
    if S8 is not None and not args.no_greedy:
        v, S, n = greedy_augment(S8, A, M)
        best.n += n
        out[f'greedy_{M}'] = v
        best.offer('d8 augmented', S, v)
        seeds.append(S)
        if best.at_floor():
            return done('augmented at floor')

    # 4. winning structured families
    rng = np.random.default_rng([args.seed, ix, iy, M])
    famp = _family_params_d8(pt8, ix, iy)
    for fam in winning_families_for(M):
        starts = transfer_params(fam, famp) + model10_family_starts(fam, d1, d2, J, h)
        v, S, p, n = optimize_family(fam, A, starts, n_random=args.fam_random,
                                     n_polish=args.fam_polish,
                                     maxfev=args.fam_maxfev, rng=rng)
        best.n += n
        out[f'famval_{fam.name}_{M}'] = v
        out[f'famp_{fam.name}_{M}'] = p
        best.offer(f'family {fam.name}', S, v)
        seeds.append(S)
        if best.at_floor():
            return done('family at floor')

    # 5. seeded global polish (no DE)
    seeds = [S for S in [best.S] + seeds if S is not None]
    S_g, mu_g, info = minimize_rate_global(
        A, M, seeds=seeds, seed=args.seed + M, de_popsize=1, de_maxiter=0,
        bundle_iters=args.bundle_iters, nm_maxfev=args.nm_maxfev)
    v_inc = _certify(best.S, A) if best.S is not None else np.inf
    if np.isfinite(mu_g) and mu_g < v_inc - TOL:
        best.v, best.S, best.label = float(mu_g), S_g, f'{best.label} +opt'
    else:
        best.v = v_inc
    out[f'mu_bundle_{M}'] = info.get('mu_bundle', np.nan)
    return done('global')


# ---------------------------------------------------------------------------
#  Stage: refine (one round)
# ---------------------------------------------------------------------------
def compute_refine(ix, iy, nx, ny, args):
    pt = d12_dir(args.out_dir, args.stride)
    pt8 = d8_dir(args.out_dir, args.stride)
    v_own, S_own, lab_own = best_known(pt, ix, iy, KEY, S_KEY)
    if S_own is None or v_own <= args.rate_tol:
        return None                                     # nothing to do
    nb = []
    for dx, dy in NEIGHBORS4:
        jx, jy = ix + dx, iy + dy
        if 0 <= jx < nx and 0 <= jy < ny:
            v, S, lab = best_known(pt, jx, jy, KEY, S_KEY)
            if S is not None:
                nb.append((v, S, lab))
    boundary = any(v <= args.rate_tol for v, _, _ in nb)

    p1, p2 = grid_vals(args.stride)
    d1, d2 = float(p1[ix]), float(p2[iy])
    A = generator(d1, d2, args.dim)
    t0 = time.perf_counter()
    best = Best(A)
    best.v, best.S, best.label = v_own, S_own, lab_own
    # propagation: neighbour d12 frames, own current d8 frame padded
    _, S8, lab8 = best_known(pt8, ix, iy, f'rate_{M_SRC}', f'S_{M_SRC}')
    cands = [(lab, S) for _, S, lab in nb]
    if S8 is not None:
        cands.append((f'd8 {lab8}', fit_columns(S8, M)))
    screened = Best(A)
    for lab, S in cands:
        screened.offer(lab, S)
    if screened.S is not None and screened.v < v_own - TOL:
        v_ref = _certify(screened.S, A)
        if v_ref < v_own - TOL:
            best.v, best.S, best.label = v_ref, screened.S, screened.label
    # quick re-optimisation at the floor boundary
    if boundary and best.v > args.rate_tol:
        seed = args.seed + (ix * ny + iy) + 100000 * args.round
        S_g, mu_g, _ = minimize_rate_global(
            A, M, seeds=[best.S] + [S for _, S, _ in nb], seed=seed,
            de_popsize=1, de_maxiter=0, bundle_iters=args.refine_bundle,
            nm_maxfev=args.refine_nm)
        if np.isfinite(mu_g) and mu_g < best.v - TOL:
            lab = best.label if best.label.endswith('+qref') else f'{best.label} +qref'
            best.v, best.S, best.label = float(mu_g), S_g, lab
    if best.v >= v_own - TOL:
        return dict(improved=False, t=time.perf_counter() - t0,
                    boundary=boundary, v_own=v_own)
    return {KEY: best.v, S_KEY: best.S, LAB_KEY: best.label,
            f'{KEY}_prev': v_own, 'boundary': boundary, 'improved': True,
            'delta1': d1, 'delta2': d2, 't': time.perf_counter() - t0}


# ---------------------------------------------------------------------------
def run_point(ix, iy, nx, ny, args) -> None:
    pt = d12_dir(args.out_dir, args.stride)
    stem = f'pt_{ix:03d}_{iy:03d}'
    if args.stage == 'seed':
        out_f = pt / f'{stem}.npz'
        if out_f.exists() and not args.force:
            print(f'[skip] {out_f.name}', flush=True)
            return
        t0 = time.perf_counter()
        res = compute_seed(ix, iy, nx, ny, args)
    else:
        out_f = pt / f'{stem}{ROUND_TAG}{args.round:02d}.npz'
        if out_f.exists() or not (pt / f'{stem}.npz').exists():
            return
        t0 = time.perf_counter()
        res = compute_refine(ix, iy, nx, ny, args)
        if res is None:
            return
        if not res['improved']:
            print(f'  ({ix},{iy}) {res["v_own"]:.6e} not improved '
                  f'(boundary={res["boundary"]}, {res["t"]:.0f}s)', flush=True)
            return
    pt.mkdir(parents=True, exist_ok=True)
    tmp = out_f.with_suffix('.tmp.npz')
    np.savez(tmp, model=MODEL, ix=ix, iy=iy, stride=args.stride, d_ext=M,
             stage=args.stage, round=args.round, rate_version=RATE_VERSION,
             rate_global_version=RATE_GLOBAL_VERSION,
             rate_families_version=RATE_FAMILIES_VERSION, **res)
    os.replace(tmp, out_f)
    prev = res.get(f'{KEY}_prev', res.get(f'rate_{M_SRC}_src', np.nan))
    print(f'  saved {out_f.name}  d{M}={res[KEY]:.6e} [{res[LAB_KEY]}] '
          f'(from {prev:.6e})  ({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('seed', 'refine'), required=True)
    p.add_argument('--round', type=int, default=0,
                   help='refine round (1..99); round r reads rounds < r')
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1)
    p.add_argument('--out_dir', type=str, default='results_model10_rate')
    p.add_argument('--stride', type=int, default=1,
                   help='stride of the seeded d_ext = 4 / 8 run')
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no_greedy', action='store_true')
    p.add_argument('--fam_random', type=int, default=4)
    p.add_argument('--fam_polish', type=int, default=2)
    p.add_argument('--fam_maxfev', type=int, default=80)
    p.add_argument('--bundle_iters', type=int, default=40)
    p.add_argument('--nm_maxfev', type=int, default=150)
    p.add_argument('--rate_tol', type=float, default=1e-6)
    p.add_argument('--refine_bundle', type=int, default=25)
    p.add_argument('--refine_nm', type=int, default=0)
    p.add_argument('--force', action='store_true',
                   help='seed stage: recompute points that already have a file')
    args = p.parse_args()
    if args.dim is None:
        args.dim = MODELS[MODEL].dim
    if args.stage == 'refine' and args.round < 1:
        sys.exit('--stage refine needs --round >= 1')

    p1, p2 = grid_vals(args.stride)
    nx, ny = len(p1), len(p2)
    n_total = nx * ny
    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            sys.exit(f'task_id must be in [0, {n_total})')
        ids = [args.task_id]
    else:
        if not (0 <= args.task_id < args.n_chunks):
            sys.exit(f'chunk id must be in [0, {args.n_chunks})')
        ids = list(range(args.task_id, n_total, args.n_chunks))
        print(f'[chunk {args.task_id}/{args.n_chunks}] d_ext={M} stage={args.stage}'
              f'{f" round={args.round}" if args.stage == "refine" else ""}: '
              f'{len(ids)} of {n_total} points', flush=True)
    for pid in ids:
        ix, iy = pid // ny, pid % ny
        try:
            run_point(ix, iy, nx, ny, args)
        except Exception as e:                              # noqa: BLE001
            print(f'  ERROR at ({ix},{iy}): {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
