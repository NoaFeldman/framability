"""
The remaining paper panels of the disordered open Dicke model
(framability_rate_disorder): stabilizer-3, random product-state chi = 10 / 40,
and an optimised Schrodinger (state) frame, each as the PAIR-AVERAGED rate

    F = sum_p w_p mu(B_p; frame)      (p = species pairs, w_p = pair weights)

the same average that dicke_disorder_rate_worker.py takes for Pauli and the
optimised observable frames (mu(L) <= (N/2) F; at width 0 there is one pair
and F is the uniform model's bond rate).  The frames are the uniform
pipeline's:

    stab3     framability_rate_frames.stabilizer_3_rate on every pair patch
    prod_<c>  framability_rate_frames.product_state_rate with the frames of
              model4_product_rate_worker.build_prod_frames (the same draws as
              the uniform rate_prod_<c>)
    schro_<d> ONE state frame S on every atom (shared; no per-species state
              frames), minimising the pair-averaged Schrodinger rate.  Seeds:
              the uniform model's optimised x_schro_<d> at the same grid point
              (scripts/submit_paper_extra_rates.sh ->
              results_<family>_N<N>_seeded/<family>_N<N>_schro/), the
              great-circle seeds of model4_rate_panels_worker.circle_state_seeds,
              the octahedron / SIC / random seeds of _build_state_inits and the
              padded smaller-d optimum of this point; every seed is scored, the
              best --n_starts are polished with the uniform optimiser's method
              (SCHRO_DEFAULT_METHOD, --maxfev evaluations each).

Combos: --families x --ns x --widths x the uniform model's grid, enumerated as
in dicke_disorder_rate_worker.py; task t of --n_tasks takes combos t,
t + n_tasks, ...

Output (one file per combo, rewritten atomically after every measure;
finished measures are skipped on a re-run unless --force):
    <out_root>/<family>_N<N>_M<M>_w<width:.2f>_extra/pt_<ix:03d>_<iy:03d>.npz
keys: family, N, M, M_eff, width, lam_ratio, y, ix, iy, pair_w, and
      F_stab3, pairs_stab3, F_prod_<c>, pairs_prod_<c>, F_schro_<d>,
      S_schro_<d>, x_schro_<d>, pairs_schro_<d>, seed_schro_<d>, t_<measure>.
Collect: scripts/dicke_disorder_extra_collect.py.

Usage (the slurm script passes the task id):
    python scripts/dicke_disorder_extra_worker.py --task_id 0 --n_tasks 200
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from trotter_lindbladian_scan import MODELS                           # noqa: E402
from framability_rate_disorder import DISORDER_VERSION, build_ensemble  # noqa: E402
from framability_rate_frames import (stabilizer_3_rate,                # noqa: E402
                                     product_state_rate, schroedinger_frame_rate)
from framability_rate_state import state_rate_objective                # noqa: E402
from optimize_framability import (_state_params_to_S, _build_state_inits,  # noqa: E402
                                  SCHRO_DEFAULT_METHOD)
from model4_product_rate_worker import build_prod_frames, PROD_RATE_CHIS  # noqa: E402
from model4_rate_panels_worker import circle_state_seeds, warm_state_seeds  # noqa: E402
from dicke_disorder_rate_worker import out_dir_of, save_atomic         # noqa: E402

MEASURES = ('stab3', 'prod', 'schro')
EXTRA_VERSION = '1.0-disorder-extra'


def extra_dir_of(root: Path, family: str, N: int, M: int, width: float) -> Path:
    d = out_dir_of(root, family, N, M, width)
    return d.with_name(d.name + '_extra')


def pair_Ls(ens) -> list:
    return [np.asarray(L).real.astype(float) for L in ens.Ls]


def fixed_avg(fn, Ls, pw):
    """(F, per-pair rates) of a fixed frame evaluated by fn(L) on every pair."""
    vals = np.array([fn(L) for L in Ls], float)
    F = float(pw @ vals) if np.all(np.isfinite(vals)) else np.inf
    return F, vals


def schro_objective(x, d, Ls, pw) -> float:
    """Pair-averaged state_rate_objective (rate + support penalty / barrier)."""
    return float(sum(w * state_rate_objective(x, d, L, 2) for w, L in zip(pw, Ls)))


def pad_x(x, d_old: int, d: int, rng) -> np.ndarray:
    """3 x d_old Bloch block -> 3 x d: repeat columns, small jitter."""
    B = np.asarray(x, float).reshape(3, d_old)
    idx = np.arange(d) % d_old
    return (B[:, idx] + 1e-3 * rng.standard_normal((3, d))).ravel()


def optimise_schro(d, Ls, pw, seeds, *, n_starts, maxfev, rng, tol=1e-9):
    """Best shared state frame of size d.  seeds: [(tag, flat x)]."""
    scored = []
    for tag, x in seeds:
        x = np.asarray(x, float).ravel()
        if x.size != 3 * d or not np.all(np.isfinite(x)):
            continue
        scored.append((schro_objective(x, d, Ls, pw), tag, x))
    scored.sort(key=lambda t: t[0])
    best_v, best_tag, best_x = scored[0]
    if best_v > tol:
        opts = dict(maxfev=maxfev, xatol=1e-6, fatol=1e-6) \
            if SCHRO_DEFAULT_METHOD == 'Nelder-Mead' else dict(maxfev=maxfev)
        for v0, tag, x0 in scored[:n_starts]:
            res = minimize(schro_objective, x0, args=(d, Ls, pw),
                           method=SCHRO_DEFAULT_METHOD, options=opts)
            if res.fun < best_v:
                best_v, best_tag, best_x = float(res.fun), f'{tag}+opt', np.asarray(res.x, float)
            if best_v <= tol:
                break
    S = _state_params_to_S(best_x, d)
    vals = np.array([schroedinger_frame_rate(S, L) for L in Ls], float)
    F = float(pw @ vals) if np.all(np.isfinite(vals)) else np.inf
    return F, vals, S, best_x, best_tag


def run_combo(family, N, width, M, ix, iy, args, root: Path, prod_frames) -> None:
    spec = MODELS[f'{family}_N{N}']
    x, y = float(spec.p1_vals[ix]), float(spec.p2_vals[iy])
    od = extra_dir_of(root, family, N, M, width)
    od.mkdir(parents=True, exist_ok=True)
    path = od / f'pt_{ix:03d}_{iy:03d}.npz'
    rec = {}
    if path.exists() and not args.force:
        with np.load(path, allow_pickle=True) as z:
            rec = {k: z[k] for k in z.files}
    want = []
    if 'stab3' in args.measures:
        want.append('F_stab3')
    if 'prod' in args.measures:
        want += [f'F_prod_{c}' for c in PROD_RATE_CHIS]
    if 'schro' in args.measures:
        want += [f'F_schro_{d}' for d in args.schro_dexts]
    tag = f'{family} N={N} w={width:.2f} ({ix},{iy}) lam/lam_c={x:g} y={y:g}'
    if all(k in rec for k in want):
        print(f'[skip] {tag}: done', flush=True)
        return
    t0 = time.time()
    ens = build_ensemble(family, N, width, M, x, y)
    Ls, pw = pair_Ls(ens), np.asarray(ens.pw, float)
    rec.update(family=family, N=N, M=M, M_eff=ens.M_eff, width=width, lam_ratio=x,
               y=y, ix=ix, iy=iy, pair_w=pw, omegas=ens.omegas,
               pairs=np.array(ens.pairs, int), version=DISORDER_VERSION,
               extra_version=EXTRA_VERSION)
    print(f'[start] {tag}: M_eff={ens.M_eff} pairs={ens.P}', flush=True)

    if 'F_stab3' in want and 'F_stab3' not in rec:
        t1 = time.time()
        rec['F_stab3'], rec['pairs_stab3'] = fixed_avg(stabilizer_3_rate, Ls, pw)
        rec['t_stab3'] = time.time() - t1
        save_atomic(path, rec)
        print(f'  stab3 F={rec["F_stab3"]:.6g} ({rec["t_stab3"]:.0f}s)', flush=True)
    for c in PROD_RATE_CHIS:
        k = f'F_prod_{c}'
        if k in want and k not in rec:
            t1 = time.time()
            D = prod_frames[c]
            rec[k], rec[f'pairs_prod_{c}'] = fixed_avg(
                lambda L: product_state_rate(c, L, D=D), Ls, pw)
            rec[f't_prod_{c}'] = time.time() - t1
            save_atomic(path, rec)
            print(f'  prod chi={c} F={rec[k]:.6g} ({rec[f"t_prod_{c}"]:.0f}s)', flush=True)

    rng = np.random.default_rng(1000003 * ix + 1009 * iy + int(round(1e4 * width)) + N)
    warm_root = REPO / f'results_{family}_N{N}_seeded'
    prev = None                                          # (d, x) of the last optimum
    for d in sorted(args.schro_dexts):
        k = f'F_schro_{d}'
        if 'schro' not in args.measures:
            break
        if k in rec:
            prev = (d, np.asarray(rec[f'x_schro_{d}'], float))
            continue
        seeds = [('warm', xw) for xw in
                 warm_state_seeds(warm_root, f'{family}_N{N}_schro', ix, iy, d)]
        seeds += [(f'circle{i}', xs) for i, xs in enumerate(circle_state_seeds(d))]
        inits, tags = _build_state_inits(d, args.n_default_seeds, rng)
        seeds += list(zip(tags, inits))
        if prev is not None:
            seeds.append((f'pad{prev[0]}', pad_x(prev[1], prev[0], d, rng)))
        t1 = time.time()
        F, vals, S, xb, stag = optimise_schro(d, Ls, pw, seeds, n_starts=args.n_starts,
                                              maxfev=args.maxfev, rng=rng)
        rec.update({k: F, f'pairs_schro_{d}': vals, f'S_schro_{d}': S,
                    f'x_schro_{d}': xb, f'seed_schro_{d}': stag,
                    f't_schro_{d}': time.time() - t1})
        save_atomic(path, rec)
        prev = (d, xb)
        print(f'  schro d={d} F={F:.6g} from {stag} ({time.time() - t1:.0f}s)', flush=True)
    print(f'[done] {tag}: {time.time() - t0:.0f}s', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--families', nargs='+', default=['dicke_dec', 'dicke_dec_k025'])
    ap.add_argument('--ns', nargs='+', type=int, default=[100])
    ap.add_argument('--widths', nargs='+', type=float, default=[0.0, 0.2, 0.5, 0.8])
    ap.add_argument('--M', type=int, default=4, help='species (frequency classes)')
    ap.add_argument('--measures', nargs='+', choices=MEASURES, default=list(MEASURES))
    ap.add_argument('--schro_dexts', nargs='+', type=int, default=[6, 8])
    ap.add_argument('--n_starts', type=int, default=3,
                    help='best-scored seeds polished per Schrodinger d_ext')
    ap.add_argument('--n_default_seeds', type=int, default=4,
                    help='octahedron / SIC / random seeds of _build_state_inits')
    ap.add_argument('--maxfev', type=int, default=2000)
    ap.add_argument('--task_id', type=int, required=True)
    ap.add_argument('--n_tasks', type=int, default=200)
    ap.add_argument('--out_root', default='results_dicke_disorder')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    root = REPO / args.out_root
    prod_frames = build_prod_frames()
    combos = []
    for fam in args.families:
        for N in args.ns:
            spec = MODELS[f'{fam}_N{N}']
            for w in args.widths:
                for ix in range(spec.N_X):
                    for iy in range(spec.N_Y):
                        combos.append((fam, N, w, ix, iy))
    mine = combos[args.task_id::args.n_tasks]
    print(f'[task {args.task_id}/{args.n_tasks}] {len(mine)} of {len(combos)} combos, '
          f'measures {args.measures}, schro d {args.schro_dexts}', flush=True)
    n_err = 0
    for fam, N, w, ix, iy in mine:
        try:
            run_combo(fam, N, w, args.M, ix, iy, args, root, prod_frames)
        except Exception:                                   # noqa: BLE001
            n_err += 1
            print(f'[error] {fam} N={N} w={w} ({ix},{iy}):', flush=True)
            traceback.print_exc()
            sys.stdout.flush()
    print(f'[task {args.task_id}] finished, {n_err} errors', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
