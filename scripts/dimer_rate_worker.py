"""
Dimer-frame Heisenberg framability rate on a model's grid, one point per call
(framability_rate_dimer.minimize_dimer_rate).

Every dimer carries the same 2-qubit frame S; the model's bond gate is
certified INSIDE a dimer (frame S) and ACROSS two dimers (frame S (x) S), and
the objective is the per-bond average

    F_split = ( mu_in + sum_c w_c mu_cut_c ) / (2 dim)      (default --mode split)

comparable with the stored per-bond product-frame rates (rate_heis_<m>).
--mode joint optimises the patch certificate instead; both values are always
stored for the final frame.

Seeds per point: Pauli, products S1 (x) S1 of the single-qubit seed library
and of every 4-column frame stored for the point by the existing pipelines
(default_seed_dirs: base / gopt / refine and seeded folders at the SAME dim;
--seed_dirs overrides; with none found, a fresh single-qubit d4 optimum),
and --dimer_dirs: 2-qubit frames of an earlier dimer run (same point).

Output: <out_dir>/<model>_dimer_<tiling>_<mode>[_s<stride>]/pt_<ix:03d>_<iy:03d>.npz
    rate_dimer_16      certified objective (the run's mode) of S_dimer_16
    rate_split_16, rate_joint_16      both objectives of the final frame
    mu_in_16, mu_cut_<c>_16           split parts;  mu_joint_<c>_16 joint parts
    S_dimer_16, label_16              frame (16 x 16, identity first), origin
    mu_seed_16, seed_name_16, mu_weights_16 (15 weights, U = 1), mu_rw_16
    (rotate + weight), mu_smooth_16 (smoothed L-BFGS), mu_polish_16,
    mu_reference_16 (per-column LP), t_16
    rw_theta_16, rw_u_16              rotate+weight optimum (U params, log w)
    with --m_ext m:  rate_dimer_<m>, rate_split_<m>, rate_joint_<m>,
                     S_dimer_<m>, mu_padded_<m>, t_<m>
    rate_pauli         closed-form Pauli rate of the bond (= Pauli dimer frame)
    prod_rate_4        single-qubit d4 rate of the best product seed
    prod_best_<k>      best stored product-frame rate per single-qubit d_ext k
Default out_dir: results_<model>_dimer (results_<model>_dim<d>_dimer when
--dim differs from the model's own).

Usage (nothing is run by default; one point or a strided chunk):
    python scripts/dimer_rate_worker.py --model model4 --task_id 2550
    python scripts/dimer_rate_worker.py --model model10 --task_id 7 --n_chunks 200
    python scripts/dimer_rate_worker.py --model model10 --dim 2 --task_id 7 --n_chunks 200
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
from framability_rate import RATE_VERSION                              # noqa: E402
from framability_rate_frames import pauli_rate                         # noqa: E402
from framability_rate_global import basis_rate                         # noqa: E402
from framability_rate_dimer import (                                    # noqa: E402
    DIMER_RATE_VERSION, DimerProblem, NQ, minimize_dimer_rate,
    stored_product_frames, default_seed_dirs)


def grid_vals(model: str, stride: int):
    m = MODELS[model]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def pt_dir_name(model: str, tiling: str, mode: str, stride: int) -> str:
    return f'{model}_dimer_{tiling}_{mode}' + ('' if stride == 1 else f'_s{stride}')


def stored_dimer_frames(dirs, ix: int, iy: int):
    """2-qubit frames S_dimer_<m> of earlier dimer runs at the same point."""
    out = []
    for d in dirs or []:
        f = Path(d) / f'pt_{ix:03d}_{iy:03d}.npz'
        if not f.exists():
            continue
        try:
            z = np.load(f, allow_pickle=True)
        except Exception:                                   # noqa: BLE001
            continue
        for k in z.files:
            if k.startswith('S_dimer_'):
                S = np.asarray(z[k], float)
                if S.ndim == 2 and S.shape[0] == NQ:
                    out.append(S)
    return out


def compute_point(model, p1, p2, ix_full, iy_full, args) -> dict:
    spec = MODELS[model]
    H1, H2, j1, j2 = spec.build(p1, p2)
    B = build_bond_lindbladian(H1, H2, j1, j2, args.dim).real
    prob = DimerProblem(B, args.dim, args.tiling, mode=args.mode)
    out: dict = {spec.p1_name: p1, spec.p2_name: p2, 'dim': args.dim,
                 'tiling': prob.tiling, 'mode': args.mode,
                 'rate_pauli': pauli_rate(B)}

    dirs = args.seed_dirs if args.seed_dirs is not None else \
        default_seed_dirs(model, args.dim, spec.dim)
    frames, stored = stored_product_frames(dirs, ix_full, iy_full)
    for k, v in stored.items():
        out[f'prod_best_{k}'] = v
    d4 = frames.get(4, [])
    dimer_seeds = stored_dimer_frames(args.dimer_dirs, ix_full, iy_full)

    t0 = time.perf_counter()
    S, F, info = minimize_dimer_rate(
        prob, mode=args.mode, m_ext=args.m_ext, product_seeds=d4,
        dimer_seeds=dimer_seeds, fresh_d4=True, rw_random=args.rw_random,
        rw_maxfev=args.rw_maxfev, n_polish=args.n_polish,
        smooth_iters=args.smooth_iters, n_kicks=args.n_kicks,
        bundle_iters=args.bundle_iters, ext_iters=args.ext_iters,
        certify=not args.no_certify, seed=args.seed)
    prod4 = [basis_rate(S1, prob.A_in) for S1 in d4]
    if 'fresh_d4' in info:
        prod4.append(info['fresh_d4'])
    out['prod_rate_4'] = min(prod4) if prod4 else np.nan
    out['n_stored_d4'] = len(d4)
    out['n_dimer_seeds'] = len(dimer_seeds)

    m = NQ
    out[f'rate_dimer_{m}'] = F
    out[f'rate_split_{m}'] = info['rate_split']
    out[f'rate_joint_{m}'] = info['rate_joint']
    for k, v in info['parts_split'].items():
        out[f'mu_{k}_{m}'] = v                       # mu_in_16, mu_cut_ba_16, ..
    for k, v in info['parts_joint'].items():
        out[f'mu_{k}_{m}'] = v                       # mu_joint_ba_16, ..
    out[f'S_dimer_{m}'] = S
    out[f'label_{m}'] = str(info['label'])
    out[f'seed_name_{m}'] = str(info['seed_name'])
    for key in ('mu_seed', 'mu_weights16', 'mu_rw', 'mu_smooth', 'mu_polish',
                'mu_reference', 'op_norm_max'):
        if key in info:
            name = 'mu_weights' if key == 'mu_weights16' else key
            out[f'{name}_{m}'] = info[key]
    out[f'rw_theta_{m}'] = info['rw_theta']
    out[f'rw_u_{m}'] = info['rw_u']
    out[f't_{m}'] = info['time16']
    if 'ext' in info:
        e = info['ext']
        me = e['m']
        out[f'rate_dimer_{me}'] = e['value']
        out[f'rate_split_{me}'] = e['rate_split']
        out[f'rate_joint_{me}'] = e['rate_joint']
        out[f'mu_padded_{me}'] = e['padded']
        out[f'S_dimer_{me}'] = e['S']
        out[f't_{me}'] = e['time']
    out['t_total'] = time.perf_counter() - t0
    return out


def run_point(model: str, ix: int, iy: int, args) -> None:
    p1, p2 = grid_vals(model, args.stride)
    pt_dir = Path(args.out_dir) / pt_dir_name(model, args.tiling, args.mode,
                                              args.stride)
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists() and not args.force:
        print(f'[skip] {out_f}', flush=True)
        return
    # stored product frames live on the FULL grid
    ix_full, iy_full = ix * args.stride, iy * args.stride
    t0 = time.perf_counter()
    spec = MODELS[model]
    print(f'[{model} dim={args.dim} {args.tiling}/{args.mode}] ({ix},{iy}) '
          f'{spec.p1_name}={p1[ix]:.3f} {spec.p2_name}={p2[iy]:.3f}', flush=True)
    try:
        res = compute_point(model, float(p1[ix]), float(p2[iy]), ix_full,
                            iy_full, args)
    except Exception as e:                                  # noqa: BLE001
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        return
    pt_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_f.with_suffix('.tmp.npz')
    np.savez(tmp, model=model, ix=ix, iy=iy, stride=args.stride,
             rate_version=RATE_VERSION, dimer_rate_version=DIMER_RATE_VERSION,
             **res)
    os.replace(tmp, out_f)                      # never leave a half-written file
    prod = res.get('prod_rate_4', np.nan)
    best_k = [k for k in res if k.startswith('prod_best_')]
    stored = min((res[k] for k in best_k), default=np.nan)
    print(f'  saved {out_f.name}  F_{args.mode}={res["rate_dimer_16"]:.5f} '
          f'(split {res["rate_split_16"]:.5f}, joint {res["rate_joint_16"]:.5f}; '
          f'product d4 {prod:.5f}, best stored product {stored:.5f}, '
          f'Pauli {res["rate_pauli"]:.5f})  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='<=1: task_id is a flat grid index; otherwise the grid '
                        'is strided over n_chunks array tasks')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--dim', type=int, default=None,
                   help="lattice dimension (default: the model's)")
    p.add_argument('--tiling', type=str, default=None,
                   help='chain (dim 1); brick (default) / columnar (dim 2)')
    p.add_argument('--mode', type=str, default='split', choices=['split', 'joint'])
    p.add_argument('--m_ext', type=int, default=None,
                   help='also polish an m_ext > 16 frame (per-column LPs, slow)')
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--seed_dirs', nargs='*', default=None,
                   help='folders of stored single-qubit frames (pt_<ix>_<iy>*.npz)')
    p.add_argument('--dimer_dirs', nargs='*', default=None,
                   help='folders of an earlier dimer run (S_dimer_* seeds)')
    p.add_argument('--rw_random', type=int, default=3)
    p.add_argument('--rw_maxfev', type=int, default=400)
    p.add_argument('--n_polish', type=int, default=3)
    p.add_argument('--smooth_iters', type=int, default=300,
                   help='L-BFGS iterations per (beta, eps) stage (0 disables)')
    p.add_argument('--n_kicks', type=int, default=2,
                   help='random kicks of the best seed per kick scale')
    p.add_argument('--bundle_iters', type=int, default=40)
    p.add_argument('--ext_iters', type=int, default=20)
    p.add_argument('--no_certify', action='store_true',
                   help='skip the per-column LP confirmation')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    spec = MODELS[args.model]
    if args.dim is None:
        args.dim = spec.dim
    if args.tiling is None:
        args.tiling = 'chain' if args.dim == 1 else 'brick'
    if args.out_dir is None:
        args.out_dir = f'results_{args.model}_dimer' if args.dim == spec.dim \
            else f'results_{args.model}_dim{args.dim}_dimer'

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
          f'{n_total} points ({nx}x{ny})', flush=True)
    for pid in ids:
        run_point(args.model, pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
