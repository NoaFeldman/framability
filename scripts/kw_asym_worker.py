"""
One array task of the KW-asymmetry scan (kw_asymmetry.py): exact evolution of
the full density matrix of model10 (local or collective jumps) from an
eta-even initial state, recording Delta S_KW(t) of arXiv:2610.23111 Eq. (14)
together with a = Tr rho^2, b = Tr rho_hat^2, c = Tr(rho rho_hat), Tr rho and
Tr rho_hat at every save time.

Task layout: kw_asymmetry.scan_points() lists the 200 parameter points (J = 1;
150 on the (h, D1 = D2 = gamma) panel, 50 on the (D1, D2) panel at h = J).
Task t of an array of --n_tasks handles points t, t + n_tasks, ... (one point
per task at --n_tasks 200).  A point whose file already exists is skipped
(resubmitting the array finishes only what is missing) unless --overwrite.

--selfcheck runs kw_asymmetry.selfcheck() (dense checks of the KW relations,
the Liouvillian duality and the matrix-free L on a small ring) first and
aborts the task if any check fails.

Output (one file per point, temp file + os.replace):
    <out_dir>/<tag>/pt_<PPP>.npz      tag = <geometry>_<jumps>_<init>[_a<alpha>]
keys: t, a, b, c, tr, tr_hat, dS_gen, dS_even, panel, h, d1, d2, J, i, j,
point, geometry, jumps, init, alpha, rtol, atol, n_steps, nfev, wall,
version.

Usage:
    python scripts/kw_asym_worker.py --task_id 0 --selfcheck
    python scripts/kw_asym_worker.py --task_id 17 --geometry ring:12 --jumps collective
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
from kw_asymmetry import (geometry_from_spec, initial_state, evolve_kw_asymmetry,  # noqa: E402
                          selfcheck, scan_points, run_tag, JUMP_KINDS, INIT_KINDS,
                          ALPHA_PAPER, KW_ASYM_VERSION)

J = 1.0


def point_file(out_dir: str, tag: str, p: int) -> Path:
    return Path(out_dir) / tag / f'pt_{p:03d}.npz'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--task_id', type=int, required=True)
    ap.add_argument('--n_tasks', type=int, default=200)
    ap.add_argument('--geometry', default='ring:12')
    ap.add_argument('--jumps', choices=JUMP_KINDS, default='local')
    ap.add_argument('--init', choices=INIT_KINDS, default='ghz_s')
    ap.add_argument('--alpha', type=float, default=ALPHA_PAPER,
                    help='GHZ amplitude of the ghz_s state (paper: sqrt(3)/2)')
    ap.add_argument('--t_max', type=float, default=3.0, help='final J t')
    ap.add_argument('--n_t', type=int, default=61, help='save times 0..t_max (inclusive)')
    ap.add_argument('--rtol', type=float, default=1e-8)
    ap.add_argument('--atol', type=float, default=1e-12)
    ap.add_argument('--out_dir', default='results_kw_asym')
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args(argv)

    if not (0 <= args.task_id < args.n_tasks):
        raise SystemExit(f'task_id must be in [0, {args.n_tasks})')
    geom = geometry_from_spec(args.geometry)
    tag = run_tag(geom.name, args.jumps, args.init, args.alpha)
    pts = scan_points()
    mine = list(range(args.task_id, len(pts), args.n_tasks))
    print(f'[kw asym] {tag}, task {args.task_id}/{args.n_tasks}: points {mine}', flush=True)

    if args.selfcheck:
        selfcheck(log=lambda s: print(s, flush=True))
        print('[kw asym] selfcheck passed', flush=True)

    ts = np.linspace(0.0, args.t_max, args.n_t)
    psi = initial_state(geom.N, args.init, args.alpha)
    for p in mine:
        f = point_file(args.out_dir, tag, p)
        if f.exists() and not args.overwrite:
            print(f'[kw asym] point {p}: {f} exists, skipped', flush=True)
            continue
        pt = pts[p]
        print(f"[kw asym] point {p}: panel {pt['panel']}, h = {pt['h']:g}, "
              f"D1 = {pt['d1']:g}, D2 = {pt['d2']:g}", flush=True)
        t0 = time.time()
        rho0 = np.outer(psi, psi.conj())
        rec = evolve_kw_asymmetry(geom, J, pt['h'], pt['d1'], pt['d2'], args.jumps,
                                  rho0, ts, rtol=args.rtol, atol=args.atol,
                                  log=lambda s: print(s, flush=True))
        del rho0
        wall = time.time() - t0
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.stem + f'.tmp{os.getpid()}.npz')
        np.savez(tmp, t=ts, **{k: rec[k] for k in ('a', 'b', 'c', 'tr', 'tr_hat',
                                                   'dS_gen', 'dS_even')},
                 panel=pt['panel'], h=pt['h'], d1=pt['d1'], d2=pt['d2'], J=J,
                 i=pt['i'], j=pt['j'], point=p, geometry=args.geometry,
                 jumps=args.jumps, init=args.init, alpha=args.alpha,
                 rtol=args.rtol, atol=args.atol, n_steps=rec['n_steps'],
                 nfev=rec['nfev'], wall=wall, version=KW_ASYM_VERSION)
        os.replace(tmp, f)
        print(f"[kw asym] point {p}: saved {f} ({rec['n_steps']} steps, "
              f"{rec['nfev']} L-evals, {wall:.0f} s); dS_even(t_max) = "
              f"{rec['dS_even'][-1]:.6g}, max|dS_gen - dS_even| = "
              f"{np.nanmax(np.abs(rec['dS_gen'] - rec['dS_even'])):.2e}", flush=True)


if __name__ == '__main__':
    main()
