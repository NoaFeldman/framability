"""
Reweighted-Pauli rate of the open Dicke model (uniform and disordered omega0)
as a function of the smallest allowed one-site weight w_min
(framability_rate_pauli_weights.py).

Why: the Heisenberg-picture normalisation may be relaxed to w_min ~ N^{-c}
(polynomial prefactor).  For a chosen model point the curve F*(w_min) says
how much the pair-averaged rate drops when the frame may use smaller
weights, and w_star (the largest w_min with F* = 0) the system size
N = w_star^{-1/c} beyond which the reweighted Pauli frame is at rate zero.

Combos (task t of --n_tasks takes combos t, t + n_tasks, ...):
  points  the labelled points of POINTS (below), or --points LABEL ...
  grid    with --grid: every point of MODELS['<family>_N<N>'] at every
          width of --grid_widths (w_star maps; optional)

Per combo, on the decreasing grid w_min = logspace(0, --log10_wmin_end, --n_w):
  shared   one weight triple (w_X, w_Y, w_Z) on every atom
  species  one triple per frequency class (width > 0 only; atoms are
           physically different, so A (x) B pair frames are allowed),
           warm-started from the shared optimum of the same level, so
           F_species <= F_shared
  F*(w), s*(w) (min-max, <= 0 certifies F = 0), optimal log-weights per
  level, w_star (bisection), the plain Pauli rate (checked against
  framability_rate_disorder.pauli_avg_rate at w = 1), the pair floors
  (spectral abscissae: F >= sum_p pw_p max(floor_p, 0) in ANY frame), and
  the generic per-column LP re-evaluation of every level's weights.

Output (atomic, one file per combo; existing files are skipped unless --force):
  <out_root>/<family>_N<N>_M<M>/points/<label>.npz
  <out_root>/<family>_N<N>_M<M>_w<width:.2f>/pt_<ix:03d>_<iy:03d>.npz
keys: family, N, M, M_eff, width, omegas, pairs, pair_w, lam_ratio, y,
      gamma_down, gamma_phi, label, expect, w_grid, F_pauli, floor_pairs,
      floor_avg, floor_max, version, and per mode in (shared, species):
      F_<mode>, s_<mode>, xF_<mode>, xs_<mode>, wstar_<mode>,
      F_ref_<mode> (generic LP of xF), ref_dev_<mode>, nonmono_<mode>, t_<mode>

Usage (nothing runs by default; the slurm script passes the task id):
    python scripts/dicke_pauli_weight_worker.py --count
    python scripts/dicke_pauli_weight_worker.py --task_id 0 --n_tasks 11
    python scripts/dicke_pauli_weight_worker.py --grid --task_id 5 --n_tasks 200
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from trotter_lindbladian_scan import MODELS                              # noqa: E402
from framability_rate_disorder import (build_ensemble, check_uniform,    # noqa: E402
                                       pauli_avg_rate, pair_floors)
from framability_rate_pauli_weights import (PAULI_WEIGHTS_VERSION,       # noqa: E402
                                            WeightModel, sweep, w_star,
                                            reference_rate)

# ---------------------------------------------------------------------------
#  Chosen points (dicke_dec, kappa = 1, N = 100; grid values of the panels
#  results_dicke_disorder/dicke_dec_N100_M4_panels.npz).  "expect" is the
#  hand prediction made BEFORE the run, from the stored panels:
#    zero        Pauli > 0 (stuck on the gamma_down-independent plateau: the
#                decay damping of Z is cancelled by the Z -> I jump at unit
#                weight), d4 optimum = 0, floor = 0  -> reweighting should
#                reach F = 0 at a finite w_star
#    edge        d4 optimum at (or just off) its zero line -> w_star small or
#                not reached: the w_min-dependent case
#    plateau     floor = 0 but the d4 optimum (which contains every
#                reweighted Pauli frame) is clearly > 0 -> expected F_inf > 0
#                (not certified)
#    certified   pair floor > 0: F >= floor_avg > 0 in ANY frame, at any
#                weights -> non-simulable regime (certified)
# ---------------------------------------------------------------------------
#  label: (width, lam/lam_c, gamma_down, expect, note)
POINTS = {
    # uniform omega0 (width 0): Pauli / d4 / floor at N = 100
    'U1': (0.0, 0.6, 0.12, 'zero',      'Pauli 0.08, d4 0, floor 0'),
    'U2': (0.0, 1.0, 0.20, 'zero',      'Pauli 0.22, d4 0, floor 0'),
    'U3': (0.0, 1.4, 0.30, 'zero',      'Pauli 0.43, d4 0 (from 0.24), floor 0'),
    'U4': (0.0, 1.6, 0.28, 'edge',      'Pauli 0.56, d4 0 exactly at 0.28, floor 0'),
    'U5': (0.0, 1.0, 0.08, 'plateau',   'Pauli 0.26, d4 0.13, floor 0'),
    'U6': (0.0, 1.8, 0.10, 'certified', 'Pauli 0.70, d4 0.44, floor 0.32'),
    # disordered omega0 (width 0.5, M = 4): Pauli / d4 shared / d4 species / floor max
    'D1': (0.5, 0.6, 0.16, 'zero',      'Pauli 0.08, d4 0/0, floor 0'),
    'D2': (0.5, 1.0, 0.24, 'zero',      'Pauli 0.22, d4 0/0, floor 0'),
    'D3': (0.5, 1.4, 0.30, 'edge',      'Pauli 0.43, d4 shared 0.01 / species 0, floor 0'),
    'D4': (0.5, 1.0, 0.06, 'plateau',   'Pauli 0.30, d4 0.18/0.17, floor 0'),
    'D5': (0.5, 1.8, 0.10, 'certified', 'Pauli 0.72, d4 0.46/0.46, floor 0.33'),
}


def save_atomic(path: Path, rec: dict) -> None:
    tmp = path.with_suffix('.tmp.npz')
    np.savez(str(tmp), **rec)
    os.replace(tmp, path)


def point_path(root: Path, family: str, N: int, M: int, label: str) -> Path:
    return root / f'{family}_N{N}_M{M}' / 'points' / f'{label}.npz'


def grid_path(root: Path, family: str, N: int, M: int, width: float, ix: int, iy: int) -> Path:
    return root / f'{family}_N{N}_M{M}_w{width:.2f}' / f'pt_{ix:03d}_{iy:03d}.npz'


def run_mode(model, pair_classes, w_grid, args, warm=None, seed=0):
    t0 = time.time()
    res = sweep(model, w_grid, n_random=args.n_random, seed=seed, warm_x=warm,
                maxiter=args.maxiter)
    ws = w_star(model, res, n_bisect=args.n_bisect)
    F_ref = np.array([reference_rate(model, x, pair_classes) for x in res['x_F']])
    return res, ws, F_ref, time.time() - t0


def run_combo(family, N, M, width, lam_ratio, y, path: Path, args, *, label='', expect='',
              ix=-1, iy=-1) -> None:
    if path.exists() and not args.force:
        print(f'[skip] {path.relative_to(REPO)}', flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = f'{family} N={N} w={width:.2f} lam/lam_c={lam_ratio:g} y={y:g} {label}'
    ens = build_ensemble(family, N, width, M, lam_ratio, y)
    w_grid = np.logspace(0.0, args.log10_wmin_end, args.n_w)
    F_pauli, pairs_pauli = pauli_avg_rate(ens)
    fl = pair_floors(ens)
    rec = dict(family=family, N=N, M=M, M_eff=ens.M_eff, width=width, omegas=ens.omegas,
               pairs=np.array(ens.pairs, int), pair_w=ens.pw, lam_ratio=lam_ratio, y=y,
               ix=ix, iy=iy, gamma_down=ens.gamma_down, gamma_phi=ens.gamma_phi,
               label=label, expect=expect, w_grid=w_grid, F_pauli=F_pauli,
               pairs_pauli=pairs_pauli, floor_pairs=fl,
               floor_avg=float(ens.pw @ np.maximum(fl, 0.0)), floor_max=float(np.max(fl)),
               version=PAULI_WEIGHTS_VERSION)
    print(f'[start] {tag}: pairs={ens.P} Pauli F={F_pauli:.5g} floor avg='
          f'{rec["floor_avg"]:.3g} max={rec["floor_max"]:.3g}', flush=True)

    modes = [('shared', [(0, 0)] * ens.P, 1)]
    if ens.M_eff > 1:
        modes.append(('species', list(ens.pairs), ens.M_eff))
    warm = None
    for mode, pcls, ncls in modes:
        model = WeightModel(ens.As, ens.pw, pcls, ncls)
        F_unit = model.F(np.zeros(model.n_x))
        if abs(F_unit - F_pauli) > 1e-9:
            raise RuntimeError(f'{tag} {mode}: unit-weight F {F_unit:.12g} != Pauli '
                               f'closed form {F_pauli:.12g}')
        res, ws, F_ref, dt = run_mode(model, pcls, w_grid, args, warm=warm,
                                      seed=args.seed + (0 if mode == 'shared' else 1))
        dev = float(np.max(np.abs(F_ref - res['F_raw'])))     # same x as F_ref
        rec.update({f'F_{mode}': res['F'], f's_{mode}': res['s'],
                    f'xF_{mode}': res['x_F'], f'xs_{mode}': res['x_s'],
                    f'wstar_{mode}': ws, f'F_ref_{mode}': F_ref, f'ref_dev_{mode}': dev,
                    f'nonmono_{mode}': res['n_nonmono_F'] + res['n_nonmono_s'],
                    f't_{mode}': dt})
        if np.any(res['F'] < rec['floor_avg'] - 1e-7):
            print(f'[WARN] {tag} {mode}: F below the pair floor average '
                  f'(min F {res["F"].min():.4g} < {rec["floor_avg"]:.4g}) -> bug', flush=True)
        if dev > 1e-6:
            print(f'[WARN] {tag} {mode}: generic LP differs from the closed form by '
                  f'{dev:.2e}', flush=True)
        print(f'[{mode}] {tag}: F(1)={res["F"][0]:.4g} F(w_end)={res["F"][-1]:.4g} '
              f's(w_end)={res["s"][-1]:.3g} w*={ws:.3g} LP dev={dev:.1e} '
              f'nonmono={rec[f"nonmono_{mode}"]} ({dt:.0f}s)', flush=True)
        if mode == 'shared' and ens.M_eff > 1:
            warm = np.tile(res['x_F'], (1, ens.M_eff))
    save_atomic(path, rec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--family', default='dicke_dec')
    ap.add_argument('--N', type=int, default=100)
    ap.add_argument('--M', type=int, default=4, help='species (frequency classes)')
    ap.add_argument('--points', nargs='*', default=None,
                    help='labels of POINTS to run (default: all); --points with no '
                         'label runs none (grid only)')
    ap.add_argument('--grid', action='store_true', help='also every grid point (w_star maps)')
    ap.add_argument('--grid_widths', nargs='+', type=float, default=[0.0, 0.5])
    ap.add_argument('--log10_wmin_end', type=float, default=-8.0)
    ap.add_argument('--n_w', type=int, default=33, help='w_min levels (4 per decade by default)')
    ap.add_argument('--n_random', type=int, default=3, help='random restarts per level')
    ap.add_argument('--n_bisect', type=int, default=30)
    ap.add_argument('--maxiter', type=int, default=500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--task_id', type=int, default=None)
    ap.add_argument('--n_tasks', type=int, default=1)
    ap.add_argument('--out_root', default='results_dicke_pauli_weight')
    ap.add_argument('--count', action='store_true', help='print the number of combos and exit')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    root = REPO / args.out_root
    spec = MODELS[f'{args.family}_N{args.N}']
    labels = list(POINTS) if args.points is None else list(args.points)
    for lb in labels:
        if lb not in POINTS:
            raise KeyError(f'unknown point {lb!r}; known: {list(POINTS)}')
    combos = [('point', lb) for lb in labels]
    if args.grid:
        combos += [('grid', (w, ix, iy)) for w in args.grid_widths
                   for ix in range(spec.N_X) for iy in range(spec.N_Y)]
    if args.count:
        print(len(combos))
        return 0
    if args.task_id is None:
        ap.error('--task_id is required (or --count)')

    dev = check_uniform(args.family, args.N)
    print(f'[check] {args.family}_N{args.N}: w = 0 pair patch = registered bond '
          f'generator (max dev {dev:.1e})', flush=True)
    mine = combos[args.task_id::args.n_tasks]
    print(f'[task {args.task_id}/{args.n_tasks}] {len(mine)} of {len(combos)} combos', flush=True)
    n_err = 0
    for kind, c in mine:
        try:
            if kind == 'point':
                width, lr, y, expect, note = POINTS[c]
                print(f'[point] {c}: expect {expect} ({note})', flush=True)
                run_combo(args.family, args.N, args.M, width, lr, y,
                          point_path(root, args.family, args.N, args.M, c), args,
                          label=c, expect=expect)
            else:
                width, ix, iy = c
                run_combo(args.family, args.N, args.M, width, float(spec.p1_vals[ix]),
                          float(spec.p2_vals[iy]),
                          grid_path(root, args.family, args.N, args.M, width, ix, iy), args,
                          ix=ix, iy=iy)
        except Exception:                                   # noqa: BLE001
            n_err += 1
            print(f'[error] {kind} {c}:', flush=True)
            traceback.print_exc()
            sys.stdout.flush()
    print(f'[task {args.task_id}] finished, {n_err} errors', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
