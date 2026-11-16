"""
Average framability rate of the open Dicke model with unequal atomic
frequencies omega0 (framability_rate_disorder), one grid point per combo.

Combos: --families x --ns x --widths x the uniform model's grid
(MODELS['<family>_N<N>'], lam/lam_c x second axis; lam_c of the MEAN omega0),
enumerated family-major; task t of --n_tasks takes combos t, t + n_tasks, ...
so every task gets a mix of widths (w = 0 points are cheap controls).

Per combo, for every d_ext in --d_exts (ascending):
    seeds   stored uniform-model frames of the same grid point (results of
            scripts/submit_dicke_rate.sh, the run's N first, then the other N
            of the family; --seed_ns overrides), the single-qubit seed
            library, the padded smaller-d optimum of this run
    shared  one frame on every atom: best seed + bundle polish of the best
            --n_starts seeds + --n_kicks kicked restarts
    species one frame per species, started from the shared optimum (so
            F_species <= F_shared), + bundle polish + kicks
    certify per-column reference LP of both final assignments (--no_certify skips)
plus the Pauli average rate and the per-pair floors (spectral abscissae).

Output (one file per combo, rewritten atomically after every d_ext; finished
d_ext are skipped on a re-run unless --force):
    <out_root>/<family>_N<N>_M<M>_w<width:.2f>/pt_<ix:03d>_<iy:03d>.npz
keys: family, N, M, M_eff, width, omegas, pairs, pair_w, lam_ratio, y,
      gamma_down, gamma_phi, version, F_pauli, pairs_pauli, floor_pairs,
      floor_avg (sum w max(floor, 0)), floor_max, and per d (suffix _<d>):
      F_seed, seed_name, F_shared, S_shared, pairs_shared, worst_shared,
      F_species, S_species, pairs_species, worst_species,
      F_shared_ref, F_species_ref, F_shared_cert, F_species_cert, t.
Width 0 runs the uniform model (M_eff = 1, F_species = F_shared): the
control against the stored uniform panels (the collect script compares).

Subset of the grid (the d_ext scan of the paper, scripts/submit_paper_dext_scan.sh):
    --points 0.5:0.1 1.0:0.1   only these (lam/lam_c : second-axis) grid values
    --init_root DIR            a combo with no file yet starts from the record of
                               DIR/<same folder>/<same file> (the main run), so
                               its finished d_ext are reused and only the new
                               d_ext are optimised (seeded by the padded
                               next-smaller optimum, as usual)

Usage (nothing runs by default; the slurm script passes the task id):
    python scripts/dicke_disorder_rate_worker.py --task_id 0 --n_tasks 200
    python scripts/dicke_disorder_rate_worker.py --families dicke_dec --ns 20 \
        --widths 0.5 --task_id 3 --n_tasks 200 --d_exts 4 8 12
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
from trotter_lindbladian_scan import MODELS, MODEL_DICKE_PARAMS     # noqa: E402
from framability_rate_dimer import stored_product_frames            # noqa: E402
from framability_rate_disorder import (DISORDER_VERSION, build_ensemble,  # noqa: E402
                                       check_uniform, optimise_point,
                                       pauli_avg_rate, pair_floors)


def out_dir_of(root: Path, family: str, N: int, M: int, width: float) -> Path:
    return root / f'{family}_N{N}_M{M}_w{width:.2f}'


def seed_dirs(family: str, N: int, seed_ns) -> list:
    """Uniform-model frame folders, the run's N first, then the nearest N."""
    avail = sorted({p['N'] for p in MODEL_DICKE_PARAMS.values() if p['family'] == family})
    ns = list(seed_ns) if seed_ns else sorted(avail, key=lambda n: (n != N, abs(np.log(n / N))))
    out = []
    for n in ns:
        base = REPO / f'results_{family}_N{n}_seeded'
        out += [base / f'{family}_N{n}_seeded', base / f'{family}_N{n}_seeded_d12']
    return out


def stored_frames_for(dirs, ix: int, iy: int, m: int) -> list:
    """Stored single-qubit frames of grid point (ix, iy) with <= m columns
    (the m-column ones first), from the folders in order."""
    exact, smaller = [], []
    for d in dirs:
        frames, _ = stored_product_frames([d], ix, iy)
        exact += frames.get(m, [])
        for k in sorted(frames):
            if k < m:
                smaller += frames[k]
    return exact + smaller


def save_atomic(path: Path, rec: dict) -> None:
    tmp = path.with_suffix('.tmp.npz')
    np.savez(str(tmp), **rec)
    os.replace(tmp, path)


def run_combo(family, N, width, M, ix, iy, args, root: Path) -> None:
    spec = MODELS[f'{family}_N{N}']
    x, y = float(spec.p1_vals[ix]), float(spec.p2_vals[iy])
    od = out_dir_of(root, family, N, M, width)
    od.mkdir(parents=True, exist_ok=True)
    path = od / f'pt_{ix:03d}_{iy:03d}.npz'
    rec = {}
    init = (out_dir_of(REPO / args.init_root, family, N, M, width) / path.name
            if args.init_root else None)
    if path.exists() and not args.force:
        with np.load(path, allow_pickle=True) as z:
            rec = {k: z[k] for k in z.files}
    elif init is not None and init.exists():
        with np.load(init, allow_pickle=True) as z:
            rec = {k: z[k] for k in z.files}
        print(f'[init] {path.name}: d_ext from {init}', flush=True)
    d_exts = sorted(args.d_exts)
    todo = [m for m in d_exts if f'F_shared_cert_{m}' not in rec]
    tag = f'{family} N={N} w={width:.2f} ({ix},{iy}) lam/lam_c={x:g} y={y:g}'
    if not todo:
        print(f'[skip] {tag}: done', flush=True)
        return
    t0 = time.time()
    ens = build_ensemble(family, N, width, M, x, y)
    if 'F_pauli' not in rec:
        Fp, pp = pauli_avg_rate(ens)
        fl = pair_floors(ens)
        rec.update(family=family, N=N, M=M, M_eff=ens.M_eff, width=width,
                   omegas=ens.omegas, pairs=np.array(ens.pairs, int), pair_w=ens.pw,
                   lam_ratio=x, y=y, ix=ix, iy=iy, gamma_down=ens.gamma_down,
                   gamma_phi=ens.gamma_phi, version=DISORDER_VERSION,
                   F_pauli=Fp, pairs_pauli=pp, floor_pairs=fl,
                   floor_avg=float(ens.pw @ np.maximum(fl, 0.0)),
                   floor_max=float(np.max(fl)))
    print(f'[start] {tag}: M_eff={ens.M_eff} pairs={ens.P} Pauli F={float(rec["F_pauli"]):.5g} '
          f'floor max={float(rec["floor_max"]):.3g}', flush=True)
    dirs = seed_dirs(family, N, args.seed_ns)
    rng = np.random.default_rng(1000003 * ix + 1009 * iy + int(round(1e4 * width)) + N)
    prev_sh = prev_sp = None
    for m in d_exts:
        if f'F_shared_cert_{m}' in rec:
            prev_sh = np.asarray(rec[f'S_shared_{m}'], float)
            prev_sp = list(np.asarray(rec[f'S_species_{m}'], float))
            continue
        r = optimise_point(ens, m, stored_frames=stored_frames_for(dirs, ix, iy, m),
                           prev_shared=prev_sh, prev_species=prev_sp,
                           n_starts=args.n_starts, bundle_iters=args.bundle_iters,
                           n_kicks=args.n_kicks, kick_scales=tuple(args.kick_scales),
                           max_seed_eval=args.max_seed_eval,
                           certify=not args.no_certify, rng=rng, verbose=args.verbose)
        rec.update(r)
        save_atomic(path, rec)
        prev_sh = r[f'S_shared_{m}']
        prev_sp = list(r[f'S_species_{m}'])
        print(f'  d={m}: seed {r[f"F_seed_{m}"]:.5g} ({r[f"seed_name_{m}"]})  '
              f'shared {r[f"F_shared_cert_{m}"]:.5g}  species {r[f"F_species_cert_{m}"]:.5g}  '
              f'worst pair {r[f"worst_shared_{m}"]:.4g}/{r[f"worst_species_{m}"]:.4g}  '
              f'{r[f"t_{m}"]:.0f}s', flush=True)
    print(f'[done] {tag}: {time.time() - t0:.0f}s', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--families', nargs='+', default=['dicke_dec', 'dicke_dec_k025'])
    ap.add_argument('--ns', nargs='+', type=int, default=[100])
    ap.add_argument('--widths', nargs='+', type=float, default=[0.0, 0.2, 0.5, 0.8])
    ap.add_argument('--M', type=int, default=4, help='species (frequency classes)')
    ap.add_argument('--d_exts', nargs='+', type=int, default=[4, 8])
    ap.add_argument('--task_id', type=int, required=True)
    ap.add_argument('--n_tasks', type=int, default=200)
    ap.add_argument('--out_root', default='results_dicke_disorder')
    ap.add_argument('--seed_ns', nargs='*', type=int, default=None,
                    help='N whose uniform frames seed the search (default: run N first, then all)')
    ap.add_argument('--n_starts', type=int, default=3)
    ap.add_argument('--bundle_iters', type=int, default=60)
    ap.add_argument('--n_kicks', type=int, default=1)
    ap.add_argument('--kick_scales', nargs='+', type=float, default=[0.03])
    ap.add_argument('--max_seed_eval', type=int, default=40)
    ap.add_argument('--points', nargs='*', default=None,
                    help='only these grid points, as P1:P2 values (default: whole grid)')
    ap.add_argument('--init_root', default=None,
                    help='start new files from the same-named record under this root')
    ap.add_argument('--no_certify', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    root = REPO / args.out_root
    # the disordered builder must contain the uniform model (fail fast)
    for fam in args.families:
        for N in args.ns:
            dev = check_uniform(fam, N)
            print(f'[check] {fam}_N{N}: w = 0 pair patch = registered bond generator '
                  f'(max dev {dev:.1e})', flush=True)

    def on_grid(vals, v):
        i = int(np.argmin(np.abs(np.asarray(vals, float) - v)))
        if abs(vals[i] - v) > 1e-9:
            raise ValueError(f'{v} is not on the grid {list(vals)}')
        return i

    combos = []
    for fam in args.families:
        for N in args.ns:
            spec = MODELS[f'{fam}_N{N}']
            if args.points:
                pts = []
                for s_pt in args.points:
                    a, b = (float(v) for v in s_pt.split(':'))
                    pts.append((on_grid(spec.p1_vals, a), on_grid(spec.p2_vals, b)))
            else:
                pts = [(ix, iy) for ix in range(spec.N_X) for iy in range(spec.N_Y)]
            for w in args.widths:
                for ix, iy in pts:
                    combos.append((fam, N, w, ix, iy))
    mine = combos[args.task_id::args.n_tasks]
    print(f'[task {args.task_id}/{args.n_tasks}] {len(mine)} of {len(combos)} combos', flush=True)
    n_err = 0
    for fam, N, w, ix, iy in mine:
        try:
            run_combo(fam, N, w, args.M, ix, iy, args, root)
        except Exception:                                   # noqa: BLE001
            n_err += 1
            print(f'[error] {fam} N={N} w={w} ({ix},{iy}):', flush=True)
            traceback.print_exc()
            sys.stdout.flush()
    print(f'[task {args.task_id}] finished, {n_err} errors', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
