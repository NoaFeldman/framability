"""
Seeded Heisenberg framability RATES of model10 (Shibata-Katsura dissipative
quantum Ising chain, bond generator with dim = 1) at d_ext = 4 and 8, built
from the closed-form frames of the rate-zero analysis.

Per grid point (Delta1, Delta2) and per d_ext m:

  1. closed-form frames (framability_rate_families.model10_frames):
       B   rescaled Pauli {I, xX, yY, Z}, y = h/D1, x = D2 y/J (padded to m)
       C   {I, xX, YZ (m-2)-gon} at x = D2/J (the finite circle frame)
       P   projector pair + regular XY (m-3)-gon (m >= 5)
  2. structured families (framability_rate_families.families_for(m)):
     affine-regular polygons in the YZ / XY plane plus axis columns,
     optimised over their 5-7 parameters by multistart Nelder-Mead from the
     analytic starts (m = 4: every {pure X, two YZ columns} frame, i.e. B and
     the skewed bases; m = 8: YZ hexagon, Z pair + XY pentagon, X pair + YZ
     pentagon, YZ square + XY pair)
  3. every frame already stored for the point by the base scan / refine /
     gopt pipelines (read-only; S_heis_4, S_heis_6, S_heis_8), and at m = 8
     the point's own m = 4 optimum padded (so rate_8 <= rate_4)
     --xfer_dirs (off by default): also the best frames of ANOTHER full-grid
     seeded run (e.g. the dim = 1 chain when this run is dim = 2) at the
     point and its neighbours within --xfer_radius, every source d_ext in
     XFER_SRC[m], padded to m (xfer_frames; read-only)
  4. unless a seed already sits at the floor mu* = 0:
     framability_rate_global.minimize_rate_global seeded with all of the
     above (its own seed library -- exact rescaled-Pauli optimum, pump / Z
     projectors, polygons -- is added internally), then DE + bundle polish +
     Nelder-Mead, certified with the independent per-column LP.

The value "structured" (fam_<m>) is the best of 1-2 alone, i.e. frames written
down directly; "rate" (rate_<m>) is the certified final value.

The closed forms are written for the dim = 1 bond.  A dim-d bond is the
dim = 1 bond at (D1/d, h/d), and the B / C weights depend on h/D1 and D2/J
only, so the same frames serve every --dim.

--fixed_frames (off by default) also stores the fixed-frame rates rate_pauli
and rate_stab3 (framability_rate_frames), for runs without a base scan.

Output: <out_dir>/<model>_seeded[_s<stride>]/pt_<ix:03d>_<iy:03d>.npz
    rate_<m>, S_<m>, label_<m>   certified rate, frame, origin of the frame
    fam_<m>, famS_<m>, famlabel_<m>   best closed-form / structured frame
    famval_<name>_<m>            per-family optimum,  famp_<name>_<m> params
    prev_best_<m>                best stored (old) value for the same m
    xfer_best_<m>                best transferred frame (--xfer_dirs only)
    rate_pauli, rate_stab3       (--fixed_frames only)
    t_<m>                        seconds
Nothing is written into the directories of the existing pipelines, so their
collect / refine scripts never see these files.

Usage:
    python scripts/model10_seeded_worker.py --task_id 0 --n_chunks 200
    python scripts/model10_seeded_worker.py --task_id 1300          # one point
    python scripts/model10_seeded_worker.py --self_check --task_id 0 --n_chunks 200
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
from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,     # noqa: E402
                                      MODEL10_J, MODEL10_H)
from framability_rate import RATE_VERSION, spectral_abscissa              # noqa: E402
from framability_rate_global import (minimize_rate_global, fit_columns,   # noqa: E402
                                     frame_rate_value, RATE_GLOBAL_VERSION)
from framability_rate_families import (families_for, model10_frames,      # noqa: E402
                                       model10_family_starts, optimize_family,
                                       self_check, RATE_FAMILIES_VERSION)
from framability_rate_frames import pauli_rate, stabilizer_3_rate         # noqa: E402
from rate_gopt_worker import stored_frames                                # noqa: E402

MODEL = 'model10'
D_EXTS_DEFAULT = (4, 8)
STORED_D_EXTS = (4, 6, 8)             # stored frame sizes used as seeds
STORED_DIRS_DEFAULT = ('results_model10_rate', 'results_model4_rate')
XFER_SRC = {4: (4,), 8: (4, 8), 12: (8, 12)}   # source d_ext transferred to m
TOL = 1e-9


def grid_vals(stride: int):
    spec = MODELS[MODEL]
    return (np.asarray(spec.p1_vals[::stride], float),
            np.asarray(spec.p2_vals[::stride], float))


def pt_dir_name(stride: int) -> str:
    """Per-point directory of this pipeline (stride-tagged off the full grid,
    so a preview run never shares file names with the full run)."""
    return f'{MODEL}_seeded' + ('' if stride == 1 else f'_s{stride}')


def generator(d1: float, d2: float, dim: int):
    spec = MODELS[MODEL]
    L = build_bond_lindbladian(*spec.build(d1, d2), dim).real
    return L.T


def load_stored(stored_dirs, ix_full: int, iy_full: int) -> dict:
    """{m: [(value, frame), ...]} from every existing <dir>/model10 of the
    earlier pipelines (base scan, refine rounds, gopt), full-grid indices."""
    out = {m: [] for m in STORED_D_EXTS}
    for d in stored_dirs:
        pt = Path(d) / MODEL
        if not pt.is_dir():
            continue
        for m, lst in stored_frames(pt, ix_full, iy_full, STORED_D_EXTS).items():
            out[m] += lst
    return out


def xfer_frames(xfer_dirs, ix_full: int, iy_full: int, m: int,
                radius: int = 1) -> list:
    """[(label, frame)]: the best frames another full-grid seeded run (e.g.
    the dim = 1 chain) holds at the full-grid point (ix_full, iy_full) and its
    neighbours within Chebyshev `radius`, for every source d_ext in
    XFER_SRC[m], padded to m columns.  Per source point the frame is the
    minimum over its worker / xeval / refine / margin files
    (model10_seeded_qrefine_worker.best_known) in <dir>/model10_seeded
    (d_ext 4, 8) or <dir>/model10_seeded_d12 (d_ext 12).  Read-only."""
    # imported here: model10_seeded_qrefine_worker imports this module
    from model10_seeded_qrefine_worker import best_known
    spec = MODELS[MODEL]
    out = []
    for d in xfer_dirs:
        for k in XFER_SRC.get(m, ()):
            name = pt_dir_name(1)
            pt = Path(d) / (name if k < 12 else name.replace('_seeded', '_seeded_d12', 1))
            if not pt.is_dir():
                continue
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    jx, jy = ix_full + dx, iy_full + dy
                    if not (0 <= jx < spec.N_X and 0 <= jy < spec.N_Y):
                        continue
                    _, S, _ = best_known(pt, jx, jy, f'rate_{k}', f'S_{k}')
                    if S is not None:
                        lab = f'xfer d{k}' + ('' if dx == dy == 0 else ' nb')
                        out.append((lab, fit_columns(S, m)))
    return out


def _certify(S, A) -> float:
    """Independent per-column LP value, clipped at the floor 0."""
    v = frame_rate_value(S, A, reference=True)
    return max(float(v), 0.0) if np.isfinite(v) and v > -TOL else float(v)


def compute_point(d1: float, d2: float, ix: int, iy: int, args) -> dict:
    A = generator(d1, d2, args.dim)
    J, h = MODEL10_J, MODEL10_H
    rng = np.random.default_rng([args.seed, ix, iy])
    out: dict = dict(delta1=d1, delta2=d2, dim=args.dim,
                     floor=spectral_abscissa(A))
    if args.fixed_frames:
        t0 = time.perf_counter()
        out['rate_pauli'] = pauli_rate(A.T)
        out['rate_stab3'] = stabilizer_3_rate(A.T)
        out['t_fixed'] = time.perf_counter() - t0
    stored = load_stored(args.stored_dirs, ix * args.stride, iy * args.stride)
    prev_S, prev_label = None, None

    for m in sorted(args.d_exts):
        t0 = time.perf_counter()
        cands = []                                       # (label, frame)

        # ---- 1. closed-form frames -------------------------------------
        for name, S in model10_frames(d1, d2, J, h, m).items():
            cands.append((f'analytic {name}', S))

        # ---- 2. structured families ------------------------------------
        for fam in families_for(m):
            v, S, p, _ = optimize_family(
                fam, A, model10_family_starts(fam, d1, d2, J, h),
                n_random=args.fam_random, n_polish=args.fam_polish,
                maxfev=args.fam_maxfev4 if m == 4 else args.fam_maxfev,
                rng=rng)
            out[f'famval_{fam.name}_{m}'] = v
            out[f'famp_{fam.name}_{m}'] = p
            cands.append((f'family {fam.name}', S))
            if v <= TOL:
                break                                    # already at the floor

        n_struct = len(cands)
        # ---- 3. stored frames, padded smaller optimum, transfers --------
        for k in STORED_D_EXTS:
            for _, S in stored[k]:
                cands.append(('stored', fit_columns(S, m)))
        if prev_S is not None:
            cands.append((prev_label, fit_columns(prev_S, m)))
        n_pre_xfer = len(cands)
        cands += xfer_frames(args.xfer_dirs, ix * args.stride, iy * args.stride,
                             m, args.xfer_radius)

        vals = np.array([frame_rate_value(S, A) for _, S in cands], float)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        if len(cands) > n_pre_xfer:
            out[f'xfer_best_{m}'] = float(vals[n_pre_xfer:].min())
        i_s = int(np.argmin(vals[:n_struct]))
        out[f'fam_{m}'] = _certify(cands[i_s][1], A)
        out[f'famS_{m}'] = cands[i_s][1]
        out[f'famlabel_{m}'] = cands[i_s][0]
        prev_vals = [v for k in STORED_D_EXTS if k <= m for v, _ in stored[k]
                     if np.isfinite(v)]
        out[f'prev_best_{m}'] = min(prev_vals) if prev_vals else np.nan

        i0 = int(np.argmin(vals))
        label, S_best, v_seed = cands[i0][0], cands[i0][1], float(vals[i0])
        mu = _certify(S_best, A) if v_seed <= TOL else np.inf
        if mu <= TOL:
            out[f'mode_{m}'] = 'seed at floor'
        else:
            # ---- 4. seeded global search --------------------------------
            basis = (m == 4)
            S_g, mu_g, info = minimize_rate_global(
                A, m, seeds=[S for _, S in cands], seed=args.seed + m,
                de_popsize=args.de_popsize4 if basis else args.de_popsize,
                de_maxiter=args.de_maxiter4 if basis else args.de_maxiter,
                bundle_iters=args.bundle_iters,
                nm_maxfev=args.nm_maxfev4 if basis else args.nm_maxfev)
            mu_seed_cert = _certify(S_best, A)
            if np.isfinite(mu_g) and mu_g < mu_seed_cert - TOL:
                name = str(info['seed_name'])
                if name.startswith('seed') and name[4:].isdigit():
                    base = cands[int(name[4:])][0]
                else:
                    base = 'library'
                label = base if mu_g >= info['mu_seed'] - TOL else f'{base} +opt'
                S_best, mu = S_g, float(mu_g)
            else:
                mu = mu_seed_cert
            out[f'mode_{m}'] = 'global'
            out[f'mu_de_{m}'] = info['mu_de']
            out[f'mu_bundle_{m}'] = info['mu_bundle']

        out[f'rate_{m}'] = mu
        out[f'S_{m}'] = S_best
        out[f'label_{m}'] = label
        out[f't_{m}'] = time.perf_counter() - t0
        prev_S, prev_label = S_best, (label if label.startswith('d4 ')
                                      else f'd4 {label}')
    return out


def run_point(ix: int, iy: int, args) -> None:
    p1, p2 = grid_vals(args.stride)
    d1, d2 = float(p1[ix]), float(p2[iy])
    pt = Path(args.out_dir) / pt_dir_name(args.stride)
    out_f = pt / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists() and not args.force:
        print(f'[skip] {out_f.name}', flush=True)
        return
    t0 = time.perf_counter()
    print(f'[{MODEL} seeded] ({ix},{iy}) delta1={d1:.3f} delta2={d2:.3f}',
          flush=True)
    try:
        res = compute_point(d1, d2, ix, iy, args)
    except Exception as e:                                  # noqa: BLE001
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        return
    pt.mkdir(parents=True, exist_ok=True)
    tmp = out_f.with_suffix('.tmp.npz')
    np.savez(tmp, model=MODEL, ix=ix, iy=iy, stride=args.stride,
             d_exts=np.array(sorted(args.d_exts)), rate_version=RATE_VERSION,
             rate_global_version=RATE_GLOBAL_VERSION,
             rate_families_version=RATE_FAMILIES_VERSION, **res)
    os.replace(tmp, out_f)                    # never leave a half-written file
    msg = '  '.join(f'd{m}={res[f"rate_{m}"]:.5f} [{res[f"label_{m}"]}] '
                    f'(struct {res[f"fam_{m}"]:.5f}, old {res[f"prev_best_{m}"]:.5f})'
                    for m in sorted(args.d_exts))
    print(f'  saved {out_f.name}  {msg}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='<=1: task_id is a flat grid index; otherwise the grid '
                        'is strided over n_chunks array tasks')
    p.add_argument('--out_dir', type=str, default='results_model10_rate')
    p.add_argument('--stored_dirs', type=str, nargs='*',
                   default=list(STORED_DIRS_DEFAULT),
                   help='existing rate dirs whose <dir>/model10 frames seed '
                        'the search (read-only; missing dirs are skipped)')
    p.add_argument('--xfer_dirs', type=str, nargs='*', default=[],
                   help='roots of other full-grid seeded runs (e.g. the dim=1 '
                        'results_model10_rate) whose best frames near the point '
                        '(d_ext in XFER_SRC[m]) seed the search (read-only; '
                        'default none)')
    p.add_argument('--xfer_radius', type=int, default=1,
                   help='Chebyshev radius (full-grid steps) of the transfers')
    p.add_argument('--fixed_frames', action='store_true',
                   help='also store rate_pauli and rate_stab3')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--d_exts', type=int, nargs='+', default=list(D_EXTS_DEFAULT))
    p.add_argument('--dim', type=int, default=None,
                   help="bond convention dimension (default: model10's, 1)")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fam_random', type=int, default=12,
                   help='random starts per structured family')
    p.add_argument('--fam_polish', type=int, default=3,
                   help='Nelder-Mead polishes per structured family')
    p.add_argument('--fam_maxfev4', type=int, default=1500,
                   help='NM evaluations per polish, 4-column (closed form)')
    p.add_argument('--fam_maxfev', type=int, default=200,
                   help='NM evaluations per polish, LP frames')
    p.add_argument('--de_popsize4', type=int, default=24)
    p.add_argument('--de_maxiter4', type=int, default=300)
    p.add_argument('--nm_maxfev4', type=int, default=20000)
    p.add_argument('--de_popsize', type=int, default=6)
    p.add_argument('--de_maxiter', type=int, default=20)
    p.add_argument('--bundle_iters', type=int, default=80)
    p.add_argument('--nm_maxfev', type=int, default=400)
    p.add_argument('--force', action='store_true',
                   help='recompute points that already have a file')
    p.add_argument('--self_check', action='store_true',
                   help='run framability_rate_families.self_check first '
                        '(chunk 0 only; logs, never aborts)')
    args = p.parse_args()
    if args.dim is None:
        args.dim = MODELS[MODEL].dim

    if args.self_check and args.task_id == 0:
        print('[self-check] framability_rate_families:', flush=True)
        try:
            print('[self-check] ' + ('passed' if self_check() else 'FAILED'),
                  flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f'[self-check] ERROR {type(e).__name__}: {e}', flush=True)

    p1, p2 = grid_vals(args.stride)
    nx, ny = len(p1), len(p2)
    n_total = nx * ny
    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            sys.exit(f'task_id must be in [0, {n_total})')
        run_point(args.task_id // ny, args.task_id % ny, args)
        return
    if not (0 <= args.task_id < args.n_chunks):
        sys.exit(f'chunk id must be in [0, {args.n_chunks})')
    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {MODEL}: {len(ids)} of '
          f'{n_total} points ({nx}x{ny}), d_ext={sorted(args.d_exts)}', flush=True)
    for pid in ids:
        run_point(pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
