"""
Heisenberg framability RATES with NON-HERMITIAN frame elements and complex
coefficients (framability_rate_global's default since 2.0) on a coarsened grid
of several models at once: models 4, 10, 14, 15 and the two uniform Dicke
gamma_down families (kappa = 1, 0.25) at N = 100 by default.

Grid: every model's own grid of trotter_lindbladian_scan.MODELS taken with
--stride (default 2 = twice the step, the same points as every second point
of the full grid, so the earlier Hermitian runs line up point by point).  All
points of all models form ONE flat list; array task t of n_chunks takes points
t, t + n_chunks, ..., so every task gets a mix of models (balanced load).

Stages (--stage):
  seed    per point: fixed real frames (Pauli, stabilizer-3; complex
          coefficients cannot improve a real frame, so these are the final
          values), then d_ext = 4 and 8 with complex frames:
            d4  minimize_rate_global (closed form, complex DE over 24
                parameters + Nelder-Mead), seeded with the point's Hermitian
                optima of the earlier run and the clock-ladder frames
            d8  multistart search: every seed (library incl. clock ladders,
                Hermitian optima of the earlier run at d4 / 8 / 12, this
                run's complex d4 padded) evaluated; the best --n_starts are
                bundle-polished (real seeds after small complex kicks, since a
                real frame is stationary in the imaginary directions), then
                Nelder-Mead, then certification with the per-column LP
          -> <out_root>/<model>_s<stride>/pt_<ix>_<iy>.npz
  d12     d_ext = 12, the same multistart search seeded with this run's
          complex d8 (padded), the earlier Hermitian d12 / d8 optima and the
          library   -> pt_<ix>_<iy>_d12.npz
  refine  one round (--round) at --d_ext: every point with a positive rate
          takes the best of its own frame, its neighbours' frames (Chebyshev
          radius --radius on the coarse grid) and its own smaller-d_ext frame
          padded, then basin hopping (complex Gaussian kicks at --scales, each
          followed by a bundle polish) until --hops or --max_seconds;
          written only if the certified rate drops
          -> pt_<ix>_<iy>_refine_d<m>_r<NN>.npz  (rate_<m>, rate_<m>_prev)

Every stored rate is certified (independent per-column LP) and is a rigorous
upper bound of the complex-frame optimum; seeding with the Hermitian optima
makes it <= the earlier Hermitian value at every point, and padding makes it
monotone in d_ext.  Earlier runs are read-only (default roots:
results_<model>_seeded, model10: results_model10_rate; folders
<model>_seeded (d_ext 4, 8) and <model>_seeded_d12).

Keys of pt_<ix>_<iy>.npz: model, ix, iy, ix_full, iy_full, stride, p1, p2,
<p1_name>, <p2_name>, dim, floor, rate_pauli, rate_stab3, real_<m> (best
earlier Hermitian value, m = 4, 8, 12), rate_<m>, S_<m> (complex 4 x m),
label_<m>, imag_<m> (||Im S|| / ||S||), t_<m>, versions.

Usage:
    python scripts/rate_complex_worker.py --stage seed --task_id 0 --n_chunks 200
    python scripts/rate_complex_worker.py --stage seed --point model4:3:5
    python scripts/rate_complex_worker.py --stage refine --d_ext 8 --round 1 \\
        --task_id 0 --n_chunks 200
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
try:
    from numpy.exceptions import ComplexWarning
except ImportError:                                         # numpy < 1.25
    ComplexWarning = np.ComplexWarning

# a complex frame must never be cast to float silently
warnings.simplefilter('error', ComplexWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, build_bond_lindbladian        # noqa: E402
from optimize_framability import _FIXED_COLS, N_FIXED_COLS                 # noqa: E402
from framability_rate import (RATE_VERSION, spectral_abscissa,             # noqa: E402
                              _as_frame_array, is_complex_frame,
                              project_frame_columns)
from framability_rate_global import (minimize_rate_global, fit_columns,    # noqa: E402
                                     frame_rate_value, bundle_rate_polish,
                                     seed_library, x_from_frame, frame_from_x,
                                     RATE_GLOBAL_VERSION)
from framability_rate_frames import pauli_rate, stabilizer_3_rate         # noqa: E402

PIPELINE_VERSION = '1.0-complex-panels'
DEFAULT_MODELS = ('model4', 'model10', 'model14', 'model15',
                  'dicke_dec_N100', 'dicke_dec_k025_N100')
D_EXTS = (4, 8, 12)
TOL = 1e-9
RATE_TOL = 1e-6          # a point counts as rate 0 below this (refine targets)


# ---------------------------------------------------------------------------
#  Grid, folders, generator
# ---------------------------------------------------------------------------
def grid(model: str, stride: int):
    spec = MODELS[model]
    return (np.asarray(spec.p1_vals, float)[::stride],
            np.asarray(spec.p2_vals, float)[::stride])


def point_list(models, stride: int) -> list:
    """Flat list of (model, ix, iy) over every model's coarse grid."""
    out = []
    for model in models:
        p1, p2 = grid(model, stride)
        out += [(model, ix, iy) for ix in range(len(p1)) for iy in range(len(p2))]
    return out


def model_dir(out_root, model: str, stride: int) -> Path:
    return Path(out_root) / f'{model}_s{stride}'


def generator(model: str, p1: float, p2: float):
    """A = L^T of the model's bond generator (its own ModelSpec.dim)."""
    spec = MODELS[model]
    return build_bond_lindbladian(*spec.build(p1, p2), spec.dim).real.T


def old_dirs(model: str, root_map: dict) -> dict:
    """{d_ext: folder} of the earlier Hermitian seeded run (read-only)."""
    root = Path(root_map.get(model) or ('results_model10_rate' if model == 'model10'
                                        else f'results_{model}_seeded'))
    return {4: root / f'{model}_seeded', 8: root / f'{model}_seeded',
            12: root / f'{model}_seeded_d12'}


def point_files(folder: Path, ix: int, iy: int) -> list:
    if not folder.is_dir():
        return []
    return sorted(f for f in folder.glob(f'pt_{ix:03d}_{iy:03d}*.npz')
                  if '.tmp' not in f.name)


def best_in(folder: Path, ix: int, iy: int, m: int):
    """Lowest (rate, frame, label) over every file of the point holding
    rate_<m> and S_<m>; (inf, None, '') if none.  Keeps complex frames."""
    best = (np.inf, None, '')
    for f in point_files(folder, ix, iy):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:                                   # noqa: BLE001
            continue
        if f'rate_{m}' not in d.files or f'S_{m}' not in d.files:
            continue
        v = float(d[f'rate_{m}'])
        if np.isfinite(v) and v < best[0]:
            lab = str(d[f'label_{m}']) if f'label_{m}' in d.files else f.stem
            best = (v, _as_frame_array(d[f'S_{m}']), lab)
    return best


def best_complex(args, model: str, ix: int, iy: int, m: int):
    """Best frame of THIS run at the coarse point (any stage / round)."""
    return best_in(model_dir(args.out_root, model, args.stride), ix, iy, m)


def best_real(args, model: str, ix: int, iy: int, m: int):
    """Best Hermitian frame of the earlier run at the same physical point."""
    s = args.stride
    return best_in(old_dirs(model, args.old_root_map)[m], ix * s, iy * s, m)


# ---------------------------------------------------------------------------
#  Search pieces
# ---------------------------------------------------------------------------
def kick(S, sigma: float, rng) -> np.ndarray:
    """Complex Gaussian kick (real and imaginary parts) of the free columns,
    projected onto the operator-norm ball."""
    shape = (S.shape[0], S.shape[1] - N_FIXED_COLS)
    z = sigma * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    free = project_frame_columns(S[:, N_FIXED_COLS:] + z)
    return np.hstack([_FIXED_COLS.astype(free.dtype), free])


def imag_fraction(S) -> float:
    S = np.asarray(S)
    n = float(np.linalg.norm(S))
    return float(np.linalg.norm(np.imag(S)) / n) if n > 0 else 0.0


def certify(S, A) -> float:
    """Independent per-column LP value, clipped at 0 (floor of TP generators)."""
    v = frame_rate_value(S, A, reference=True)
    return max(float(v), 0.0) if np.isfinite(v) and v > -TOL else float(v)


def evaluate_seeds(A, m: int, seeds) -> list:
    """[(value, label, frame)] of the seeds fitted to m columns, ascending
    (singular / failed frames dropped)."""
    out = []
    for lab, S in seeds:
        if S is None:
            continue
        S = fit_columns(S, m)
        v = frame_rate_value(S, A)
        if np.isfinite(v):
            out.append((float(v), lab, S))
    out.sort(key=lambda t: t[0])
    return out


def multistart(A, m: int, seeds, args, rng, t_end: float):
    """Seeds -> bundle polishes of the best n_starts (real seeds also from
    n_kicks complex kicks) -> Nelder-Mead -> certification.
    Returns (rate, frame, label, info)."""
    cands = evaluate_seeds(A, m, seeds)
    if not cands:
        return np.inf, None, 'none', dict(mu_seed=np.inf)
    best_v, best_lab, best_S = cands[0]
    info = dict(mu_seed=best_v, n_seeds=len(cands))
    if best_v > TOL:
        starts = []
        for v0, lab, S0 in cands[:args.n_starts]:
            starts.append((lab, S0))
            if not is_complex_frame(S0):
                starts += [(f'{lab} kick', kick(S0, args.kick0, rng))
                           for _ in range(args.n_kicks)]
        for lab, S1 in starts:
            if time.time() > t_end or best_v <= TOL:
                break
            vb, Sb = bundle_rate_polish(S1, A, n_iter=args.bundle_iters)
            if np.isfinite(vb) and vb < best_v - TOL:
                best_v, best_lab, best_S = float(vb), f'{lab} +bundle', Sb
    info['mu_bundle'] = best_v
    if best_v > TOL and args.nm_maxfev > 0 and time.time() < t_end:
        from scipy.optimize import minimize
        obj = lambda x: frame_rate_value(frame_from_x(x, m), A)   # noqa: E731
        r = minimize(obj, x_from_frame(best_S, True), method='Nelder-Mead',
                     options=dict(maxfev=args.nm_maxfev, xatol=1e-9, fatol=1e-12))
        if np.isfinite(r.fun) and r.fun < best_v - TOL:
            best_v, best_lab, best_S = float(r.fun), f'{best_lab} +nm', frame_from_x(r.x, m)
    info['mu_nm'] = best_v
    mu = certify(best_S, A)
    info['mu_reference'] = mu
    return max(mu, best_v) if np.isfinite(mu) else mu, best_S, best_lab, info


def library_seeds(A, m: int) -> list:
    return [(f'library {k}', S) for k, S in seed_library(m, A).items()]


# ---------------------------------------------------------------------------
#  Stages
# ---------------------------------------------------------------------------
def save(path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name[:-4] + '.tmp.npz')
    np.savez(tmp, rate_version=RATE_VERSION, rate_global_version=RATE_GLOBAL_VERSION,
             pipeline_version=PIPELINE_VERSION, **kw)
    os.replace(tmp, path)                      # never leave a half-written file


def point_header(model: str, ix: int, iy: int, args) -> dict:
    spec = MODELS[model]
    p1, p2 = grid(model, args.stride)
    v1, v2 = float(p1[ix]), float(p2[iy])
    return {'model': model, 'ix': ix, 'iy': iy, 'ix_full': ix * args.stride,
            'iy_full': iy * args.stride, 'stride': args.stride, 'p1': v1, 'p2': v2,
            spec.p1_name: v1, spec.p2_name: v2, 'dim': spec.dim}


def stage_seed(model: str, ix: int, iy: int, args) -> None:
    out_f = model_dir(args.out_root, model, args.stride) / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists() and not args.force:
        print(f'[skip] {model} {out_f.name}', flush=True)
        return
    t_pt = time.perf_counter()
    out = point_header(model, ix, iy, args)
    A = generator(model, out['p1'], out['p2'])
    rng = np.random.default_rng([args.seed, ix, iy, len(model)])
    out['floor'] = spectral_abscissa(A)
    out['rate_pauli'] = pauli_rate(A.T)
    out['rate_stab3'] = stabilizer_3_rate(A.T)
    real = {m: best_real(args, model, ix, iy, m) for m in D_EXTS}
    for m in D_EXTS:
        out[f'real_{m}'] = real[m][0]
    real_seeds = [(f'hermitian d{k} ({real[k][2]})', real[k][1]) for k in D_EXTS
                  if real[k][1] is not None]

    # ---- d_ext = 4: closed form, complex DE -------------------------------
    t0 = time.perf_counter()
    S4, mu4, info = minimize_rate_global(
        A, 4, seeds=[S for _, S in real_seeds], seed=args.seed + 4,
        de_popsize=args.de_popsize4, de_maxiter=args.de_maxiter4,
        nm_maxfev=args.nm_maxfev4)
    name = str(info['seed_name'])
    lab4 = (real_seeds[int(name[4:])][0] if name.startswith('seed') and name[4:].isdigit()
            else f'library {name}')
    if mu4 < info['mu_seed'] - TOL:
        lab4 += ' +opt'
    out.update({'rate_4': mu4, 'S_4': S4, 'label_4': lab4, 'imag_4': imag_fraction(S4),
                'mu_de_4': info['mu_de'], 't_4': time.perf_counter() - t0})

    # ---- d_ext = 8: multistart -------------------------------------------
    t0 = time.perf_counter()
    seeds = [('complex d4', S4)] + real_seeds + library_seeds(A, 8)
    mu8, S8, lab8, info = multistart(A, 8, seeds, args, rng,
                                     time.time() + args.max_seconds_seed)
    out.update({'rate_8': mu8, 'S_8': S8, 'label_8': lab8, 'imag_8': imag_fraction(S8),
                'mu_seed_8': info['mu_seed'], 'mu_bundle_8': info.get('mu_bundle', np.nan),
                't_8': time.perf_counter() - t0})
    save(out_f, **out)
    print(f'  saved {model} {out_f.name}: d4={mu4:.5f} [{lab4}] d8={mu8:.5f} [{lab8}] '
          f'(hermitian {real[4][0]:.5f} / {real[8][0]:.5f}, Pauli {out["rate_pauli"]:.4f}) '
          f'({time.perf_counter() - t_pt:.0f}s)', flush=True)


def stage_d12(model: str, ix: int, iy: int, args) -> None:
    folder = model_dir(args.out_root, model, args.stride)
    out_f = folder / f'pt_{ix:03d}_{iy:03d}_d12.npz'
    if out_f.exists() and not args.force:
        print(f'[skip] {model} {out_f.name}', flush=True)
        return
    v8, S8, lab8 = best_complex(args, model, ix, iy, 8)
    if S8 is None:
        print(f'[wait] {model} ({ix},{iy}): no d_ext=8 frame yet', flush=True)
        return
    t0 = time.perf_counter()
    out = point_header(model, ix, iy, args)
    A = generator(model, out['p1'], out['p2'])
    rng = np.random.default_rng([args.seed, ix, iy, len(model), 12])
    real = {m: best_real(args, model, ix, iy, m) for m in (8, 12)}
    seeds = [(f'complex d8 ({lab8})', S8)]
    seeds += [(f'hermitian d{k} ({real[k][2]})', real[k][1]) for k in (12, 8)
              if real[k][1] is not None]
    seeds += library_seeds(A, 12)
    a = argparse.Namespace(**{**vars(args), 'n_starts': args.n_starts12,
                              'n_kicks': args.n_kicks12,
                              'bundle_iters': args.bundle_iters12,
                              'nm_maxfev': args.nm_maxfev12})
    mu, S, lab, info = multistart(A, 12, seeds, a, rng,
                                  time.time() + args.max_seconds_d12)
    out.update({'rate_12': mu, 'S_12': S, 'label_12': lab, 'imag_12': imag_fraction(S),
                'real_12': real[12][0], 'mu_seed_12': info['mu_seed'],
                't_12': time.perf_counter() - t0})
    save(out_f, **out)
    print(f'  saved {model} {out_f.name}: d12={mu:.5f} [{lab}] (d8 {v8:.5f}, hermitian '
          f'd12 {real[12][0]:.5f}) ({time.perf_counter() - t0:.0f}s)', flush=True)


def neighbours(model: str, ix: int, iy: int, radius: int, stride: int) -> list:
    p1, p2 = grid(model, stride)
    return [(ix + dx, iy + dy) for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            if (dx or dy) and 0 <= ix + dx < len(p1) and 0 <= iy + dy < len(p2)]


def stage_refine(model: str, ix: int, iy: int, args) -> None:
    m, r = args.d_ext, args.round
    folder = model_dir(args.out_root, model, args.stride)
    out_f = folder / f'pt_{ix:03d}_{iy:03d}_refine_d{m}_r{r:02d}.npz'
    if out_f.exists():
        return
    own_v, own_S, own_lab = best_complex(args, model, ix, iy, m)
    if own_S is None or own_v <= RATE_TOL:
        return                                   # not seeded yet / already rate 0
    t0 = time.perf_counter()
    t_end = time.time() + args.max_seconds
    hdr = point_header(model, ix, iy, args)
    A = generator(model, hdr['p1'], hdr['p2'])
    rng = np.random.default_rng([args.seed, ix, iy, len(model), m, r])
    seeds = [(own_lab, own_S)]
    for jx, jy in neighbours(model, ix, iy, args.radius, args.stride):
        _, Sn, labn = best_complex(args, model, jx, jy, m)
        if Sn is not None:
            seeds.append((f'nb ({jx},{jy}) {labn}', Sn))
    for k in D_EXTS:
        if k < m:
            _, Sk, labk = best_complex(args, model, ix, iy, k)
            if Sk is not None:
                seeds.append((f'own d{k} padded', Sk))
    cands = evaluate_seeds(A, m, seeds)
    best_v, best_lab, best_S = cands[0] if cands else (own_v, own_lab, own_S)
    mode = 'transfer' if best_v < own_v - TOL else 'own'
    hop = 0
    for hop in range(1, args.hops + 1):
        if time.time() > t_end or best_v <= TOL:
            break
        sigma = args.scales[(hop - 1) % len(args.scales)]
        vb, Sb = bundle_rate_polish(kick(best_S, sigma, rng), A, n_iter=args.polish_iters)
        if np.isfinite(vb) and vb < best_v - TOL:
            best_v, best_S = float(vb), Sb
            best_lab, mode = f'{best_lab} +hop', 'hop'
    if best_v >= own_v - TOL:
        return
    mu = certify(best_S, A)
    if not (np.isfinite(mu) and mu < own_v - TOL):
        return
    save(out_f, **hdr, **{f'rate_{m}': mu, f'rate_{m}_prev': own_v, f'S_{m}': best_S,
                          f'label_{m}': best_lab, f'imag_{m}': imag_fraction(best_S),
                          'mode': mode, 'round': r, 'hops': hop,
                          't': time.perf_counter() - t0})
    print(f'  saved {model} {out_f.name}: d{m} {own_v:.5f} -> {mu:.5f} [{mode}] '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


STAGES = {'seed': stage_seed, 'd12': stage_d12, 'refine': stage_refine}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=sorted(STAGES), required=True)
    p.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    p.add_argument('--stride', type=int, default=2,
                   help='grid stride (2 = twice the step of the model grid)')
    p.add_argument('--out_root', default='results_rate_complex')
    p.add_argument('--old_roots', nargs='*', default=[],
                   help='model:root overrides of the earlier Hermitian runs')
    p.add_argument('--task_id', type=int, default=0)
    p.add_argument('--n_chunks', type=int, default=1)
    p.add_argument('--point', default=None, help='one point, model:ix:iy (coarse grid)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force', action='store_true')
    # seed stage
    p.add_argument('--de_popsize4', type=int, default=24)
    p.add_argument('--de_maxiter4', type=int, default=300)
    p.add_argument('--nm_maxfev4', type=int, default=20000)
    p.add_argument('--n_starts', type=int, default=3, help='d8: seeds polished')
    p.add_argument('--n_kicks', type=int, default=2,
                   help='complex kicks per real seed (d8)')
    p.add_argument('--kick0', type=float, default=0.05)
    p.add_argument('--bundle_iters', type=int, default=60)
    p.add_argument('--nm_maxfev', type=int, default=400)
    p.add_argument('--max_seconds_seed', type=float, default=3600.0,
                   help='d8 search budget per point (checked between polishes)')
    # d12 stage
    p.add_argument('--n_starts12', type=int, default=2)
    p.add_argument('--n_kicks12', type=int, default=1)
    p.add_argument('--bundle_iters12', type=int, default=30)
    p.add_argument('--nm_maxfev12', type=int, default=0)
    p.add_argument('--max_seconds_d12', type=float, default=2400.0)
    # refine stage
    p.add_argument('--d_ext', type=int, choices=D_EXTS, default=8)
    p.add_argument('--round', type=int, default=1)
    p.add_argument('--radius', type=int, default=1)
    p.add_argument('--hops', type=int, default=12)
    p.add_argument('--scales', type=float, nargs='+', default=[0.02, 0.06, 0.15])
    p.add_argument('--polish_iters', type=int, default=40)
    p.add_argument('--max_seconds', type=float, default=1800.0)
    args = p.parse_args()
    for mdl in args.models:
        if mdl not in MODELS:
            sys.exit(f'unknown model {mdl!r}')
    args.old_root_map = dict(s.split(':', 1) for s in args.old_roots)

    run = STAGES[args.stage]
    if args.point:
        mdl, ix, iy = args.point.split(':')
        run(mdl, int(ix), int(iy), args)
        return
    pts = point_list(args.models, args.stride)
    if not (0 <= args.task_id < max(args.n_chunks, 1)):
        sys.exit(f'task_id must be in [0, {args.n_chunks})')
    mine = pts[args.task_id::max(args.n_chunks, 1)]
    print(f'[rate complex {args.stage}' + (f' d{args.d_ext} r{args.round}'
                                           if args.stage == 'refine' else '')
          + f'] chunk {args.task_id}/{args.n_chunks}: {len(mine)} of {len(pts)} points '
          f'({", ".join(args.models)}; stride {args.stride})', flush=True)
    for mdl, ix, iy in mine:
        try:
            run(mdl, ix, iy, args)
        except Exception as e:                              # noqa: BLE001
            print(f'  ERROR {mdl} ({ix},{iy}): {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
