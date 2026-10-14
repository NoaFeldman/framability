"""
Reliable optimised Heisenberg framability rates of model10 at chosen d_ext
(default 6 and 10) at a few points of the diagonal Delta_1 = Delta_2 = Delta
(default 0.5, 0.6, 0.7): an ensemble of independent searches per
(point, d_ext), then rounds of cross-restart basin hopping, so the reported
value comes with a convergence test (scripts/model10_dext_scan_collect.py).

Task layout (both stages): combos = points x d_exts, point-major; task t of
an array of --n_tasks works on combo t % n_combos as replica t // n_combos
(200 tasks, 3 points x 2 d_ext -> 33-34 replicas per combo).

--stage search  (one independent restart per task)
  Replica r is COLD when r % --cold_every == --cold_every - 1, else SEEDED.
  1. pipeline frames (seeded replicas only): the best frame per d_ext the
     earlier pipelines hold at the point and its 8 neighbours -- original
     scan + nrefine / qrefine / gopt files (S_heis_4/6/8, every
     --stored_dirs) and the seeded runs (best_known over the worker / xeval /
     qrefine / margin / rrefine files of model10_seeded and
     model10_seeded_d12) -- fitted to m columns: smaller frames padded,
     the point's own larger frames shrunk by greedy column removal
     (greedy_reduce), and the point's best smaller frame grown by
     greedy_augment.
  2. closed forms B / C_{m-2} / P_{m-3} (every replica).
  3. structured families with m columns (every replica): the pipeline's
     (families_for, winning_families_for), every two-plane split
     yz<a>xy<b>, and the d12-probe shapes (warped YZ polygon, latitude
     rings, Z pair + YZ + XY polygons), multistart Nelder-Mead from analytic
     starts and replica-specific random starts; seeded replicas also start
     from the point's stored d_ext 4 / 8 / 12 family optima.
  4. framability_rate_global.minimize_rate_global seeded with the best
     --seed_top frames of 1-3: differential evolution with a
     replica-specific seed, bundle polish, Nelder-Mead.
  5. basin hopping: margin_polish (bundle descent over all near-binding
     columns; at positive rate its objective is the rate), then Gaussian
     kicks of the free columns at --scales, each followed by margin_polish,
     until --hop_seconds.
  6. certification with the independent per-column LP.
  COLD replicas never see a frame from an earlier numerical optimisation
  (only 2-3 from analytic and random starts, minimize_rate_global's seed
  library and random DE members): an independent check of the seeded value.

--stage hop --round R  (cross-restart basin hopping)
  Start pool: every frame of the combo from the search stage and hop rounds
  < R, the --xfer_top best frames of the other target points at the same m,
  the scan's best frame of every smaller d_ext at the point (rounds < R)
  padded, and the pipelines' frames at the point (d_ext <= m, padded);
  scored on this generator and deduplicated by value.  Replica k starts from
  the (k mod --hop_top)-th best distinct frame and hops with its own RNG
  until --hop_seconds.  A file whose rate beats the combo's best before
  the round by more than --improve_tol prints IMPROVED (the driver counts
  them).

Every task writes exactly one file, via a temp file + os.replace, with
names unique per stage / round / combo / replica:
  <out_dir>/model10_dext_scan/m<MM>/pt_<ix>_<iy>/search_r<RRR>.npz
  <out_dir>/model10_dext_scan/m<MM>/pt_<ix>_<iy>/hop<NN>_r<RRR>.npz
keys: rate (certified), S, label, kind (seeded / cold / hop), rate_start,
rate_global (search), rate_prev / improved (hop), famval_<family> (search),
t.  Nothing is written into the pipeline directories.

Usage (the whole chain: scripts/submit_model10_dext_scan.sh):
    python scripts/model10_dext_scan_worker.py --stage search --task_id 0
    python scripts/model10_dext_scan_worker.py --stage hop --round 1 --task_id 0
    python scripts/model10_dext_scan_worker.py --stage search --task_id 0 \
        --deltas 0.5 0.6 0.7 --d_exts 4 6 8 10 12      # every d_ext, 13 replicas
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
from optimize_framability import N_FIXED_COLS                             # noqa: E402
from framability_rate import RATE_VERSION                                 # noqa: E402
from framability_rate_global import (minimize_rate_global, fit_columns,   # noqa: E402
                                     frame_rate_value, RATE_GLOBAL_VERSION)
from framability_rate_margin import (excess, margin_polish,              # noqa: E402
                                     RATE_MARGIN_VERSION)
from framability_rate_families import (families_for, winning_families_for,  # noqa: E402
                                       yz_xy_poly, optimize_family,
                                       greedy_augment, transfer_params,
                                       RATE_FAMILIES_VERSION)
from model10_seeded_worker import (MODEL, grid_vals, generator,           # noqa: E402
                                   weights_frame, closed_forms, family_starts,
                                   STORED_DIRS_DEFAULT)
from model10_seeded_qrefine_worker import best_known                      # noqa: E402
from model10_d12_worker import d12_dir, d8_dir                            # noqa: E402
from model10_d12_seed_probe_worker import (yzw_poly, lat_poly,            # noqa: E402
                                           yz_lat_poly, lat_lat_poly,
                                           zpair_yz_xy_poly, yz_part_starts,
                                           starts_for as probe_starts_for,
                                           _famp)
from model10_rrefine_worker import perturb                                # noqa: E402
from rate_gopt_worker import stored_frames                                # noqa: E402

SCAN_VERSION = '1.0-dext-scan'
SCAN_DIR = 'model10_dext_scan'
TOL = 1e-9
PIPE_D_EXTS = (4, 8, 12)          # seeded pipeline: model10_seeded (4, 8), _d12 (12)
STORED_D_EXTS = (4, 6, 8)         # S_heis_<k> of the original scan / refine / gopt
# differential evolution (popsize multiplier, generations) per d_ext: LP cost
# grows ~ m^4, so the budget shrinks with m (d_ext 10: ~1 h at ~0.5 s per LP)
DE_BUDGET = {4: (24, 300), 6: (12, 150), 8: (8, 60), 10: (6, 30), 12: (4, 15)}
FAMILY_GROUPS = ('base', 'split', 'probe')
PROBE_KINDS = ('yzw', 'lat', 'yz+lat', 'lat+lat', 'zp+yz+xy', 'yz+xy')
STAGE_CODE = {'search': 0, 'hop': 1}


# ---------------------------------------------------------------------------
#  Layout
# ---------------------------------------------------------------------------
def combo_dir(out_dir: str, m: int, ix: int, iy: int) -> Path:
    return Path(out_dir) / SCAN_DIR / f'm{m:02d}' / f'pt_{ix:03d}_{iy:03d}'


def search_file(out_dir: str, m: int, ix: int, iy: int, r: int) -> Path:
    return combo_dir(out_dir, m, ix, iy) / f'search_r{r:03d}.npz'


def hop_file(out_dir: str, m: int, ix: int, iy: int, rnd: int, r: int) -> Path:
    return combo_dir(out_dir, m, ix, iy) / f'hop{rnd:02d}_r{r:03d}.npz'


def diag_points(deltas, model: str = MODEL) -> list:
    """[(ix, iy)] of the grid points Delta_1 = Delta_2 = Delta."""
    p1, p2 = grid_vals(1, model)
    out = []
    for d in deltas:
        ix, iy = int(np.argmin(np.abs(p1 - d))), int(np.argmin(np.abs(p2 - d)))
        if abs(p1[ix] - d) > 1e-9 or abs(p2[iy] - d) > 1e-9:
            raise SystemExit(f'Delta = {d:g} is not a grid point')
        out.append((ix, iy))
    return out


def combos(deltas, d_exts, model: str = MODEL) -> list:
    """[(ix, iy, m)], point-major."""
    return [(ix, iy, m) for ix, iy in diag_points(deltas, model)
            for m in sorted(set(d_exts))]


def task_combo(task_id: int, n_tasks: int, cmb: list):
    """((ix, iy, m), replica) of an array task."""
    if n_tasks < len(cmb):
        raise SystemExit(f'{n_tasks} tasks < {len(cmb)} (point, d_ext) combos')
    if not (0 <= task_id < n_tasks):
        raise SystemExit(f'task_id must be in [0, {n_tasks})')
    return cmb[task_id % len(cmb)], task_id // len(cmb)


def rng_seed(args, ix: int, iy: int, m: int, replica: int, stage: str,
             rnd: int = 0) -> int:
    """Per-task seed (fits differential_evolution's 32-bit seed)."""
    ss = np.random.SeedSequence([args.seed, ix, iy, m, replica,
                                 STAGE_CODE[stage], rnd])
    return int(ss.generate_state(1)[0] % (2 ** 31 - 1))


# ---------------------------------------------------------------------------
#  Frames
# ---------------------------------------------------------------------------
def certify(S, A) -> float:
    """Independent per-column LP value, clipped at the floor 0."""
    v = frame_rate_value(S, A, reference=True)
    return max(float(v), 0.0) if np.isfinite(v) and v > -TOL else float(v)


def greedy_reduce(S, A, m: int):
    """Shrink S to m columns one column at a time, each time dropping the free
    column whose removal raises the rate least (a column whose removal loses
    full support scores inf).  Returns (value, S_m, n_evals); falls back to
    the plain trim if every removal fails."""
    S = np.asarray(S, float).copy()
    v, n_ev = np.inf, 0
    while S.shape[1] > m:
        best_v, best_S = np.inf, None
        for j in range(N_FIXED_COLS, S.shape[1]):
            T = np.delete(S, j, axis=1)
            val = frame_rate_value(T, A)
            n_ev += 1
            if np.isfinite(val) and val < best_v:
                best_v, best_S = float(val), T
        if best_S is None:
            S = fit_columns(S, m)
            return float(frame_rate_value(S, A)), S, n_ev + 1
        S, v = best_S, best_v
    return v, S, n_ev


def _lowest(lst):
    """(value, frame) with the lowest finite stored value (first if none)."""
    if not lst:
        return None
    fin = [t for t in lst if np.isfinite(t[0])]
    return min(fin, key=lambda t: t[0]) if fin else lst[0]


def pipeline_frames(args, ix: int, iy: int, radius: int = 1) -> list:
    """[(label, k, frame, own)]: the best frame per d_ext k that the earlier
    pipelines hold at (ix, iy) and its neighbours within Chebyshev `radius`
    (own = the point itself).  Read-only."""
    spec = MODELS[args.model]
    pt_seed, pt_d12 = d8_dir(args.out_dir, 1, args.model), d12_dir(args.out_dir, 1, args.model)
    out = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < spec.N_X and 0 <= jy < spec.N_Y):
                continue
            own = dx == dy == 0
            tag = '' if own else f' nb({dx:+d},{dy:+d})'
            base: dict = {}
            for d in args.stored_dirs:
                pt = Path(d) / args.model
                if pt.is_dir():
                    for k, lst in stored_frames(pt, jx, jy, STORED_D_EXTS).items():
                        base.setdefault(k, []).extend(lst)
            for k, lst in sorted(base.items()):
                b = _lowest(lst)
                if b is not None:
                    out.append((f'base d{k}{tag}', k, b[1], own))
            for k in PIPE_D_EXTS:
                _, S, lab = best_known(pt_d12 if k == 12 else pt_seed, jx, jy,
                                       f'rate_{k}', f'S_{k}')
                if S is not None:
                    out.append((f'seeded d{k} {lab}{tag}', k, S, own))
    return out


def seed_pool(args, A, m: int, pipe: list) -> list:
    """[(label, frame)] with m columns from the pipelines' frames: same size
    as is, smaller ones padded, the point's own larger ones reduced
    greedily, and the point's best smaller frame grown greedily."""
    out = []
    small = (np.inf, None, '')
    for lab, k, S, own in pipe:
        if k == m:
            out.append((lab, S))
        elif k < m:
            out.append((f'{lab} padded', fit_columns(S, m)))
            if own:
                v = frame_rate_value(S, A)
                if np.isfinite(v) and v < small[0]:
                    small = (float(v), S, lab)
        elif own and not args.no_reduce:
            v, Sr, _ = greedy_reduce(S, A, m)
            if np.isfinite(v):
                out.append((f'{lab} reduced', Sr))
    if small[1] is not None and not args.no_augment:
        v, Sa, _ = greedy_augment(small[1], A, m)
        if np.isfinite(v):
            out.append((f'{small[2]} augmented', Sa))
    return out


def scan_families(m: int, groups) -> list:
    """Structured families with exactly m columns (deduplicated by name)."""
    fams: dict = {}

    def add(f):
        if f.m == m:
            fams.setdefault(f.name, f)

    if 'base' in groups:
        for f in families_for(m) + winning_families_for(m):
            add(f)
    if 'split' in groups:
        for a in range(2, m - 3):                         # yz a-gon + xy b-gon
            add(yz_xy_poly(a, m - 2 - a))
    if 'probe' in groups and m >= 6:
        add(yzw_poly(m - 2))
        add(lat_poly(m - 2))
        for a in range(2, m - 3):
            add(yz_lat_poly(a, m - 2 - a))
        for a in range(2, (m - 2) // 2 + 1):
            add(lat_lat_poly(a, m - 2 - a))
        for a in range(2, m - 5):
            add(zpair_yz_xy_poly(a, m - 4 - a))
    return list(fams.values())


def fam_starts(fam, args, d1, d2, Sw, famp: dict, yz_starts: list,
               famp12: dict) -> list:
    """Transferred stored optima first, then analytic starts (capped)."""
    st = transfer_params(fam, famp)
    if fam.kind in ('yz', 'xy', 'xp', 'yzxy', 'yz+xy'):
        st += family_starts(args.model, fam, d1, d2, Sw)
    if fam.kind in PROBE_KINDS:
        st += probe_starts_for(fam, yz_starts, famp12, args.max_starts)
    return [np.asarray(p, float) for p in st[:args.max_starts]]


def basin_hop(S0, A, rng, args):
    """margin_polish of S0, then Gaussian kicks (scales cycled) each followed
    by margin_polish, keeping a kick only if it lowers F = max over the
    non-identity columns; stops at --hop_seconds / --max_hops or once the
    frame has margin --margin_goal.  Returns (F, S, n_hops, n_kept)."""
    t0 = time.perf_counter()
    F_best, S_best = excess(S0, A), np.asarray(S0, float)
    F, S, _ = margin_polish(S_best, A, n_iter=args.polish_iters)
    if np.isfinite(F) and F < F_best:
        F_best, S_best = float(F), S
    n_hops = n_kept = 0
    while n_hops < args.max_hops and time.perf_counter() - t0 < args.hop_seconds \
            and F_best > -args.margin_goal:
        sigma = args.scales[n_hops % len(args.scales)]
        F, S, _ = margin_polish(perturb(S_best, sigma, rng), A,
                                n_iter=args.polish_iters)
        n_hops += 1
        if np.isfinite(F) and F < F_best - args.gain_tol:
            F_best, S_best = float(F), S
            n_kept += 1
    return F_best, S_best, n_hops, n_kept


def _point(args, ix: int, iy: int):
    spec = MODELS[args.model]
    p1, p2 = grid_vals(1, args.model)
    d1, d2 = float(p1[ix]), float(p2[iy])
    return spec, d1, d2, generator(d1, d2, args.dim, args.model)


# ---------------------------------------------------------------------------
#  Stage: search (one independent restart)
# ---------------------------------------------------------------------------
def compute_search(ix: int, iy: int, m: int, r: int, args) -> dict:
    spec, d1, d2, A = _point(args, ix, iy)
    Sw = weights_frame(args.model, A)
    cold = args.cold_every > 0 and r % args.cold_every == args.cold_every - 1
    sd = rng_seed(args, ix, iy, m, r, 'search')
    rng = np.random.default_rng(sd)
    t0 = time.perf_counter()
    out = {spec.p1_name: d1, spec.p2_name: d2, 'dim': args.dim,
           'kind': 'cold' if cold else 'seeded', 'rng_seed': sd}
    cands = []                                            # (label, frame)

    # ---- 1. pipeline frames (seeded only) ----------------------------------
    famp8, famp12 = {}, {}
    if not cold:
        cands += seed_pool(args, A, m, pipeline_frames(args, ix, iy))
        pt_seed = d8_dir(args.out_dir, 1, args.model)
        famp8 = {**_famp(pt_seed, ix, iy, 4), **_famp(pt_seed, ix, iy, 8)}
        famp12 = _famp(d12_dir(args.out_dir, 1, args.model), ix, iy, 12)
    out['n_pipeline'] = len(cands)

    # ---- 2. closed forms ------------------------------------------------------
    for name, S in closed_forms(args.model, d1, d2, m, Sw).items():
        cands.append((f'analytic {name}', S))

    # ---- 3. structured families ---------------------------------------------
    yz_starts = yz_part_starts(args.model, d1, d2, A, famp8, famp12)
    fam_vals = []
    for fam in scan_families(m, args.families):
        starts = fam_starts(fam, args, d1, d2, Sw, {**famp8, **famp12},
                            yz_starts, famp12)
        v, S, p, _ = optimize_family(
            fam, A, starts, n_random=args.fam_random, n_polish=args.fam_polish,
            maxfev=args.fam_maxfev4 if m == 4 else args.fam_maxfev, rng=rng)
        out[f'famval_{fam.name}'] = float(v)
        out[f'famp_{fam.name}'] = np.asarray(p, float)
        fam_vals.append(float(v))
        cands.append((f'family {fam.name}', S))
    out['rate_fam'] = min(fam_vals) if fam_vals else np.nan
    out['t_seeds'] = time.perf_counter() - t0

    # ---- 4. seeded global search -----------------------------------------
    vals = np.array([frame_rate_value(S, A) for _, S in cands], float)
    vals = np.where(np.isfinite(vals), vals, np.inf)
    order = [int(i) for i in np.argsort(vals) if np.isfinite(vals[i])]
    if not order:
        raise RuntimeError('no finite seed frame')
    out['rate_pool'] = float(vals[order[0]])
    out['label_pool'] = cands[order[0]][0]
    seeds = [cands[i][1] for i in order[:args.seed_top]]
    pop, gens = DE_BUDGET.get(m, (6, 25))
    S_g, mu_g, info = minimize_rate_global(
        A, m, seeds=seeds, seed=sd,
        de_popsize=args.de_popsize if args.de_popsize > 0 else pop,
        de_maxiter=args.de_maxiter if args.de_maxiter >= 0 else gens,
        bundle_iters=args.bundle_iters,
        nm_maxfev=args.nm_maxfev4 if m == 4 else args.nm_maxfev)
    name = str(info['seed_name'])
    if name.startswith('seed') and name[4:].isdigit():
        label = cands[order[int(name[4:])]][0]
    else:
        label = f'library {name}'
    if mu_g < info['mu_seed'] - args.gain_tol:
        label = f'{label} +opt'
    out.update(rate_global=float(mu_g), rate_de=float(info['mu_de']),
               t_global=float(info['time']))

    # ---- 5. basin hopping ----------------------------------------------------
    F, S_h, n_hops, n_kept = basin_hop(S_g, A, rng, args)
    rate = certify(S_h, A)
    if np.isfinite(rate) and rate < mu_g - args.gain_tol:
        S_best, label = S_h, f'{label} +hop'
    else:
        S_best, rate = S_g, float(mu_g)
    out.update(rate=rate, S=S_best, label=label, rate_start=out['rate_pool'],
               hops=n_hops, hops_kept=n_kept, excess=float(F),
               t=time.perf_counter() - t0)
    return out


# ---------------------------------------------------------------------------
#  Stage: hop (cross-restart basin hopping, one round)
# ---------------------------------------------------------------------------
def frames_on_disk(out_dir: str, m: int, ix: int, iy: int, before_round: int) -> list:
    """[(value, frame, label, source)] of the combo's search files and hop
    rounds < before_round, ascending in the stored certified value."""
    d = combo_dir(out_dir, m, ix, iy)
    files = sorted(d.glob('search_r[0-9][0-9][0-9].npz'))
    files += [f for f in sorted(d.glob('hop[0-9][0-9]_r[0-9][0-9][0-9].npz'))
              if int(f.name[3:5]) < before_round]
    out = []
    for f in files:
        try:
            z = np.load(f, allow_pickle=True)
            v = float(z['rate'])
            if np.isfinite(v):
                out.append((v, np.asarray(z['S'], float), str(z['label']), f.stem))
        except Exception:                                   # noqa: BLE001
            continue
    out.sort(key=lambda t: t[0])
    return out


def compute_hop(ix: int, iy: int, m: int, k: int, rnd: int, args) -> dict:
    spec, d1, d2, A = _point(args, ix, iy)
    t0 = time.perf_counter()
    own = frames_on_disk(args.out_dir, m, ix, iy, rnd)
    v_prev = own[0][0] if own else np.inf
    pool = [(f'{src}: {lab}', S) for _, S, lab, src in own]
    for jx, jy in diag_points(args.deltas, args.model):
        if (jx, jy) != (ix, iy):
            for _, S, lab, src in frames_on_disk(args.out_dir, m, jx, jy, rnd)[:args.xfer_top]:
                pool.append((f'xfer pt_{jx:03d}_{jy:03d} {src}: {lab}', S))
    for km in sorted(set(args.d_exts)):
        if km < m:
            for _, S, lab, src in frames_on_disk(args.out_dir, km, ix, iy, rnd)[:1]:
                pool.append((f'd{km} {src}: {lab} padded', fit_columns(S, m)))
    for lab, km, S, _ in pipeline_frames(args, ix, iy, radius=0):
        if km <= m:
            pool.append((lab if km == m else f'{lab} padded', fit_columns(S, m)))
    if not pool:
        raise RuntimeError('empty start pool')

    vals = np.array([frame_rate_value(S, A) for _, S in pool], float)
    distinct = []                                         # (value, index)
    for i in np.argsort(np.where(np.isfinite(vals), vals, np.inf)):
        v = float(vals[i])
        if not np.isfinite(v):
            break
        if all(abs(v - u) > args.dedupe_tol * max(1.0, abs(u)) for u, _ in distinct):
            distinct.append((v, int(i)))
    if not distinct:
        raise RuntimeError('no finite start frame')
    rank = k % min(args.hop_top, len(distinct))
    v_start, i0 = distinct[rank]
    label, S0 = pool[i0]

    sd = rng_seed(args, ix, iy, m, k, 'hop', rnd)
    F, S_h, n_hops, n_kept = basin_hop(S0, A, np.random.default_rng(sd), args)
    rate = certify(S_h, A)
    v0 = certify(S0, A)
    if not (np.isfinite(rate) and rate <= v0):
        S_h, rate = S0, v0
    elif rate < v0 - args.gain_tol:
        label = f'{label} +hop'
    return {spec.p1_name: d1, spec.p2_name: d2, 'dim': args.dim, 'kind': 'hop',
            'rng_seed': sd, 'rate': rate, 'S': S_h, 'label': label,
            'rate_start': v_start, 'start_rank': rank, 'n_distinct': len(distinct),
            'rate_prev': v_prev, 'improved': bool(rate < v_prev - args.improve_tol),
            'hops': n_hops, 'hops_kept': n_kept, 'excess': float(F),
            't': time.perf_counter() - t0}


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('search', 'hop'), required=True)
    p.add_argument('--round', type=int, default=0,
                   help='hop round (1..99); round R reads search + rounds < R')
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_tasks', type=int, default=200,
                   help='array size; task t -> combo t %% n_combos, replica '
                        't // n_combos')
    p.add_argument('--deltas', type=float, nargs='+', default=[0.5, 0.6, 0.7],
                   help='diagonal points Delta_1 = Delta_2 = Delta (grid values)')
    p.add_argument('--d_exts', type=int, nargs='+', default=[6, 10])
    p.add_argument('--model', type=str, default=MODEL, choices=sorted(MODELS))
    p.add_argument('--out_dir', type=str, default='results_model10_rate')
    p.add_argument('--stored_dirs', type=str, nargs='*',
                   default=list(STORED_DIRS_DEFAULT),
                   help='roots whose <dir>/<model> holds the original scan '
                        '(read-only; missing dirs are skipped)')
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cold_every', type=int, default=3,
                   help='search: every cold_every-th replica is cold (0: none)')
    p.add_argument('--families', nargs='+', default=list(FAMILY_GROUPS),
                   choices=FAMILY_GROUPS)
    p.add_argument('--max_starts', type=int, default=8,
                   help='analytic / transferred starts per family')
    p.add_argument('--fam_random', type=int, default=6)
    p.add_argument('--fam_polish', type=int, default=2)
    p.add_argument('--fam_maxfev', type=int, default=120)
    p.add_argument('--fam_maxfev4', type=int, default=1500)
    p.add_argument('--no_reduce', action='store_true',
                   help='skip greedy column removal of larger pipeline frames')
    p.add_argument('--no_augment', action='store_true')
    p.add_argument('--seed_top', type=int, default=24,
                   help='best seed frames handed to minimize_rate_global')
    p.add_argument('--de_popsize', type=int, default=0, help='0: DE_BUDGET[m]')
    p.add_argument('--de_maxiter', type=int, default=-1, help='-1: DE_BUDGET[m]')
    p.add_argument('--bundle_iters', type=int, default=80)
    p.add_argument('--nm_maxfev', type=int, default=400)
    p.add_argument('--nm_maxfev4', type=int, default=20000)
    p.add_argument('--polish_iters', type=int, default=100)
    p.add_argument('--scales', type=float, nargs='+', default=[0.02, 0.06, 0.15, 0.3])
    p.add_argument('--hop_seconds', type=float, default=3600.0)
    p.add_argument('--max_hops', type=int, default=500)
    p.add_argument('--margin_goal', type=float, default=1e-3)
    p.add_argument('--gain_tol', type=float, default=1e-9,
                   help='smallest decrease that counts as a step / kick gain')
    p.add_argument('--improve_tol', type=float, default=1e-5,
                   help='hop: a round IMPROVES a combo only if it lowers the '
                        'best rate by more than this (the driver stop rule)')
    p.add_argument('--hop_top', type=int, default=5,
                   help='hop: replicas start from the best hop_top distinct frames')
    p.add_argument('--xfer_top', type=int, default=3,
                   help='hop: best frames taken from each other target point')
    p.add_argument('--dedupe_tol', type=float, default=1e-7)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    if args.dim is None:
        args.dim = MODELS[args.model].dim
    if args.stage == 'hop' and args.round < 1:
        sys.exit('--stage hop needs --round >= 1')

    cmb = combos(args.deltas, args.d_exts, args.model)
    (ix, iy, m), r = task_combo(args.task_id, args.n_tasks, cmb)
    if args.stage == 'search':
        out_f = search_file(args.out_dir, m, ix, iy, r)
    else:
        out_f = hop_file(args.out_dir, m, ix, iy, args.round, r)
    head = (f'[{args.model} dext scan] {args.stage}'
            f'{f" round {args.round}" if args.stage == "hop" else ""} task '
            f'{args.task_id}/{args.n_tasks}: pt ({ix},{iy}) d_ext={m} replica {r} '
            f'({len(cmb)} combos)')
    print(head, flush=True)
    if out_f.exists() and not args.force:
        print(f'[skip] {out_f}', flush=True)
        return
    t0 = time.perf_counter()
    try:
        if args.stage == 'search':
            res = compute_search(ix, iy, m, r, args)
        else:
            res = compute_hop(ix, iy, m, r, args.round, args)
    except Exception as e:                                  # noqa: BLE001
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        return
    out_f.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_f.with_suffix('.tmp.npz')
    np.savez(tmp, model=args.model, ix=ix, iy=iy, d_ext=m, replica=r,
             stage=args.stage, round=args.round, scan_version=SCAN_VERSION,
             rate_version=RATE_VERSION, rate_global_version=RATE_GLOBAL_VERSION,
             rate_families_version=RATE_FAMILIES_VERSION,
             rate_margin_version=RATE_MARGIN_VERSION, **res)
    os.replace(tmp, out_f)                    # never leave a half-written file
    extra = (f'  IMPROVED from {res["rate_prev"]:.8e}' if res.get('improved') else '')
    print(f'  saved {out_f.name}  [{res["kind"]}] start {res["rate_start"]:.8e} -> '
          f'{res["rate"]:.8e} [{res["label"]}]  {res["hops_kept"]}/{res["hops"]} '
          f'hops kept{extra}  ({time.perf_counter() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
