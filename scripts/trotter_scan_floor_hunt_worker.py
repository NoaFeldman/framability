"""
Floor hunt for Trotter-scan models with NO framable point yet: heavy
re-optimisation of the most promising grid points, trying to find the first
point with opt_fra == 1.

Quick refining / island polishing propagate an existing floor outward, so
they are useless when no point sits at the floor.  What still applies:

  * the spectral-radius floor is exactly 1 for every trace-preserving gate,
    so the target value is KNOWN everywhere — a floor-targeted Polyak
    subgradient polish (analytic LP-dual gradient) can close a last-mile gap
    that Powell leaves at the kink a floor optimum generically is;
  * frames still transfer between nearby gates, so the grid minima ("elite"
    points) and all 4-connected neighbours supply warm starts;
  * a genuinely global search (dual annealing) with a serious budget probes
    whether the remaining gap is an optimisation artifact.

If the heavy hunt drives the grid minimum to 1 (up to --fra_tol) the old
minimum was an optimiser artifact and the island pipeline can take over to
grow the region.  If it stalls on a plateau above 1, that is (numerical)
evidence the model genuinely has no framable point at this grid.

Two phases:

  RANK  (single cheap job)
    python .../trotter_scan_floor_hunt_worker.py --model model2 --round 1 --rank
    Scans best-known values over the whole grid, ranks points by
    min(opt_fra_4, opt_fra_6), stores the n_select lowest (plus the elite
    frames) in  <out_dir>/<model>/fhunt_rank_r<NN>.npz .

  HUNT  (array job over the selected points)
    python .../trotter_scan_floor_hunt_worker.py --model model2 --round 1 \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
    Per point and key: Powell over self + all-neighbour + elite (+ embedded
    d4) seeds, dual annealing warm-started from the best, then the Polyak
    floor polish.  Values are confirmed with the reference per-column LP.

Writes: <out_dir>/<model>/pt_fhunt_r<NN>_<ix>_<iy>.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS
from dissipative_PT import (
    optimise_framability, frame_from_params, params_from_frame,
)
from optimize_framability import minimize_framability, polyak_floor_polish
from trotter_refine_common import (
    KEYS, base_path, best_known, neighbor_frames, build_gate,
    eval_frame_reference, embed_S4_to_S6,
)

TOL = 1e-9


def rank_file(out_dir: Path, model, rnd: int) -> Path:
    return Path(out_dir) / model.name / f'fhunt_rank_r{rnd:02d}.npz'


# ---------------------------------------------------------------------------
#  RANK phase
# ---------------------------------------------------------------------------
def run_rank(model, args) -> None:
    out_dir = Path(args.out_dir)
    t0 = time.perf_counter()
    print(f'[{model.name}] ranking {model.N_TOTAL} points by best-known '
          f'min(opt_fra_4, opt_fra_6) ...', flush=True)

    rows = []          # (score, point_id)
    vals = {k: {} for k in KEYS}
    frames = {k: {} for k in KEYS}
    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            if not base_path(out_dir, model, ix, iy).exists():
                continue
            score = np.inf
            for key, (s_key, _) in KEYS.items():
                v, S = best_known(out_dir, model, ix, iy, key, s_key)
                vals[key][(ix, iy)] = v
                frames[key][(ix, iy)] = S
                score = min(score, v)
            if np.isfinite(score):
                rows.append((score, ix * model.N_Y + iy))

    rows.sort()
    n_floor = sum(1 for s, _ in rows if s <= 1.0 + args.fra_tol)
    if n_floor:
        print(f'[{model.name}] NOTE: {n_floor} point(s) already at the floor '
              f'— the island pipeline (cross-eval + polish) is the right tool '
              f'for this model; the hunt will still run on the selection.',
              flush=True)

    sel = rows[:args.n_select]
    point_ids = np.array([pid for _, pid in sel], dtype=int)
    scores    = np.array([s   for s, _  in sel], dtype=float)

    payload = {'point_ids': point_ids, 'scores': scores,
               'round': np.array(args.round)}
    # elite frames: the n_elite lowest points' frames, per key
    for key, (s_key, _) in KEYS.items():
        elites = []
        for _, pid in rows[:args.n_elite]:
            S = frames[key].get((pid // model.N_Y, pid % model.N_Y))
            if S is not None:
                elites.append(S)
        if elites:
            payload[f'elite_{s_key}'] = np.stack(elites)

    out = rank_file(out_dir, model, args.round)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(f'[{model.name}] wrote {out.name}: {len(sel)} point(s) selected, '
          f'grid minimum = {rows[0][0]:.6f} at point_id {rows[0][1]} '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


# ---------------------------------------------------------------------------
#  HUNT phase
# ---------------------------------------------------------------------------
def hunt_key(gate, key: str, d_ext: int, self_val: float, self_S,
             nb_list, elite_frames, args, seed: int, extra_frames=None):
    """Powell over all seeds -> dual annealing warm start -> Polyak polish."""
    maxfev = {'opt_fra_4': args.fra_maxfev_4, 'opt_fra_6': args.fra_maxfev_6}[key]

    seed_frames = ([self_S] if self_S is not None else [])
    seed_frames += [S for _, S in nb_list]
    seed_frames += list(elite_frames)
    seed_frames += list(extra_frames or [])
    seeds = [params_from_frame(S) for S in seed_frames]

    best_val, best_S = self_val, self_S

    # Stage 1: Powell restarts over every seed.
    f, x = optimise_framability(gate, d_ext, n_restarts=args.n_restarts,
                                maxfev=maxfev, seed=seed,
                                extra_init_xs=seeds if seeds else None,
                                return_x=True)
    if np.isfinite(f) and f < best_val - TOL:
        S = frame_from_params(x, d_ext)
        f_ref = eval_frame_reference(S, gate)
        if np.isfinite(f_ref) and f_ref < best_val - TOL:
            best_val, best_S = f_ref, S

    # Stage 2: global search (dual annealing), warm-started from the best.
    if best_val > 1.0 + args.fra_tol and args.da_maxfev > 0:
        x0 = params_from_frame(best_S) if best_S is not None else None
        _, f_da, x_da = minimize_framability(
            gate.real, d_ext_single=d_ext, method='dual_annealing',
            maxfev=args.da_maxfev, seed=seed + 17, verbose=False,
            use_complex=False, return_x=True,
            extra_init_xs=[x0] if x0 is not None else None)
        if np.isfinite(f_da) and f_da < best_val - TOL:
            S = frame_from_params(np.asarray(x_da, dtype=float), d_ext)
            f_ref = eval_frame_reference(S, gate)
            if np.isfinite(f_ref) and f_ref < best_val - TOL:
                best_val, best_S = f_ref, S

    # Stage 3: floor-targeted Polyak polish.
    if best_S is not None and best_val > 1.0 + args.fra_tol:
        f_pol, S_pol = polyak_floor_polish(best_S, gate, target=1.0,
                                           n_iter=args.polish_iters)
        if np.isfinite(f_pol) and f_pol < best_val - TOL:
            f_ref = eval_frame_reference(S_pol, gate)
            if np.isfinite(f_ref) and f_ref < best_val - TOL:
                best_val, best_S = f_ref, S_pol

    return best_val, best_S


def run_point(model, point_id: int, elite: dict, args) -> None:
    out_dir = Path(args.out_dir)
    ix = point_id // model.N_Y
    iy = point_id %  model.N_Y
    out = out_dir / model.name / f'pt_fhunt_r{args.round:02d}_{ix:03d}_{iy:03d}.npz'

    if out.exists():
        print(f'[skip] {model.name}/{out.name} already exists', flush=True)
        return

    base = base_path(out_dir, model, ix, iy)
    if not base.exists():
        print(f'ERROR: base scan file {base} not found', file=sys.stderr)
        sys.exit(1)

    info = {}
    for key, (s_key, _) in KEYS.items():
        self_val, self_S = best_known(out_dir, model, ix, iy, key, s_key)
        nb_list = neighbor_frames(out_dir, model, ix, iy, key, s_key)
        info[key] = (self_val, self_S, nb_list)

    f4_known = info['opt_fra_4'][0]
    f6_known = info['opt_fra_6'][0]
    seed = args.seed + point_id + 100000 * args.round
    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} floor hunt round '
          f'{args.round} {model.p1_name}={model.p1_vals[ix]:.3f} '
          f'{model.p2_name}={model.p2_vals[iy]:.3f}  '
          f'best d4={f4_known:.6f} d6={f6_known:.6f}', flush=True)

    gate = build_gate(model, ix, iy, np.load(base))

    results = {}
    for off, (key, (s_key, d_ext)) in enumerate(KEYS.items()):
        self_val, self_S, nb_list = info[key]
        extra = []
        if key == 'opt_fra_6':
            if info['opt_fra_4'][1] is not None:
                extra.append(embed_S4_to_S6(info['opt_fra_4'][1]))
            extra += [embed_S4_to_S6(S) for _, S in info['opt_fra_4'][2]]
            extra += [embed_S4_to_S6(S) for S in elite.get('opt_S_4', [])]
        results[key] = hunt_key(gate, key, d_ext, self_val, self_S, nb_list,
                                elite.get(s_key, []), args, seed + off,
                                extra_frames=extra)
        print(f'  d{d_ext}: {self_val:.6f} -> {results[key][0]:.6f}', flush=True)

    # cross-d_ext step: embed the best d=4 frame into d=6
    f4, S4 = results['opt_fra_4']
    f6, S6 = results['opt_fra_6']
    if f4 < f6 - TOL and S4 is not None:
        S6e = embed_S4_to_S6(S4)
        f6e = eval_frame_reference(S6e, gate)
        if np.isfinite(f6e) and f6e < f6 - TOL:
            print(f'  cross-embed d4->d6: {f6:.6f} -> {f6e:.6f}', flush=True)
            results['opt_fra_6'] = (f6e, S6e)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'round': np.array(args.round),
               'ix': np.array(ix), 'iy': np.array(iy)}
    for key, (s_key, _) in KEYS.items():
        val, S = results[key]
        payload[key] = np.array(val)
        if S is not None:
            payload[s_key] = np.asarray(S)
    np.savez(out, **payload)
    hit = min(results['opt_fra_4'][0], results['opt_fra_6'][0]) <= 1.0 + args.fra_tol
    print(f'  saved {out.name}: d4 {f4_known:.6f}->{results["opt_fra_4"][0]:.6f}  '
          f'd6 {f6_known:.6f}->{results["opt_fra_6"][0]:.6f}  '
          f'{"*** FLOOR REACHED ***  " if hit else ""}'
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',        type=str, required=True, choices=list(MODELS))
    p.add_argument('--round',        type=int, required=True)
    p.add_argument('--rank',         action='store_true',
                   help='run the RANK phase (single job) instead of hunting')
    p.add_argument('--task_id',      type=int, default=None)
    p.add_argument('--n_chunks',     type=int, default=1)
    p.add_argument('--out_dir',      type=str, default='results_trotter_v3')
    p.add_argument('--n_select',     type=int, default=200,
                   help='number of lowest-value points to hunt')
    p.add_argument('--n_elite',      type=int, default=5,
                   help='number of grid-minimum frames used as elite seeds')
    p.add_argument('--n_restarts',   type=int, default=3)
    p.add_argument('--fra_maxfev_4', type=int, default=3000)
    p.add_argument('--fra_maxfev_6', type=int, default=2000)
    p.add_argument('--da_maxfev',    type=int, default=6000,
                   help='dual-annealing evaluation budget (0 disables)')
    p.add_argument('--polish_iters', type=int, default=500,
                   help='Polyak floor-polish subgradient steps')
    p.add_argument('--fra_tol',      type=float, default=1e-6)
    p.add_argument('--seed',         type=int, default=0)
    args = p.parse_args()

    model = MODELS[args.model]

    if args.rank:
        run_rank(model, args)
        return

    if args.task_id is None:
        print('ERROR: --task_id is required in the HUNT phase', file=sys.stderr)
        sys.exit(1)

    rf = rank_file(Path(args.out_dir), model, args.round)
    if not rf.exists():
        print(f'ERROR: ranking file {rf} not found — run the RANK phase first:'
              f'\n  python scripts/trotter_scan_floor_hunt_worker.py '
              f'--model {model.name} --round {args.round} --rank',
              file=sys.stderr)
        sys.exit(1)
    r = np.load(rf)
    point_ids = [int(pid) for pid in r['point_ids']]
    elite = {}
    for key, (s_key, _) in KEYS.items():
        ek = f'elite_{s_key}'
        if ek in r:
            elite[s_key] = [np.asarray(S, dtype=float) for S in r[ek]]

    if not (0 <= args.task_id < max(1, args.n_chunks)):
        print(f'ERROR: task_id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)
    mine = point_ids[args.task_id::max(1, args.n_chunks)]
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name} floor hunt '
          f'round {args.round}: {len(mine)} point(s)', flush=True)
    for pid in mine:
        run_point(model, pid, elite, args)


if __name__ == '__main__':
    main()
