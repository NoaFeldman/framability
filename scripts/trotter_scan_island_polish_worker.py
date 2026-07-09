"""
Island / boundary polish for the Trotter-scan optimised framabilities
(opt_fra_4 / opt_fra_6): the optimisation stage that follows a cross-eval
sweep (scripts/trotter_scan_cross_eval.py).

Targets the same boundary points as the quick refine — best-known opt_fra > 1
with at least one 4-connected neighbour at the framable floor (== 1 up to
--fra_tol), plus the cross d4->d6 embedding step — but fixes the failure
modes the islands exposed:

  1. ALL 4 neighbours' frames are used as warm-start seeds (not just the
     single best one): optimal frames come in distinct symmetry branches and
     the branch can switch across the grid.
  2. Larger Powell budgets by default (the old fra_maxfev_6=500 gave Powell
     roughly one iteration on 20 parameters).
  3. A floor-targeted Polyak subgradient polish (analytic LP-dual gradient,
     known target = 1) is run from the best frame whenever the Powell result
     is still above the floor — Powell stalls at the kink the floor optimum
     generically is.

Every stored value is confirmed with the reference per-column LP so it stays
consistent with the rest of the pipeline.

Reads:  <out_dir>/<model>/pt_<ix>_<iy>.npz + every refinement file
        (pt_refine_r*, pt_qrefine_r*, pt_xeval_r*, pt_polish_r*, pt_fhunt_r*)
Writes: <out_dir>/<model>/pt_polish_r<NN>_<ix>_<iy>.npz   (boundary points only)
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
from optimize_framability import polyak_floor_polish
from trotter_refine_common import (
    KEYS, base_path, best_known, neighbor_frames, build_gate,
    eval_frame_reference, embed_S4_to_S6,
)

TOL = 1e-9


def refine_key(gate, key: str, d_ext: int, self_val: float, self_S,
               nb_list, args, seed: int, extra_frames=None):
    """One key's full polish: Powell over all seeds, then Polyak floor polish.

    Returns (value, frame) — never worse than (self_val, self_S)."""
    maxfev = {'opt_fra_4': args.fra_maxfev_4, 'opt_fra_6': args.fra_maxfev_6}[key]

    seed_frames = ([self_S] if self_S is not None else [])
    seed_frames += [S for _, S in nb_list]
    seed_frames += list(extra_frames or [])
    seeds = [params_from_frame(S) for S in seed_frames]

    f, x = optimise_framability(gate, d_ext, n_restarts=args.n_restarts,
                                maxfev=maxfev, seed=seed,
                                extra_init_xs=seeds if seeds else None,
                                return_x=True)
    best_val, best_S = self_val, self_S
    if np.isfinite(f) and f < best_val - TOL:
        S = frame_from_params(x, d_ext)
        f_ref = eval_frame_reference(S, gate)
        if np.isfinite(f_ref) and f_ref < best_val - TOL:
            best_val, best_S = f_ref, S

    # Floor-targeted last-mile polish from the best frame so far.
    if best_S is not None and best_val > 1.0 + args.fra_tol:
        f_pol, S_pol = polyak_floor_polish(best_S, gate, target=1.0,
                                           n_iter=args.polish_iters)
        if np.isfinite(f_pol) and f_pol < best_val - TOL:
            f_ref = eval_frame_reference(S_pol, gate)
            if np.isfinite(f_ref) and f_ref < best_val - TOL:
                best_val, best_S = f_ref, S_pol

    return best_val, best_S


def run_point(model, point_id: int, args) -> None:
    out_dir = Path(args.out_dir)
    ix = point_id // model.N_Y
    iy = point_id %  model.N_Y
    out = out_dir / model.name / f'pt_polish_r{args.round:02d}_{ix:03d}_{iy:03d}.npz'

    if out.exists():
        print(f'[skip] {model.name}/{out.name} already exists', flush=True)
        return

    base = base_path(out_dir, model, ix, iy)
    if not base.exists():
        print(f'ERROR: base scan file {base} not found — run the scan first',
              file=sys.stderr)
        sys.exit(1)

    # ── boundary detection over best-known values ────────────────────────────
    info, todo = {}, []
    for key, (s_key, _) in KEYS.items():
        self_val, self_S = best_known(out_dir, model, ix, iy, key, s_key)
        nb_list = neighbor_frames(out_dir, model, ix, iy, key, s_key)
        info[key] = (self_val, self_S, nb_list)
        nb_best = nb_list[0][0] if nb_list else np.inf
        if self_val > 1.0 + args.fra_tol and nb_best <= 1.0 + args.fra_tol:
            todo.append(key)

    f4_known, S4_known = info['opt_fra_4'][0], info['opt_fra_4'][1]
    f6_known = info['opt_fra_6'][0]
    cross = (f4_known < f6_known - TOL and S4_known is not None
             and f6_known > 1.0 + args.fra_tol)

    if not todo and not cross:
        return   # interior point — nothing to do, no file written

    seed = args.seed + point_id + 100000 * args.round
    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} polish round '
          f'{args.round} {model.p1_name}={model.p1_vals[ix]:.3f} '
          f'{model.p2_name}={model.p2_vals[iy]:.3f}  keys={todo or "none"} '
          f'cross={cross}  best d4={f4_known:.6f} d6={f6_known:.6f}',
          flush=True)

    gate = build_gate(model, ix, iy, np.load(base))

    results = {}
    for off, (key, (s_key, d_ext)) in enumerate(KEYS.items()):
        self_val, self_S, nb_list = info[key]
        if key not in todo:
            results[key] = (self_val, self_S)
            continue
        # d6 additionally seeds with the embedded self/neighbour d4 frames
        extra = []
        if key == 'opt_fra_6':
            if S4_known is not None:
                extra.append(embed_S4_to_S6(S4_known))
            extra += [embed_S4_to_S6(S) for _, S in info['opt_fra_4'][2]]
        results[key] = refine_key(gate, key, d_ext, self_val, self_S,
                                  nb_list, args, seed + off, extra_frames=extra)
        print(f'  d{d_ext}: {self_val:.6f} -> {results[key][0]:.6f} '
              f'({len(nb_list)} neighbour seed(s))', flush=True)

    # ── cross-d_ext step: embed the best d=4 frame into d=6 ─────────────────
    f4, S4 = results['opt_fra_4']
    f6, S6 = results['opt_fra_6']
    if f4 < f6 - TOL and S4 is not None:
        f6c, S6c = refine_key(gate, 'opt_fra_6', 6, f6, S6, [],
                              args, seed + 7, extra_frames=[embed_S4_to_S6(S4)])
        if f6c < f6 - TOL:
            print(f'  cross-seed d4->d6: {f6:.6f} -> {f6c:.6f}', flush=True)
            results['opt_fra_6'] = (f6c, S6c)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'round': np.array(args.round),
               'ix': np.array(ix), 'iy': np.array(iy)}
    for key, (s_key, _) in KEYS.items():
        val, S = results[key]
        payload[key] = np.array(val)
        if S is not None:
            payload[s_key] = np.asarray(S)
    np.savez(out, **payload)
    print(f'  saved {out.name}: d4 {f4_known:.6f}->{results["opt_fra_4"][0]:.6f}  '
          f'd6 {f6_known:.6f}->{results["opt_fra_6"][0]:.6f}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',        type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id',      type=int, required=True)
    p.add_argument('--n_chunks',     type=int, default=1)
    p.add_argument('--round',        type=int, required=True,
                   help='polish round (1, 2, ...)')
    p.add_argument('--out_dir',      type=str, default='results_trotter_v3')
    p.add_argument('--n_restarts',   type=int, default=3)
    p.add_argument('--fra_maxfev_4', type=int, default=3000)
    p.add_argument('--fra_maxfev_6', type=int, default=2000)
    p.add_argument('--polish_iters', type=int, default=300,
                   help='Polyak floor-polish subgradient steps')
    p.add_argument('--fra_tol',      type=float, default=1e-6,
                   help='tolerance for opt_fra == 1 (framable floor)')
    p.add_argument('--seed',         type=int, default=0)
    args = p.parse_args()

    model = MODELS[args.model]
    N = model.N_TOTAL

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        run_point(model, args.task_id, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name} polish round '
          f'{args.round}: {len(point_ids)} points', flush=True)
    for pid in point_ids:
        run_point(model, pid, args)


if __name__ == '__main__':
    main()
