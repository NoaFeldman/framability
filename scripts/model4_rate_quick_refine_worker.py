"""
Quick neighbour refining of the optimised HEISENBERG framability RATES
(rate_heis_4 / rate_heis_6) of the model4 rate scan -- one round, one point.

This is the rate-picture twin of scripts/trotter_dtbase_line_quick_refine_worker.py
(and of the results_trotter_v3 quick-refine pipeline it copies), with the
finite-dt gate optimiser replaced by the dt-free generator optimiser:

    finite dt : optimise_framability(gate, d)          floor = 1
    rate      : framability_rate.minimize_rate(L^T, d) floor = 0

The "quick" boundary rule is the established one, restated at the rate floor.
A grid point (ix, iy) is touched for key k only if its best-known rate sits
ABOVE the floor while at least one 4-connected neighbour sits AT the floor
(both to within --rate_tol).  Everything else is left untouched -- far cheaper
than a full re-optimisation, and it sharpens exactly the boundary of the
region where the frame does not inflate.  A point also qualifies through the
cross-d_ext step alone: if its best d=4 rate is below its best d=6 rate (which
can only be an optimiser artifact -- six columns contain four), the d=4 frame
is embedded into d=6 via dissipative_PT.embed_frame_params and re-optimised.

Each round does two stages per qualifying key, both reusing framability_rate:

  1. propagation (cheap, no optimisation): every neighbour's frame is scored
     on THIS point's generator with the batched LP generator_log_norm.  A
     finite value is a rigorous upper bound on the point's true minimum, so
     keeping min(own, best neighbour) is always sound; an improvement is
     re-confirmed with the independent per-column LP
     generator_log_norm_reference before it is accepted.  This is the same
     two-LP accept rule as framability_rate.neighbor_refine_rates, sharded
     one point per task instead of run as a whole-grid Gauss-Seidel loop.

  2. quick re-optimisation: minimize_rate seeded with the point's own frame
     plus every neighbour frame, at reduced restarts/maxfev.  The incumbent is
     kept unless the new value strictly beats it, so a round can never make a
     point worse.

Rounds are sequential -- round r reads the base file plus every earlier round,
so the floor propagates outward one ring per round.  Run 10 rounds via
scripts/submit_model4_rate_quick_refine.sh.

--model picks the rate-panel model (model4_rate_panels_worker.SUPPORTED_MODELS,
default model4), e.g. model10, the Shibata-Katsura dissipative quantum Ising
chain (https://arxiv.org/abs/1904.12505); the bond generator uses the model's
own ModelSpec.dim unless --dim is given.

Reads:  results_<model>_rate/<model>/pt_<ix>_<iy>.npz              (base scan)
        results_<model>_rate/<model>/pt_<ix>_<iy>_qrefine_r*.npz   (earlier rounds)
Writes: results_<model>_rate/<model>/pt_<ix>_<iy>_qrefine_r<NN>.npz
        (qualifying points only; interior points write nothing)

Points with no base file are skipped silently, so this is safe to run over the
whole grid however much of the base scan has landed.

Usage:
    python scripts/model4_rate_quick_refine_worker.py --round 1 --task_id 0 --n_chunks 200
    python scripts/model4_rate_quick_refine_worker.py --round 1 --task_id 17   # single point
    python scripts/model4_rate_quick_refine_worker.py --model model10 --round 1 --task_id 0 --n_chunks 200
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
from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,   # noqa: E402
                                      DIM_DEFAULT)
from dissipative_PT import embed_frame_params                           # noqa: E402
from optimize_framability import _FIXED_COLS, _kron_power               # noqa: E402
from framability_rate import (minimize_rate, generator_log_norm,        # noqa: E402
                              generator_log_norm_reference, RATE_VERSION)
from model4_rate_panels_worker import (MODEL_NAME,                      # noqa: E402
                                       SUPPORTED_MODELS)

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TOL = 1e-9

# The framability-rate floor.  mu*(D) >= max Re lambda(A), which is 0 for a
# trace-preserving generator (the steady state pins an eigenvalue at 0), so
# mu* = 0 is the rate-picture image of framability = 1.
RATE_FLOOR = 0.0

# rate key -> (frame key, d_ext_single)
KEYS = {'rate_heis_4': ('S_heis_4', 4), 'rate_heis_6': ('S_heis_6', 6)}


def _pt_paths(pt_dir: Path, ix: int, iy: int):
    """Base file + every refine round written so far for this point: quick
    (_qrefine_r*) and full (_nrefine_r*, scripts/model4_rate_nb_refine_worker.py)."""
    base = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    rounds = sorted(pt_dir.glob(f'pt_{ix:03d}_{iy:03d}_*refine_r*.npz'))
    rounds += sorted(pt_dir.glob(f'pt_{ix:03d}_{iy:03d}_gopt*.npz'))   # global re-opt
    return [p for p in [base, *rounds] if p.exists()]


def best_known(pt_dir: Path, ix: int, iy: int, key: str, s_key: str):
    """Lowest (rate, frame) over the base file and every quick-refine round."""
    best_val, best_S = np.inf, None
    for f in _pt_paths(pt_dir, ix, iy):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if key not in d.files:
            continue
        v = float(d[key])
        if np.isfinite(v) and v < best_val:
            best_val = v
            best_S = np.asarray(d[s_key], float) if s_key in d.files else None
    return best_val, best_S


def neighbor_frames(pt_dir: Path, model, ix: int, iy: int, key: str, s_key: str):
    """(value, frame) of every 4-connected neighbour's best-known result,
    ascending by value.  Every neighbour is used as a seed, not just the best:
    a worse neighbour can still carry the frame that transfers to this point."""
    out = []
    for dx, dy in NEIGHBORS:
        jx, jy = ix + dx, iy + dy
        if 0 <= jx < model.N_X and 0 <= jy < model.N_Y:
            v, S = best_known(pt_dir, jx, jy, key, s_key)
            if S is not None:
                out.append((v, S))
    out.sort(key=lambda t: t[0])
    return out


def _embed_S(S4: np.ndarray, d_small: int, d_large: int) -> np.ndarray:
    """Embed a d_small frame into d_large by replicating its last free column.

    Reuses dissipative_PT.embed_frame_params on the free-column block (the
    identity column is pinned and re-attached here), so the monotonicity
    argument carries over verbatim: kron(S_large, S_large) contains every
    column of kron(S_small, S_small), hence the larger frame's rate can only
    be <= the smaller one's -- a sound warm start.
    """
    free = np.asarray(S4, float)[:, 1:]
    padded = embed_frame_params(free.ravel(), d_small, d_large)
    return np.hstack([_FIXED_COLS, padded.reshape(4, d_large - 1)])


def propagate(A, own_val, own_S, nb_list):
    """Stage 1: best neighbour frame scored on this point's own generator.

    Returns (value, frame, source_index_in_nb_list) improving on the incumbent,
    or the incumbent unchanged.  The cheap batched LP screens; only a candidate
    that beats the incumbent is paid for with the independent per-column LP.
    """
    best_v, best_S, best_j = own_val, own_S, -1
    for j, (_, S_nb) in enumerate(nb_list):
        v = generator_log_norm(_kron_power(S_nb, 2), A)
        if np.isfinite(v) and v < best_v - TOL:
            v_ref = generator_log_norm_reference(_kron_power(S_nb, 2), A)
            if np.isfinite(v_ref) and v_ref < best_v - TOL:
                best_v, best_S, best_j = v_ref, S_nb.copy(), j
    return best_v, best_S, best_j


def run_point(model, point_id: int, args) -> None:
    pt_dir = Path(args.out_dir) / model.name
    ix, iy = point_id // model.N_Y, point_id % model.N_Y
    p1, p2 = float(model.p1_vals[ix]), float(model.p2_vals[iy])

    out = pt_dir / f'pt_{ix:03d}_{iy:03d}_qrefine_r{args.round:02d}.npz'
    if out.exists():
        print(f'[skip] {out.name} already exists', flush=True)
        return
    if not (pt_dir / f'pt_{ix:03d}_{iy:03d}.npz').exists():
        return                      # point not scanned yet -- nothing to refine

    # ── boundary detection: best-known self vs best-known neighbours ────────
    info, todo = {}, []
    for key, (s_key, _) in KEYS.items():
        self_val, self_S = best_known(pt_dir, ix, iy, key, s_key)
        nb_list = neighbor_frames(pt_dir, model, ix, iy, key, s_key)
        nb_val = nb_list[0][0] if nb_list else np.inf
        info[key] = (self_val, self_S, nb_val, nb_list)
        if (self_val > RATE_FLOOR + args.rate_tol
                and nb_val <= RATE_FLOOR + args.rate_tol):
            todo.append(key)

    v4, S4 = info['rate_heis_4'][0], info['rate_heis_4'][1]
    v6 = info['rate_heis_6'][0]
    cross = (v4 < v6 - TOL and S4 is not None
             and v6 > RATE_FLOOR + args.rate_tol)

    if not np.isfinite(v4) and not np.isfinite(v6):
        return                      # base file has neither key -- skip quietly
    if not todo and not cross:
        return                      # interior point -- no file written

    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} round {args.round} '
          f'{model.p1_name}={p1:.3f} {model.p2_name}={p2:.3f}  '
          f'keys={todo or "none"} cross={cross}  '
          f'best d4={v4:.6e} d6={v6:.6e}', flush=True)

    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    L = build_bond_lindbladian(H1, H2, jumps1, jumps2, args.dim).real
    A = L.T
    seed = args.seed + point_id + 100000 * args.round

    maxfev = {'rate_heis_4': args.maxfev_4, 'rate_heis_6': args.maxfev_6}
    results = {k: (info[k][0], info[k][1]) for k in KEYS}

    for off, key in enumerate(KEYS):
        if key not in todo:
            continue
        self_val, self_S, nb_val, nb_list = info[key]
        d_ext = KEYS[key][1]

        # stage 1: propagation (a handful of LPs)
        val, S, j = propagate(A, self_val, self_S, nb_list)
        if j >= 0:
            print(f'  d{d_ext}: propagated {self_val:.6e} -> {val:.6e} '
                  f'from neighbour {j}', flush=True)

        # stage 2: quick re-optimisation seeded with own + all neighbour frames
        seeds = ([S] if S is not None else []) + [f for _, f in nb_list]
        S_new, mu_new, _ = minimize_rate(
            A, d_ext, n_restarts=args.n_restarts, maxfev=maxfev[key],
            seed=seed + off, verbose=False, polish_iters=args.polish,
            extra_init_S=seeds or None)
        if np.isfinite(mu_new) and mu_new < val - TOL:
            val, S = mu_new, S_new
        results[key] = (val, S)
        print(f'  d{d_ext}: {self_val:.6e} -> {val:.6e}  '
              f'(floor neighbour {nb_val:.6e})  '
              f'[{time.perf_counter() - t0:.0f}s]', flush=True)

    # ── cross-d_ext step: embed the best d=4 frame into d=6 ────────────────
    if cross:
        v6_now, S6_now = results['rate_heis_6']
        S4_now = results['rate_heis_4'][1]
        if S4_now is not None:
            seed6 = _embed_S(S4_now, 4, 6)
            v_emb = generator_log_norm_reference(_kron_power(seed6, 2), A)
            if np.isfinite(v_emb) and v_emb < v6_now - TOL:
                v6_now, S6_now = v_emb, seed6
            S_new, mu_new, _ = minimize_rate(
                A, 6, n_restarts=args.n_restarts, maxfev=args.maxfev_6,
                seed=seed + 7, verbose=False, polish_iters=args.polish,
                extra_init_S=[seed6] + ([S6_now] if S6_now is not None else []))
            if np.isfinite(mu_new) and mu_new < v6_now - TOL:
                v6_now, S6_now = mu_new, S_new
            print(f'  cross d4->d6: {results["rate_heis_6"][0]:.6e} -> '
                  f'{v6_now:.6e}', flush=True)
            results['rate_heis_6'] = (v6_now, S6_now)

    improved = {k: results[k][0] < info[k][0] - TOL for k in KEYS}
    if not any(improved.values()):
        print(f'  no improvement; nothing written '
              f'({time.perf_counter() - t0:.0f}s)', flush=True)
        return

    # {p1_name: p1, p2_name: p2} keeps model4's gamma / gamma_p keys unchanged
    payload = dict(model=model.name, ix=ix, iy=iy,
                   **{model.p1_name: p1, model.p2_name: p2},
                   dim=args.dim, round=args.round, rate_version=RATE_VERSION,
                   rate_floor=RATE_FLOOR, rate_tol=args.rate_tol)
    for key, (s_key, _) in KEYS.items():
        payload[key] = results[key][0]
        payload[f'{key}_prev'] = info[key][0]
        payload[f'{key}_improved'] = improved[key]
        if results[key][1] is not None:
            payload[s_key] = results[key][1]
    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    print(f'  saved {out.name}  d4={results["rate_heis_4"][0]:.6e} '
          f'd6={results["rate_heis_6"][0]:.6e}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, default=MODEL_NAME,
                   choices=SUPPORTED_MODELS,
                   help='rate-panel model whose scan is refined')
    p.add_argument('--round',    type=int, required=True,
                   help='quick-refine round (1..10); round r reads r-1')
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--out_dir',  type=str, default=None,
                   help='default results_<model>_rate')
    p.add_argument('--dim',      type=int, default=None,
                   help="bond Trotter convention; default: the model's "
                        f'ModelSpec.dim (DIM_DEFAULT = {DIM_DEFAULT})')
    p.add_argument('--rate_tol', type=float, default=1e-6,
                   help='a rate within this of 0 counts as sitting on the '
                        'framable floor (matches the collect script\'s '
                        'CONTOUR_TOL, so the refined boundary and the plotted '
                        'white contour mean the same thing)')
    p.add_argument('--n_restarts', type=int, default=3,
                   help='minimize_rate restarts ("quick": the base scan used 8)')
    p.add_argument('--maxfev_4', type=int, default=1000)
    p.add_argument('--maxfev_6', type=int, default=500)
    p.add_argument('--polish',   type=int, default=150)
    p.add_argument('--seed',     type=int, default=0)
    args = p.parse_args()

    model = MODELS[args.model]
    if args.out_dir is None:
        args.out_dir = f'results_{args.model}_rate'
    if args.dim is None:
        args.dim = model.dim
    n_total = model.N_TOTAL

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
            sys.exit(1)
        run_point(model, args.task_id, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] round {args.round}: '
          f'{len(ids)} of {n_total} points', flush=True)
    for pid in ids:
        run_point(model, pid, args)


if __name__ == '__main__':
    main()
