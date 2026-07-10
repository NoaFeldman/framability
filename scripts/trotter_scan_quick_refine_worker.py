"""
Quick neighbour-seeded refinement of the optimised framabilities (opt_fra_4 /
opt_fra_6) for the generic Trotter Lindbladian scan.

Unlike trotter_scan_refine_worker.py (which re-optimises every grid point),
this quick variant only touches BOUNDARY points: those whose best-known
framability exceeds 1 while at least one 4-connected neighbour sits at the
framable floor (opt_fra == 1, both up to --fra_tol).  All other points are
left untouched, so each round is much cheaper and sharpens the boundary of
the framable region.  A point also qualifies for the cross-d_ext step alone:
if its best d=4 value is below its best d=6 value, the d=4 frame is embedded
into d=6 and re-optimised (guarantees opt_fra_6 <= opt_fra_4).

Run several sequential rounds (--round 1, 2, ...); each round reads the base
scan plus every earlier refine AND quick-refine file, so the framable floor
propagates outward across the grid one ring per round.

Grid (matches the model's p1_vals / p2_vals):
    ix in 0 .. N_X-1 ; iy in 0 .. N_Y-1 ; point_id = ix * N_Y + iy

Reads:  <out_dir>/<model>/pt_<ix>_<iy>.npz          (base scan; self + neighbours)
        <out_dir>/<model>/pt_refine_r*_<ix>_<iy>.npz
        <out_dir>/<model>/pt_qrefine_r*_<ix>_<iy>.npz
Writes: <out_dir>/<model>/pt_qrefine_r<NN>_<ix>_<iy>.npz   (boundary points only)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (
    MODELS, bond_trotter_gate, choose_dt, DIM_DEFAULT,
)
from dissipative_PT import (
    optimise_framability, embed_frame_params, frame_from_params, params_from_frame,
)

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TOL = 1e-9

# key -> (frame key, d_ext_single)
KEYS = {'opt_fra_4': ('opt_S_4', 4), 'opt_fra_6': ('opt_S_6', 6)}


def _base_path(out_dir: Path, model, ix: int, iy: int) -> Path:
    return out_dir / model.name / f'pt_{ix:03d}_{iy:03d}.npz'


def _all_paths(out_dir: Path, model, ix: int, iy: int):
    """Base scan file + every refinement file for this point (all flavours:
    full refine, quick refine, cross-eval sweep, island polish, floor hunt)."""
    d = out_dir / model.name
    paths = [_base_path(out_dir, model, ix, iy)]
    for pat in ('pt_refine_r*', 'pt_qrefine_r*', 'pt_xeval_r*',
                'pt_polish_r*', 'pt_fhunt_r*', 'pt_verdict_r*'):
        paths += sorted(d.glob(f'{pat}_{ix:03d}_{iy:03d}.npz'))
    return [p for p in paths if p.exists()]


def _best_known(out_dir: Path, model, ix: int, iy: int, key: str, s_key: str):
    """Lowest-framability (value, frame S) over base + every refine file."""
    best_val, best_S = np.inf, None
    for f in _all_paths(out_dir, model, ix, iy):
        try:
            d = np.load(f)
        except Exception:
            continue
        if key not in d or s_key not in d:
            continue
        v = float(d[key])
        if np.isfinite(v) and v < best_val:
            S = np.asarray(d[s_key], dtype=float)
            if np.all(np.isfinite(S)):
                best_val, best_S = v, S
    return best_val, best_S


def _neighbor_frames(out_dir: Path, model, ix: int, iy: int, key: str, s_key: str):
    """(value, frame) of the best-known result of EVERY 4-connected neighbour,
    ordered by ascending value.  All of them are used as seeds: optimal frames
    come in distinct symmetry branches and the branch can switch across the
    grid, so the single best neighbour may be the wrong one to copy while
    another neighbour's frame transfers cleanly."""
    out = []
    for dx, dy in NEIGHBORS:
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < model.N_X and 0 <= ny < model.N_Y:
            v, S = _best_known(out_dir, model, nx, ny, key, s_key)
            if S is not None:
                out.append((v, S))
    out.sort(key=lambda t: t[0])
    return out


def run_point(model, point_id: int, args) -> None:
    out_dir = Path(args.out_dir)
    ix = point_id // model.N_Y
    iy = point_id %  model.N_Y
    out = out_dir / model.name / f'pt_qrefine_r{args.round:02d}_{ix:03d}_{iy:03d}.npz'

    if out.exists():
        print(f'[skip] {model.name}/{out.name} already exists', flush=True)
        return

    base = _base_path(out_dir, model, ix, iy)
    if not base.exists():
        print(f'ERROR: base scan file {base} not found — run the scan first',
              file=sys.stderr)
        sys.exit(1)

    # ── boundary detection: best-known self vs best-known neighbours ────────
    info = {}
    todo = []
    for key, (s_key, _) in KEYS.items():
        self_val, self_S = _best_known(out_dir, model, ix, iy, key, s_key)
        nb_list = _neighbor_frames(out_dir, model, ix, iy, key, s_key)
        nb_val = nb_list[0][0] if nb_list else np.inf
        info[key] = (self_val, self_S, nb_val, nb_list)
        if self_val > 1.0 + args.fra_tol and nb_val <= 1.0 + args.fra_tol:
            todo.append(key)

    f4_known = info['opt_fra_4'][0]
    f6_known = info['opt_fra_6'][0]
    cross = (f4_known < f6_known - TOL
             and info['opt_fra_4'][1] is not None
             and f6_known > 1.0 + args.fra_tol)

    if not todo and not cross:
        return   # interior point — nothing to do, no file written

    d_base = np.load(base)
    dim = int(d_base['dim']) if 'dim' in d_base else DIM_DEFAULT
    dt  = float(d_base['dt']) if 'dt' in d_base else None
    p1 = float(model.p1_vals[ix])
    p2 = float(model.p2_vals[iy])
    seed = args.seed + point_id + 100000 * args.round

    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} quick round {args.round} '
          f'{model.p1_name}={p1:.3f} {model.p2_name}={p2:.3f}  '
          f'boundary keys={todo or "none"} cross={cross}  '
          f'best d4={f4_known:.6f} d6={f6_known:.6f}', flush=True)

    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    if dt is None:   # legacy pt file without a stored dt: re-derive it
        dt = model.dt if model.dt is not None else choose_dt(H1, H2, jumps1, jumps2)
    gate = bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)

    maxfev = {'opt_fra_4': args.fra_maxfev_4, 'opt_fra_6': args.fra_maxfev_6}
    results = {}
    for off, (key, (s_key, d_ext)) in enumerate(KEYS.items()):
        self_val, self_S, nb_val, nb_list = info[key]
        if key not in todo:
            results[key] = (self_val, self_S)
            continue
        # seed with the point's own frame plus EVERY neighbour's frame
        seed_frames = ([self_S] if self_S is not None else [])
        seed_frames += [S for _, S in nb_list]
        seeds = [params_from_frame(S) for S in seed_frames]
        f, x = optimise_framability(gate, d_ext, n_restarts=args.n_restarts,
                                    maxfev=maxfev[key], seed=seed + off,
                                    extra_init_xs=seeds if seeds else None,
                                    return_x=True)
        if self_S is not None and self_val <= f:
            results[key] = (self_val, self_S)
        else:
            results[key] = (f, frame_from_params(x, d_ext))
        print(f'  d{d_ext}: {self_val:.6f} -> {results[key][0]:.6f} '
              f'(framable neighbour {nb_val:.6f})', flush=True)

    # ── cross-d_ext step: embed the best d=4 frame into d=6 ─────────────────
    f4, S4 = results['opt_fra_4']
    f6, S6 = results['opt_fra_6']
    if f4 < f6 - TOL and S4 is not None:
        x6_seed = embed_frame_params(params_from_frame(S4), 4, 6)
        f6c, x6c = optimise_framability(gate, 6, n_restarts=args.n_restarts,
                                        maxfev=args.fra_maxfev_6, seed=seed + 1,
                                        extra_init_xs=[x6_seed], return_x=True)
        if f6c < f6:
            print(f'  cross-seed d4->d6: {f6:.6f} -> {f6c:.6f}', flush=True)
            f6, S6 = f6c, frame_from_params(x6c, 6)
            results['opt_fra_6'] = (f6, S6)

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
                   help='quick-refinement round (1, 2, ...)')
    p.add_argument('--out_dir',      type=str, default='results_trotter_v3')
    p.add_argument('--n_restarts',   type=int, default=3)
    p.add_argument('--fra_maxfev_4', type=int, default=1000)
    p.add_argument('--fra_maxfev_6', type=int, default=500)
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
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name} quick round '
          f'{args.round}: {len(point_ids)} points', flush=True)
    for pid in point_ids:
        run_point(model, pid, args)


if __name__ == '__main__':
    main()
