"""
Item 5: quick neighbour-seeded re-optimisation of opt_fra_4 / opt_fra_6 on the
DT_BASE-line grid (scripts/trotter_dtbase_line_worker.py), one fixed DT_BASE
(base_idx, one of the bottom N_BASE_KEEP=10 values -- item 1) at a time.

This is the same "quick" boundary rule as the established
scripts/trotter_scan_quick_refine_worker.py pipeline (results_trotter_v3),
adapted to read/write the results_dtbase_line/<tag>/base_<idx>.npz layout
instead of pt_<ix>_<iy>.npz: a grid point (ix, iy) in the model's (p1, p2)
grid is only touched if its best-known value exceeds 1 + fra_tol while at
least one 4-connected neighbour sits at the framable floor (opt_fra == 1, both
up to fra_tol).  All other points are left untouched -- much cheaper than a
full re-optimisation and it sharpens the boundary of the framable region.  A
point also qualifies for the cross-d_ext step alone: if its best d=4 value is
below its best d=6 value, the d=4 frame is embedded into d=6 and re-optimised.

Run 10 sequential rounds (--round 1..10, per the task spec's "10 stages of
neighbor refining"); each round reads the base file plus every earlier
quick-refine round, so the framable floor propagates outward one ring per
round.  Neighbour frames are seeded directly from the flat parameter vectors
opt_x_4/opt_x_6 that trotter_dtbase_line_worker.py now stores (no separate S
<-> x conversion needed -- optimise_framability's return_x=True and
extra_init_xs both speak the same flat encoding).

Grid (matches the model's p1_vals / p2_vals, same convention as
trotter_scan_quick_refine_worker.py):
    ix in 0 .. N_X-1 ; iy in 0 .. N_Y-1 ; point_id = ix * N_Y + iy

Reads:  results_dtbase_line/<tag>/base_<base_idx:03d>.npz                (self + neighbours)
        results_dtbase_line/<tag>/base_<base_idx:03d>_qrefine_r*.npz     (earlier rounds)
Writes: results_dtbase_line/<tag>/base_<base_idx:03d>_qrefine_r<NN>.npz  (boundary points only)

Points whose base file doesn't exist yet (most of the 51x51 grid, at any given
time) are silently skipped -- not an error -- so this is safe to run over the
full grid regardless of how much of the DT_BASE sweep has completed.

Usage (single point, single round):
    python scripts/trotter_dtbase_line_quick_refine_worker.py \
        --model model3 --base_idx 0 --round 1 --task_id 0

Usage (strided across an array):
    python scripts/trotter_dtbase_line_quick_refine_worker.py \
        --model model3 --base_idx 0 --round 1 \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, bond_trotter_gate, choose_dt, DIM_DEFAULT
from dissipative_PT import optimise_framability, embed_frame_params
from trotter_dtbase_line_worker import (  # noqa: E402  (scripts/ is on the path)
    base_grid, point_tag,
)

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TOL = 1e-9

# key -> (flat-param key, d_ext_single)
KEYS = {'opt_fra_4': ('opt_x_4', 4), 'opt_fra_6': ('opt_x_6', 6)}


def _point_dir(out_dir: Path, model_name: str, p1: float, p2: float) -> Path:
    return out_dir / point_tag(model_name, p1, p2)


def _base_path(out_dir: Path, model_name: str, p1: float, p2: float,
               base_idx: int) -> Path:
    return _point_dir(out_dir, model_name, p1, p2) / f'base_{base_idx:03d}.npz'


def _all_paths(out_dir: Path, model_name: str, p1: float, p2: float, base_idx: int):
    """Base file + every quick-refine round for this (point, base_idx)."""
    d = _point_dir(out_dir, model_name, p1, p2)
    paths = [_base_path(out_dir, model_name, p1, p2, base_idx)]
    paths += sorted(d.glob(f'base_{base_idx:03d}_qrefine_r*.npz'))
    return [p for p in paths if p.exists()]


def _best_known(out_dir: Path, model_name: str, p1: float, p2: float,
                base_idx: int, key: str, x_key: str):
    """Lowest-framability (value, flat_x) over the base file + every quick-refine
    round for this point at this base_idx."""
    best_val, best_x = np.inf, None
    for f in _all_paths(out_dir, model_name, p1, p2, base_idx):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        if key not in d.files:
            continue
        v = float(d[key])
        if np.isfinite(v) and v < best_val:
            x = np.asarray(d[x_key], dtype=float) if x_key in d.files else None
            best_val, best_x = v, x
    return best_val, best_x


def _neighbor_x(out_dir: Path, model, ix: int, iy: int, base_idx: int,
                key: str, x_key: str):
    """(value, flat_x) of the best-known result of EVERY 4-connected neighbour in
    the model's (p1, p2) grid, ordered by ascending value (see
    trotter_scan_quick_refine_worker._neighbor_frames for why every neighbour,
    not just the best, is used as a seed)."""
    out = []
    for dx, dy in NEIGHBORS:
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < model.N_X and 0 <= ny < model.N_Y:
            p1n, p2n = float(model.p1_vals[nx]), float(model.p2_vals[ny])
            v, x = _best_known(out_dir, model.name, p1n, p2n, base_idx, key, x_key)
            if x is not None:
                out.append((v, x))
    out.sort(key=lambda t: t[0])
    return out


def run_point(model, point_id: int, base_idx: int, args) -> None:
    out_dir = Path(args.out_dir)
    ix = point_id // model.N_Y
    iy = point_id % model.N_Y
    p1, p2 = float(model.p1_vals[ix]), float(model.p2_vals[iy])

    pt_dir = _point_dir(out_dir, model.name, p1, p2)
    out = pt_dir / f'base_{base_idx:03d}_qrefine_r{args.round:02d}.npz'
    if out.exists():
        print(f'[skip] {pt_dir.name}/{out.name} already exists', flush=True)
        return

    base = _base_path(out_dir, model.name, p1, p2, base_idx)
    if not base.exists():
        return   # point not swept at this base_idx yet -- nothing to refine

    # ── boundary detection: best-known self vs best-known neighbours ────────
    info = {}
    todo = []
    for key, (x_key, _) in KEYS.items():
        self_val, self_x = _best_known(out_dir, model.name, p1, p2, base_idx,
                                       key, x_key)
        nb_list = _neighbor_x(out_dir, model, ix, iy, base_idx, key, x_key)
        nb_val = nb_list[0][0] if nb_list else np.inf
        info[key] = (self_val, self_x, nb_val, nb_list)
        if self_val > 1.0 + args.fra_tol and nb_val <= 1.0 + args.fra_tol:
            todo.append(key)

    f4_known = info['opt_fra_4'][0]
    f6_known = info['opt_fra_6'][0]
    cross = (f4_known < f6_known - TOL
             and info['opt_fra_4'][1] is not None
             and f6_known > 1.0 + args.fra_tol)

    if not np.isfinite(f4_known) and not np.isfinite(f6_known):
        return   # base file has neither key yet (not backfilled) -- skip quietly
    if not todo and not cross:
        return   # interior point -- nothing to do, no file written

    d_base = np.load(base, allow_pickle=True)
    dim = int(d_base['dim']) if 'dim' in d_base.files else DIM_DEFAULT
    base_val = float(base_grid()[base_idx])
    dt = float(d_base['dt']) if 'dt' in d_base.files else None
    seed = args.seed + point_id + 100000 * args.round + 1000000 * base_idx

    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} base_idx={base_idx} '
          f'(DT_BASE={base_val:.3f}) quick round {args.round} '
          f'{model.p1_name}={p1:.3f} {model.p2_name}={p2:.3f}  '
          f'boundary keys={todo or "none"} cross={cross}  '
          f'best d4={f4_known:.6f} d6={f6_known:.6f}', flush=True)

    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    if dt is None:
        dt = choose_dt(H1, H2, jumps1, jumps2, base=base_val)
    gate = bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)

    maxfev = {'opt_fra_4': args.fra_maxfev_4, 'opt_fra_6': args.fra_maxfev_6}
    results = {}
    for off, (key, (x_key, d_ext)) in enumerate(KEYS.items()):
        self_val, self_x, nb_val, nb_list = info[key]
        if key not in todo:
            results[key] = (self_val, self_x)
            continue
        seeds = ([self_x] if self_x is not None else []) + [x for _, x in nb_list]
        f, x = optimise_framability(gate, d_ext, n_restarts=args.n_restarts,
                                    maxfev=maxfev[key], seed=seed + off,
                                    extra_init_xs=seeds if seeds else None,
                                    return_x=True)
        if self_x is not None and self_val <= f:
            results[key] = (self_val, self_x)
        else:
            results[key] = (f, x)
        print(f'  d{d_ext}: {self_val:.6f} -> {results[key][0]:.6f} '
              f'(framable neighbour {nb_val:.6f})', flush=True)

    # ── cross-d_ext step: embed the best d=4 frame into d=6 ─────────────────
    f4, x4 = results['opt_fra_4']
    f6, x6 = results['opt_fra_6']
    if f4 < f6 - TOL and x4 is not None:
        x6_seed = embed_frame_params(x4, 4, 6)
        f6c, x6c = optimise_framability(gate, 6, n_restarts=args.n_restarts,
                                        maxfev=args.fra_maxfev_6, seed=seed + 1,
                                        extra_init_xs=[x6_seed], return_x=True)
        if f6c < f6:
            print(f'  cross-seed d4->d6: {f6:.6f} -> {f6c:.6f}', flush=True)
            f6, x6 = f6c, x6c
            results['opt_fra_6'] = (f6, x6)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {'round': np.array(args.round), 'base_idx': np.array(base_idx),
              'ix': np.array(ix), 'iy': np.array(iy)}
    for key, (x_key, _) in KEYS.items():
        val, x = results[key]
        payload[key] = np.array(val)
        if x is not None:
            payload[x_key] = np.asarray(x)
    np.savez(out, **payload)
    print(f'  saved {pt_dir.name}/{out.name}: '
          f'd4 {f4_known:.6f}->{results["opt_fra_4"][0]:.6f}  '
          f'd6 {f6_known:.6f}->{results["opt_fra_6"][0]:.6f}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',        type=str, required=True, choices=list(MODELS))
    p.add_argument('--base_idx',     type=int, required=True,
                   help='which of the bottom-10 DT_BASE values to refine (0..9)')
    p.add_argument('--task_id',      type=int, required=True)
    p.add_argument('--n_chunks',     type=int, default=1)
    p.add_argument('--round',        type=int, required=True,
                   help='quick-refinement round (1 .. 10)')
    p.add_argument('--out_dir',      type=str, default='results_dtbase_line')
    p.add_argument('--n_restarts',   type=int, default=3)
    p.add_argument('--fra_maxfev_4', type=int, default=1000)
    p.add_argument('--fra_maxfev_6', type=int, default=500)
    p.add_argument('--fra_tol',      type=float, default=1e-6,
                   help='tolerance for opt_fra == 1 (framable floor)')
    p.add_argument('--seed',         type=int, default=0)
    args = p.parse_args()

    model = MODELS[args.model]
    N = model.N_TOTAL
    if not (0 <= args.base_idx < len(base_grid())):
        print(f'ERROR: base_idx must be in [0, {len(base_grid())})', file=sys.stderr)
        sys.exit(1)

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        run_point(model, args.task_id, args.base_idx, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name} base_idx='
          f'{args.base_idx} quick round {args.round}: {len(point_ids)} points',
          flush=True)
    for pid in point_ids:
        run_point(model, pid, args.base_idx, args)


if __name__ == '__main__':
    main()
