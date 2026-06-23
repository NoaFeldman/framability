"""
Neighbour-seeded refinement of the optimised framabilities (opt_fra_4 /
opt_fra_6) for the generic Trotter Lindbladian scan.

For each grid point and each frame size d_ext_single in {4, 6}:
  1. Gather the best startup frame for this point (its base scan result plus any
     earlier refine round) and the best 4-connected neighbour's frame.
  2. Re-optimise this point's framability seeded from those frames; never
     regress below the best startup value.
Then a cross-d_ext step: if opt_fra_4 < opt_fra_6, embed the optimal d=4 frame
into d=6 and re-optimise d=6 from it (guarantees opt_fra_6 <= opt_fra_4).

Run 5 sequential rounds (--round 1 .. 5), each reading every earlier round so
information propagates across the grid.

Grid (matches the model's p1_vals / p2_vals):
    ix in 0 .. N_X-1 ; iy in 0 .. N_Y-1 ; point_id = ix * N_Y + iy

Reads:  <out_dir>/<model>/pt_<ix>_<iy>.npz          (base scan; self + neighbours)
Writes: <out_dir>/<model>/pt_refine_r<NN>_<ix>_<iy>.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (
    MODELS, FRA_REFINE_KEYS, bond_trotter_gate, DT_DEFAULT, DIM_DEFAULT,
)
from dissipative_PT import (
    optimise_framability, embed_frame_params, frame_from_params, params_from_frame,
)

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TOL = 1e-9


def _base_path(out_dir: Path, model, ix: int, iy: int) -> Path:
    return out_dir / model.name / f'pt_{ix:03d}_{iy:03d}.npz'


def _refine_paths(out_dir: Path, model, ix: int, iy: int):
    """Base scan file + every refine file for this point (any round)."""
    d = out_dir / model.name
    paths = [_base_path(out_dir, model, ix, iy)]
    paths += sorted(d.glob(f'pt_refine_r*_{ix:03d}_{iy:03d}.npz'))
    return [p for p in paths if p.exists()]


def _best_known(out_dir: Path, model, ix: int, iy: int, key: str, s_key: str):
    """Lowest-framability (value, frame S) over base + every refine file."""
    best_val, best_S = np.inf, None
    for f in _refine_paths(out_dir, model, ix, iy):
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


def _best_neighbor(out_dir: Path, model, ix: int, iy: int, key: str, s_key: str):
    best_val, best_S = np.inf, None
    for dx, dy in NEIGHBORS:
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < model.N_X and 0 <= ny < model.N_Y:
            v, S = _best_known(out_dir, model, nx, ny, key, s_key)
            if S is not None and v < best_val:
                best_val, best_S = v, S
    return best_val, best_S


def _refine_dext(out_dir, model, ix, iy, d_ext_single, key, s_key, gate,
                 n_restarts, maxfev, seed):
    """Re-optimise one frame size seeded from the best self + neighbour frames;
    never regresses below the best known startup."""
    self_val, self_S = _best_known(out_dir, model, ix, iy, key, s_key)
    nb_val,   nb_S   = _best_neighbor(out_dir, model, ix, iy, key, s_key)

    seeds = []
    if self_S is not None:
        seeds.append(params_from_frame(self_S))
    if nb_S is not None:
        seeds.append(params_from_frame(nb_S))

    print(f'  d{d_ext_single}: startup best={self_val:.6f} '
          f'(neighbour best={nb_val:.6f})', flush=True)
    f, x = optimise_framability(gate, d_ext_single, n_restarts=n_restarts,
                                maxfev=maxfev, seed=seed,
                                extra_init_xs=seeds if seeds else None,
                                return_x=True)
    if self_S is not None and self_val <= f:
        return self_val, params_from_frame(self_S)
    return f, x


def run_point(model, point_id: int, args) -> None:
    out_dir = Path(args.out_dir)
    ix = point_id // model.N_Y
    iy = point_id %  model.N_Y
    out = out_dir / model.name / f'pt_refine_r{args.round:02d}_{ix:03d}_{iy:03d}.npz'

    if out.exists():
        print(f'[skip] {model.name}/{out.name} already exists', flush=True)
        return

    base = _base_path(out_dir, model, ix, iy)
    if not base.exists():
        print(f'ERROR: base scan file {base} not found — run the scan first',
              file=sys.stderr)
        sys.exit(1)

    d_base = np.load(base)
    m4 = float(d_base['opt_fra_4'])
    m6 = float(d_base['opt_fra_6'])
    dim = int(d_base['dim']) if 'dim' in d_base else DIM_DEFAULT
    dt  = float(d_base['dt']) if 'dt' in d_base else DT_DEFAULT
    p1 = float(model.p1_vals[ix])
    p2 = float(model.p2_vals[iy])
    seed = args.seed + point_id

    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} round {args.round} '
          f'{model.p1_name}={p1:.3f} {model.p2_name}={p2:.3f}  '
          f'base d4={m4:.6f} d6={m6:.6f}', flush=True)

    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    gate = bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)

    f4, x4 = _refine_dext(out_dir, model, ix, iy, 4, 'opt_fra_4', 'opt_S_4',
                          gate, args.n_restarts, args.fra_maxfev_4, seed)
    f6, x6 = _refine_dext(out_dir, model, ix, iy, 6, 'opt_fra_6', 'opt_S_6',
                          gate, args.n_restarts, args.fra_maxfev_6, seed + 1)

    if f4 < f6 - TOL and x4 is not None:
        x6_seed = embed_frame_params(x4, 4, 6)
        f6c, x6c = optimise_framability(gate, 6, n_restarts=args.n_restarts,
                                        maxfev=args.fra_maxfev_6, seed=seed + 1,
                                        extra_init_xs=[x6_seed], return_x=True)
        if f6c < f6:
            print(f'  cross-seed d4->d6: {f6:.6f} -> {f6c:.6f}', flush=True)
            f6, x6 = f6c, x6c

    base_S4 = np.asarray(d_base['opt_S_4']) if 'opt_S_4' in d_base else frame_from_params(None, 4)
    base_S6 = np.asarray(d_base['opt_S_6']) if 'opt_S_6' in d_base else frame_from_params(None, 6)
    S4 = frame_from_params(x4, 4) if x4 is not None else base_S4
    S6 = frame_from_params(x6, 6) if x6 is not None else base_S6

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             round=np.array(args.round),
             opt_fra_4=np.array(f4), opt_fra_6=np.array(f6),
             opt_S_4=S4, opt_S_6=S6,
             base_fra_4=np.array(m4), base_fra_6=np.array(m6),
             ix=np.array(ix), iy=np.array(iy))
    print(f'  saved {out.name}: d4 {m4:.6f}->{f4:.6f}  d6 {m6:.6f}->{f6:.6f}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',        type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id',      type=int, required=True)
    p.add_argument('--n_chunks',     type=int, default=1)
    p.add_argument('--round',        type=int, required=True,
                   help='refinement round (1..5)')
    p.add_argument('--out_dir',      type=str, default='results_trotter')
    p.add_argument('--n_restarts',   type=int, default=5)
    p.add_argument('--fra_maxfev_4', type=int, default=1000)
    p.add_argument('--fra_maxfev_6', type=int, default=500)
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
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name} round {args.round}: '
          f'{len(point_ids)} points', flush=True)
    for pid in point_ids:
        run_point(model, pid, args)


if __name__ == '__main__':
    main()
