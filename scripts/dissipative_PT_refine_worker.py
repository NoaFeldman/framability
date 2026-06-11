"""
Neighbor-seeded refinement of the dissipative-PT optimised framabilities.

Grid (must match dissipative_PT.H_LIST / GAMMA_LIST):
    ih in 0..N_H-1   (h)
    ig in 0..N_G-1   (gamma)
    task_id = ih * N_G + ig   (0 … N_TOTAL-1 = 199)

For each grid point and each frame size d_ext ∈ {4, 6}:
  1. Find the 4-connected neighbour with the lowest opt_fra (read from the
     base scan npz files).
  2. If that neighbour beats this point, recompute the neighbour's optimal
     single-qubit frame (1 restart) and re-optimise this point seeded with it.

Then a cross-d_ext step: if opt_fra_4 < opt_fra_6, embed the optimal d=4 frame
into d=6 (last column multiplied) and re-optimise d=6 from that seed.  Because
kron(S_6, S_6) then contains every column of kron(S_4, S_4), this guarantees
opt_fra_6 ≤ opt_fra_4 (a strictly larger frame can only lower framability).

Reads:  <out_dir>/dpt_<ih:02d>_<ig:02d>.npz          (base scan; self + neighbours)
Writes: <out_dir>/dpt_refine_<ih:02d>_<ig:02d>.npz   (refined opt_fra_4, opt_fra_6)

Usage:
    python scripts/dissipative_PT_refine_worker.py --task_id 42 --out_dir results_dpt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dissipative_PT import (bond_trotter_gate, optimise_framability,
                            embed_frame_params, frame_from_params, params_from_frame,
                            H_LIST, GAMMA_LIST, N_H, N_G, N_TOTAL,
                            J_DEFAULT, DT_DEFAULT)

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TOL = 1e-9


def _base_path(out_dir: Path, ih: int, ig: int) -> Path:
    return out_dir / f'dpt_{ih:02d}_{ig:02d}.npz'


def _load_fra(out_dir: Path, ih: int, ig: int, key: str):
    f = _base_path(out_dir, ih, ig)
    if not f.exists():
        return None
    d = np.load(f)
    return float(d[key]) if key in d else None


def _load_frame(out_dir: Path, ih: int, ig: int, s_key: str):
    """Stored optimal frame S (4 × d_ext_single), or None if not available."""
    f = _base_path(out_dir, ih, ig)
    if not f.exists():
        return None
    d = np.load(f)
    if s_key not in d:
        return None
    S = np.asarray(d[s_key], dtype=float)
    return S if np.all(np.isfinite(S)) else None


def _best_neighbor(out_dir: Path, ih: int, ig: int, key: str):
    """Lowest-opt_fra 4-connected neighbour: returns (value, (ni, nj)) or (inf, None)."""
    best_val, best = np.inf, None
    for di, dj in NEIGHBORS:
        ni, nj = ih + di, ig + dj
        if 0 <= ni < N_H and 0 <= nj < N_G:
            v = _load_fra(out_dir, ni, nj, key)
            if v is not None and v < best_val:
                best_val, best = v, (ni, nj)
    return best_val, best


def _refine_dext(out_dir: Path, ih: int, ig: int, d_ext_single: int,
                 key: str, s_key: str, my_val: float, gate_self: np.ndarray,
                 J: float, dt: float, n_restarts: int, maxfev: int, seed: int):
    """Neighbour-seeded refinement of one frame size.

    Returns (f_best, x_best); x_best is None if this point was not re-optimised.
    """
    best_val, best_nb = _best_neighbor(out_dir, ih, ig, key)
    if best_nb is None or best_val >= my_val - TOL:
        return my_val, None

    ni, nj = best_nb
    # Neighbour seed: use the stored optimal frame; recompute only if absent
    # (e.g. base scan produced before frames were saved).
    S_nb = _load_frame(out_dir, ni, nj, s_key)
    if S_nb is not None:
        x_nb = params_from_frame(S_nb)
    else:
        gate_nb = bond_trotter_gate(J, H_LIST[ni], GAMMA_LIST[nj], dt)
        _, x_nb = optimise_framability(gate_nb, d_ext_single, n_restarts=1,
                                       maxfev=maxfev, seed=seed, return_x=True)
    if x_nb is None:
        return my_val, None

    print(f'  d{d_ext_single}: neighbour ({ni},{nj}) {best_val:.6f} '
          f'< self {my_val:.6f} → seeding', flush=True)
    f, x = optimise_framability(gate_self, d_ext_single, n_restarts=n_restarts,
                                maxfev=maxfev, seed=seed,
                                extra_init_xs=[x_nb], return_x=True)
    return (f, x) if f < my_val else (my_val, None)


def run_point(point_id: int, args) -> None:
    """Refine one grid point (skips if its refine npz already exists)."""
    out_dir = Path(args.out_dir)
    ih = point_id // N_G
    ig = point_id %  N_G
    round_tag = f'_r{args.round:02d}' if args.round > 0 else ''
    out = out_dir / f'dpt_refine{round_tag}_{ih:02d}_{ig:02d}.npz'

    if out.exists():
        print(f'[skip] {out.name} already exists', flush=True)
        return

    base = _base_path(out_dir, ih, ig)
    if not base.exists():
        print(f'ERROR: base scan file {base} not found — run the scan first',
              file=sys.stderr)
        sys.exit(1)

    d_base = np.load(base)
    m4 = float(d_base['opt_fra_4'])
    m6 = float(d_base['opt_fra_6'])
    J  = float(d_base['J'])  if 'J'  in d_base else J_DEFAULT
    dt = float(d_base['dt']) if 'dt' in d_base else DT_DEFAULT
    h, gamma = H_LIST[ih], GAMMA_LIST[ig]
    seed = args.seed + point_id

    t0 = time.perf_counter()
    print(f'[point {point_id}/{N_TOTAL}] ih={ih} ig={ig} h={h:.2f} gamma={gamma:.2f} '
          f'dt={dt}  base: d4={m4:.6f} d6={m6:.6f}', flush=True)

    gate_self = bond_trotter_gate(J, h, gamma, dt)

    # ── neighbour-seeded refinement per frame size ────────────────────────────
    f4, x4 = _refine_dext(out_dir, ih, ig, 4, 'opt_fra_4', 'opt_S_4', m4,
                          gate_self, J, dt, args.n_restarts, args.fra_maxfev_4, seed)
    f6, x6 = _refine_dext(out_dir, ih, ig, 6, 'opt_fra_6', 'opt_S_6', m6,
                          gate_self, J, dt, args.n_restarts, args.fra_maxfev_6, seed + 1)

    # ── cross-d_ext: if d=4 beats d=6, seed d=6 from the embedded d=4 frame ───
    if f4 < f6 - TOL:
        if x4 is None:                       # need the optimal d=4 frame to embed
            f4b, x4 = optimise_framability(gate_self, 4, n_restarts=args.n_restarts,
                                           maxfev=args.fra_maxfev_4, seed=seed,
                                           return_x=True)
            f4 = min(f4, f4b)
        if x4 is not None:
            x6_seed = embed_frame_params(x4, 4, 6)
            f6c, x6c = optimise_framability(gate_self, 6, n_restarts=args.n_restarts,
                                            maxfev=args.fra_maxfev_6, seed=seed + 1,
                                            extra_init_xs=[x6_seed], return_x=True)
            if f6c < f6:
                print(f'  cross-seed d4→d6: {f6:.6f} → {f6c:.6f}  (d4={f4:.6f})',
                      flush=True)
                f6, x6 = f6c, x6c

    # Refined frames: re-optimised frame where improved, else carry the base frame.
    base_S4 = np.asarray(d_base['opt_S_4']) if 'opt_S_4' in d_base else frame_from_params(None, 4)
    base_S6 = np.asarray(d_base['opt_S_6']) if 'opt_S_6' in d_base else frame_from_params(None, 6)
    S4 = frame_from_params(x4, 4) if x4 is not None else base_S4
    S6 = frame_from_params(x6, 6) if x6 is not None else base_S6

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             round=np.array(args.round),
             opt_fra_4=np.array(f4), opt_fra_6=np.array(f6),
             opt_S_4=S4, opt_S_6=S6,
             base_fra_4=np.array(m4), base_fra_6=np.array(m6),
             ih=np.array(ih), ig=np.array(ig))

    elapsed = time.perf_counter() - t0
    print(f'  saved {out.name}: d4 {m4:.6f}→{f4:.6f}  d6 {m6:.6f}→{f6:.6f}  '
          f'({elapsed:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',      type=int, required=True,
                   help='array index: point id when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks',     type=int, default=1,
                   help='split the N_TOTAL grid into this many array tasks (strided)')
    p.add_argument('--round',        type=int, default=0,
                   help='refinement round (0 = legacy name dpt_refine_*.npz, '
                        '1+ = dpt_refine_r<N>_*.npz)')
    p.add_argument('--out_dir',      type=str, default='results_dpt')
    p.add_argument('--n_restarts',   type=int, default=5)
    p.add_argument('--fra_maxfev_4', type=int, default=1000)
    p.add_argument('--fra_maxfev_6', type=int, default=500)
    p.add_argument('--seed',         type=int, default=0)
    args = p.parse_args()

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N_TOTAL):
            print(f'ERROR: task_id must be in [0, {N_TOTAL})', file=sys.stderr)
            sys.exit(1)
        run_point(args.task_id, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N_TOTAL, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {len(point_ids)} points: '
          f'{point_ids}', flush=True)
    for pid in point_ids:
        run_point(pid, args)


if __name__ == '__main__':
    main()
