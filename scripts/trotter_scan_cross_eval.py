"""
Cross-evaluation sweep for the Trotter-scan optimised framabilities:
frame label-propagation with NO optimisation at all.

For every grid point and each key (opt_fra_4 / opt_fra_6) the stored optimal
frames of all 4-connected neighbours (plus the d4->d6 embeddings) are
evaluated DIRECTLY on the point's own bond gate — one batched LP per frame.
Any finite value is a rigorous upper bound on the point's minimal
framability, so

    new_val = min(stored_val, min over candidate frames)

is always sound.  Improved (value, frame) pairs are updated in memory and the
sweep repeats (Gauss-Seidel: an improved frame is immediately available to
its own neighbours) until a full sweep changes nothing.  This erases every
'island' whose neighbour frame transfers exactly, and for the rest it records
the best cross-evaluation value — the diagnostic separating "optimiser
stalled just above the floor" (cross ~ 1+eps) from "frame branch switch"
(cross >> 1).

The whole grid fits in one job: per point and sweep it is <= 13 batched LPs,
a few ms each.  Improved points are written as

    <out_dir>/<model>/pt_xeval_r<NN>_<ix>_<iy>.npz

which every refinement worker and the final collect pick up (values are
confirmed with the reference per-column LP before being stored, so they are
consistent with the optimiser-produced numbers).

Usage:
    python scripts/trotter_scan_cross_eval.py --model model1 --round 1 \
        [--out_dir results_trotter_v3] [--tol 1e-9] [--fra_tol 1e-6] \
        [--max_sweeps 200]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS
from trotter_refine_common import (
    KEYS, NEIGHBORS, base_path, best_known, build_gate,
    eval_frame_fast, eval_frame_reference, embed_S4_to_S6,
)


def load_state(out_dir: Path, model):
    """Best-known (value, frame) per point per key, plus the per-point gates."""
    nx, ny = model.N_X, model.N_Y
    vals = {k: np.full((nx, ny), np.inf) for k in KEYS}
    frames = {k: {} for k in KEYS}
    gates = {}
    missing = 0
    for ix in range(nx):
        for iy in range(ny):
            base = base_path(out_dir, model, ix, iy)
            if not base.exists():
                missing += 1
                continue
            gates[(ix, iy)] = build_gate(model, ix, iy, np.load(base))
            for key, (s_key, _) in KEYS.items():
                v, S = best_known(out_dir, model, ix, iy, key, s_key)
                vals[key][ix, iy] = v
                if S is not None:
                    frames[key][(ix, iy)] = S
    if missing:
        print(f'WARNING: {missing} base point file(s) missing — those points '
              f'are skipped.', flush=True)
    return vals, frames, gates


def count_islands(vals: np.ndarray, fra_tol: float):
    """Points > 1 whose existing 4-connected neighbours ALL sit at the floor."""
    nx, ny = vals.shape
    islands = []
    for ix in range(nx):
        for iy in range(ny):
            v = vals[ix, iy]
            if not np.isfinite(v) or v <= 1.0 + fra_tol:
                continue
            nbs = [vals[ix + dx, iy + dy] for dx, dy in NEIGHBORS
                   if 0 <= ix + dx < nx and 0 <= iy + dy < ny]
            nbs = [w for w in nbs if np.isfinite(w)]
            if nbs and all(w <= 1.0 + fra_tol for w in nbs):
                islands.append((ix, iy, v))
    return islands


def candidate_frames(key: str, ix: int, iy: int, model, frames: dict):
    """All cross-evaluation candidates for (ix, iy): every neighbour's current
    frame; for d6 additionally the self and neighbour d4 frames embedded."""
    cands = []
    for dx, dy in NEIGHBORS:
        p = (ix + dx, iy + dy)
        if 0 <= p[0] < model.N_X and 0 <= p[1] < model.N_Y:
            S = frames[key].get(p)
            if S is not None:
                cands.append(S)
    if key == 'opt_fra_6':
        for p in [(ix, iy)] + [(ix + dx, iy + dy) for dx, dy in NEIGHBORS]:
            S4 = frames['opt_fra_4'].get(p)
            if S4 is not None:
                cands.append(embed_S4_to_S6(S4))
    return cands


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',      type=str, required=True, choices=list(MODELS))
    p.add_argument('--round',      type=int, required=True,
                   help='label of the pt_xeval_r<NN> output files')
    p.add_argument('--out_dir',    type=str, default='results_trotter_v3')
    p.add_argument('--tol',        type=float, default=1e-9,
                   help='minimal accepted improvement')
    p.add_argument('--fra_tol',    type=float, default=1e-6,
                   help='tolerance for opt_fra == 1 in the island diagnostic')
    p.add_argument('--max_sweeps', type=int, default=200)
    args = p.parse_args()

    model = MODELS[args.model]
    out_dir = Path(args.out_dir)
    t0 = time.perf_counter()

    print(f'[{model.name}] loading best-known state and building '
          f'{model.N_TOTAL} gates ...', flush=True)
    vals, frames, gates = load_state(out_dir, model)

    islands_before = {k: count_islands(vals[k], args.fra_tol) for k in KEYS}
    for k in KEYS:
        print(f'[{model.name}] {k}: {len(islands_before[k])} island(s) before '
              f'the sweep', flush=True)

    improved: set = set()          # (ix, iy) with any improved key
    n_evals = 0

    for sweep in range(1, args.max_sweeps + 1):
        n_changed = 0
        for ix in range(model.N_X):
            for iy in range(model.N_Y):
                if (ix, iy) not in gates:
                    continue
                gate = gates[(ix, iy)]
                for key in KEYS:
                    cur = vals[key][ix, iy]
                    best_f, best_S = np.inf, None
                    for S in candidate_frames(key, ix, iy, model, frames):
                        f = eval_frame_fast(S, gate)
                        n_evals += 1
                        if np.isfinite(f) and f < best_f:
                            best_f, best_S = f, S
                    if best_S is None or best_f >= cur - args.tol:
                        continue
                    # confirm with the reference evaluator before accepting
                    f_ref = eval_frame_reference(best_S, gate)
                    if np.isfinite(f_ref) and f_ref < cur - args.tol:
                        vals[key][ix, iy] = f_ref
                        frames[key][(ix, iy)] = best_S.copy()
                        improved.add((ix, iy))
                        n_changed += 1
        print(f'[{model.name}] sweep {sweep}: {n_changed} value(s) improved '
              f'({n_evals} LP evals total, {time.perf_counter()-t0:.0f}s)',
              flush=True)
        if n_changed == 0:
            break

    # ── write improved points ────────────────────────────────────────────────
    mdir = out_dir / model.name
    mdir.mkdir(parents=True, exist_ok=True)
    for (ix, iy) in sorted(improved):
        payload = {'round': np.array(args.round),
                   'ix': np.array(ix), 'iy': np.array(iy)}
        for key, (s_key, _) in KEYS.items():
            payload[key] = np.array(vals[key][ix, iy])
            S = frames[key].get((ix, iy))
            if S is not None:
                payload[s_key] = np.asarray(S)
        np.savez(mdir / f'pt_xeval_r{args.round:02d}_{ix:03d}_{iy:03d}.npz',
                 **payload)
    print(f'[{model.name}] wrote {len(improved)} pt_xeval_r{args.round:02d} '
          f'file(s).', flush=True)

    # ── island diagnostic after the sweep ────────────────────────────────────
    for key in KEYS:
        islands = count_islands(vals[key], args.fra_tol)
        print(f'\n[{model.name}] {key}: {len(islands_before[key])} island(s) '
              f'before -> {len(islands)} after.', flush=True)
        for ix, iy, v in islands:
            gate = gates[(ix, iy)]
            crosses = [eval_frame_fast(S, gate)
                       for S in candidate_frames(key, ix, iy, model, frames)]
            crosses = [c for c in crosses if np.isfinite(c)]
            c_best = min(crosses) if crosses else np.inf
            kind = ('optimiser stall (cross-eval ~ floor)'
                    if c_best <= 1.0 + 50 * (v - 1.0)
                    else 'frame branch switch (cross-eval far from floor)')
            print(f'    ({ix:3d},{iy:3d})  {model.p1_name}='
                  f'{model.p1_vals[ix]:.3f} {model.p2_name}='
                  f'{model.p2_vals[iy]:.3f}  val={v:.6f}  '
                  f'best cross={c_best:.6f}  [{kind}]', flush=True)

    print(f'\n[{model.name}] cross-eval done in '
          f'{time.perf_counter() - t0:.0f}s.', flush=True)


if __name__ == '__main__':
    main()
