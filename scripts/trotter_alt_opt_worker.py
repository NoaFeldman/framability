"""
Re-optimise the Heisenberg framabilities (d_ext_single in {4, 6}) of every
stored trotter_lindbladian_scan point (models 1-5, results_trotter_v3) with
the certificate-based 'alternating' scheme (optimize_framability,
OPT_VERSION), updating each pt_<ix>_<iy>.npz *in place*.

Only these keys are (re)written:

    opt_fra_4, opt_S_4, opt_fra_6, opt_S_6   best-known value / frame
                                             (never degraded: min of the old
                                             stored value and the new run)
    prev_fra_4/6, prev_S_4/6   the pre-alternating (Powell) value / frame,
                               captured once on first migration and then
                               preserved across re-runs
    alt_fra_4/6                raw value reached by the alternating scheme
                               (+ Polyak floor polish) at this point -- the
                               "new scheme" side of the comparison heatmap
    alt_opt_version            optimize_framability.OPT_VERSION stamp; a point
                               already carrying the current stamp is skipped

Every other stored quantity (sign problem, NESS data, stabilizer / Pauli /
product / Schrodinger framabilities, gamma_ch1, dephasing group, dt, ...) is
copied through untouched.  The gate is rebuilt from the model builder with the
*stored* dt and dim, so it is bit-identical to the gate the point was
originally scanned with (cross-checked against the stored spectral floor).

The new-scheme value is computed from the scheme's own standard restarts (it
is NOT warm-started from the old optimum), so alt_fra_* vs prev_fra_* is a
fair scheme-vs-scheme comparison; the best-of update on opt_fra_* is what the
scan figures use.

Grid layout: the points of all requested models are concatenated into one
global list (model order = the --models order, point order = ix * N_Y + iy)
and strided across --n_chunks array tasks.

Usage (one strided chunk of a 200-task array over all five models):
    python scripts/trotter_alt_opt_worker.py \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200

Usage (single point of one model, for testing):
    python scripts/trotter_alt_opt_worker.py --models model1 --task_id 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, bond_trotter_gate
from optimize_framability import (
    minimize_framability, polyak_floor_polish, spectral_floor, OPT_VERSION,
)
from dissipative_PT import frame_from_params

D_EXTS = (4, 6)


def global_points(model_names: list[str]) -> list[tuple[str, int]]:
    """Concatenated (model_name, point_id) list over the requested models."""
    pts = []
    for name in model_names:
        m = MODELS[name]
        pts.extend((name, pid) for pid in range(m.N_TOTAL))
    return pts


def _reopt_one(gate: np.ndarray, de: int, seed: int, args,
               cur_val: float, cur_S: np.ndarray):
    """Run the alternating scheme (+ optional Polyak floor polish) at one
    d_ext_single.  Returns (alt_raw, best_val, best_S) where best never
    degrades the stored incumbent (cur_val, cur_S)."""
    _, f_alt, x_alt = minimize_framability(
        gate, de, n_restarts=args.n_restarts, method='alternating',
        maxfev=args.maxfev, seed=seed, verbose=False, return_x=True)
    S_alt = frame_from_params(x_alt, de)
    if args.polish_iters > 0 and np.isfinite(f_alt):
        f_pol, S_pol = polyak_floor_polish(S_alt, gate,
                                           n_iter=args.polish_iters)
        if np.isfinite(f_pol) and f_pol < f_alt:
            f_alt, S_alt = f_pol, S_pol
    if np.isfinite(f_alt) and f_alt < cur_val:
        return float(f_alt), float(f_alt), S_alt
    return float(f_alt), float(cur_val), np.asarray(cur_S)


def process_point(name: str, point_id: int, gidx: int, args) -> None:
    model = MODELS[name]
    ix = point_id // model.N_Y
    iy = point_id % model.N_Y
    f = Path(args.in_dir) / name / f'pt_{ix:03d}_{iy:03d}.npz'
    if not f.exists():
        print(f'[miss] {name}/pt_{ix:03d}_{iy:03d}.npz not scanned yet -- skip',
              flush=True)
        return

    z = np.load(f, allow_pickle=True)
    if (not args.force and 'alt_opt_version' in z.files
            and str(z['alt_opt_version']) == OPT_VERSION):
        print(f'[skip] {name}/{f.name} already at {OPT_VERSION}', flush=True)
        return
    data = {k: z[k] for k in z.files}
    z.close()

    p1, p2 = float(data['p1']), float(data['p2'])
    dt, dim = float(data['dt']), int(data['dim'])
    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    gate = bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)

    # Rebuild cross-check: the spectral floor is a cheap fingerprint of the gate.
    fl = spectral_floor(gate)
    if abs(fl - float(data['floor'])) > 1e-6:
        print(f'[WARN] {name}/{f.name}: rebuilt floor {fl:.8f} != stored '
              f'{float(data["floor"]):.8f} -- gate mismatch?', flush=True)

    t0 = time.perf_counter()
    line = [f'[{gidx}] {name} {model.p1_name}={p1:.3f} {model.p2_name}={p2:.3f}']
    for k, de in enumerate(D_EXTS):
        cur_val = float(data[f'opt_fra_{de}'])
        cur_S = np.asarray(data[f'opt_S_{de}'])
        # capture the pre-alternating scheme once
        if f'prev_fra_{de}' not in data:
            data[f'prev_fra_{de}'] = np.array(cur_val)
            data[f'prev_S_{de}'] = cur_S
        alt_raw, best_val, best_S = _reopt_one(
            gate, de, args.seed + 2 * gidx + k, args, cur_val, cur_S)
        data[f'alt_fra_{de}'] = np.array(alt_raw)
        data[f'opt_fra_{de}'] = np.array(best_val)
        data[f'opt_S_{de}'] = best_S
        line.append(f'd{de}: old={float(data[f"prev_fra_{de}"]):.6f} '
                    f'alt={alt_raw:.6f} best={best_val:.6f}')
    data['alt_opt_version'] = np.array(OPT_VERSION)

    # Atomic in-place update (unique temp name -> no cross-job collisions).
    tmp = f.with_name(f'{f.stem}.tmp{os.getpid()}.npz')
    np.savez(tmp, **data)
    os.replace(tmp, f)
    line.append(f'({time.perf_counter() - t0:.0f}s)')
    print('  '.join(line), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',  type=int, required=True,
                   help='global point id when --n_chunks=1, else chunk id')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='stride the global point list over this many tasks')
    p.add_argument('--in_dir',   type=str, default='results_trotter_v3')
    p.add_argument('--models',   type=str,
                   default='model1,model2,model3,model4,model5',
                   help='comma-separated model names (defines the point order)')
    p.add_argument('--n_restarts',   type=int, default=5)
    p.add_argument('--maxfev',       type=int, default=2000,
                   help='sweep budget of the alternating scheme per d_ext')
    p.add_argument('--polish_iters', type=int, default=150,
                   help='Polyak floor-polish steps after the alternating '
                        'scheme (0 disables)')
    p.add_argument('--seed',  type=int, default=0)
    p.add_argument('--force', action='store_true',
                   help='re-run points already stamped with the current '
                        'alt_opt_version')
    args = p.parse_args()

    names = [s.strip() for s in args.models.split(',') if s.strip()]
    for n in names:
        if n not in MODELS:
            print(f'ERROR: unknown model {n!r}', file=sys.stderr)
            sys.exit(1)
    pts = global_points(names)
    N = len(pts)

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        ids = [args.task_id]
    else:
        if not (0 <= args.task_id < args.n_chunks):
            print(f'ERROR: chunk id must be in [0, {args.n_chunks})',
                  file=sys.stderr)
            sys.exit(1)
        ids = list(range(args.task_id, N, args.n_chunks))
        print(f'[chunk {args.task_id}/{args.n_chunks}] {len(ids)} points '
              f'of {N} total ({OPT_VERSION})', flush=True)

    for gidx in ids:
        name, pid = pts[gidx]
        process_point(name, pid, gidx, args)


if __name__ == '__main__':
    main()
