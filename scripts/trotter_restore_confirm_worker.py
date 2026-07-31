"""
Stage 1 (CLUSTER, parallel) of undoing the flawed alternating-scheme re-run:
RESTORE the correct opt_fra_4/6 of every trotter_lindbladian_scan point from the
OLD frames -- no optimisation, just reference-LP re-certification.

Why this is needed
------------------
The alternating re-optimisation (scripts/trotter_alt_opt_worker.py) overwrote
opt_fra_de with min(old_value, alt_value), where alt_value comes from the fast
batched objective optimize_framability._get_framability_fast.  That objective
(a) under-reports on some frames and (b) lets the frame collapse onto a
gate-invariant subspace (vacuous certificate), so opt_fra was pulled spuriously
toward 1 and the scan figures show framable regions that are not framable.

The correct value is the true framability of the best OLD frame, and those
frames are still on disk:
    * this point's own stored opt_S_de and prev_S_de (the pre-alternating Powell
      frame, captured once by the alt worker), and
    * the same frames from the pre-re-run filesystem snapshot --old_dir
      (default results_trotter_v3_old).
Each candidate is re-certified with the hardened reference evaluator
dissipative_PT._framability_lp, which INTERNALLY rejects rank-deficient
(collapsed) frames by returning +inf.  We keep the frame with the lowest true
framability -- this alone restores opt_fra everywhere a valid old frame exists.

For every point and d_ext in {4, 6} it rewrites, in place:
    opt_fra_de     <- min true framability over all found valid old frames
    opt_S_de       <- the winning frame
    restore_fra_de  the same value (provenance)
    unconf_fra_de   the pre-restore opt_fra_de (kept if already present)
    restore_version stamp; a point already at the current stamp is skipped
Every other stored quantity is copied through untouched.  The gate is rebuilt
from the stored p1, p2, dt, dim (spectral-floor cross-check) so it is
bit-identical to the scanned gate.

A (point, d_ext) whose old frames do NOT certify (best value +inf, or the point
was claimed framable but no valid frame reaches <= 1 + tol) is written to a
per-chunk manifest fragment restore_manifest_chunk<task>.npz -- the ONLY points
Stage 2 (scripts/trotter_recompute.slurm.sh) must re-optimise.  Per-chunk
fragments avoid cross-task write collisions; merge them with
scripts/trotter_restore_collect.py.

Grid layout matches trotter_alt_opt_worker.py: the points of all requested
models are concatenated into one global list (model order = --models order,
point order = ix * N_Y + iy) and strided across --n_chunks array tasks.

    python scripts/trotter_restore_confirm_worker.py \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
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
import dissipative_PT as dp

D_EXTS = (4, 6)
FRA_ONE_TOL = 1e-6
FLOOR_TOL = 1e-6
RESTORE_VERSION = '1.0-restore-oldframes'
MODELS_IDX = {name: i for i, name in enumerate(MODELS)}


def global_points(model_names: list[str]) -> list[tuple[str, int]]:
    """Concatenated (model_name, point_id) list over the requested models."""
    pts: list[tuple[str, int]] = []
    for name in model_names:
        pts.extend((name, pid) for pid in range(MODELS[name].N_TOTAL))
    return pts


def _certify(S, gate) -> float:
    """True framability of a stored real frame S via the hardened reference LP
    (returns +inf for a collapsed / non-full-support / unreachable frame)."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.size == 0 or not np.all(np.isfinite(S)):
        return float('inf')
    return dp._framability_lp(dp._kron_power(S, 2), gate)


def _old_frames(data: dict, old_dir: Path, name: str, ix: int, iy: int,
                de: int) -> list[np.ndarray]:
    """Every candidate OLD frame for this (point, d_ext): the point's own
    opt_S/prev_S plus, if the backup snapshot exists, its opt_S/prev_S too."""
    cands: list[np.ndarray] = []
    for key in (f'opt_S_{de}', f'prev_S_{de}'):
        if key in data:
            cands.append(np.asarray(data[key], dtype=float))
    if old_dir is not None:
        of = old_dir / name / f'pt_{ix:03d}_{iy:03d}.npz'
        if of.exists():
            try:
                oz = np.load(of, allow_pickle=True)
                for key in (f'opt_S_{de}', f'prev_S_{de}'):
                    if key in oz.files:
                        cands.append(np.asarray(oz[key], dtype=float))
                oz.close()
            except Exception:
                pass
    return cands


def process_point(name: str, point_id: int, args, manifest: list) -> None:
    model = MODELS[name]
    ix = point_id // model.N_Y
    iy = point_id % model.N_Y
    f = Path(args.in_dir) / name / f'pt_{ix:03d}_{iy:03d}.npz'
    if not f.exists():
        return

    z = np.load(f, allow_pickle=True)
    if (not args.force and 'restore_version' in z.files
            and str(z['restore_version']) == RESTORE_VERSION):
        z.close()
        return
    data = {k: z[k] for k in z.files}
    z.close()

    p1, p2 = float(data['p1']), float(data['p2'])
    dt, dim = float(data['dt']), int(data['dim'])
    H1, H2, j1, j2 = model.build(p1, p2)
    gate = bond_trotter_gate(H1, H2, j1, j2, dim, dt)
    fl = dp.spectral_floor(gate)
    if abs(fl - float(data['floor'])) > FLOOR_TOL:
        print(f'[WARN] {name}/{f.name}: rebuilt floor {fl:.8f} != stored '
              f'{float(data["floor"]):.8f} -- gate mismatch?', flush=True)

    old_dir = Path(args.old_dir) if args.old_dir else None
    t0 = time.perf_counter()
    line = [f'{name}/pt_{ix:03d}_{iy:03d} ({p1:.3f},{p2:.3f})']
    for de in D_EXTS:
        pre = float(data[f'opt_fra_{de}'])           # flawed / current value
        best_val, best_S = float('inf'), None
        for S in _old_frames(data, old_dir, name, ix, iy, de):
            v = _certify(S, gate)
            if np.isfinite(v) and v < best_val:
                best_val, best_S = v, S

        if f'unconf_fra_{de}' not in data:            # capture pre-restore once
            data[f'unconf_fra_{de}'] = np.array(pre)

        if best_S is not None:
            data[f'opt_fra_{de}'] = np.array(best_val)
            data[f'opt_S_{de}'] = best_S
            data[f'restore_fra_{de}'] = np.array(best_val)
        else:
            data[f'restore_fra_{de}'] = np.array(np.inf)

        # A point needs cluster re-optimisation when no old frame certifies, or
        # it was claimed framable but no valid old frame reaches <= 1 + tol.
        claimed = float(data[f'unconf_fra_{de}']) <= 1.0 + FRA_ONE_TOL
        restored_framable = np.isfinite(best_val) and best_val <= 1.0 + FRA_ONE_TOL
        if (not np.isfinite(best_val)) or (claimed and not restored_framable):
            manifest.append((MODELS_IDX[name], ix, iy, de, p1, p2,
                             float(data[f'unconf_fra_{de}']),
                             best_val if np.isfinite(best_val) else np.inf,
                             int(claimed)))
        bv = 'inf' if not np.isfinite(best_val) else f'{best_val:.6f}'
        line.append(f'd{de}: was={pre:.6f} restored={bv}')

    data['restore_version'] = np.array(RESTORE_VERSION)
    if not args.dry_run:
        tmp = f.with_name(f'{f.stem}.tmp{os.getpid()}.npz')
        np.savez(tmp, **data)
        os.replace(tmp, f)
    line.append(f'({time.perf_counter() - t0:.1f}s)')
    print('  '.join(line), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',  type=int, required=True,
                   help='global point id when --n_chunks=1, else chunk id')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='stride the global point list over this many tasks')
    p.add_argument('--in_dir',   type=str, default='results_trotter_v3')
    p.add_argument('--old_dir',  type=str, default='results_trotter_v3_old',
                   help='pre-re-run snapshot searched for extra old frames; '
                        'pass "" to skip and use only in-file frames')
    p.add_argument('--models',   type=str,
                   default='model1,model2,model3,model4,model5')
    p.add_argument('--manifest_dir', type=str, default='.',
                   help='directory for the per-chunk manifest fragment')
    p.add_argument('--force', action='store_true',
                   help='re-restore points already at RESTORE_VERSION')
    p.add_argument('--dry_run', action='store_true',
                   help='report only; do not modify any npz or write a manifest')
    args = p.parse_args()

    names = [s.strip() for s in args.models.split(',') if s.strip()]
    for n in names:
        if n not in MODELS:
            print(f'ERROR: unknown model {n!r}', file=sys.stderr)
            sys.exit(1)
    pts = global_points(names)
    N = len(pts)

    if args.n_chunks <= 1:
        ids = [args.task_id] if 0 <= args.task_id < N else []
    else:
        if not (0 <= args.task_id < args.n_chunks):
            print(f'ERROR: chunk id must be in [0, {args.n_chunks})',
                  file=sys.stderr)
            sys.exit(1)
        ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[restore {args.task_id}/{args.n_chunks}] {len(ids)} of {N} points '
          f'({RESTORE_VERSION})', flush=True)

    manifest: list = []
    for gi in ids:
        name, pid = pts[gi]
        process_point(name, pid, args, manifest)

    # Per-chunk manifest fragment (unique name -> no cross-task collision).
    if not args.dry_run:
        frag = Path(args.manifest_dir) / f'restore_manifest_chunk{args.task_id:04d}.npz'
        rows = np.array(manifest, dtype=float).reshape(-1, 9)
        np.savez(frag, rows=rows,
                 columns=np.array(['model_idx', 'ix', 'iy', 'de', 'p1', 'p2',
                                   'unconf_fra', 'restored_fra', 'was_claimed']),
                 model_names=np.array(list(MODELS)))
        print(f'[restore {args.task_id}] {len(manifest)} (point,d_ext) need '
              f'recompute -> {frag}', flush=True)


if __name__ == '__main__':
    main()
