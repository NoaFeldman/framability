"""
Stage 2 of the alternating-scheme cleanup: RE-OPTIMISE only the manifest points.

Reads the recompute manifest produced by scripts/trotter_alt_confirm.py (the
small set of points the scan claimed framable that are NOT, after confirming the
best stored frame with the reference LP) and re-optimises each with a
*correct-objective* run of optimize_framability.minimize_framability.

Correctness relies on the confirmation gate now built into minimize_framability:
the search objective _get_framability_fast can under-report, so the value it
returns is re-checked against dissipative_PT._framability_lp before it is
reported.  Every value this worker writes is therefore reference-LP confirmed;
in addition each candidate frame is re-confirmed here explicitly (belt and
braces, incl. the Polyak-polished frame whose own evaluator is also the fast one).

Seeds per point/d_ext: the point's own stored opt_S and prev_S frames, plus the
four grid-neighbour opt_S frames (neighbour seeding escapes the branch/stall
artefacts documented in the refine pipeline) — all fed as extra_init_xs on top
of the standard ixyz + random restarts.

Only opt_fra_de / opt_S_de are updated, and only when the new reference-confirmed
value beats the stored one (never degraded).  recomp_fra_de and recompute_version
are stamped for provenance; a point already at the current stamp is skipped.

Grid layout: unique manifest points are concatenated and strided across
--n_chunks array tasks (both d_ext of a point handled by the same task -> one
read/write per file, no cross-task collision).

    python scripts/trotter_recompute_worker.py --task_id $SLURM_ARRAY_TASK_ID \
        --n_chunks 200 --manifest recompute_manifest.npz
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
from optimize_framability import minimize_framability, polyak_floor_polish, OPT_VERSION
from dissipative_PT import (
    _framability_lp, _kron_power, frame_from_params, spectral_floor,
)

RECOMPUTE_VERSION = f'{OPT_VERSION}-recomp1'
FRA_ONE_TOL = 1e-6


def _seed_x(S) -> np.ndarray | None:
    """Flat parameter seed (free columns raveled) from a real frame S — matches
    optimize_framability._params_to_D(use_complex=False)."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[1] < 2 or not np.all(np.isfinite(S)):
        return None
    return S[:, 1:].ravel()


def _confirm(S, gate) -> float:
    S = np.asarray(S, dtype=float)
    if not np.all(np.isfinite(S)):
        return float('inf')
    return _framability_lp(_kron_power(S, 2), gate)


def _neighbour_seeds(in_dir: Path, name: str, ix: int, iy: int, de: int) -> list:
    seeds = []
    model = MODELS[name]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        jx, jy = ix + dx, iy + dy
        if not (0 <= jx < model.N_X and 0 <= jy < model.N_Y):
            continue
        nf = in_dir / name / f'pt_{jx:03d}_{jy:03d}.npz'
        if not nf.exists():
            continue
        try:
            nz = np.load(nf, allow_pickle=True)
            if f'opt_S_{de}' in nz.files:
                s = _seed_x(nz[f'opt_S_{de}'])
                if s is not None:
                    seeds.append(s)
            nz.close()
        except Exception:
            pass
    return seeds


def _reoptimise(gate, de, seeds, args):
    """Return (best_val, best_S), both reference-LP confirmed."""
    extra = [s for s in seeds if s is not None]
    _, f_opt, x_opt = minimize_framability(
        gate, de, n_restarts=args.n_restarts, method='alternating',
        maxfev=args.maxfev, seed=args.seed, verbose=False, return_x=True,
        extra_init_xs=extra or None)
    S_opt = frame_from_params(x_opt, de)
    best_val, best_S = _confirm(S_opt, gate), S_opt          # re-confirm frame
    if args.polish_iters > 0 and np.all(np.isfinite(S_opt)):
        f_pol, S_pol = polyak_floor_polish(S_opt, gate, n_iter=args.polish_iters)
        v_pol = _confirm(S_pol, gate)                        # confirm polished
        if np.isfinite(v_pol) and v_pol < best_val:
            best_val, best_S = v_pol, S_pol
    return best_val, np.asarray(best_S)


def process_point(in_dir: Path, name: str, ix: int, iy: int,
                  d_exts: list, args) -> None:
    f = in_dir / name / f'pt_{ix:03d}_{iy:03d}.npz'
    if not f.exists():
        print(f'[miss] {name}/pt_{ix:03d}_{iy:03d}.npz', flush=True)
        return
    z = np.load(f, allow_pickle=True)
    if (not args.force and 'recompute_version' in z.files
            and str(z['recompute_version']) == RECOMPUTE_VERSION):
        print(f'[skip] {name}/{f.name} already at {RECOMPUTE_VERSION}', flush=True)
        z.close()
        return
    data = {k: z[k] for k in z.files}
    z.close()

    p1, p2 = float(data['p1']), float(data['p2'])
    dt, dim = float(data['dt']), int(data['dim'])
    H1, H2, j1, j2 = MODELS[name].build(p1, p2)
    gate = bond_trotter_gate(H1, H2, j1, j2, dim, dt)
    if abs(spectral_floor(gate) - float(data['floor'])) > 1e-6:
        print(f'[WARN] {name}/{f.name}: gate/floor mismatch', flush=True)

    t0 = time.perf_counter()
    line = [f'{name}/pt_{ix:03d}_{iy:03d} ({p1:.3f},{p2:.3f})']
    for de in d_exts:
        stored = float(data[f'opt_fra_{de}'])
        seeds = [_seed_x(data[f'opt_S_{de}'])]
        if f'prev_S_{de}' in data:
            seeds.append(_seed_x(data[f'prev_S_{de}']))
        seeds += _neighbour_seeds(in_dir, name, ix, iy, de)
        new_val, new_S = _reoptimise(gate, de, seeds, args)
        data[f'recomp_fra_{de}'] = np.array(new_val)
        improved = np.isfinite(new_val) and new_val < stored - 1e-12
        if improved:
            data[f'opt_fra_{de}'] = np.array(new_val)
            data[f'opt_S_{de}'] = new_S
        tag = 'framable' if (np.isfinite(new_val)
                             and new_val <= 1.0 + FRA_ONE_TOL) else '>1'
        line.append(f'd{de}: stored={stored:.6f} new={new_val:.6f} '
                    f'{"IMPROVED" if improved else "kept"} [{tag}]')
    data['recompute_version'] = np.array(RECOMPUTE_VERSION)

    tmp = f.with_name(f'{f.stem}.tmp{os.getpid()}.npz')
    np.savez(tmp, **data)
    os.replace(tmp, f)
    line.append(f'({time.perf_counter() - t0:.0f}s)')
    print('  '.join(line), flush=True)


def load_points(manifest: Path) -> list:
    """Unique (model_name, ix, iy) -> sorted list of d_exts, from the manifest."""
    m = np.load(manifest, allow_pickle=True)
    rows = m['rows']
    names = list(m['model_names'])
    pts: dict = {}
    for r in rows:
        name = names[int(r[0])]
        key = (name, int(r[1]), int(r[2]))
        pts.setdefault(key, set()).add(int(r[3]))
    return [(n, ix, iy, sorted(de)) for (n, ix, iy), de in sorted(pts.items())]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1)
    p.add_argument('--in_dir',   type=str, default='results_trotter_v3')
    p.add_argument('--manifest', type=str, default='recompute_manifest.npz')
    p.add_argument('--n_restarts',   type=int, default=12)
    p.add_argument('--maxfev',       type=int, default=6000)
    p.add_argument('--polish_iters', type=int, default=300)
    p.add_argument('--seed',  type=int, default=0)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    points = load_points(Path(args.manifest))
    N = len(points)
    if N == 0:
        print('manifest is empty -- nothing to recompute.', flush=True)
        return

    if args.n_chunks <= 1:
        ids = [args.task_id] if 0 <= args.task_id < N else []
    else:
        ids = list(range(args.task_id, N, args.n_chunks))
    if not ids:
        print(f'[task {args.task_id}] no points in this chunk '
              f'({N} total)', flush=True)
        return
    print(f'[task {args.task_id}/{args.n_chunks}] {len(ids)} of {N} points '
          f'({RECOMPUTE_VERSION})', flush=True)

    for gi in ids:
        name, ix, iy, d_exts = points[gi]
        process_point(in_dir, name, ix, iy, d_exts, args)


if __name__ == '__main__':
    main()
