"""
In-place upgrade of Trotter-scan results from TLS version 2.2 to 2.3.

The only computed quantity that changed in 2.3 is lpdo_max: for models with
lpdo_init='plus' (model2, model4) the relaxation path now starts from |+>^N
instead of |0>^N.  For those models this worker recomputes lpdo_max only,
preserving every other stored field -- in particular the expensive optimised
framabilities (which may already carry neighbour-refinement improvements
merged back into the base npz).  For models whose computation is unchanged
(model1, model3) it just re-stamps the version, so the base scan worker will
skip them instead of recomputing from scratch.

Files that do not exist, or whose version is not 2.2, are left untouched --
the base scan worker (trotter_scan_worker.py) recomputes those in full.

Usage (strided across a 200-task array):
    python scripts/trotter_scan_patch_worker.py --model model2 \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (
    MODELS, TLS_VERSION, build_full_lindbladian_model, lpdo_init_vector,
    LATTICE_LX, LATTICE_LY, LPDO_PATH_DT, LPDO_PATH_FIDELITY,
)
from dissipative_PT import steady_state_and_decay, pauli_to_rho
from analysis import compute_max_bond_dim

PATCH_FROM = '2.2'


def _recompute_lpdo_max(model, p1: float, p2: float) -> float:
    """lpdo_max along the model's lpdo_init -> NESS relaxation path."""
    N = LATTICE_LX * LATTICE_LY
    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    L_full = build_full_lindbladian_model(H1, H2, jumps1, jumps2)
    c_ss, _ = steady_state_and_decay(L_full, N=N)
    if c_ss is None:
        return float('nan')
    rho = pauli_to_rho(c_ss, N=N)
    try:
        _, lpdo_max = compute_max_bond_dim(
            L_full, rho, None, N=N, init=lpdo_init_vector(model, N),
            dt=LPDO_PATH_DT, fidelity_threshold=LPDO_PATH_FIDELITY)
        return float(lpdo_max)
    except Exception:
        return float('nan')


def patch_point(model, point_id: int, args) -> None:
    ix = point_id // model.N_Y
    iy = point_id %  model.N_Y
    out = Path(args.out_dir) / model.name / f'pt_{ix:03d}_{iy:03d}.npz'

    if not out.exists():
        return                       # base scan worker computes it in full
    try:
        d = dict(np.load(out, allow_pickle=True))
    except Exception as e:
        print(f'[leave] {model.name}/{out.name}: unreadable ({e})', flush=True)
        return
    ver = str(d['code_version']) if 'code_version' in d else ''
    if ver == TLS_VERSION:
        print(f'[skip] {model.name}/{out.name} already at {TLS_VERSION}', flush=True)
        return
    if ver != PATCH_FROM:
        print(f'[leave] {model.name}/{out.name} at version "{ver}" != {PATCH_FROM} '
              f'-> base scan will recompute', flush=True)
        return

    t0 = time.perf_counter()
    if model.lpdo_init != 'zero':
        p1 = float(model.p1_vals[ix])
        p2 = float(model.p2_vals[iy])
        old = float(d['lpdo_max']) if 'lpdo_max' in d else float('nan')
        d['lpdo_max'] = np.array(_recompute_lpdo_max(model, p1, p2))
        print(f'[patch] {model.name}/{out.name}: lpdo_max {old:.6f} -> '
              f'{float(d["lpdo_max"]):.6f} (init |+>^N)  '
              f'({time.perf_counter() - t0:.0f}s)', flush=True)
    else:
        print(f'[stamp] {model.name}/{out.name}: {PATCH_FROM} -> {TLS_VERSION} '
              f'(no recompute needed)', flush=True)

    d['code_version'] = np.array(TLS_VERSION)
    np.savez(out, **d)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1)
    p.add_argument('--out_dir',  type=str, default='results_trotter')
    args = p.parse_args()

    model = MODELS[args.model]
    N = model.N_TOTAL

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        patch_point(model, args.task_id, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name}: '
          f'{len(point_ids)} points', flush=True)
    for pid in point_ids:
        patch_point(model, pid, args)


if __name__ == '__main__':
    main()
