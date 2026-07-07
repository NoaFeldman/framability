"""
Per-point cluster worker for the 2D dissipative phase transition scan.

Parameter grid (J=1 fixed):
    h      in [0.1*i for i in range(-10, 10)]  ->  -1.0 … 0.9, N_H=20
    gamma  in [0.1*i for i in range(10)]        ->   0.0 … 0.9, N_G=10

task_id = ih * N_G + ig   (0 … N_H*N_G-1 = 199)

Output: <out_dir>/dpt_<ih:02d>_<ig:02d>.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dissipative_PT import (compute_point, bond_trotter_gate, spectral_floor,
                             DPT_VERSION,
                             H_LIST, GAMMA_LIST, N_H, N_G, N_TOTAL,
                             J_DEFAULT, DT_DEFAULT)

J  = J_DEFAULT
DT = DT_DEFAULT


def _is_current(out: Path) -> bool:
    """True iff `out` exists and already carries the floor + current version."""
    if not out.exists():
        return False
    try:
        d = np.load(out, allow_pickle=True)
        return ('code_version' in d
                and str(d['code_version']) == DPT_VERSION
                and 'floor' in d)
    except Exception:
        return False


def _try_backfill_floor(out: Path) -> bool:
    """Non-destructively add the floor + version stamp to a valid old result.

    The dissipative-PT *optimiser* is unchanged in this code version — only the
    floor was added — and the base npz may already hold values improved by the
    neighbour-refinement rounds (merged back in by dissipative_PT_refine_collect).
    Re-running compute_point would overwrite those refined opt_fra values with
    plain base-scan ones, so for an existing file that already has a valid
    optimisation we only backfill the floor (rebuilt from the stored gate
    parameters) and re-stamp the version, preserving every other field.

    Returns True if the file was backfilled (caller should skip recomputation),
    False if the file is missing/incomplete and a full compute is required.
    """
    if not out.exists():
        return False
    try:
        d = dict(np.load(out, allow_pickle=True))
    except Exception:
        return False
    if 'opt_fra_4' not in d or 'opt_fra_6' not in d:
        return False   # incomplete -> let the caller recompute from scratch
    if not (np.isfinite(float(d['opt_fra_4'])) and np.isfinite(float(d['opt_fra_6']))):
        return False
    h_  = float(d['h'])     if 'h'  in d else None
    g_  = float(d['gamma']) if 'gamma' in d else None
    J_  = float(d['J'])     if 'J'  in d else J_DEFAULT
    dt_ = float(d['dt'])    if 'dt' in d else DT_DEFAULT
    if h_ is None or g_ is None:
        return False
    d['floor'] = np.array(spectral_floor(bond_trotter_gate(J_, h_, g_, dt_)))
    d['code_version'] = np.array(DPT_VERSION)
    np.savez(out, **d)
    return True


def run_point(point_id: int, args) -> None:
    """Compute and save one grid point (skips if its npz already exists)."""
    ih    = point_id // N_G
    ig    = point_id %  N_G
    h     = H_LIST[ih]
    gamma = GAMMA_LIST[ig]
    out   = Path(args.out_dir) / f'dpt_{ih:02d}_{ig:02d}.npz'

    if _is_current(out):
        print(f'[skip] {out.name} already has floor + version {DPT_VERSION}',
              flush=True)
        return
    elif out.exists():
        # Old file: the optimiser is unchanged, and this file may hold
        # refinement-improved opt_fra — backfill the floor instead of
        # recomputing (which would clobber the refined values).
        if _try_backfill_floor(out):
            print(f'[backfill] {out.name}: added floor, preserved opt_fra '
                  f'(version {DPT_VERSION})', flush=True)
            return
        print(f'[recompute] {out.name} exists but is incomplete -> recomputing',
              flush=True)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print(f'[point {point_id}/{N_TOTAL}] ih={ih} ig={ig} '
          f'h={h:.2f} gamma={gamma:.2f} J={J} dt={args.dt}', flush=True)

    res = compute_point(
        h=h, J=J, gamma=gamma, dt=args.dt,
        fra_restarts=args.fra_restarts,
        fra_maxfev_4=args.fra_maxfev_4,
        fra_maxfev_6=args.fra_maxfev_6,
        sign_restarts=args.sign_restarts,
        seed=args.seed + point_id,
        Lx=2, Ly=2, verbose=True,
    )

    np.savez(out,
             h=np.array(h), J=np.array(J), gamma=np.array(gamma), dt=np.array(args.dt),
             pauli_fra  = np.array(res['pauli_fra']),
             floor      = np.array(res['floor']),
             opt_fra_4  = np.array(res['opt_fra_4']),
             opt_fra_6  = np.array(res['opt_fra_6']),
             opt_S_4    = np.asarray(res['opt_S_4']),
             opt_S_6    = np.asarray(res['opt_S_6']),
             sign_init  = np.array(res['sign_init']),
             sign_opt   = np.array(res['sign_opt']),
             chan_stab  = np.array(res['chan_stab']),
             decay_rate = np.array(res['decay_rate']),
             ss_vn      = np.array(res['ss_vn']),
             mean_mag   = np.array(res['mean_mag']),
             neg_half   = np.array(res['neg_half']),
             max_lpdo   = np.array(res['max_lpdo']),
             ih=np.array(ih), ig=np.array(ig),
             code_version=np.array(DPT_VERSION))

    elapsed = time.perf_counter() - t_start
    print(f'  saved {out.name}  ({elapsed:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',      type=int,   required=True,
                   help='array index: point id when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks',     type=int,   default=1,
                   help='split the N_TOTAL grid into this many array tasks (strided)')
    p.add_argument('--out_dir',      type=str,   default='results_dpt')
    p.add_argument('--dt',           type=float, default=DT)
    p.add_argument('--fra_restarts', type=int,   default=5)
    p.add_argument('--fra_maxfev_4', type=int,   default=1000)
    p.add_argument('--fra_maxfev_6', type=int,   default=500)
    p.add_argument('--sign_restarts',type=int,   default=10)
    p.add_argument('--seed',         type=int,   default=0)
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
