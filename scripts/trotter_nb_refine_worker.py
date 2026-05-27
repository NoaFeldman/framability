"""
Neighbor-seeded refinement of the Lindbladian Trotter framability scan.

For each grid point (d_idx, ig, igp) the worker:
  1. Looks at the 4-connected neighbors in (ig, igp) space.
  2. If the best neighbor has a lower framability, re-runs
     minimize_framability seeded with that neighbor's saved x vector.
  3. Saves the result (improved or not) to a small npz file.

task_id = d_idx * N_GAMMA * N_GP + ig * N_GP + igp
  — identical layout to lindbladian_trotter_worker.py.

Requires:
  - trotter_summary.npz (or the per-task trotter_<d>_*.npz files) to
    already exist in --out_dir so framability values can be compared.
  - Per-task trotter_<d>_<ig:03d>_<igp:03d>.npz files to exist so the
    neighbor's x vector can be loaded as a warm-start seed.

Output:
  <out_dir>/trotter_nb_<d>_<ig:03d>_<igp:03d>.npz
    keys: framability, D, x, improved (bool), gamma, gamma_p, d_ext_single
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from optimize_framability import minimize_framability, DEFAULT_METHOD

D_EXT_SINGLES = [4, 6]
N_D           = len(D_EXT_SINGLES)
GAMMA_MAX     = 8.0
GP_MAX        = 4.0
GAMMA_STEP    = 0.2
N_GAMMA       = int(round(GAMMA_MAX  / GAMMA_STEP)) + 1   # 41
N_GP          = int(round(GP_MAX     / GAMMA_STEP)) + 1   # 21
NEIGHBORS     = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _load_fra_summary(out_dir: str) -> np.ndarray | None:
    """Return (N_D, N_GAMMA, N_GP) array from summary file, or None."""
    p = os.path.join(out_dir, 'trotter_summary.npz')
    if os.path.exists(p):
        return np.load(p)['framability']
    return None


def _load_fra_from_files(out_dir: str) -> np.ndarray:
    """Build (N_D, N_GAMMA, N_GP) array by reading individual task npz files."""
    fra = np.full((N_D, N_GAMMA, N_GP), np.nan)
    for di, d in enumerate(D_EXT_SINGLES):
        for ig in range(N_GAMMA):
            for igp in range(N_GP):
                f = os.path.join(out_dir, f'trotter_{d}_{ig:03d}_{igp:03d}.npz')
                if os.path.exists(f):
                    fra[di, ig, igp] = float(np.load(f)['framability'])
    return fra


def _make_gate(J: float, gamma: float, gamma_p: float, dt: float) -> np.ndarray:
    L = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)
    return expm(L * dt).real


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',    type=int,   required=True,
                        help=f'0..{N_D * N_GAMMA * N_GP - 1}')
    parser.add_argument('--out_dir',    type=str,   default='results_trotter')
    parser.add_argument('--n_restarts', type=int,   default=5)
    parser.add_argument('--maxfev',     type=int,   default=2000)
    parser.add_argument('--max_iter',   type=int,   default=500)
    parser.add_argument('--J',          type=float, default=1.0)
    parser.add_argument('--dt',         type=float, default=0.01)
    parser.add_argument('--method',     type=str,   default=DEFAULT_METHOD)
    parser.add_argument('--force_real', action='store_true')
    args = parser.parse_args()

    total = N_D * N_GAMMA * N_GP
    if not (0 <= args.task_id < total):
        print(f'ERROR: task_id {args.task_id} out of range (0..{total-1})',
              file=sys.stderr)
        sys.exit(1)

    d_idx = args.task_id // (N_GAMMA * N_GP)
    rem   = args.task_id  % (N_GAMMA * N_GP)
    ig    = rem // N_GP
    igp   = rem  % N_GP

    d_ext_single = D_EXT_SINGLES[d_idx]
    gamma   = GAMMA_STEP * ig
    gamma_p = GAMMA_STEP * igp

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f'trotter_nb_{d_ext_single}_{ig:03d}_{igp:03d}.npz'
    )

    # Load framability grid
    fra_all = _load_fra_summary(args.out_dir)
    if fra_all is None:
        print('[info] trotter_summary.npz not found, loading from per-task files',
              flush=True)
        fra_all = _load_fra_from_files(args.out_dir)

    my_val = fra_all[d_idx, ig, igp]
    if not np.isfinite(my_val):
        print(f'({d_ext_single},{ig},{igp}) no base value — skip', flush=True)
        return

    # Find the best 4-connected neighbor (only in ig/igp dimensions)
    best_nb_val = np.inf
    best_ni, best_nj = -1, -1
    for di, dj in NEIGHBORS:
        ni, nj = ig + di, igp + dj
        if 0 <= ni < N_GAMMA and 0 <= nj < N_GP:
            nb_val = fra_all[d_idx, ni, nj]
            if np.isfinite(nb_val) and nb_val < best_nb_val:
                best_nb_val = nb_val
                best_ni, best_nj = ni, nj

    if best_nb_val >= my_val - 1e-9:
        print(f'({d_ext_single},{ig},{igp}) fra={my_val:.6f}  '
              f'best_nb={best_nb_val:.6f} → skip (no better neighbor)', flush=True)
        # Still write a placeholder so this task is not repeated next round
        np.savez(out_path,
                 framability=np.array(my_val),
                 D=np.load(os.path.join(args.out_dir,
                           f'trotter_{d_ext_single}_{ig:03d}_{igp:03d}.npz'))['D'],
                 x=np.load(os.path.join(args.out_dir,
                           f'trotter_{d_ext_single}_{ig:03d}_{igp:03d}.npz'))['x'],
                 improved=np.array(False),
                 gamma=np.array(gamma), gamma_p=np.array(gamma_p),
                 d_ext_single=np.array(d_ext_single))
        return

    print(f'({d_ext_single},{ig},{igp}) fra={my_val:.6f}  '
          f'best_nb=({best_ni},{best_nj}) {best_nb_val:.6f} → refining ...',
          flush=True)

    use_complex = False if args.force_real else None

    # Load neighbor's x as warm-start seed
    nb_npz_path = os.path.join(
        args.out_dir, f'trotter_{d_ext_single}_{best_ni:03d}_{best_nj:03d}.npz'
    )
    x_nb = None
    if os.path.exists(nb_npz_path):
        x_nb = np.load(nb_npz_path)['x']
    else:
        print(f'  [warn] neighbor file {nb_npz_path} not found, skipping seed',
              flush=True)

    gate_self = _make_gate(args.J, gamma, gamma_p, args.dt)

    t0 = time.perf_counter()
    D_opt, f_refined, x_new = minimize_framability(
        gate_self, d_ext_single=d_ext_single,
        n_restarts=args.n_restarts,
        method=args.method,
        max_iter=args.max_iter,
        maxfev=args.maxfev,
        verbose=False,
        extra_init_xs=[x_nb] if x_nb is not None else None,
        return_x=True,
        use_complex=use_complex,
    )
    elapsed = time.perf_counter() - t0

    improved = bool(f_refined < my_val - 1e-9)
    print(f'({d_ext_single},{ig},{igp}) {"improved" if improved else "no change"}: '
          f'{my_val:.6f} → {f_refined:.6f}  elapsed={elapsed:.1f}s', flush=True)

    np.savez(out_path,
             framability=np.array(f_refined),
             D=D_opt,
             x=x_new,
             improved=np.array(improved),
             gamma=np.array(gamma),
             gamma_p=np.array(gamma_p),
             d_ext_single=np.array(d_ext_single))


if __name__ == '__main__':
    main()
