"""
Worker: find optimal S for one (gamma, gamma') point and save it.

Each SLURM task handles one point from PARAMS_LIST (indexed by SLURM_ARRAY_TASK_ID).
Searches over random seeds until framability drops below the stored scan value + TOL,
then saves the S matrix and achieved framability.

Output
------
    <out_dir>/tex_S_<task_id:04d>.npy   shape (4, 6)  — best S found
    <out_dir>/tex_f_<task_id:04d>.npy   scalar float  — achieved framability

Usage
-----
    python tex_S_worker.py --task_id 3 --out_dir results
"""

import argparse
import os

import numpy as np
from scipy.linalg import expm

from framability import extended_pauli_D
from optimize_framability import minimize_framability, DEFAULT_METHOD, _project_columns
from analysis import compute_steady_state

# (gamma, gamma') points to compute S for
PARAMS_LIST = [(6.0, 0.0), (7.0, 0.0), (0.0, 0.6), (2.4, 0.4)]

# Stop searching when fra < stored_fra + TOL
TOL = 1e-4

GAMMA_STEP = 0.2
J = 1.0
DT = 0.01 * GAMMA_STEP   # 0.002


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, required=True,
                   help='Index into PARAMS_LIST (0-based); maps to SLURM_ARRAY_TASK_ID.')
    p.add_argument('--out_dir', type=str, default='results')
    p.add_argument('--n_restarts', type=int, default=5,
                   help='Restarts per seed.')
    p.add_argument('--maxfev', type=int, default=1000,
                   help='Max function evaluations per restart.')
    p.add_argument('--max_seeds', type=int, default=500,
                   help='Maximum number of seeds to try.')
    args = p.parse_args()

    task_id = args.task_id
    if task_id < 0 or task_id >= len(PARAMS_LIST):
        raise ValueError(f'task_id {task_id} out of range (0–{len(PARAMS_LIST)-1})')

    gamma, gp = PARAMS_LIST[task_id]
    os.makedirs(args.out_dir, exist_ok=True)
    out_S = os.path.join(args.out_dir, f'tex_S_{task_id:04d}.npy')
    out_f = os.path.join(args.out_dir, f'tex_f_{task_id:04d}.npy')

    # Target: stored optimized framability from scan_full
    scan_full = np.load(os.path.join(args.out_dir, 'scan_full.npy'))
    ig  = int(round(gamma / GAMMA_STEP))
    igp = int(round(gp    / GAMMA_STEP))
    fra_stored = float(scan_full[ig, igp, 3])
    target = fra_stored + TOL

    print(f'[task {task_id}] (gamma={gamma}, gp={gp})  '
          f'stored_fra={fra_stored:.8f}  target<{target:.8f}', flush=True)

    rho_ss, L = compute_steady_state(J, gamma, gp)
    gate = expm(DT * L).real
    d_ext_single = int(round(np.sqrt(extended_pauli_D().shape[1])))

    best_f, best_x = np.inf, None
    for seed in range(args.max_seeds):
        _, f, x = minimize_framability(
            gate, d_ext_single=d_ext_single, n_restarts=args.n_restarts,
            method=DEFAULT_METHOD, max_iter=200, maxfev=args.maxfev,
            verbose=False, return_x=True, seed=seed,
        )
        if f < best_f:
            best_f = f
            best_x = x.copy()
            print(f'[task {task_id}] seed={seed}: fra={best_f:.8f}', flush=True)
        if best_f < target:
            print(f'[task {task_id}] reached target at seed={seed}', flush=True)
            break
    else:
        print(f'[task {task_id}] exhausted {args.max_seeds} seeds; '
              f'best fra={best_f:.8f}', flush=True)

    S = _project_columns(best_x.reshape(4, d_ext_single))
    np.save(out_S, S)
    np.save(out_f, np.array([best_f]))
    print(f'[task {task_id}] saved {out_S}  fra={best_f:.8f}', flush=True)


if __name__ == '__main__':
    main()
