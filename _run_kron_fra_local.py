"""
Local parallel runner: compute the Kronecker-optimised framability for every
(gamma, gamma') grid point and save the result as kron_fra_grid.npy so that
scan_collect.py can add it as a new panel in the figure.

Usage
-----
    python _run_kron_fra_local.py
"""
import numpy as np
import os
import multiprocessing as mp

N_PTS       = 41
J           = 1.0
GAMMA_STEP  = 0.2
OUT_DIR     = 'results'
D_EXT_SINGLE = 6        # single-qubit frame columns; D = kron(S,S) has 36 cols
N_RESTARTS  = 5
MAXFEV      = 1000


def _process(tid):
    from two_qubit_lindbladian import numeric_two_qubit_lindbladian
    from framability import extended_pauli_D, heisenberg_framability
    from optimize_framability import minimize_framability, DEFAULT_METHOD
    from scipy.linalg import expm

    n   = N_PTS
    ig  = tid // n
    igp = tid %  n

    gamma = GAMMA_STEP * ig
    gp    = GAMMA_STEP * igp

    L    = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gp)
    dt   = 0.01 * GAMMA_STEP
    gate = expm(dt * L).real

    # safety baseline: extended-Pauli framability
    D_ext = extended_pauli_D()
    baseline = heisenberg_framability(D_ext, gate)

    _, f_opt = minimize_framability(
        gate, D_EXT_SINGLE,
        n_restarts=N_RESTARTS, method=DEFAULT_METHOD,
        maxfev=MAXFEV, verbose=False,
    )
    f_opt = min(f_opt, baseline)
    return ig, igp, f_opt


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    out_path = os.path.join(OUT_DIR, 'kron_fra_grid.npy')
    # Load any existing partial results
    if os.path.exists(out_path):
        grid = np.load(out_path)
        print(f'Resuming from existing {out_path}')
    else:
        grid = np.full((N_PTS, N_PTS), np.inf)

    # Collect tasks that still need computing
    tids = [ig * N_PTS + igp
            for ig in range(N_PTS) for igp in range(N_PTS)
            if not np.isfinite(grid[ig, igp])]

    total   = len(tids)
    workers = max(1, mp.cpu_count() - 1)
    print(f'Computing {total} / {N_PTS*N_PTS} grid points on {workers} workers '
          f'(d_ext_single={D_EXT_SINGLE}, n_restarts={N_RESTARTS}) ...', flush=True)

    done = 0
    with mp.Pool(workers) as pool:
        for ig, igp, f_opt in pool.imap_unordered(_process, tids, chunksize=1):
            grid[ig, igp] = f_opt
            done += 1
            print(f'  [{done:3d}/{total}] ({ig:2d},{igp:2d})  kron_fra={f_opt:.6f}',
                  flush=True)
            if done % 20 == 0 or done == total:
                np.save(out_path, grid)

    np.save(out_path, grid)
    print(f'\nSaved {out_path}')

    print('\nRegenerating plot ...')
    import subprocess, sys
    subprocess.run(
        [sys.executable, 'scan_collect.py',
         '--n_pts', str(N_PTS), '--J', str(J),
         '--gamma_step', str(GAMMA_STEP), '--out_dir', OUT_DIR],
        check=True,
    )
    print('Done.')
