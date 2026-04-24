"""
Compute Robustness of Magic (RoM) for all (gamma, gamma') grid points.

Saves  <out_dir>/rom_grid.npy   shape (n_pts, n_pts).
Then re-runs scan_collect.py to update scan_full.npy and the figure.

Reference
---------
Hamaguchi, Hamada & Yoshioka, "Handbook for Quantifying Robustness of
Magic", Quantum 8, 1461 (2024).  arXiv:2311.01362

Usage
-----
    python compute_rom.py [--n_pts 20] [--J 1.0] [--gamma_step 0.1]
                          [--out_dir results] [--N 2]
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np

# ── defaults (match the rest of the scan pipeline) ────────────────────────
N_PTS = 20
J = 1.0
GAMMA_STEP = 0.1
OUT_DIR = 'results'
N_QUBITS = 2


def _process(task):
    """Compute RoM for one grid point."""
    tid, n_pts, J_val, gamma_step, N = task
    from analysis import compute_steady_state, compute_rom

    ig = tid // n_pts
    igp = tid % n_pts
    gamma = gamma_step * ig
    gp = gamma_step * igp

    rho_ss, _ = compute_steady_state(J_val, gamma, gp, N=N)
    rom = compute_rom(rho_ss, N=N)
    return tid, ig, igp, rom


def main():
    p = argparse.ArgumentParser(
        description='Compute Robustness of Magic for the full scan grid.'
    )
    p.add_argument('--n_pts', type=int, default=N_PTS)
    p.add_argument('--J', type=float, default=J)
    p.add_argument('--gamma_step', type=float, default=GAMMA_STEP)
    p.add_argument('--out_dir', type=str, default=OUT_DIR)
    p.add_argument('--N', type=int, default=N_QUBITS)
    args = p.parse_args()

    n = args.n_pts
    total = n * n
    os.makedirs(args.out_dir, exist_ok=True)

    rom_grid = np.zeros((n, n))
    tasks = [(tid, n, args.J, args.gamma_step, args.N)
             for tid in range(total)]
    workers = max(1, mp.cpu_count() - 1)
    print(f'Computing RoM for {total} grid points on {workers} workers...',
          flush=True)

    done = 0
    with mp.Pool(workers) as pool:
        for tid, ig, igp, rom in pool.imap_unordered(_process, tasks,
                                                     chunksize=1):
            rom_grid[ig, igp] = rom
            done += 1
            if done % 20 == 0 or done == total:
                print(f'  [{done:3d}/{total}] last: ig={ig} igp={igp} '
                      f'RoM={rom:.6f}', flush=True)

    rom_path = os.path.join(args.out_dir, 'rom_grid.npy')
    np.save(rom_path, rom_grid)
    print(f'Saved RoM grid to {rom_path}')

    # Re-run collect to fold RoM into scan_full.npy and regenerate the plot
    print('Regenerating plot...')
    subprocess.run(
        [sys.executable, 'scan_collect.py',
         '--n_pts', str(n), '--J', str(args.J),
         '--gamma_step', str(args.gamma_step), '--out_dir', args.out_dir],
        check=True,
    )


if __name__ == '__main__':
    main()
