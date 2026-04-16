"""Local parallel runner for the extra properties."""
import numpy as np
import os
import multiprocessing as mp

N_PTS = 41
J = 1.0
GAMMA_STEP = 0.2
OUT_DIR = 'results'
N_QUBITS = 2
PRODUCT_CHI = 6


def _process(tid):
    from scipy.linalg import expm
    from analysis import (compute_steady_state, compute_magnetization_x,
                          compute_ss_bond_entropy)
    from framability import dyadic_stabilizer_framability, product_state_framability
    n = N_PTS
    ig  = tid // n
    igp = tid %  n
    gamma = GAMMA_STEP * ig
    gp    = GAMMA_STEP * igp
    out = os.path.join(OUT_DIR, f'point_extra_{ig:04d}_{igp:04d}.npy')

    if os.path.exists(out):
        existing = np.load(out)
        if existing.shape == (4,):
            return tid, None, None, None, None  # already up to date

    rho_ss, L = compute_steady_state(J, gamma, gp, N=N_QUBITS)
    se     = compute_ss_bond_entropy(rho_ss, N=N_QUBITS)
    mgx    = compute_magnetization_x(rho_ss, N=N_QUBITS)
    dt     = 0.01 * GAMMA_STEP
    gate   = expm(dt * L).real
    sf     = dyadic_stabilizer_framability(gate, n_qubits=N_QUBITS)
    pf     = product_state_framability(PRODUCT_CHI, gate)
    np.save(out, np.array([se, mgx, sf, pf]))
    return tid, se, mgx, sf, pf


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    total   = N_PTS * N_PTS
    tids    = list(range(total))
    workers = max(1, mp.cpu_count() - 1)
    print(f'Running {total} tasks on {workers} workers...', flush=True)
    done = 0
    with mp.Pool(workers) as pool:
        for tid, se, mgx, sf, pf in pool.imap_unordered(_process, tids, chunksize=1):
            done += 1
            if se is not None:
                ig, igp = tid // N_PTS, tid % N_PTS
                print(f'  [{done:3d}/{total}] ig={ig:2d} igp={igp:2d}  '
                      f'ss_ent={se:.4f}  mag_x={mgx:.4f}  '
                      f'stab_fra={sf:.6f}  product_fra={pf:.6f}', flush=True)
            else:
                print(f'  [{done:3d}/{total}] task {tid:3d} skipped (exists)',
                      flush=True)
    print('All done. Running collect...')
    import subprocess, sys
    subprocess.run([sys.executable, 'scan_collect.py',
                    '--n_pts', str(N_PTS), '--J', str(J),
                    '--gamma_step', str(GAMMA_STEP), '--out_dir', OUT_DIR],
                   check=True)
