"""
Refine scan results at outlier data points using neighbor seeding.

A grid point (ig, igp) is flagged as an outlier when its stored min_fra
is significantly above the best 4-connected neighbor's value:

    min_fra[ig, igp]  >  best_neighbor  *  (1 + rel_tol)
                         AND the absolute gap  >  abs_tol

For each outlier the script:
  1.  Picks the neighbor with the lowest stored min_fra.
  2.  Re-runs minimize_framability on the NEIGHBOR's gate (1 restart) to
      capture its locally-optimal parameter vector x_nb.
  3.  Re-runs minimize_framability on the OUTLIER's gate, seeding with
      x_nb as an extra restart in addition to the standard ones.
  4.  If the refined value is strictly lower than the stored value (and
      also satisfies the ext_fra safety clamp) the row .npy file is
      updated in-place.

Multiple passes are performed until no data point improves.

Usage
-----
    python refine_scan.py [--n_pts 20] [--J 1.0] [--gamma_step 0.1]
                          [--out_dir results] [--rel_tol 0.05]
                          [--abs_tol 1e-3] [--max_passes 10]
                          [--n_restarts 3] [--maxfev 1000]
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from framability import extended_pauli_D, get_framability
from optimize_framability import minimize_framability, DEFAULT_METHOD

# Column indices in row .npy files (must match scan_worker.py)
COL_EXT_FRA = 4
COL_MIN_FRA = 5


def _make_gate(J, gamma, gamma_p, gamma_step):
    """Build the Lindbladian propagator for a single grid point."""
    L = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)
    dt = 0.01 * gamma_step
    M = expm(dt * L)
    return M.real


def _load_data(out_dir, n_pts):
    """Load all row files into a (n_pts, n_pts, 8) array."""
    rows = []
    missing = []
    for ig in range(n_pts):
        path = os.path.join(out_dir, f'row_{ig:04d}.npy')
        if not os.path.exists(path):
            missing.append(path)
        else:
            rows.append(np.load(path))
    if missing:
        print('ERROR: missing result files:', file=sys.stderr)
        for f in missing:
            print(f'  {f}', file=sys.stderr)
        sys.exit(1)
    return np.stack(rows)   # (n_pts, n_pts, 8), axis0=ig, axis1=igp


def _save_row(data, ig, out_dir):
    """Overwrite row_{ig:04d}.npy with the current data slice."""
    path = os.path.join(out_dir, f'row_{ig:04d}.npy')
    np.save(path, data[ig])   # data[ig] has shape (n_pts, 8)


def _detect_outliers(min_fra, rel_tol, abs_tol):
    """
    Return a list of (ig, igp, best_ni, best_nj, current_val, best_nb_val)
    for all points that are significantly above their best neighbor.
    """
    n = min_fra.shape[0]
    outliers = []
    for ig in range(n):
        for igp in range(n):
            v = min_fra[ig, igp]
            best_nb_val = np.inf
            best_ni, best_nj = -1, -1
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ni, nj = ig + di, igp + dj
                if 0 <= ni < n and 0 <= nj < n:
                    nb_val = min_fra[ni, nj]
                    if nb_val < best_nb_val:
                        best_nb_val = nb_val
                        best_ni, best_nj = ni, nj
            if best_ni == -1:
                continue
            gap_abs = v - best_nb_val
            gap_rel = gap_abs / max(best_nb_val, 1e-15)
            if gap_abs > abs_tol and gap_rel > rel_tol:
                outliers.append((ig, igp, best_ni, best_nj, v, best_nb_val))
    return outliers


def refine(n_pts=20, J=1.0, gamma_step=0.1, out_dir='results',
           rel_tol=0.05, abs_tol=1e-3, max_passes=10,
           n_restarts=3, maxfev=1000):
    """
    Iteratively refine outlier data points using neighbor-seeded optimization.
    """
    d_ext = extended_pauli_D().shape[1]   # 36

    print(f'Loading data from {out_dir}/ ...')
    data = _load_data(out_dir, n_pts)   # (n_pts, n_pts, 8)

    for pass_idx in range(1, max_passes + 1):
        min_fra = data[:, :, COL_MIN_FRA]
        ext_fra = data[:, :, COL_EXT_FRA]

        outliers = _detect_outliers(min_fra, rel_tol, abs_tol)
        if not outliers:
            print(f'Pass {pass_idx}: no outliers detected — converged.')
            break

        print(f'Pass {pass_idx}: {len(outliers)} outlier(s) found.')
        n_improved = 0

        for k, (ig, igp, ni, nj, old_val, nb_val) in enumerate(outliers):
            gamma    = gamma_step * ig
            gamma_p  = gamma_step * igp
            gamma_n  = gamma_step * ni
            gamma_p_n = gamma_step * nj

            print(f'  [{k+1}/{len(outliers)}] '
                  f'({ig},{igp}) min_fra={old_val:.5f}  '
                  f'neighbor ({ni},{nj}) min_fra={nb_val:.5f}', end='', flush=True)

            gate_outlier  = _make_gate(J, gamma,   gamma_p,   gamma_step)
            gate_neighbor = _make_gate(J, gamma_n, gamma_p_n, gamma_step)

            # Step 1: get the neighbor's locally-optimal parameter vector
            _, _, x_nb = minimize_framability(
                gate_neighbor, d_ext=d_ext, mode='kronecker',
                n_restarts=1, method=DEFAULT_METHOD,
                maxfev=maxfev, verbose=False, return_x=True,
            )

            # Step 2: re-optimise the outlier, seeded with the neighbor's x
            _, f_refined, _ = minimize_framability(
                gate_outlier, d_ext=d_ext, mode='kronecker',
                n_restarts=n_restarts, method=DEFAULT_METHOD,
                maxfev=maxfev, verbose=False,
                extra_init_xs=[x_nb],
            )

            # Safety clamp: cannot be worse than the extended-Pauli frame
            f_refined = min(f_refined, ext_fra[ig, igp])

            if f_refined < old_val - 1e-9:
                data[ig, igp, COL_MIN_FRA] = f_refined
                _save_row(data, ig, out_dir)
                n_improved += 1
                print(f'  → improved to {f_refined:.5f}  '
                      f'(Δ = {old_val - f_refined:.5f})')
            else:
                print(f'  → no improvement ({f_refined:.5f})')

        print(f'Pass {pass_idx}: improved {n_improved}/{len(outliers)} point(s).')
        if n_improved == 0:
            print('No improvements in this pass — stopping.')
            break

    # Update scan_full.npy if it exists
    combined_path = os.path.join(out_dir, 'scan_full.npy')
    np.save(combined_path, data)
    print(f'Updated {combined_path}')

    return data


def main():
    p = argparse.ArgumentParser(
        description='Refine scan outliers using neighbor-seeded optimization.'
    )
    p.add_argument('--n_pts',      type=int,   default=20,
                   help='Grid size N (scan grid is N×N, default 20)')
    p.add_argument('--J',          type=float, default=1.0,
                   help='Coupling constant J (default 1.0)')
    p.add_argument('--gamma_step', type=float, default=0.1,
                   help='Grid spacing (default 0.1)')
    p.add_argument('--out_dir',    type=str,   default='results',
                   help='Directory containing row_XXXX.npy files')
    p.add_argument('--rel_tol',    type=float, default=0.05,
                   help='Relative gap threshold for outlier detection (default 0.05)')
    p.add_argument('--abs_tol',    type=float, default=1e-3,
                   help='Absolute gap threshold for outlier detection (default 1e-3)')
    p.add_argument('--max_passes', type=int,   default=10,
                   help='Maximum refinement passes (default 10)')
    p.add_argument('--n_restarts', type=int,   default=3,
                   help='Standard restarts per outlier refinement (default 3)')
    p.add_argument('--maxfev',     type=int,   default=1000,
                   help='Max function evaluations per restart (default 1000)')
    args = p.parse_args()

    refine(
        n_pts=args.n_pts,
        J=args.J,
        gamma_step=args.gamma_step,
        out_dir=args.out_dir,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        max_passes=args.max_passes,
        n_restarts=args.n_restarts,
        maxfev=args.maxfev,
    )


if __name__ == '__main__':
    main()
