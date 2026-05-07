"""
Worker: neighbor-seeded refinement of optimised framability (results_opt/).

For each grid point (ig, igp) this script:
  1. Loads the current best fra from opt_fra_<ig>_<igp>.npy (if present).
  2. Loads the 1-qubit S matrix from opt_S_<ig>_<igp>.npy (if present) and
     uses its free columns as an extra restart seed.
  3. Finds the 4-connected neighbor with the lowest current fra and loads
     its opt_S file as a further extra seed.
  4. Re-runs minimize_framability with all gathered seeds.
  5. Saves the updated:
       opt_fra_<ig:04d>_<igp:04d>.npy  — shape (2,): [min_fra, pauli_fra]
       opt_S_<ig:04d>_<igp:04d>.npy   — shape (4, d_ext_single): 1-qubit S

Files are only overwritten when a strictly better fra is found (or when
opt_S did not previously exist).

Usage
-----
    python opt_refine_worker.py --task_id 42 --n_pts 41 \\
                                --J 1.0 --gamma_step 0.2 --out_dir results_opt
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from framability import extended_pauli_D, heisenberg_framability
from optimize_framability import (
    minimize_framability, DEFAULT_METHOD,
    _params_to_D, _FIXED_COLS, N_FIXED_COLS,
    _project_columns_bloch,
)

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _x_from_S(S):
    """Extract the free-column parameter vector from a full S matrix."""
    return S[:, N_FIXED_COLS:].ravel().copy()


def _S_from_x(x, n_s, d_ext_single):
    """Reconstruct normalised S (shape n_s × d_ext_single) from x."""
    return _params_to_D(x, n_s, d_ext_single, n_qubits=1)


def _load_fra(out_dir, ig, igp):
    """Return (min_fra, pauli_fra) from opt_fra file, or (inf, inf)."""
    fp = os.path.join(out_dir, f'opt_fra_{ig:04d}_{igp:04d}.npy')
    if os.path.exists(fp):
        arr = np.load(fp)
        return float(arr[0]), float(arr[1])
    return np.inf, np.inf


def _load_S(out_dir, ig, igp):
    """Return S array (4, d_ext_single) or None."""
    fp = os.path.join(out_dir, f'opt_S_{ig:04d}_{igp:04d}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',    type=int,   required=True)
    p.add_argument('--n_pts',      type=int,   default=41)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str,   default='results_opt')
    p.add_argument('--n_restarts', type=int,   default=5)
    p.add_argument('--maxfev',     type=int,   default=1000)
    args = p.parse_args()

    n   = args.n_pts
    ig  = args.task_id // n
    igp = args.task_id %  n

    if ig >= n:
        print(f'ERROR: task_id {args.task_id} out of range for {n}x{n} grid',
              file=sys.stderr)
        sys.exit(1)

    gamma = args.gamma_step * ig
    gp    = args.gamma_step * igp
    dt    = 0.01 * args.gamma_step

    os.makedirs(args.out_dir, exist_ok=True)

    # ── build gate ────────────────────────────────────────────────────────
    L    = numeric_two_qubit_lindbladian(J=args.J, gamma=gamma, gamma_p=gp)
    gate = expm(dt * L).real

    D_ext = extended_pauli_D()
    d_ext_single = int(round(np.sqrt(D_ext.shape[1])))  # = 6
    n_s = 4                                               # qubit_d^2

    pauli_fra = float(np.max(np.sum(np.abs(gate), axis=1)))

    # ── load current best ─────────────────────────────────────────────────
    cur_fra, _ = _load_fra(args.out_dir, ig, igp)
    cur_S      = _load_S(args.out_dir, ig, igp)

    # ── gather extra seeds from current S and best neighbor ───────────────
    extra_init_xs = []

    if cur_S is not None:
        extra_init_xs.append(_x_from_S(cur_S))

    best_nb_fra = np.inf
    best_nb_S   = None
    for di, dj in NEIGHBORS:
        ni, nj = ig + di, igp + dj
        if 0 <= ni < n and 0 <= nj < n:
            nb_fra, _ = _load_fra(args.out_dir, ni, nj)
            if nb_fra < best_nb_fra:
                nb_S = _load_S(args.out_dir, ni, nj)
                if nb_S is not None:
                    best_nb_fra = nb_fra
                    best_nb_S   = nb_S

    if best_nb_S is not None:
        extra_init_xs.append(_x_from_S(best_nb_S))

    # ── optimise ──────────────────────────────────────────────────────────
    D_opt, min_fra, x_opt = minimize_framability(
        gate,
        d_ext_single=d_ext_single,
        n_restarts=args.n_restarts,
        method=DEFAULT_METHOD,
        max_iter=200,
        maxfev=args.maxfev,
        verbose=False,
        extra_init_xs=extra_init_xs if extra_init_xs else None,
        return_x=True,
    )

    # sanity floor: ext-Pauli D
    ext_fra = heisenberg_framability(D_ext, gate)
    min_fra = float(min(min_fra, ext_fra))

    # extract 1-qubit S from best x_opt
    S_opt = _S_from_x(x_opt, n_s, d_ext_single)  # shape (4, 6)

    # ── save (always write S; only improve fra) ───────────────────────────
    fra_path = os.path.join(args.out_dir, f'opt_fra_{ig:04d}_{igp:04d}.npy')
    S_path   = os.path.join(args.out_dir, f'opt_S_{ig:04d}_{igp:04d}.npy')

    improved = min_fra < cur_fra - 1e-9
    save_fra = improved or not os.path.exists(fra_path)
    save_S   = improved or cur_S is None

    if save_fra:
        np.save(fra_path, np.array([min_fra, pauli_fra]))
    if save_S:
        np.save(S_path, S_opt)

    tag = 'improved' if improved else ('new-S' if cur_S is None else 'unchanged')
    print(
        f'({ig},{igp}) gamma={gamma:.3f} gp={gp:.3f}  '
        f'min_fra={min_fra:.6f}  pauli_fra={pauli_fra:.6f}  [{tag}]',
        flush=True,
    )


if __name__ == '__main__':
    main()
