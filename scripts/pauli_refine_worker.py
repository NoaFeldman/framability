"""
Post-processing worker: refine grid points where min_fra > pauli_fra
by seeding the Kronecker-frame optimisation with the Pauli (identity)
starting point.

Background
----------
The Pauli framability of a gate M is the max-row 1-norm of M, equal to
heisenberg_framability(D, M) when D = I_n.  In Kronecker mode (D = kron(S, S)),
D = I_16 is represented as kron(S_id, S_id) where S_id is the
"cycling-identity" matrix (shape n_s × d_single, n_s = qubit_d**2 = 4):

    S_id[:, j] = e_{j % n_s}

kron(S_id, S_id) contains all 16 standard-basis vectors as distinct
columns (covering I_16) plus (d_ext - 16) duplicate columns; duplicate
columns do not affect the L1 minimisation, so:

    heisenberg_framability(kron(S_id, S_id), M) == pauli_framability(M)

Seeding the optimiser at S_id therefore starts at exactly the Pauli
baseline and gradient descent can only improve from there.

If stored results show min_fra > pauli_fra (which can happen when the
optimizer diverged in an earlier run), this script re-runs the
optimisation with x_pauli as an *extra* restart seed, applies a hard
upper-bound clamp (min(result, pauli_fra)), and overwrites the row file
in place when an improvement is found.

Column layout of row_XXXX.npy  (must match scan_worker.py)
-----------------------------------------------------------
    0 entropy   1 negativity   2 magic      3 pauli_fra
    4 ext_fra   5 min_fra      6 dec_rate   7 chi

Usage (single row)
------------------
    python pauli_refine_worker.py \\
        --task_id 5 --n_pts 20 --J 1.0 --gamma_step 0.1 --out_dir results

SLURM array
-----------
    sbatch --array=0-19 pauli_refine_array.sh
"""

import argparse
import os
import sys

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian, qubit_d
from framability import extended_pauli_D
from optimize_framability import minimize_framability, DEFAULT_METHOD

# Column indices in row .npy files (must match scan_worker.py)
COL_PAULI_FRA = 3
COL_EXT_FRA   = 4
COL_MIN_FRA   = 5


def _make_gate(J, gamma, gamma_p, gamma_step):
    """Build the real Lindbladian propagator for a single grid point."""
    L = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)
    dt = 0.01 * gamma_step
    return expm(dt * L).real


def _pauli_init_x(d_ext_single):
    """
    Flat parameter vector for the cycling-identity starting point.

    Constructs S_id of shape (n_s, d_ext_single), n_s = qubit_d**2,
    with S_id[:, j] = e_{j % n_s}.  Returns S_id.ravel()
    (length n_s * d_ext_single).

    kron_n(S_id) has all standard-basis vectors as columns, so
    heisenberg_framability(kron_n(S_id), gate) == pauli_framability(gate).
    """
    n_s = qubit_d ** 2   # 4 for two qubits
    S_id = np.zeros((n_s, d_ext_single))
    for j in range(d_ext_single):
        S_id[j % n_s, j] = 1.0
    return S_id.ravel()


def main():
    p = argparse.ArgumentParser(
        description='Re-optimise grid points where min_fra > pauli_fra '
                    'using the Pauli (identity) frame as an extra seed.'
    )
    p.add_argument('--task_id',    type=int,   required=True,
                   help='Row index (gamma index); maps to SLURM_ARRAY_TASK_ID.')
    p.add_argument('--n_pts',      type=int,   default=20,
                   help='Number of grid points per axis (default: 20).')
    p.add_argument('--J',          type=float, default=1.0,
                   help='Coupling constant J (default: 1.0).')
    p.add_argument('--gamma_step', type=float, default=0.1,
                   help="Grid spacing for gamma and gamma' (default: 0.1).")
    p.add_argument('--out_dir',    type=str,   default='results',
                   help='Directory containing row_XXXX.npy files '
                        '(default: results/).')
    p.add_argument('--n_restarts', type=int,   default=5,
                   help='Random restarts per data point (default: 5). '
                        'The Pauli seed is always appended as an extra restart.')
    p.add_argument('--maxfev',     type=int,   default=1000,
                   help='Max function evaluations per restart (default: 1000).')
    p.add_argument('--max_iter',   type=int,   default=200,
                   help='Max iterations per restart (default: 200).')
    args = p.parse_args()

    ig = args.task_id
    if ig < 0 or ig >= args.n_pts:
        print(f'ERROR: task_id {ig} out of range [0, {args.n_pts - 1}]',
              file=sys.stderr)
        sys.exit(1)

    row_path = os.path.join(args.out_dir, f'row_{ig:04d}.npy')
    if not os.path.exists(row_path):
        print(f'ERROR: {row_path} not found — run scan_worker first.',
              file=sys.stderr)
        sys.exit(1)

    row_data = np.load(row_path).copy()   # (n_pts, 8); copy to avoid aliasing

    gamma   = args.gamma_step * ig
    D_ext   = extended_pauli_D()
    d_ext_single = int(round(np.sqrt(D_ext.shape[1])))   # 6
    x_pauli = _pauli_init_x(d_ext_single)

    print(f'[task {ig}] gamma={gamma:.4f}  scanning {args.n_pts} gamma\' values '
          f'for min_fra > pauli_fra', flush=True)

    n_improved = 0
    for igp in range(args.n_pts):
        pauli_fra = row_data[igp, COL_PAULI_FRA]
        ext_fra   = row_data[igp, COL_EXT_FRA]
        min_fra   = row_data[igp, COL_MIN_FRA]

        if min_fra <= pauli_fra + 1e-9:
            # Already at or below the Pauli baseline; nothing to do.
            continue

        gp   = args.gamma_step * igp
        gate = _make_gate(args.J, gamma, gp, args.gamma_step)

        print(
            f'[task {ig}] col {igp}  gamma\'={gp:.4f}: '
            f'min_fra={min_fra:.6f} > pauli_fra={pauli_fra:.6f}  → refining …',
            flush=True,
        )

        _, new_min_fra = minimize_framability(
            gate,
            d_ext_single  = d_ext_single,
            n_restarts    = args.n_restarts,
            method        = DEFAULT_METHOD,
            max_iter      = args.max_iter,
            maxfev        = args.maxfev,
            verbose       = False,
            extra_init_xs = [x_pauli],
        )

        # Hard upper-bound: the identity frame always attains pauli_fra, so
        # the true minimum cannot exceed it.  ext_fra is an additional safe
        # upper bound from the known extended-Pauli frame.
        new_min_fra = min(new_min_fra, pauli_fra, ext_fra)

        if new_min_fra < min_fra - 1e-12:
            print(
                f'[task {ig}]   improved: {min_fra:.6f} → {new_min_fra:.6f}',
                flush=True,
            )
            row_data[igp, COL_MIN_FRA] = new_min_fra
            n_improved += 1
        else:
            print(
                f'[task {ig}]   no further improvement '
                f'(best found: {new_min_fra:.6f})',
                flush=True,
            )

    if n_improved > 0:
        np.save(row_path, row_data)
        print(
            f'[task {ig}] saved updated row '
            f'({n_improved}/{args.n_pts} points improved).',
            flush=True,
        )
    else:
        print(f'[task {ig}] no improvements found; file unchanged.', flush=True)


if __name__ == '__main__':
    main()
