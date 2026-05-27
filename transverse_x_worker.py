"""
Worker: scan (gamma, gamma') for the two-qubit Lindbladian with an added
transverse-field term  H = J*Z⊗Z + h*(X⊗I + I⊗X).

For each grid point computes:
  1. Pauli framability    -- max row L1-norm of exp(L*dt)
  2. opt_fra_4            -- optimised framability, d_ext_single=4
  3. opt_fra_6            -- optimised framability, d_ext_single=6
  4. max_lpdo_entropy     -- max bond entropy during evolution to steady state

task_id = ig * N_GP + igp
  ig  in 0..N_GAMMA-1   gamma   = GAMMA_STEP * ig,   up to GAMMA_MAX=8
  igp in 0..N_GP-1      gamma_p = GAMMA_STEP * igp,  up to GP_MAX=4

Total tasks: 41 * 21 = 861  (task_ids 0..860)

Output: <out_dir>/transverse_x_<ig:03d>_<igp:03d>.npz
  keys: pauli_fra, opt_fra_4, opt_fra_6, max_lpdo_entropy,
        gamma, gamma_p, J, h, dt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

from optimize_framability import minimize_framability

GAMMA_MAX  = 8.0
GP_MAX     = 4.0
GAMMA_STEP = 0.2
N_GAMMA    = int(round(GAMMA_MAX  / GAMMA_STEP)) + 1   # 41
N_GP       = int(round(GP_MAX     / GAMMA_STEP)) + 1   # 21

_I2 = np.eye(2, dtype=complex)
_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_I2, _sx, _sy, _sz]
_BASIS_2Q = [np.kron(p1, p2) for p1 in _PAULIS for p2 in _PAULIS]   # 16 elements


def _superop_lindblad(H, jump_ops, basis):
    """Build the Lindbladian superoperator in a Pauli-string basis."""
    n = len(basis)
    d = basis[0].shape[0]
    L = np.zeros((n, n), dtype=complex)
    for j, Bj in enumerate(basis):
        res = -1j * (H @ Bj - Bj @ H)
        for A in jump_ops:
            Adag = A.conj().T
            res += A @ Bj @ Adag - 0.5 * (Adag @ A @ Bj + Bj @ Adag @ A)
        for i, Bi in enumerate(basis):
            L[i, j] = np.trace(Bi @ res) / d
    return L.real


def build_lindbladian(J: float, gamma: float, gamma_p: float, h: float) -> np.ndarray:
    """16×16 Lindbladian for H = J*ZZ + h*(XI + IX), same Lindblad ops as base model."""
    mp = 0.5 * np.array([[1, 1], [-1, -1]], dtype=complex)   # |−><+|
    pm = mp.conj().T

    H = (J  * np.kron(_sz, _sz)
       + h  * (np.kron(_sx, _I2) + np.kron(_I2, _sx)))

    jump_ops = [
        np.sqrt(gamma)   * np.kron(mp, _I2),   # relaxation qubit 0
        np.sqrt(gamma)   * np.kron(_I2, mp),   # relaxation qubit 1
        np.sqrt(gamma_p) * np.kron(_sz, _I2),  # dephasing qubit 0
        np.sqrt(gamma_p) * np.kron(_I2, _sz),  # dephasing qubit 1
    ]
    return _superop_lindblad(H, jump_ops, _BASIS_2Q)


def _steady_state_rho(L: np.ndarray) -> np.ndarray | None:
    """Return the 4×4 steady-state density matrix (null eigenvector of L)."""
    eigenvalues, eigenvectors = np.linalg.eig(L)
    idx = np.argmin(np.abs(eigenvalues))
    ss_vec = eigenvectors[:, idx].real
    # Normalise: ss_vec[0] encodes the coefficient of II, which equals 1/4
    norm = ss_vec[0] * 4
    if abs(norm) < 1e-12:
        return None
    ss_vec = ss_vec / norm
    rho = sum(ss_vec[i] * _BASIS_2Q[i] for i in range(16))
    rho = (rho + rho.conj().T) / 2
    return rho


def _pauli_fra(gate: np.ndarray) -> float:
    return float(np.max(np.sum(np.abs(gate), axis=1)))


def _max_lpdo_entropy(L: np.ndarray, rho_ss: np.ndarray,
                      max_steps: int = 5000) -> float:
    """Max LPDO bond entropy during time evolution from |+Y>⊗2 to steady state."""
    from lpdo import (purification_sqrt, tensorize_to_lpdo,
                      truncate_and_validate, _bures_fidelity, _bond_entropy)

    dt = 0.01 * GAMMA_STEP    # 0.002, matching analysis.compute_max_bond_dim convention
    M  = expm(dt * L).real

    # Initial state: (II + IY + YI + YY)/4  — |+Y>⊗2
    v = np.zeros(16)
    for bits in range(4):
        idx = 0
        for k in range(2):
            pauli_idx = 2 if (bits >> (2 - 1 - k)) & 1 else 0   # Y=2, I=0
            idx += pauli_idx * (4 ** (2 - 1 - k))
        v[idx] = 1.0
    v /= 4.0

    basis_arr = np.array(_BASIS_2Q)
    max_entropy = 0.0

    for _ in range(max_steps):
        rho = np.einsum('i,ijk->jk', v, basis_arr)
        rho = (rho + rho.conj().T) / 2
        try:
            X_lp = purification_sqrt(rho)
            A1, A2, _ = tensorize_to_lpdo(X_lp, d=2)
            _, _, _, _, info = truncate_and_validate(rho, A1, A2, d=2)
            max_entropy = max(max_entropy, _bond_entropy(info['singular_values']))
        except Exception:
            pass

        if _bures_fidelity(rho, rho_ss) >= 0.9:
            break
        v_new = M @ v
        if np.allclose(v, v_new, atol=1e-15):
            break
        v = v_new

    return max_entropy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',    type=int,   required=True,
                        help=f'0..{N_GAMMA * N_GP - 1}')
    parser.add_argument('--out_dir',    type=str,   default='results_transverse_x')
    parser.add_argument('--n_restarts', type=int,   default=20)
    parser.add_argument('--max_iter',   type=int,   default=500)
    parser.add_argument('--J',          type=float, default=1.0)
    parser.add_argument('--h',          type=float, default=1.0,
                        help='Transverse-field strength for h*(XI + IX).')
    parser.add_argument('--dt',         type=float, default=0.01)
    parser.add_argument('--max_lpdo_steps', type=int, default=5000)
    parser.add_argument('--seed',       type=int,   default=0)
    args = parser.parse_args()

    total = N_GAMMA * N_GP
    if not (0 <= args.task_id < total):
        print(f'ERROR: task_id out of range (0..{total - 1})', file=sys.stderr)
        sys.exit(1)

    ig  = args.task_id // N_GP
    igp = args.task_id  % N_GP
    gamma   = GAMMA_STEP * ig
    gamma_p = GAMMA_STEP * igp

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'transverse_x_{ig:03d}_{igp:03d}.npz')
    if os.path.exists(out_path):
        print(f'Skip: {out_path} already exists', flush=True)
        return

    print(f'[task {args.task_id}] gamma={gamma:.2f}  gamma_p={gamma_p:.2f}  '
          f'J={args.J}  h={args.h}', flush=True)

    t0 = time.perf_counter()

    L    = build_lindbladian(args.J, gamma, gamma_p, args.h)
    gate = expm(L * args.dt).real

    # ── 1. Pauli framability ──────────────────────────────────────────────────
    pauli_fra = _pauli_fra(gate)

    # ── 2-3. Optimised framability ────────────────────────────────────────────
    _, opt_fra_4, _ = minimize_framability(
        gate, d_ext_single=4,
        n_restarts=args.n_restarts, max_iter=args.max_iter,
        seed=args.seed + args.task_id, verbose=False,
        return_x=True, use_complex=False,
    )
    _, opt_fra_6, _ = minimize_framability(
        gate, d_ext_single=6,
        n_restarts=args.n_restarts, max_iter=args.max_iter,
        seed=args.seed + args.task_id, verbose=False,
        return_x=True, use_complex=False,
    )

    # ── 4. Max LPDO entropy ───────────────────────────────────────────────────
    rho_ss = _steady_state_rho(L)
    if rho_ss is not None:
        max_lpdo_entropy = _max_lpdo_entropy(L, rho_ss, args.max_lpdo_steps)
    else:
        max_lpdo_entropy = np.nan

    elapsed = time.perf_counter() - t0

    np.savez(out_path,
             pauli_fra       = np.array(pauli_fra),
             opt_fra_4       = np.array(opt_fra_4),
             opt_fra_6       = np.array(opt_fra_6),
             max_lpdo_entropy= np.array(max_lpdo_entropy),
             gamma           = np.array(gamma),
             gamma_p         = np.array(gamma_p),
             J               = np.array(args.J),
             h               = np.array(args.h),
             dt              = np.array(args.dt))
    print(f'[task {args.task_id}] saved  pauli_fra={pauli_fra:.4f}  '
          f'opt4={opt_fra_4:.4f}  opt6={opt_fra_6:.4f}  '
          f'lpdo_ent={max_lpdo_entropy:.4f}  elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
