"""
Worker: optimised sign problem of the two-qubit Lindbladian Trotter step
exp(L*dt), with or without a transverse field.

For a single (gamma, gamma') grid point:
  1. Build L for H = J*Z⊗Z + h*(X⊗I + I⊗X) and the usual jump operators.
  2. Form gate = exp(L*dt).real (16x16 superop in the Pauli basis).
  3. Convert to the matrix basis (4x4 ⊗ 4x4 on density-matrix space) via the
     change-of-basis B = stack of Pauli basis vectors → not needed: we work
     directly with the 16x16 superoperator, which is what enters the sign
     problem  s(gate) = |tr(gate)/tr(|gate|)|.
  4. Search over a 3-parameter local rotation R(nx,ny,nz) = exp(iπ(nx X +
     ny Y + nz Z)).  R acts on a single qubit; the corresponding superop on
     the Pauli basis is M(R), and the n-qubit local change of basis on the
     superop is M(R)⊗n.  We minimise s(M(R)⊗2 . gate . M(R)⊗2^†).

task_id = ig * N_GP + igp
  ig  in 0..N_GAMMA-1   gamma   = GAMMA_STEP * ig,   up to GAMMA_MAX = 8
  igp in 0..N_GP-1      gamma_p = GAMMA_STEP * igp,  up to GP_MAX    = 4

Total tasks: 41 * 21 = 861  (task_ids 0..860)

Output: <out_dir>/sign_<tag>_<ig:03d>_<igp:03d>.npz
  keys: sign_init, sign_opt, n_opt, gamma, gamma_p, J, h, dt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sign_problem import minimize_sign_problem, sign_problem

# ── grid ─────────────────────────────────────────────────────────────────────
GAMMA_MAX  = 8.0
GP_MAX     = 4.0
GAMMA_STEP = 0.2
N_GAMMA    = int(round(GAMMA_MAX  / GAMMA_STEP)) + 1   # 41
N_GP       = int(round(GP_MAX     / GAMMA_STEP)) + 1   # 21

# ── basis ────────────────────────────────────────────────────────────────────
_I2 = np.eye(2, dtype=complex)
_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS  = [_I2, _sx, _sy, _sz]
_BASIS_2Q = [np.kron(p1, p2) for p1 in _PAULIS for p2 in _PAULIS]
_BASIS_ARR_2Q = np.array(_BASIS_2Q)                                  # (16,4,4)


# ── Lindbladian ──────────────────────────────────────────────────────────────
def _superop_lindblad(H: np.ndarray, jump_ops: list[np.ndarray],
                      basis: list[np.ndarray]) -> np.ndarray:
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


def build_lindbladian(J: float, gamma: float, gamma_p: float,
                      h: float) -> np.ndarray:
    mp = 0.5 * np.array([[1, 1], [-1, -1]], dtype=complex)        # |−⟩⟨+|
    H = J * np.kron(_sz, _sz) + h * (np.kron(_sx, _I2) + np.kron(_I2, _sx))
    jump_ops = [
        np.sqrt(gamma)   * np.kron(mp,  _I2),
        np.sqrt(gamma)   * np.kron(_I2, mp ),
        np.sqrt(gamma_p) * np.kron(_sz, _I2),
        np.sqrt(gamma_p) * np.kron(_I2, _sz),
    ]
    return _superop_lindblad(H, jump_ops, _BASIS_2Q)


# ── single-qubit rotation → superop in the Pauli basis ──────────────────────
def _single_qubit_superop_in_pauli_basis(R: np.ndarray) -> np.ndarray:
    """4x4 matrix M with rho_basis -> R rho R^† in the {I,X,Y,Z} basis.

    M[i,j] = (1/2) tr(P_i R P_j R^†).
    """
    M = np.zeros((4, 4), dtype=complex)
    for i, Pi in enumerate(_PAULIS):
        for j, Pj in enumerate(_PAULIS):
            M[i, j] = np.trace(Pi @ R @ Pj @ R.conj().T) / 2.0
    return M.real if np.allclose(M.imag, 0, atol=1e-12) else M


def _rotate_gate_pauli_basis(gate: np.ndarray, n_vec: np.ndarray) -> np.ndarray:
    """Apply local change of basis R^⊗2 to a 16x16 Pauli-basis superop."""
    H1 = n_vec[0] * _sx + n_vec[1] * _sy + n_vec[2] * _sz
    R = expm(1j * np.pi * H1)
    M = _single_qubit_superop_in_pauli_basis(R)
    M_full = np.kron(M, M)                                          # (16,16)
    return M_full @ gate @ M_full.conj().T


# ── search wrapped to use the proper rotation in Pauli basis ─────────────────
def minimize_sign_problem_pauli_basis(gate: np.ndarray,
                                      n_restarts: int = 30,
                                      method: str = 'BFGS',
                                      seed: int = 0,
                                      verbose: bool = False
                                      ) -> tuple[float, np.ndarray]:
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)

    def obj(params: np.ndarray) -> float:
        return sign_problem(_rotate_gate_pauli_basis(gate, params))

    best_val = sign_problem(gate)
    best_n = np.zeros(3)
    if verbose:
        print(f'[init] s(gate) = {best_val:.6f}')
    for r in range(n_restarts):
        x0 = rng.standard_normal(3)
        x0 /= max(np.linalg.norm(x0), 1e-14)
        res = minimize(obj, x0, method=method)
        f_cand = float(res.fun)
        if f_cand < best_val:
            best_val = f_cand
            best_n = res.x.copy()
        if verbose:
            print(f'  restart {r + 1:2d}/{n_restarts}:  '
                  f'f={f_cand:.6f}  best={best_val:.6f}')
    return best_val, best_n


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',    type=int,   required=True,
                        help=f'0..{N_GAMMA * N_GP - 1}')
    parser.add_argument('--out_dir',    type=str,   required=True,
                        help='Output directory (e.g. results_sign_problem_nofield).')
    parser.add_argument('--tag',        type=str,   required=True,
                        help='Short tag used in output filenames (e.g. nofield).')
    parser.add_argument('--J',          type=float, default=1.0)
    parser.add_argument('--h',          type=float, default=0.0,
                        help='Transverse-field strength for h*(XI + IX). '
                             '0.0 = no field; 1.0 = standard transverse case.')
    parser.add_argument('--dt',         type=float, default=0.01)
    parser.add_argument('--n_restarts', type=int,   default=30)
    parser.add_argument('--method',     type=str,   default='BFGS')
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
    out_path = os.path.join(
        args.out_dir, f'sign_{args.tag}_{ig:03d}_{igp:03d}.npz'
    )
    if os.path.exists(out_path):
        print(f'Skip: {out_path} already exists', flush=True)
        return

    print(f'[task {args.task_id}] gamma={gamma:.2f}  gamma_p={gamma_p:.2f}  '
          f'J={args.J}  h={args.h}', flush=True)

    L    = build_lindbladian(args.J, gamma, gamma_p, args.h)
    gate = expm(L * args.dt).real

    t0 = time.perf_counter()
    sign_init = sign_problem(gate)
    sign_opt, n_opt = minimize_sign_problem_pauli_basis(
        gate, n_restarts=args.n_restarts, method=args.method,
        seed=args.seed + args.task_id, verbose=False,
    )
    elapsed = time.perf_counter() - t0

    np.savez(out_path,
             sign_init = np.array(sign_init),
             sign_opt  = np.array(sign_opt),
             n_opt     = np.array(n_opt),
             gamma     = np.array(gamma),
             gamma_p   = np.array(gamma_p),
             J         = np.array(args.J),
             h         = np.array(args.h),
             dt        = np.array(args.dt))
    print(f'[task {args.task_id}] saved  s_init={sign_init:.4f}  '
          f's_opt={sign_opt:.4f}  n={n_opt}  elapsed={elapsed:.1f}s',
          flush=True)


if __name__ == '__main__':
    main()
