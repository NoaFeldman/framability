"""
Worker: sign problem of the 6-qubit star+plaquette Lindbladian Trotter step
exp(L*dt) on the 20 x 20 grid (gamma_s, gamma_p) in [0, 4) with step 0.2.

Lindbladian (from two_d_lindbladian.six_qubit_lindbladian):
    H   = h (X_u + X_r) + lam (Z_u + Z_r)
    L_s = sqrt(gamma_s) X_u X_r X_d X_l
    L_p = sqrt(gamma_p) Z_u Z_r Z_ur Z_ru

Sign problem:
    s(U) = |sum(U) / sum(|U|)|   --  s = 1 means no sign problem (best),
                                     s -> 0 is severe cancellation.

We MAXIMISE s over translationally-invariant local rotations R^{otimes 6}:
    R(n) = exp(i pi (n_x X + n_y Y + n_z Z))
The Pauli-basis 4x4 superop M(R) is applied to each of the 12 axes of the
4096 x 4096 gate via tensor contractions, avoiding the full 4096 x 4096
Kronecker product.

Parallelisation: N_TOTAL = 400 grid points are split across --n_jobs tasks.

Per-point output: <out_dir>/sign_six_<ig:03d>_<igp:03d>.npz with keys:
    sign_init, sign_opt, n_opt, gamma_s, gamma_p, h, lam, dt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sign_problem import sign_problem
from two_d_lindbladian import six_qubit_lindbladian, N_QUBITS_6Q


# ── grid ─────────────────────────────────────────────────────────────────────
GAMMA_STEP = 0.2
N_GRID     = int(round(4.0 / GAMMA_STEP))   # 20
DT         = GAMMA_STEP / 10.0              # 0.02
N_TOTAL    = N_GRID * N_GRID                # 400


# ── Pauli single-qubit basis ─────────────────────────────────────────────────
_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_I2, _SX, _SY, _SZ]


def _single_qubit_superop_in_pauli_basis(R: np.ndarray) -> np.ndarray:
    """4x4 superop M with rho_{Pauli} -> R rho R^†.

    M[i, j] = (1/2) tr(P_i R P_j R^†).
    For unitary R, M is real orthogonal.
    """
    M = np.zeros((4, 4), dtype=complex)
    for i, Pi in enumerate(_PAULIS):
        for j, Pj in enumerate(_PAULIS):
            M[i, j] = np.trace(Pi @ R @ Pj @ R.conj().T) / 2.0
    return M.real if np.allclose(M.imag, 0, atol=1e-12) else M


def _apply_local_rotation(gate: np.ndarray, M: np.ndarray,
                          n_qubits: int = N_QUBITS_6Q) -> np.ndarray:
    """Compute (M^{otimes n}) gate (M^{otimes n})^T via tensor contractions.

    gate: (d, d) with d = 4**n
    M:    (4, 4) real
    Avoids forming the full d x d Kronecker product M^{otimes n}.
    Cost: O(n * 4**(2 n + 1)) flops vs O(4**(3 n)) for the dense product.
    """
    d = 4 ** n_qubits
    g = gate.reshape((4,) * (2 * n_qubits))
    # Output (first n axes): g -> M . g
    for k in range(n_qubits):
        g = np.tensordot(M, g, axes=([1], [k]))   # new axis at 0
        g = np.moveaxis(g, 0, k)
    # Input (last n axes): g -> g . M^T which contracts axis k of g with
    # axis 1 of M, so the same tensordot pattern as above on axes n..2n-1.
    for k in range(n_qubits):
        axis = n_qubits + k
        g = np.tensordot(M, g, axes=([1], [axis]))
        g = np.moveaxis(g, 0, axis)
    return g.reshape(d, d)


def _rotated_sign(gate: np.ndarray, n_vec: np.ndarray) -> float:
    """sign_problem((M(R))^{otimes 6} . gate . (...)^T) for R = R(n_vec)."""
    H1 = n_vec[0] * _SX + n_vec[1] * _SY + n_vec[2] * _SZ
    R = expm(1j * np.pi * H1)
    M = _single_qubit_superop_in_pauli_basis(R)
    if np.max(np.abs(M.imag)) > 1e-10:
        return float('nan')
    rotated = _apply_local_rotation(gate, M.real)
    return float(sign_problem(rotated))


def _maximise_sign(gate: np.ndarray,
                   n_restarts: int = 10,
                   seed: int = 0,
                   verbose: bool = False) -> tuple[float, np.ndarray]:
    """Maximise s(rotated gate) over (nx, ny, nz) via BFGS on -s."""
    rng = np.random.default_rng(seed)
    best_val = float(sign_problem(gate))
    best_n   = np.zeros(3)

    def neg_obj(params: np.ndarray) -> float:
        return -_rotated_sign(gate, params)

    for r in range(n_restarts):
        x0 = rng.standard_normal(3)
        x0 /= max(np.linalg.norm(x0), 1e-14)
        res = minimize(neg_obj, x0, method='BFGS')
        f_cand = float(-res.fun)
        if f_cand > best_val:
            best_val = f_cand
            best_n   = res.x.copy()
        if verbose:
            print(f'    restart {r+1:2d}/{n_restarts}  f={f_cand:.6f}  '
                  f'best={best_val:.6f}', flush=True)
    return best_val, best_n


# ── per-point computation ────────────────────────────────────────────────────
def _process_point(ig: int, igp: int, args) -> None:
    gamma_s = GAMMA_STEP * ig
    gamma_p = GAMMA_STEP * igp
    out_path = Path(args.out_dir) / f'sign_six_{ig:03d}_{igp:03d}.npz'
    if out_path.exists():
        print(f'  skip {out_path.name} (exists)', flush=True)
        return

    t_start = time.perf_counter()
    print(f'[ig={ig:02d} igp={igp:02d}]  gamma_s={gamma_s:.2f}  gamma_p={gamma_p:.2f}  '
          f'h={args.h}  lam={args.lam}', flush=True)

    t0 = time.perf_counter()
    L = six_qubit_lindbladian(gamma_s=gamma_s, gamma_p=gamma_p,
                              h=args.h, lam=args.lam).real
    print(f'  L built  ({time.perf_counter()-t0:.1f}s)', flush=True)

    t0 = time.perf_counter()
    gate = expm(L * args.dt).real
    print(f'  expm(L*dt)  ({time.perf_counter()-t0:.1f}s)', flush=True)

    t0 = time.perf_counter()
    sign_init = float(sign_problem(gate))
    sign_opt, n_opt = _maximise_sign(
        gate, n_restarts=args.n_restarts,
        seed=args.seed + ig * N_GRID + igp,
        verbose=False,
    )
    print(f'  s_init={sign_init:.4f}  s_opt={sign_opt:.4f}  '
          f'n_opt={n_opt}  ({time.perf_counter()-t0:.1f}s)', flush=True)

    np.savez(out_path,
             sign_init = np.array(sign_init),
             sign_opt  = np.array(sign_opt),
             n_opt     = np.array(n_opt),
             gamma_s   = np.array(gamma_s),
             gamma_p   = np.array(gamma_p),
             h         = np.array(args.h),
             lam       = np.array(args.lam),
             dt        = np.array(args.dt))
    print(f'  saved {out_path.name}  (total {time.perf_counter()-t_start:.1f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',    type=int, required=True)
    p.add_argument('--n_jobs',     type=int, required=True,
                   help=f'Total array tasks (N_TOTAL={N_TOTAL} grid points split).')
    p.add_argument('--out_dir',    type=str, required=True)
    p.add_argument('--h',          type=float, default=1.0)
    p.add_argument('--lam',        type=float, default=1.0)
    p.add_argument('--dt',         type=float, default=DT)
    p.add_argument('--n_restarts', type=int,   default=10)
    p.add_argument('--seed',       type=int,   default=0)
    args = p.parse_args()

    if not (0 <= args.task_id < args.n_jobs):
        print(f'ERROR: task_id {args.task_id} out of range [0, {args.n_jobs})',
              file=sys.stderr)
        sys.exit(1)

    pts_per_job = (N_TOTAL + args.n_jobs - 1) // args.n_jobs
    start = args.task_id * pts_per_job
    stop  = min(N_TOTAL, start + pts_per_job)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f'[task {args.task_id}/{args.n_jobs}] grid linear-indices '
          f'[{start}, {stop})  out_dir={args.out_dir}', flush=True)

    for lin in range(start, stop):
        ig  = lin // N_GRID
        igp = lin %  N_GRID
        try:
            _process_point(ig, igp, args)
        except Exception as e:
            print(f'  !! point (ig={ig}, igp={igp}) failed: '
                  f'{type(e).__name__}({e!r})', flush=True)


if __name__ == '__main__':
    main()
