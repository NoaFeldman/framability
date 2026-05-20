"""
Worker: minimize framability of a random two-qubit gate
    U = exp(i * (alpha * XX + beta * YY + gamma * ZZ))
where XX = X⊗X, YY = Y⊗Y, ZZ = Z⊗Z are 2-qubit Pauli products.

The 10 angle triples (alpha, beta, gamma) are drawn uniformly from
[0, pi/2)^3 using a fixed master seed, so all workers agree on the
assignment without communication.

task_id = d_idx * N_SAMPLES + sample_idx
  d_idx      in 0..1   D_EXT_SINGLES = [4, 6]
  sample_idx in 0..9

Total tasks: 20  (task_id 0..19)

Output: <out_dir>/random_kron_<d>_<sample_idx:02d>.npz
  keys: framability, D (16, d_ext), x, alpha, beta, gamma, d_ext_single
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

from optimize_framability import minimize_framability, DEFAULT_METHOD

D_EXT_SINGLES = [4, 6]
N_D           = len(D_EXT_SINGLES)
N_SAMPLES     = 10
MASTER_SEED   = 42

_I2 = np.eye(2,  dtype=complex)
_X  = np.array([[0, 1], [1, 0]],    dtype=complex)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z  = np.array([[1, 0], [0, -1]],   dtype=complex)
_P1 = [_I2, _X, _Y, _Z]
PAULIS_2Q = [np.kron(a, b) for a in _P1 for b in _P1]


def _generate_angles() -> np.ndarray:
    """Return (N_SAMPLES, 3) array of (alpha, beta, gamma) in [0, pi/2)^3."""
    rng = np.random.default_rng(MASTER_SEED)
    return rng.uniform(0.0, np.pi / 2.0, size=(N_SAMPLES, 3))


def build_gate_superop(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """16×16 real superoperator of U = exp(i*(alpha*XX + beta*YY + gamma*ZZ))."""
    XX   = np.kron(_X, _X)
    YY   = np.kron(_Y, _Y)
    ZZ   = np.kron(_Z, _Z)
    U    = expm(1j * (alpha * XX + beta * YY + gamma * ZZ))

    n = 16
    L = np.zeros((n, n), dtype=float)
    for j, Bj in enumerate(PAULIS_2Q):
        img = U @ Bj @ U.conj().T
        for i, Bi in enumerate(PAULIS_2Q):
            L[i, j] = (np.trace(Bi.conj().T @ img) / 4).real
    return L


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',    type=int,   required=True,
                        help=f'0..{N_D * N_SAMPLES - 1}')
    parser.add_argument('--out_dir',    type=str,   default='results_random_kron')
    parser.add_argument('--n_restarts', type=int,   default=10)
    parser.add_argument('--maxfev',     type=int,   default=2000)
    parser.add_argument('--max_iter',   type=int,   default=500)
    parser.add_argument('--method',     type=str,   default=DEFAULT_METHOD)
    parser.add_argument('--seed',       type=int,   default=0)
    args = parser.parse_args()

    total = N_D * N_SAMPLES
    if not (0 <= args.task_id < total):
        print(f'ERROR: task_id out of range (0..{total - 1})', file=sys.stderr)
        sys.exit(1)

    d_idx      = args.task_id // N_SAMPLES
    sample_idx = args.task_id  % N_SAMPLES
    d_ext_single = D_EXT_SINGLES[d_idx]

    angles              = _generate_angles()
    alpha, beta, gamma  = angles[sample_idx]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f'random_kron_{d_ext_single}_{sample_idx:02d}.npz'
    )
    if os.path.exists(out_path):
        print(f'Skip: {out_path} already exists', flush=True)
        return

    print(f'[task {args.task_id}] d_ext_single={d_ext_single}  '
          f'sample={sample_idx}  '
          f'alpha={alpha:.6f}  beta={beta:.6f}  gamma={gamma:.6f}', flush=True)

    gate = build_gate_superop(float(alpha), float(beta), float(gamma))

    t0 = time.perf_counter()
    D_opt, f_opt, x_opt = minimize_framability(
        gate, d_ext_single=d_ext_single,
        n_restarts=args.n_restarts,
        method=args.method,
        max_iter=args.max_iter,
        maxfev=args.maxfev,
        seed=args.seed + args.task_id,
        verbose=True,
        return_x=True,
    )
    elapsed = time.perf_counter() - t0

    np.savez(out_path,
             framability=np.array(f_opt),
             D=D_opt,
             x=x_opt,
             alpha=np.array(alpha),
             beta=np.array(beta),
             gamma=np.array(gamma),
             d_ext_single=np.array(d_ext_single))
    print(f'[task {args.task_id}] saved {out_path}  '
          f'fra={f_opt:.6f}  elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
