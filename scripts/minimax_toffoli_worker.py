"""
Worker: minimax framability for gate set {H, Toffoli} under 3-qubit
depolarisation N_p ⊗ N_p ⊗ N_p.

For each (d_ext_single, p) find S (shape 4 × d_ext_single) minimising:
    max_{g in {H, Toffoli}}  framability(kron(S,S,S), N_p^{x3} . g_super)

H is lifted to 3 qubits as H ⊗ I ⊗ I.
Toffoli (CCX) acts on all 3 qubits with controls on qubits 0,1 and
target on qubit 2.

task_id = d_idx * N_P + p_idx
  d_idx in 0..N_D-1   D_EXT_SINGLES = [4, 6]
  p_idx in 0..N_P-1   P_VALUES = 0.00 .. 0.10 step 0.01 (N_P=11)

Total tasks: 2 * 11 = 22  (task_id 0..21)

Note: d_ext_single=8 gives d_ext=512 for 3 qubits, which results in an
LP with ~524k variables — impractical. Only [4, 6] are included here.

Output: <out_dir>/minimax_toffoli_<d>_<pi:02d>.npz
  keys: framability (2,), worst, D (64, d_ext), S (4, d_ext_single),
        x, d_ext_single, p, gates
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

from optimize_framability import (
    N_FIXED_COLS,
    _FIXED_COLS,
    _build_inits,
    _get_framability_fast,
    _kron_power,
    _project_columns_bloch,
)

# ── parameter grid ────────────────────────────────────────────────────────────
D_EXT_SINGLES = [4, 6]
P_VALUES      = [0.01 * i for i in range(11)]
N_D, N_P      = len(D_EXT_SINGLES), len(P_VALUES)
N_S_ROWS      = 4    # qubit_d^2 for one qubit
N_QUBITS      = 3

GATES = ['H', 'Toffoli']

# ── 3-qubit Pauli basis ───────────────────────────────────────────────────────
_I2 = np.eye(2, dtype=complex)
_X  = np.array([[0, 1], [1, 0]], dtype=complex)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z  = np.array([[1, 0], [0, -1]], dtype=complex)
_P1 = [_I2, _X, _Y, _Z]
PAULIS_3Q = [np.kron(np.kron(a, b), c)
             for a in _P1 for b in _P1 for c in _P1]  # 64 elements


def _superop_3q(U: np.ndarray) -> np.ndarray:
    """64×64 real superoperator for rho -> U rho U† in the 3-qubit Pauli basis."""
    n = 64
    L = np.zeros((n, n), dtype=float)
    for j, Bj in enumerate(PAULIS_3Q):
        img = U @ Bj @ U.conj().T
        for i, Bi in enumerate(PAULIS_3Q):
            L[i, j] = (np.trace(Bi.conj().T @ img) / 8).real
    return L


def _depol_3q(p: float) -> np.ndarray:
    """N_p ⊗ N_p ⊗ N_p diagonal channel in the 3-qubit Pauli basis."""
    diag = np.array(
        [(1.0 - 4 * p) ** ((a != 0) + (b != 0) + (c != 0))
         for a in range(4) for b in range(4) for c in range(4)],
        dtype=float,
    )
    return np.diag(diag)


def build_channel_3q(gate_label: str, p: float) -> np.ndarray:
    """Return the 64×64 depolarised channel for H or Toffoli on 3 qubits."""
    H_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
    # Toffoli (CCX): |0..6> unchanged, |6><->|7> (|110><->|111>)
    Toffoli = np.eye(8, dtype=complex)
    Toffoli[6, 6] = 0.0; Toffoli[6, 7] = 1.0
    Toffoli[7, 7] = 0.0; Toffoli[7, 6] = 1.0

    if gate_label == 'H':
        U = np.kron(H_mat, np.eye(4, dtype=complex))   # H ⊗ I ⊗ I
    elif gate_label == 'Toffoli':
        U = Toffoli
    else:
        raise ValueError(gate_label)

    return _depol_3q(p) @ _superop_3q(U)


# ── optimiser helpers ─────────────────────────────────────────────────────────

def _build_S(params: np.ndarray, d_ext_single: int) -> np.ndarray:
    n_free = d_ext_single - N_FIXED_COLS
    free = _project_columns_bloch(params[:N_S_ROWS * n_free].reshape(N_S_ROWS, n_free))
    return np.hstack([_FIXED_COLS, free])


def _ixyz_xy_init(d_ext_single: int, a: float = 1.0) -> np.ndarray:
    """S = [I | X, Y, Z, a(X+Y)/√2, a(X-Y)/√2, ...]  (real, free columns only)."""
    n_free = d_ext_single - N_FIXED_COLS
    base = np.array([
        [0.0, 0.0, 0.0, 0.0,           0.0          ],
        [1.0, 0.0, 0.0, a/np.sqrt(2),  a/np.sqrt(2) ],
        [0.0, 1.0, 0.0, a/np.sqrt(2), -a/np.sqrt(2) ],
        [0.0, 0.0, 1.0, 0.0,           0.0          ],
    ])
    free = np.zeros((N_S_ROWS, n_free))
    k = min(n_free, base.shape[1])
    free[:, :k] = base[:, :k]
    return free.ravel()


def make_objective(channels: list[np.ndarray], d_ext_single: int):
    def obj(params: np.ndarray) -> float:
        S = _build_S(params, d_ext_single)
        D = _kron_power(S, N_QUBITS)
        return float(np.max([_get_framability_fast(D, ch) for ch in channels]))
    return obj


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',    type=int,   required=True,
                        help=f'0..{N_D * N_P - 1}')
    parser.add_argument('--out_dir',    type=str,   default='results_minimax_toffoli')
    parser.add_argument('--n_restarts', type=int,   default=20)
    parser.add_argument('--maxfev',     type=int,   default=2000)
    parser.add_argument('--max_iter',   type=int,   default=500)
    parser.add_argument('--seed',       type=int,   default=0)
    parser.add_argument('--method',     type=str,   default='SLSQP',
                        help='SLSQP / Powell / Nelder-Mead')
    args = parser.parse_args()

    if not (0 <= args.task_id < N_D * N_P):
        print(f'ERROR: task_id out of range (0..{N_D * N_P - 1})', file=sys.stderr)
        sys.exit(1)

    d_idx        = args.task_id // N_P
    p_idx        = args.task_id  % N_P
    d_ext_single = D_EXT_SINGLES[d_idx]
    p            = P_VALUES[p_idx]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f'minimax_toffoli_{d_ext_single}_{p_idx:02d}.npz'
    )
    if os.path.exists(out_path):
        print(f'Skip: {out_path} already exists', flush=True)
        return

    print(f'[task {args.task_id}] d_ext_single={d_ext_single}  p={p:.3f}  '
          f'method={args.method}  n_restarts={args.n_restarts}', flush=True)

    channels  = [build_channel_3q(g, float(p)) for g in GATES]
    objective = make_objective(channels, d_ext_single)

    n_free   = d_ext_single - N_FIXED_COLS
    n_params = N_S_ROWS * n_free
    d_ext    = d_ext_single ** N_QUBITS
    rng      = np.random.default_rng(args.seed + 1000 * args.task_id)
    inits    = [_ixyz_xy_init(d_ext_single, a=0.84)] + _build_inits(
        N_S_ROWS, d_ext_single, d_ext, args.n_restarts, rng, use_complex=False)

    if args.method == 'SLSQP':
        opts = {'maxiter': args.max_iter, 'ftol': 1e-8}
    elif args.method == 'Powell':
        opts = {'maxiter': args.max_iter, 'maxfev': args.maxfev,
                'ftol': 1e-7, 'xtol': 1e-7}
    else:
        opts = {'maxiter': args.max_iter, 'maxfev': args.maxfev}

    best_val = np.inf
    best_x   = None
    t0 = time.perf_counter()

    for i, x0 in enumerate(inits):
        # _build_inits with use_complex=False returns length n_params arrays
        x0_real = x0[:n_params]
        f_x0 = objective(x0_real)
        if f_x0 < best_val:
            best_val, best_x = f_x0, x0_real.copy()
        res    = minimize(objective, x0_real, method=args.method, options=opts)
        f_cand = objective(res.x)
        if f_cand < best_val:
            best_val, best_x = f_cand, res.x.copy()
        print(f'  restart {i + 1}/{len(inits)}:  '
              f'f_init={f_x0:.6f}  f_opt={f_cand:.6f}  best={best_val:.6f}',
              flush=True)

    elapsed  = time.perf_counter() - t0
    S_opt    = _build_S(best_x, d_ext_single)
    D_opt    = _kron_power(S_opt, N_QUBITS)
    per_gate = np.array([_get_framability_fast(D_opt, ch) for ch in channels])
    worst    = float(np.max(per_gate))

    np.savez(out_path,
             framability=per_gate,
             worst=worst,
             D=D_opt,
             S=S_opt,
             x=best_x,
             d_ext_single=np.array(d_ext_single),
             p=np.array(p),
             gates=np.array(GATES))
    print(f'[task {args.task_id}] saved {out_path}  '
          f'worst={worst:.6f}  per_gate={per_gate}  '
          f'elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
