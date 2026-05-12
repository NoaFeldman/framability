"""
Per-task worker: minimise over a Kronecker-structured frame D = kron(S, S) the
worst-case framability across the gate set {H, T, CNOT} under 2-qubit
depolarisation with rate p.

For a given (d_ext_single, p):

    min_S   max_{g in {H, T, CNOT}}  framability(D=kron(S,S), N_p^{x2} . g_super)

S has shape (4, d_ext_single).  The first two columns of S are fixed to I and
Z (matching optimize_framability._FIXED_COLS); the remaining free columns are
parameterised by a real vector of length 4*(d_ext_single-2) and normalised
column-wise via the Bloch projection used by optimize_framability.

Output: <out_dir>/minimax_<d>_<pi:02d>.npz
  framability: (3,)   per-gate framability at D_opt, in order [H, T, CNOT]
  worst:       ()     max of the above (the minimised objective)
  D:           (16, d_ext)  optimal frame
  S:           (4, d_ext_single)  single-qubit factor
  x:           (n_params,)  raw parameter vector
  d_ext_single: ()    int
  p:           ()     float

Task layout (set by --task_id):
    task_id = d_idx * N_P + p_idx
    d_idx in 0..N_D-1  (D_EXT_SINGLES = [4, 6, 8])
    p_idx in 0..N_P-1  (P_VALUES = [0.01*i for i in range(11)])
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

from optimize_framability import (
    DEFAULT_METHOD,
    N_FIXED_COLS,
    _FIXED_COLS,
    _build_inits,
    _get_framability_fast,
    _kron_power,
    _params_to_D,
)
from sweep_depol_gates_worker import GATES, build_channel


# ── parameter grid ───────────────────────────────────────────────────────────
D_EXT_SINGLES = [4, 6, 8]
P_VALUES = [0.01 * i for i in range(11)]
N_D = len(D_EXT_SINGLES)
N_P = len(P_VALUES)
N_GATES = len(GATES)
N_S_ROWS = 4  # qubit_d^2
N_QUBITS = 2


# ── objective ────────────────────────────────────────────────────────────────
def per_gate_framabilities(D: np.ndarray, channels: list[np.ndarray]) -> np.ndarray:
    return np.array([_get_framability_fast(D, ch) for ch in channels])


def make_objective(channels: list[np.ndarray], d_ext_single: int):
    """Return a callable f(params) = max_g framability(D, channel_g)."""
    def obj(params: np.ndarray) -> float:
        D = _params_to_D(params, N_S_ROWS, d_ext_single, N_QUBITS)
        return float(np.max(per_gate_framabilities(D, channels)))
    return obj


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help=f'0..{N_D * N_P - 1}; task_id = d_idx * N_P + p_idx.')
    parser.add_argument('--out_dir', type=str, default='results_minimax_frame')
    parser.add_argument('--n_restarts', type=int, default=20)
    parser.add_argument('--maxfev', type=int, default=2000)
    parser.add_argument('--max_iter', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default=DEFAULT_METHOD)
    args = parser.parse_args()

    if args.task_id < 0 or args.task_id >= N_D * N_P:
        print(f'ERROR: task_id {args.task_id} out of range '
              f'(0..{N_D * N_P - 1}).', file=sys.stderr)
        sys.exit(1)

    d_idx = args.task_id // N_P
    p_idx = args.task_id % N_P
    d_ext_single = D_EXT_SINGLES[d_idx]
    p = P_VALUES[p_idx]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'minimax_{d_ext_single}_{p_idx:02d}.npz')
    if os.path.exists(out_path):
        print(f'Skip: {out_path} already exists', flush=True)
        return

    print(f'[task {args.task_id}] d_ext_single={d_ext_single}  p={p:.3f}  '
          f'method={args.method}  n_restarts={args.n_restarts}', flush=True)

    channels = [build_channel(g, float(p)) for g in GATES]
    objective = make_objective(channels, d_ext_single)

    n_free = d_ext_single - N_FIXED_COLS
    n_params = N_S_ROWS * n_free
    d_ext = d_ext_single ** N_QUBITS
    rng = np.random.default_rng(args.seed + 1000 * args.task_id)

    inits = _build_inits(N_S_ROWS, d_ext_single, d_ext, args.n_restarts, rng)

    if args.method == 'cobyqa':
        opts = {'maxfev': args.maxfev}
    elif args.method == 'Powell':
        opts = {'maxiter': args.max_iter, 'maxfev': args.maxfev,
                'ftol': 1e-7, 'xtol': 1e-7}
    else:
        opts = {'maxiter': args.max_iter, 'maxfev': args.maxfev}

    best_val = np.inf
    best_x = None
    t0 = time.perf_counter()
    for i, x0 in enumerate(inits):
        f_x0 = objective(x0)
        if f_x0 < best_val:
            best_val = f_x0
            best_x = x0.copy()
        res = minimize(objective, x0, method=args.method, options=opts)
        f_cand = objective(res.x)
        if f_cand < best_val:
            best_val = f_cand
            best_x = res.x.copy()
        print(f'  restart {i + 1}/{len(inits)}:  '
              f'f_init={f_x0:.6f}  f_opt={f_cand:.6f}  '
              f'best={best_val:.6f}', flush=True)

    elapsed = time.perf_counter() - t0

    D_opt = _params_to_D(best_x, N_S_ROWS, d_ext_single, N_QUBITS)
    free = D_opt  # not needed; recompute S for inspection
    # Recover S = first single-qubit factor; D = kron(S, S) by construction
    n_free = d_ext_single - N_FIXED_COLS
    free_cols = best_x.reshape(N_S_ROWS, n_free)
    # Apply same projection used inside _params_to_D
    from optimize_framability import _project_columns_bloch
    S_opt = np.hstack([_FIXED_COLS, _project_columns_bloch(free_cols)])

    per_gate = per_gate_framabilities(D_opt, channels)
    worst = float(np.max(per_gate))

    np.savez(out_path,
             framability=per_gate, worst=worst,
             D=D_opt, S=S_opt, x=best_x,
             d_ext_single=np.array(d_ext_single),
             p=np.array(p),
             gates=np.array(GATES))
    print(f'[task {args.task_id}] saved {out_path}  '
          f'worst={worst:.6f}  per_gate={per_gate}  '
          f'elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
