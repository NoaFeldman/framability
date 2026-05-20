"""
Per-task worker: minimise over a Kronecker-structured frame D = kron(S, S) the
worst-case framability across a gate set under 2-qubit depolarisation with
rate p.

Supported gate sets (selected via --gate_set):
    H_CNOT_T      {H, T, CNOT}        (default, backward compatible)
    H_CNOT_sqrtT  {H, CNOT, sqrtT}

For a given (d_ext_single, p):

    min_S   max_{g in gate_set}  framability(D=kron(S,S), N_p^{x2} . g_super)

S has shape (4, d_ext_single).  The first two columns of S are fixed to I and
Z (matching optimize_framability._FIXED_COLS); the remaining free columns are
parameterised by a real vector of length 4*(d_ext_single-2) and normalised
column-wise via the Bloch projection used by optimize_framability.

Output: <out_dir>/minimax_<d>_<pi:02d>.npz
  framability: (n_gates,) per-gate framability at D_opt
  worst:       ()     max of the above (the minimised objective)
  D:           (16, d_ext)  optimal frame
  S:           (4, d_ext_single)  single-qubit factor
  x:           (n_params,)  raw parameter vector
  d_ext_single: ()    int
  p:           ()     float
  gates:       (n_gates,)  gate labels

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
    _project_columns_bloch,
)
from sweep_depol_gates_worker import build_channel


# ── gate set definitions ─────────────────────────────────────────────────────
GATE_SETS = {
    'H_CNOT_T':     ['CNOT', 'H', 'T'],
    'H_CNOT_sqrtT': ['H', 'CNOT', 'sqrtT'],
}

# ── parameter grid ───────────────────────────────────────────────────────────
D_EXT_SINGLES = [4, 6, 8]
P_VALUES = [0.01 * i for i in range(11)]
N_D = len(D_EXT_SINGLES)
N_P = len(P_VALUES)
N_S_ROWS = 4  # qubit_d^2
N_QUBITS = 2


# ── objective ────────────────────────────────────────────────────────────────
def per_gate_framabilities(D: np.ndarray, channels: list[np.ndarray]) -> np.ndarray:
    return np.array([_get_framability_fast(D, ch) for ch in channels])


def _build_S(params: np.ndarray, d_ext_single: int) -> np.ndarray:
    """Decode params into the single-qubit factor S = [_FIXED_COLS | free].

    Free columns are projected onto |c_I| + ||c_XYZ||_2 <= 1.  Matches the
    parameterisation used inside optimize_framability._params_to_D.
    """
    n_free = d_ext_single - N_FIXED_COLS
    free = _project_columns_bloch(params.reshape(N_S_ROWS, n_free))
    return np.hstack([_FIXED_COLS, free])


def _ext_pauli_xy_init(d_ext_single: int, a: float = 0.84) -> np.ndarray:
    """Flat param vector whose decoded S matches the generalised X-Y
    extended-Pauli frame with scale ``a`` (default a = 0.84).

    Single-qubit free columns are [X, Y, a*(X+Y)/sqrt(2), a*(X-Y)/sqrt(2)]:

        free = [[0,   0,    0,        0       ],
                [1,   0,   a/√2,    a/√2     ],
                [0,   1,   a/√2,   -a/√2     ],
                [0,   0,    0,        0       ]]

    For d_ext_single = 6 these are exactly the 4 free columns of the
    a-scaled extended-Pauli D.  For smaller d (4) the first (d-2) columns
    are used; for larger d (8) extra columns are zero-padded.
    """
    n_free = d_ext_single - N_FIXED_COLS
    base = np.array([
        [0.0,  0.0,  0.0,         0.0        ],
        [1.0,  0.0,  a/np.sqrt(2), a/np.sqrt(2)],
        [0.0,  1.0,  a/np.sqrt(2),-a/np.sqrt(2)],
        [0.0,  0.0,  0.0,         0.0        ],
    ])
    free = np.zeros((N_S_ROWS, n_free))
    k = min(n_free, base.shape[1])
    free[:, :k] = base[:, :k]
    return free.ravel()


def make_objective(channels: list[np.ndarray], d_ext_single: int):
    """Return f(params) = max_g framability(D, channel_g)."""
    def obj(params: np.ndarray) -> float:
        S = _build_S(params, d_ext_single)
        D = _kron_power(S, N_QUBITS)
        return float(np.max(per_gate_framabilities(D, channels)))
    return obj


def make_diag_constraint(d_ext_single: int):
    """Return a function whose output (4,) must be >=0:
        (S S^T)_{ii} - 1 >= 0 for i = 0..3.

    Equivalent to diag(D D^T) >= 1 elementwise when D = kron(S, S).
    Rows 0 and 3 of S are pinned by _FIXED_COLS so those entries are
    >= 1 automatically; the binding constraints are on rows 1 (X) and
    2 (Y).
    """
    def g(params: np.ndarray) -> np.ndarray:
        S = _build_S(params, d_ext_single)
        return np.einsum('ij,ij->i', S, S) - 1.0
    return g


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
    parser.add_argument('--method', type=str, default='SLSQP',
                        help='Optimizer. SLSQP/trust-constr/COBYLA support '
                             'the diag(SS^T)>=1 inequality constraint.')
    parser.add_argument('--gate_set', type=str, default='H_CNOT_T',
                        choices=list(GATE_SETS.keys()),
                        help='Gate set to optimise over (default: H_CNOT_T).')
    args = parser.parse_args()

    GATES = GATE_SETS[args.gate_set]
    N_GATES = len(GATES)

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
    diag_g = make_diag_constraint(d_ext_single)
    constraints = ({'type': 'ineq', 'fun': diag_g},)

    n_free = d_ext_single - N_FIXED_COLS
    n_params = N_S_ROWS * n_free
    d_ext = d_ext_single ** N_QUBITS
    rng = np.random.default_rng(args.seed + 1000 * args.task_id)

    inits = _build_inits(N_S_ROWS, d_ext_single, d_ext, args.n_restarts, rng,
                         use_complex=False)
    # Prepend the X-Y extended-Pauli starting point.  For p above ~0.04
    # this point alone achieves worst-case framability = 1 across
    # {H, T, CNOT}, so without it the optimiser frequently misses the
    # global optimum at higher p.
    inits = [_ext_pauli_xy_init(d_ext_single)] + inits

    if args.method == 'SLSQP':
        opts = {'maxiter': args.max_iter, 'ftol': 1e-8}
    elif args.method == 'trust-constr':
        opts = {'maxiter': args.max_iter, 'xtol': 1e-8, 'gtol': 1e-8}
    elif args.method == 'COBYLA':
        opts = {'maxiter': args.max_iter, 'rhobeg': 0.1, 'catol': 1e-8}
    elif args.method == 'cobyqa':
        opts = {'maxfev': args.maxfev}
    elif args.method == 'Powell':
        opts = {'maxiter': args.max_iter, 'maxfev': args.maxfev,
                'ftol': 1e-7, 'xtol': 1e-7}
    else:
        opts = {'maxiter': args.max_iter, 'maxfev': args.maxfev}

    use_constraints = args.method in ('SLSQP', 'trust-constr', 'COBYLA')

    best_val = np.inf
    best_x = None
    t0 = time.perf_counter()
    for i, x0 in enumerate(inits):
        f_x0 = objective(x0)
        feas_x0 = bool(np.all(diag_g(x0) >= -1e-8))
        if feas_x0 and f_x0 < best_val:
            best_val = f_x0
            best_x = x0.copy()
        if use_constraints:
            res = minimize(objective, x0, method=args.method,
                           constraints=constraints, options=opts)
        else:
            res = minimize(objective, x0, method=args.method, options=opts)
        f_cand = objective(res.x)
        feas_cand = bool(np.all(diag_g(res.x) >= -1e-8))
        if feas_cand and f_cand < best_val:
            best_val = f_cand
            best_x = res.x.copy()
        print(f'  restart {i + 1}/{len(inits)}:  '
              f'f_init={f_x0:.6f}(feas={feas_x0})  '
              f'f_opt={f_cand:.6f}(feas={feas_cand})  '
              f'best={best_val:.6f}', flush=True)

    elapsed = time.perf_counter() - t0

    S_opt = _build_S(best_x, d_ext_single)
    D_opt = _kron_power(S_opt, N_QUBITS)

    per_gate = per_gate_framabilities(D_opt, channels)
    worst = float(np.max(per_gate))
    diag_SST = np.einsum('ij,ij->i', S_opt, S_opt)
    constraint_ok = bool(np.all(diag_SST >= 1.0 - 1e-8))

    np.savez(out_path,
             framability=per_gate, worst=worst,
             D=D_opt, S=S_opt, x=best_x,
             diag_SST=diag_SST,
             constraint_ok=np.array(constraint_ok),
             d_ext_single=np.array(d_ext_single),
             p=np.array(p),
             gates=np.array(GATES))
    print(f'[task {args.task_id}] saved {out_path}  '
          f'worst={worst:.6f}  per_gate={per_gate}  '
          f'diag(S S^T)={diag_SST}  constraint_ok={constraint_ok}  '
          f'elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
