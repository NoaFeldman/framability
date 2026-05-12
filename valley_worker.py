"""
Valley-edge worker for the two-qubit Lindbladian framability.

For one task (a (gamma, gamma_p) grid point and a d_ext_single value):

  1. Build the Lindbladian gate U(dt) = expm(dt * L(J, gamma, gamma_p)).real.
  2. Find S_opt minimising  f(D) = heisenberg_framability(D, gate)
     with the Kronecker structure D = kron(S, S), S of shape (4, d_ext_single).
     The first two columns of S are fixed to I and Z (matching
     optimize_framability._FIXED_COLS); free columns are normalised under
     the *equality* constraint |c_I| + ||c_XYZ||_2 == 1.
  3. Walk from x_opt in `valley_param_size` (default 10) distinct
     directions in parameter space, finding the largest step alpha along
     each direction such that the framability stays within
     [f_opt, f_opt + plateau_tol].  Directions are Gram-Schmidt
     orthogonalised against the *step vectors* of previously discovered
     edge points to spread them out.

Single-gate framability does not need an inequality on the column norms:
columns can always be rescaled to unit Bloch norm without changing the
LP feasible set up to a scalar, so we keep the equality |c_I| + ||c_XYZ||_2 = 1.

Output:
    <out_dir>/valley_<gi>_<gpi>_d<d>.npz
    keys:
        gamma, gamma_p, J, dt, d_ext_single, plateau_tol
        f_opt, x_opt (n_params,), D_opt (16, d_ext)
        edge_xs (K, n_params), edge_Ds (K, 16, d_ext)
        edge_fs (K,), edge_alphas (K,), edge_step_norms (K,)

task_id layout (when using --task_id):
    task_id = gi * N_GPS + gpi          gi in 0..N_GAMMAS-1, gpi in 0..N_GAMMA_PS-1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from two_qubit_lindbladian import (
    numeric_two_qubit_lindbladian,
    pauli_string_dim,
    qubit_d,
)
from optimize_framability import (
    DEFAULT_METHOD,
    N_FIXED_COLS,
    _FIXED_COLS,
    _build_inits,
    _get_framability_fast,
    _kron_power,
)


# ── parameter grid (used when --task_id is given) ────────────────────────────
# Explicit list of (gamma, gamma_p) points; task_id indexes this list.
POINTS = [
    (6.0, 0.0),
    (3.0, 0.8),
]
N_TASKS = len(POINTS)

N_S_ROWS  = qubit_d ** 2     # 4
N_QUBITS  = 2


# ── frame parameterisation (equality column norm) ────────────────────────────
def _project_columns_bloch_eq(M: np.ndarray) -> np.ndarray:
    """Columns of M (4xk) with |c_I| + ||c_XYZ||_2 == 1 (hard equality)."""
    c_I   = np.abs(M[0:1, :])
    bloch = np.linalg.norm(M[1:4, :], axis=0, keepdims=True)
    total = c_I + bloch
    return M / np.maximum(total, 1e-12)


def _build_S(params: np.ndarray, d_ext_single: int) -> np.ndarray:
    n_free = d_ext_single - N_FIXED_COLS
    free = _project_columns_bloch_eq(params.reshape(N_S_ROWS, n_free))
    return np.hstack([_FIXED_COLS, free])


def _params_to_D(params: np.ndarray, d_ext_single: int) -> np.ndarray:
    return _kron_power(_build_S(params, d_ext_single), N_QUBITS)


def _objective(params: np.ndarray, d_ext_single: int, gate: np.ndarray) -> float:
    D = _params_to_D(params, d_ext_single)
    return float(_get_framability_fast(D, gate))


# ── extended-Pauli (a=0.84) seeding -----------------------------------------
def _ext_pauli_xy_init(d_ext_single: int, a: float = 0.84) -> np.ndarray:
    n_free = d_ext_single - N_FIXED_COLS
    base = np.array([
        [0.0, 0.0, 0.0,           0.0          ],
        [1.0, 0.0, a/np.sqrt(2),  a/np.sqrt(2) ],
        [0.0, 1.0, a/np.sqrt(2), -a/np.sqrt(2) ],
        [0.0, 0.0, 0.0,           0.0          ],
    ])
    free = np.zeros((N_S_ROWS, n_free))
    k = min(n_free, base.shape[1])
    free[:, :k] = base[:, :k]
    return free.ravel()


# ── stage 1: minimise framability ────────────────────────────────────────────
def _find_min(gate, d_ext_single, n_restarts, seed, method,
              max_iter, maxfev, verbose):
    rng = np.random.default_rng(seed)
    d_ext = d_ext_single ** N_QUBITS
    inits = _build_inits(N_S_ROWS, d_ext_single, d_ext, n_restarts, rng)
    inits = [_ext_pauli_xy_init(d_ext_single)] + inits

    if method == 'cobyqa':
        opts = {'maxfev': maxfev}
    elif method == 'Powell':
        opts = {'maxiter': max_iter, 'maxfev': maxfev,
                'ftol': 1e-8, 'xtol': 1e-8}
    else:
        opts = {'maxiter': max_iter, 'maxfev': maxfev}

    fobj = lambda x: _objective(x, d_ext_single, gate)
    best_val = np.inf
    best_x = None
    for i, x0 in enumerate(inits):
        f0 = fobj(x0)
        if f0 < best_val:
            best_val = f0
            best_x = x0.copy()
        res = minimize(fobj, x0, method=method, options=opts)
        fc = fobj(res.x)
        if fc < best_val:
            best_val = fc
            best_x = res.x.copy()
        if verbose:
            print(f'  restart {i + 1}/{len(inits)}:  '
                  f'f_init={f0:.6f}  f_opt={fc:.6f}  best={best_val:.6f}',
                  flush=True)
    return best_x, best_val


# ── stage 2: walk to the edge of the plateau along a given direction ────────
def _walk_to_edge(x0, direction, d_ext_single, gate, f_min,
                  tol, init_step=0.1, max_expansions=60,
                  alpha_cap=1e4, bisect_iters=40):
    """Find the largest alpha >= 0 such that f(x0 + alpha*direction) <= f_min + tol.

    Strategy: exponential search (double alpha) until f exceeds the
    threshold or alpha hits ``alpha_cap``; then bisect.  Returns
    ``(x_edge, alpha)``.
    """
    threshold = f_min + tol
    fobj = lambda x: _objective(x, d_ext_single, gate)

    alpha = init_step
    last_ok = 0.0
    last_bad = None
    for _ in range(max_expansions):
        f = fobj(x0 + alpha * direction)
        if f <= threshold:
            last_ok = alpha
            alpha *= 2.0
            if alpha > alpha_cap:
                break
        else:
            last_bad = alpha
            break

    if last_bad is None:
        # Never exceeded the threshold; treat the cap as the edge.
        return x0 + last_ok * direction, last_ok

    lo, hi = last_ok, last_bad
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        f = fobj(x0 + mid * direction)
        if f <= threshold:
            lo = mid
        else:
            hi = mid
    return x0 + lo * direction, lo


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, default=None,
                        help=f'0..{N_TASKS - 1}; task_id = gi*N_GPS + gpi.  '
                             'Overrides --gamma / --gamma_p when given.')
    parser.add_argument('--gamma',   type=float, default=None)
    parser.add_argument('--gamma_p', type=float, default=None)
    parser.add_argument('--J',       type=float, default=1.0)
    parser.add_argument('--dt',      type=float, default=0.01)
    parser.add_argument('--d_ext_single', type=int, default=6)
    parser.add_argument('--n_restarts',   type=int, default=20)
    parser.add_argument('--max_iter',     type=int, default=500)
    parser.add_argument('--maxfev',       type=int, default=2000)
    parser.add_argument('--method',       type=str, default=DEFAULT_METHOD)
    parser.add_argument('--seed',         type=int, default=0)
    parser.add_argument('--valley_param_size', type=int, default=10)
    parser.add_argument('--plateau_tol',  type=float, default=1e-4,
                        help='Edge tolerance: f(x_edge) <= f_opt + plateau_tol.')
    parser.add_argument('--init_step',    type=float, default=0.1,
                        help='Initial alpha for exponential edge search.')
    parser.add_argument('--out_dir',      type=str, default='results_valley')
    parser.add_argument('--verbose',      action='store_true')
    args = parser.parse_args()

    # --- resolve task_id / (gamma, gamma_p) ---------------------------------
    if args.task_id is not None:
        if args.task_id < 0 or args.task_id >= N_TASKS:
            print(f'ERROR: task_id {args.task_id} out of range '
                  f'(0..{N_TASKS - 1}).', file=sys.stderr)
            sys.exit(1)
        gamma, gamma_p = POINTS[args.task_id]
        tag = f'task{args.task_id:02d}'
    else:
        if args.gamma is None or args.gamma_p is None:
            print('ERROR: provide --task_id, or both --gamma and --gamma_p.',
                  file=sys.stderr)
            sys.exit(1)
        gamma   = args.gamma
        gamma_p = args.gamma_p
        tag = f'g{gamma:.3f}_gp{gamma_p:.3f}'

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f'valley_{tag}_d{args.d_ext_single}.npz')
    if os.path.exists(out_path):
        print(f'Skip: {out_path} already exists', flush=True)
        return

    # --- build the 2q gate ---------------------------------------------------
    L = numeric_two_qubit_lindbladian(args.J, gamma, gamma_p)
    gate = expm(args.dt * L).real

    print(f'[task gamma={gamma}, gamma_p={gamma_p}] J={args.J}  dt={args.dt}  '
          f'd_ext_single={args.d_ext_single}  '
          f'method={args.method}  n_restarts={args.n_restarts}', flush=True)

    # --- stage 1: minimise framability --------------------------------------
    t0 = time.perf_counter()
    x_opt, f_opt = _find_min(
        gate, args.d_ext_single,
        args.n_restarts, args.seed, args.method,
        args.max_iter, args.maxfev, args.verbose,
    )
    t_min = time.perf_counter() - t0
    print(f'[stage 1] f_opt={f_opt:.6f}  ({t_min:.1f}s)', flush=True)

    # --- stage 2: walk to the edge in diverse directions --------------------
    rng = np.random.default_rng(args.seed + 12345 + 1000 *
                                (args.task_id if args.task_id is not None else 0))
    n_params = x_opt.size
    K = args.valley_param_size
    d_ext = args.d_ext_single ** N_QUBITS

    edge_xs    = np.zeros((K, n_params), dtype=float)
    edge_alphas = np.zeros(K, dtype=float)
    edge_fs    = np.zeros(K, dtype=float)
    edge_Ds    = np.zeros((K, pauli_string_dim, d_ext), dtype=float)
    edge_step_norms = np.zeros(K, dtype=float)

    step_basis: list[np.ndarray] = []  # orthonormalised previous steps
    rejected_for_zero_step = 0

    t1 = time.perf_counter()
    for k in range(K):
        # Sample direction orthogonal to all previous *step vectors*.
        direction = None
        for _ in range(40):
            v = rng.standard_normal(n_params)
            for u in step_basis:
                v = v - np.dot(v, u) * u
            nv = np.linalg.norm(v)
            if nv > 1e-8:
                direction = v / nv
                break
        if direction is None:
            v = rng.standard_normal(n_params)
            direction = v / np.linalg.norm(v)

        x_edge, alpha = _walk_to_edge(
            x_opt, direction, args.d_ext_single, gate, f_opt,
            tol=args.plateau_tol, init_step=args.init_step,
        )
        f_edge = _objective(x_edge, args.d_ext_single, gate)
        D_edge = _params_to_D(x_edge, args.d_ext_single)
        step = x_edge - x_opt
        nstep = float(np.linalg.norm(step))

        edge_xs[k]         = x_edge
        edge_alphas[k]     = alpha
        edge_fs[k]         = f_edge
        edge_Ds[k]         = D_edge
        edge_step_norms[k] = nstep

        if nstep > 1e-10:
            # Re-orthogonalise (steps may differ from direction by tiny amounts).
            u = step / nstep
            for w in step_basis:
                u = u - np.dot(u, w) * w
            un = np.linalg.norm(u)
            if un > 1e-10:
                step_basis.append(u / un)
        else:
            rejected_for_zero_step += 1

        print(f'  edge {k + 1}/{K}:  alpha={alpha:.4f}  '
              f'f_edge={f_edge:.6f}  step_norm={nstep:.4f}', flush=True)

    t_walk = time.perf_counter() - t1

    D_opt = _params_to_D(x_opt, args.d_ext_single)
    np.savez(
        out_path,
        gamma=np.array(gamma), gamma_p=np.array(gamma_p),
        J=np.array(args.J), dt=np.array(args.dt),
        d_ext_single=np.array(args.d_ext_single),
        plateau_tol=np.array(args.plateau_tol),
        f_opt=np.array(f_opt), x_opt=x_opt, D_opt=D_opt,
        edge_xs=edge_xs, edge_Ds=edge_Ds,
        edge_fs=edge_fs, edge_alphas=edge_alphas,
        edge_step_norms=edge_step_norms,
        rejected_for_zero_step=np.array(rejected_for_zero_step),
    )
    print(f'[saved] {out_path}  stage1={t_min:.1f}s  stage2={t_walk:.1f}s', flush=True)


if __name__ == '__main__':
    main()
