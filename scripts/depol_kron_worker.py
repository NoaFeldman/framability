"""
Worker: optimise framability of  depol_2q(p) ∘ g  where
    g = exp(i*(alpha*XX + beta*YY + gamma*ZZ))

The 10 angle triples (alpha, beta, gamma) are drawn from [0, pi/2)^3 with
MASTER_SEED=42, matching random_kron_worker.py.

task_id = sample_idx * N_D * N_P + d_idx * N_P + p_idx
  sample_idx in 0..N_SAMPLES-1   (N_SAMPLES = 10)
  d_idx      in 0..N_D-1         (D_EXT_SINGLES = [4, 6, 8])
  p_idx      in 0..N_P-1         (P_VALUES = 0.00..0.09)

Total tasks: 300  (task_id 0..299)

Output: <out_dir>/depol_kron_<d>_<sample_idx:02d>_<p_idx:02d>.npz
  keys: framability, D, S, x, alpha, beta, gamma, p, d_ext_single, sample_idx
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimize_framability import (
    N_FIXED_COLS,
    OPT_VERSION,
    _FIXED_COLS,
    _get_framability_fast,
    _kron_power,
    _project_columns_bloch,
    spectral_floor,
)
from sweep_depol_gates_worker import _depol_2q, _superop_2q

# ── parameter grid ────────────────────────────────────────────────────────────
MASTER_SEED   = 42
N_SAMPLES     = 10
D_EXT_SINGLES = [4, 6, 8]
P_VALUES      = [0.01 * i for i in range(10)]   # 0.00 .. 0.09
N_D, N_P      = len(D_EXT_SINGLES), len(P_VALUES)
N_S_ROWS      = 4    # qubit d^2
N_QUBITS      = 2

# ── Pauli helpers ─────────────────────────────────────────────────────────────
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _generate_angles() -> np.ndarray:
    """Return (N_SAMPLES, 3) array of (alpha, beta, gamma) in [0, pi/2)^3."""
    rng = np.random.default_rng(MASTER_SEED)
    return rng.uniform(0.0, np.pi / 2.0, size=(N_SAMPLES, 3))


def build_gate_superop(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """16×16 real superoperator for U = exp(i*(alpha*XX+beta*YY+gamma*ZZ))."""
    XX = np.kron(_X, _X)
    YY = np.kron(_Y, _Y)
    ZZ = np.kron(_Z, _Z)
    U  = expm(1j * (alpha * XX + beta * YY + gamma * ZZ))
    return _superop_2q(U)


def build_channel(alpha: float, beta: float, gamma: float, p: float) -> np.ndarray:
    """16×16 channel: depol_2q(p) ∘ superop(U)."""
    return _depol_2q(p) @ build_gate_superop(alpha, beta, gamma)


# ── optimiser ─────────────────────────────────────────────────────────────────

def _build_S(params: np.ndarray, d_ext_single: int) -> np.ndarray:
    n_free = d_ext_single - N_FIXED_COLS
    free   = _project_columns_bloch(params.reshape(N_S_ROWS, n_free))
    return np.hstack([_FIXED_COLS, free])


def _ixyz_xy_init(d_ext_single: int, a: float = 0.84) -> np.ndarray:
    b = a / np.sqrt(2)
    # 7 structured directions: X, Y, Z, (X+Y)/√2, (X-Y)/√2, (X+Z)/√2, (Y+Z)/√2
    base = np.array([
        [0., 0., 0., 0.,  0.,  0.,  0.],
        [1., 0., 0., b,   b,   b,   0.],
        [0., 1., 0., b,  -b,   0.,  b ],
        [0., 0., 1., 0.,  0.,  b,   b ],
    ])
    n_free = d_ext_single - N_FIXED_COLS
    free   = np.zeros((N_S_ROWS, n_free))
    k      = min(n_free, base.shape[1])
    free[:, :k] = base[:, :k]
    return free.ravel()


def _random_rotated_inits(d_ext_single: int, n_inits: int,
                          rng: np.random.Generator) -> list[np.ndarray]:
    """n_inits Haar-random SO(3)-rotated Pauli frames (I row = 0)."""
    n_free = d_ext_single - N_FIXED_COLS
    inits  = []
    for _ in range(n_inits):
        G = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(G)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        free         = np.zeros((N_S_ROWS, n_free))
        k_rot        = min(3, n_free)
        free[1:4, :k_rot] = Q[:, :k_rot]
        for j in range(3, n_free):
            v    = rng.standard_normal(3)
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v /= norm
            free[1:4, j] = v
        inits.append(_project_columns_bloch(free).ravel())
    return inits


def _is_current(out_path: str) -> bool:
    """True iff out_path exists and was produced by the current OPT_VERSION.

    Results written by an older version of the framability optimiser (or with
    no version stamp / no floor) are treated as stale so the optimisation is
    re-run.
    """
    if not os.path.exists(out_path):
        return False
    try:
        d = np.load(out_path, allow_pickle=True)
        return ('code_version' in d
                and str(d['code_version']) == OPT_VERSION
                and 'floor' in d)
    except Exception:
        return False


def make_objective(channel: np.ndarray, d_ext_single: int):
    def obj(params: np.ndarray) -> float:
        S = _build_S(params, d_ext_single)
        D = _kron_power(S, N_QUBITS)
        return float(_get_framability_fast(D, channel))
    return obj


def make_diag_constraint(d_ext_single: int):
    def g(params: np.ndarray) -> np.ndarray:
        S = _build_S(params, d_ext_single)
        return np.einsum('ij,ij->i', S, S) - 1.0
    return g


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',    type=int, required=True,
                        help=f'0..{N_D * N_P * N_SAMPLES - 1}')
    parser.add_argument('--out_dir',    type=str, default='results_depol_kron')
    parser.add_argument('--n_restarts', type=int, default=40)
    parser.add_argument('--max_iter',   type=int, default=500)
    parser.add_argument('--seed',       type=int, default=0)
    parser.add_argument('--method',     type=str, default='SLSQP')
    args = parser.parse_args()

    total = N_D * N_P * N_SAMPLES
    if not (0 <= args.task_id < total):
        print(f'ERROR: task_id out of range (0..{total - 1})', file=sys.stderr)
        sys.exit(1)

    sample_idx   = args.task_id // (N_D * N_P)
    rem          = args.task_id  %  (N_D * N_P)
    d_idx        = rem // N_P
    p_idx        = rem  %  N_P
    d_ext_single = D_EXT_SINGLES[d_idx]
    p            = P_VALUES[p_idx]

    angles              = _generate_angles()
    alpha, beta, gamma  = angles[sample_idx]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f'depol_kron_{d_ext_single}_{sample_idx:02d}_{p_idx:02d}.npz',
    )
    if _is_current(out_path):
        print(f'Skip: {out_path} already exists (version {OPT_VERSION})',
              flush=True)
        return
    elif os.path.exists(out_path):
        print(f'Rerun: {out_path} exists but was made with an older code '
              f'version (need {OPT_VERSION}) -> re-optimising', flush=True)

    print(f'[task {args.task_id}] d={d_ext_single}  sample={sample_idx}  '
          f'p={p:.3f}  alpha={alpha:.4f}  beta={beta:.4f}  gamma={gamma:.4f}',
          flush=True)

    channel    = build_channel(float(alpha), float(beta), float(gamma), float(p))
    objective  = make_objective(channel, d_ext_single)
    diag_g     = make_diag_constraint(d_ext_single)
    constraints = ({'type': 'ineq', 'fun': diag_g},)

    rng    = np.random.default_rng(args.seed + 1000 * args.task_id)
    inits  = [_ixyz_xy_init(d_ext_single)] + _random_rotated_inits(
        d_ext_single, args.n_restarts, rng)

    if args.method == 'SLSQP':
        opts = {'maxiter': args.max_iter, 'ftol': 1e-8}
    else:
        opts = {'maxiter': args.max_iter}

    best_val = np.inf
    best_x   = None
    t0       = time.perf_counter()

    for i, x0 in enumerate(inits):
        f0 = objective(x0)
        if np.all(diag_g(x0) >= -1e-8) and f0 < best_val:
            best_val, best_x = f0, x0.copy()
        res    = minimize(objective, x0, method=args.method,
                          constraints=constraints, options=opts)
        f_opt  = objective(res.x)
        if np.all(diag_g(res.x) >= -1e-8) and f_opt < best_val:
            best_val, best_x = f_opt, res.x.copy()
        print(f'  restart {i + 1}/{len(inits)}:  '
              f'f0={f0:.5f}  f_opt={f_opt:.5f}  best={best_val:.5f}', flush=True)

    elapsed = time.perf_counter() - t0
    S_opt   = _build_S(best_x, d_ext_single)
    D_opt   = _kron_power(S_opt, N_QUBITS)
    fra     = float(_get_framability_fast(D_opt, channel))
    floor   = spectral_floor(channel)   # framability floor = spectral radius

    np.savez(out_path,
             framability  = np.array(fra),
             floor        = np.array(floor),
             D            = D_opt,
             S            = S_opt,
             x            = best_x,
             alpha        = np.array(alpha),
             beta         = np.array(beta),
             gamma        = np.array(gamma),
             p            = np.array(p),
             d_ext_single = np.array(d_ext_single),
             sample_idx   = np.array(sample_idx),
             code_version = np.array(OPT_VERSION))
    print(f'[task {args.task_id}] saved {out_path}  '
          f'fra={fra:.6f}  floor={floor:.6f}  gap={fra - floor:.6f}  '
          f'elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
