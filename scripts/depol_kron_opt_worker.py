"""
Worker: heavily optimise the Heisenberg-picture framability of four fixed
2-qubit gates, each of the form

    channel = depol_2q(p) . superop( exp(i*(alpha*XX + beta*YY + gamma*ZZ)) )

and -- crucially -- record EVERY restart's converged frame, not only the
best one, so the collect step can select a *robustly reachable* optimum
(the frame reached by the most restarts among the near-optimal ones) rather
than a fragile knife-edge minimum.

The four target gates (GATES below) are

    g1_p0.00 : (alpha, beta, gamma) = (sqrt(0.5), exp(-1), pi),  p = 0.00
    g1_p0.08 : (alpha, beta, gamma) = (sqrt(0.5), exp(-1), pi),  p = 0.08
    g2_p0.00 : (alpha, beta, gamma) = (0.3, 0.3, 0.0),           p = 0.00
    g2_p0.08 : (alpha, beta, gamma) = (0.3, 0.3, 0.0),           p = 0.08

For every gate the framability is minimised for d_ext_single in [4, 6, 8]
(full frame d_ext = d_ext_single ** 2).  Each (gate, d) cell is spread over
N_BATCHES independent random-seed batches (parallel array tasks); the collect
step pools every restart across all batches, clusters them into basins, and
keeps the widest near-optimal basin.

Each batch task runs, and records the converged frame of, every restart of:
  * real-only frame search      (use_complex=False)
  * auto/complex frame search    (use_complex auto; admissible for d>=6)
  * analytic Polyak floor-polish of the best few real frames
The full per-restart pool (framability + single-qubit frame S) is saved so the
collector can measure reachability.

Task layout
-----------
    task_id = gate_idx * (N_D * N_BATCHES) + d_idx * N_BATCHES + batch_idx
      gate_idx  in 0..N_GATES-1   (N_GATES  = 4)
      d_idx     in 0..N_D-1       (D_EXT_SINGLES = [4, 6, 8])
      batch_idx in 0..N_BATCHES-1

Total tasks: N_GATES * N_D * N_BATCHES.

Output: <out_dir>/<label>_d<d>_b<batch:02d>.npz
  Per-restart pool (for robust-basin selection):
    pool_fra    (R,)         framability of each restart's converged frame
    pool_S      (R, 4, d)    single-qubit frame S (complex; real cases imag=0)
    pool_stage  (R,)         'real' | 'complex' | 'polish'
  Best-of-batch (convenience):
    framability, floor, gap, D, S, use_complex, stage
  Metadata: alpha, beta, gamma, p, d_ext_single, gate_label, batch_idx,
            code_version
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimize_framability import (
    N_FIXED_COLS,
    OPT_VERSION,
    _FIXED_COLS,
    _build_inits,
    _get_framability_fast,
    _kron_power,
    _project_columns_bloch,
    polyak_floor_polish,
    spectral_floor,
)
from sweep_depol_gates_worker import _depol_2q, _superop_2q

# ── target gates ──────────────────────────────────────────────────────────────
# (label, (alpha, beta, gamma), p)
GATES = [
    ('g1_p0.00', (float(np.sqrt(0.5)), float(np.exp(-1.0)), float(np.pi)), 0.00),
    ('g1_p0.08', (float(np.sqrt(0.5)), float(np.exp(-1.0)), float(np.pi)), 0.08),
    ('g2_p0.00', (0.3, 0.3, 0.0), 0.00),
    ('g2_p0.08', (0.3, 0.3, 0.0), 0.08),
]
D_EXT_SINGLES = [4, 6, 8]
N_BATCHES     = 16

N_GATES  = len(GATES)
N_D      = len(D_EXT_SINGLES)
N_S_ROWS = 4     # single-qubit d^2
N_QUBITS = 2
PAULI_STRING_DIM = 16
_PENALTY_BASE = 1e3

# ── Pauli helpers ─────────────────────────────────────────────────────────────
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def build_gate_superop(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """16x16 real superoperator for U = exp(i*(alpha*XX + beta*YY + gamma*ZZ))."""
    XX = np.kron(_X, _X)
    YY = np.kron(_Y, _Y)
    ZZ = np.kron(_Z, _Z)
    U  = expm(1j * (alpha * XX + beta * YY + gamma * ZZ))
    return _superop_2q(U)


def build_channel(alpha: float, beta: float, gamma: float, p: float) -> np.ndarray:
    """16x16 real channel: depol_2q(p) . superop(U)."""
    return _depol_2q(p) @ build_gate_superop(alpha, beta, gamma)


# ── frame decoding ────────────────────────────────────────────────────────────

def _x_to_S(x: np.ndarray, d_ext_single: int, use_complex: bool) -> np.ndarray:
    """Decode a flat parameter vector into the single-qubit frame S (4 x d).

    Mirrors optimize_framability._params_to_D but returns S (complex dtype so
    real and complex frames can share one storage array; real frames get a
    zero imaginary part)."""
    n_free = d_ext_single - N_FIXED_COLS
    half   = N_S_ROWS * n_free
    x = np.asarray(x, dtype=float)
    if use_complex:
        free = _project_columns_bloch(
            x[:half].reshape(N_S_ROWS, n_free)
            + 1j * x[half:].reshape(N_S_ROWS, n_free))
    else:
        free = _project_columns_bloch(x[:half].reshape(N_S_ROWS, n_free))
    return np.hstack([_FIXED_COLS.astype(complex), free.astype(complex)])


def _framability_of_S(S: np.ndarray, channel: np.ndarray) -> float:
    return float(_get_framability_fast(_kron_power(S, N_QUBITS), channel))


def _make_objective(channel, d_ext_single, use_complex):
    """framability objective with the smooth rank-deficiency barrier."""
    n_free = d_ext_single - N_FIXED_COLS
    half   = N_S_ROWS * n_free

    def obj(params):
        params = np.asarray(params, dtype=float)
        if use_complex:
            free = _project_columns_bloch(
                params[:half].reshape(N_S_ROWS, n_free)
                + 1j * params[half:].reshape(N_S_ROWS, n_free))
            S = np.hstack([_FIXED_COLS.astype(complex), free])
        else:
            free = _project_columns_bloch(params[:half].reshape(N_S_ROWS, n_free))
            S = np.hstack([_FIXED_COLS, free]).astype(float)
        D = _kron_power(S, N_QUBITS)
        f = _get_framability_fast(D, channel)
        if np.isfinite(f):
            return f
        sigma_min = float(np.linalg.svd(D, compute_uv=False)[-1])
        return _PENALTY_BASE * (1.0 + 1.0 / max(sigma_min, 1e-12))

    return obj


def _run_restart_pool(channel, d_ext_single, use_complex, n_restarts,
                      max_iter, maxfev, tol, rng):
    """Run n_restarts local minimisations, recording EVERY converged frame.

    Returns (fra_list, S_list) with S_list a list of (4 x d) complex frames.
    Only finite-framability (feasible) results are kept.
    """
    d_ext = d_ext_single ** N_QUBITS
    inits = _build_inits(N_S_ROWS, d_ext_single, d_ext, n_restarts, rng,
                         use_complex=use_complex)
    obj   = _make_objective(channel, d_ext_single, use_complex)
    opts  = {'maxiter': max_iter, 'maxfev': maxfev, 'fatol': tol, 'xatol': tol}

    fra_list, S_list = [], []
    for x0 in inits:
        res = minimize(obj, x0, method='Nelder-Mead', options=opts)
        S   = _x_to_S(res.x, d_ext_single, use_complex)
        f   = _framability_of_S(S, channel)
        if np.isfinite(f):
            fra_list.append(f)
            S_list.append(S)
    return fra_list, S_list


# ── staleness check (re-run only when out of date) ────────────────────────────

def _is_current(out_path: str) -> bool:
    if not os.path.exists(out_path):
        return False
    try:
        d = np.load(out_path, allow_pickle=True)
        return ('code_version' in d
                and str(d['code_version']) == OPT_VERSION
                and 'pool_fra' in d)
    except Exception:
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    total = N_GATES * N_D * N_BATCHES
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',    type=int, required=True,
                        help=f'0..{total - 1}')
    parser.add_argument('--out_dir',    type=str, default='results_depol_kron_opt')
    parser.add_argument('--n_restarts', type=int, default=60,
                        help='random restarts per search (real and complex each)')
    parser.add_argument('--max_iter',   type=int, default=2000)
    parser.add_argument('--maxfev',     type=int, default=12000)
    parser.add_argument('--polish_iter', type=int, default=4000,
                        help='Polyak floor-polish subgradient steps')
    parser.add_argument('--n_polish',   type=int, default=3,
                        help='how many of the best real frames to floor-polish')
    parser.add_argument('--tol',        type=float, default=1e-9)
    parser.add_argument('--seed',       type=int, default=0,
                        help='base seed; the per-task seed also folds in task_id')
    args = parser.parse_args()

    if not (0 <= args.task_id < total):
        print(f'ERROR: task_id out of range (0..{total - 1})', file=sys.stderr)
        sys.exit(1)

    gate_idx  = args.task_id // (N_D * N_BATCHES)
    rem       = args.task_id  %  (N_D * N_BATCHES)
    d_idx     = rem // N_BATCHES
    batch_idx = rem  %  N_BATCHES

    label, (alpha, beta, gamma), p = GATES[gate_idx]
    d_ext_single = D_EXT_SINGLES[d_idx]
    d_ext        = d_ext_single ** N_QUBITS
    use_complex_auto = (d_ext >= 2 * PAULI_STRING_DIM)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f'{label}_d{d_ext_single}_b{batch_idx:02d}.npz')

    if _is_current(out_path):
        print(f'Skip: {out_path} already at version {OPT_VERSION}', flush=True)
        return

    base_seed = args.seed + 1_000_003 * args.task_id
    print(f'[task {args.task_id}] {label}  d={d_ext_single}  batch={batch_idx}  '
          f'alpha={alpha:.6f} beta={beta:.6f} gamma={gamma:.6f} p={p:.3f}  '
          f'(complex={"auto-on" if use_complex_auto else "off"})', flush=True)

    channel = build_channel(alpha, beta, gamma, p)
    floor   = spectral_floor(channel)

    t0 = time.perf_counter()
    pool_fra, pool_S, pool_stage = [], [], []

    # ── real-only restart pool ───────────────────────────────────────────────
    rng_real = np.random.default_rng(base_seed)
    fra_r, S_r = _run_restart_pool(
        channel, d_ext_single, False, args.n_restarts,
        args.max_iter, args.maxfev, args.tol, rng_real)
    pool_fra += fra_r; pool_S += S_r; pool_stage += ['real'] * len(fra_r)
    print(f'  [real]    {len(fra_r)} feasible restarts  '
          f'min={min(fra_r) if fra_r else np.nan:.8f}', flush=True)

    # ── auto/complex restart pool (only when it adds freedom, d>=6) ──────────
    if use_complex_auto:
        rng_cplx = np.random.default_rng(base_seed + 7)
        fra_c, S_c = _run_restart_pool(
            channel, d_ext_single, True, args.n_restarts,
            args.max_iter, args.maxfev, args.tol, rng_cplx)
        pool_fra += fra_c; pool_S += S_c; pool_stage += ['complex'] * len(fra_c)
        print(f'  [complex] {len(fra_c)} feasible restarts  '
              f'min={min(fra_c) if fra_c else np.nan:.8f}', flush=True)

    # ── analytic Polyak floor-polish of the best few REAL frames ─────────────
    if fra_r and args.n_polish > 0:
        order = np.argsort(fra_r)[:args.n_polish]
        for j in order:
            S_seed = S_r[j].real            # polish operates on real frames
            f_pol, S_pol = polyak_floor_polish(
                S_seed, channel, target=floor, n_iter=args.polish_iter,
                tol=1e-12, verbose=False)
            S_pol_c = S_pol.astype(complex)
            f_chk   = _framability_of_S(S_pol_c, channel)
            if np.isfinite(f_chk):
                pool_fra.append(f_chk); pool_S.append(S_pol_c)
                pool_stage.append('polish')
        print(f'  [polish]  {args.n_polish} floor-polishes  '
              f'min={min(pool_fra):.8f}', flush=True)

    if not pool_fra:
        print(f'[task {args.task_id}] WARNING: no feasible restart -- '
              f'writing empty pool', flush=True)

    pool_fra_arr   = np.array(pool_fra, dtype=float)
    pool_S_arr     = (np.stack(pool_S, axis=0) if pool_S
                      else np.zeros((0, N_S_ROWS, d_ext_single), dtype=complex))
    pool_stage_arr = np.array(pool_stage, dtype=object)

    # best-of-batch convenience fields
    if pool_fra:
        b_i   = int(np.argmin(pool_fra_arr))
        S_best = pool_S_arr[b_i]
        real_best = np.max(np.abs(S_best.imag)) < 1e-12
        best_fra = float(pool_fra_arr[b_i])
        best_D   = _kron_power(S_best, N_QUBITS)
        best_stage = str(pool_stage_arr[b_i])
    else:
        S_best = np.zeros((N_S_ROWS, d_ext_single), dtype=complex)
        real_best = True
        best_fra = np.inf
        best_D   = np.zeros((PAULI_STRING_DIM, d_ext), dtype=complex)
        best_stage = 'none'

    elapsed = time.perf_counter() - t0
    np.savez(
        out_path,
        pool_fra     = pool_fra_arr,
        pool_S       = pool_S_arr,
        pool_stage   = pool_stage_arr.astype(str),
        framability  = np.array(best_fra),
        floor        = np.array(floor),
        gap          = np.array(best_fra - floor),
        D            = best_D,
        S            = S_best,
        use_complex  = np.array(not real_best),
        stage        = np.array(best_stage),
        alpha        = np.array(alpha),
        beta         = np.array(beta),
        gamma        = np.array(gamma),
        p            = np.array(p),
        d_ext_single = np.array(d_ext_single),
        gate_label   = np.array(label),
        batch_idx    = np.array(batch_idx),
        code_version = np.array(OPT_VERSION),
    )
    print(f'[task {args.task_id}] saved {out_path}  '
          f'n_pool={len(pool_fra)}  best={best_fra:.8f}  floor={floor:.8f}  '
          f'gap={best_fra - floor:.8f}  winner={best_stage}  '
          f'elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
