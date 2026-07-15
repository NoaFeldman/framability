"""
Worker: heavily optimise the Heisenberg-picture framability of four fixed
2-qubit gates, each of the form

    channel = depol_2q(p) . superop( exp(i*(alpha*XX + beta*YY + gamma*ZZ)) )

The four target gates (GATES below) are

    g1_p0.00 : (alpha, beta, gamma) = (sqrt(0.5), exp(-1), pi),  p = 0.00
    g1_p0.08 : (alpha, beta, gamma) = (sqrt(0.5), exp(-1), pi),  p = 0.08
    g2_p0.00 : (alpha, beta, gamma) = (0.3, 0.3, 0.0),           p = 0.00
    g2_p0.08 : (alpha, beta, gamma) = (0.3, 0.3, 0.0),           p = 0.08

For every gate the framability is minimised for d_ext_single in [4, 6, 8]
(full frame d_ext = d_ext_single ** 2).  Because we want the *best possible*
framability, the search per (gate, d) cell is spread over N_BATCHES
independent random-seed batches (parallel array tasks); each batch runs a
large block of restarts and the collect step keeps the global minimum.

Each batch task performs three complementary searches and keeps the best:
  1. auto/complex frame search  (minimize_framability, use_complex=auto)
     -- complex frames are admissible (and can be strictly better) once
        d_ext >= 2 * pauli_string_dim, i.e. for d_ext_single in {6, 8}.
  2. real-only frame search     (minimize_framability, use_complex=False)
  3. analytic Polyak floor-polish of the best real candidate
     (polyak_floor_polish) -- drives the last-mile gap to the spectral floor
     using the LP-dual subgradient (real 2-qubit gate: exact).

Task layout
-----------
    task_id = gate_idx * (N_D * N_BATCHES) + d_idx * N_BATCHES + batch_idx
      gate_idx  in 0..N_GATES-1   (N_GATES  = 4)
      d_idx     in 0..N_D-1       (D_EXT_SINGLES = [4, 6, 8])
      batch_idx in 0..N_BATCHES-1

Total tasks: N_GATES * N_D * N_BATCHES.

Output: <out_dir>/<label>_d<d>_b<batch:02d>.npz
  keys: framability, floor, gap, D, S (real cases only, else empty), x,
        use_complex, stage, alpha, beta, gamma, p, d_ext_single, gate_label,
        batch_idx, code_version
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimize_framability import (
    N_FIXED_COLS,
    OPT_VERSION,
    _FIXED_COLS,
    _get_framability_fast,
    _kron_power,
    _project_columns_bloch,
    minimize_framability,
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

def _real_x_to_S(x: np.ndarray, d_ext_single: int) -> np.ndarray:
    """Decode a real (use_complex=False) parameter vector into S (4 x d)."""
    n_free = d_ext_single - N_FIXED_COLS
    free   = _project_columns_bloch(np.asarray(x, dtype=float).reshape(N_S_ROWS,
                                                                       n_free))
    return np.hstack([_FIXED_COLS, free])


def _framability_of(D: np.ndarray, channel: np.ndarray) -> float:
    return float(_get_framability_fast(D, channel))


# ── staleness check (re-run only when out of date) ────────────────────────────

def _is_current(out_path: str) -> bool:
    if not os.path.exists(out_path):
        return False
    try:
        d = np.load(out_path, allow_pickle=True)
        return ('code_version' in d
                and str(d['code_version']) == OPT_VERSION
                and 'floor' in d)
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
                        help='random restarts per search (complex and real each)')
    parser.add_argument('--max_iter',   type=int, default=2000)
    parser.add_argument('--maxfev',     type=int, default=12000)
    parser.add_argument('--polish_iter', type=int, default=4000,
                        help='Polyak floor-polish subgradient steps')
    parser.add_argument('--method',     type=str, default='Nelder-Mead')
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

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir, f'{label}_d{d_ext_single}_b{batch_idx:02d}.npz')

    if _is_current(out_path):
        print(f'Skip: {out_path} already at version {OPT_VERSION}', flush=True)
        return

    # Distinct, reproducible seed per (task, batch); large multiplier avoids
    # overlap between the two searches launched inside this task.
    base_seed = args.seed + 1_000_003 * args.task_id
    print(f'[task {args.task_id}] {label}  d={d_ext_single}  batch={batch_idx}  '
          f'alpha={alpha:.6f} beta={beta:.6f} gamma={gamma:.6f} p={p:.3f}',
          flush=True)

    channel = build_channel(alpha, beta, gamma, p)
    floor   = spectral_floor(channel)

    t0 = time.perf_counter()

    best = {'f': np.inf, 'D': None, 'S': None, 'x': None,
            'use_complex': False, 'stage': 'none'}

    def _offer(f, D, S, x, use_complex, stage):
        if np.isfinite(f) and f < best['f']:
            best.update(f=float(f), D=np.asarray(D), S=S, x=np.asarray(x),
                        use_complex=bool(use_complex), stage=stage)

    # ── 1. auto/complex frame search ─────────────────────────────────────────
    D_c, f_c, x_c = minimize_framability(
        channel, d_ext_single, n_restarts=args.n_restarts,
        method=args.method, max_iter=args.max_iter, maxfev=args.maxfev,
        tol=args.tol, seed=base_seed, verbose=False, return_x=True)
    # auto-select of use_complex mirrors minimize_framability's own rule.
    d_ext = d_ext_single ** N_QUBITS
    use_complex_auto = (d_ext >= 2 * 16)
    _offer(f_c, D_c, None, x_c, use_complex_auto, 'complex_search')
    print(f'  [1] complex/auto search: f={f_c:.8f}', flush=True)

    # ── 2. real-only frame search ────────────────────────────────────────────
    D_r, f_r, x_r = minimize_framability(
        channel, d_ext_single, n_restarts=args.n_restarts,
        method=args.method, max_iter=args.max_iter, maxfev=args.maxfev,
        tol=args.tol, seed=base_seed + 7, verbose=False, return_x=True,
        use_complex=False)
    S_r = _real_x_to_S(x_r, d_ext_single)
    _offer(f_r, D_r, S_r, x_r, False, 'real_search')
    print(f'  [2] real search:         f={f_r:.8f}', flush=True)

    # ── 3. analytic Polyak floor-polish of the best real candidate ───────────
    # Seed the polish from whichever real frame is currently best (the real
    # search result, and -- if it happens to be real -- the complex-search one).
    seeds_S = [S_r]
    f_pol_best, S_pol_best = np.inf, None
    for S_seed in seeds_S:
        f_pol, S_pol = polyak_floor_polish(
            S_seed, channel, target=floor, n_iter=args.polish_iter,
            tol=1e-12, verbose=False)
        if f_pol < f_pol_best:
            f_pol_best, S_pol_best = f_pol, S_pol
    if S_pol_best is not None:
        D_pol = _kron_power(S_pol_best, N_QUBITS)
        f_pol_chk = _framability_of(D_pol, channel)
        x_pol = S_pol_best[:, N_FIXED_COLS:].ravel()
        _offer(f_pol_chk, D_pol, S_pol_best, x_pol, False, 'real_polish')
        print(f'  [3] Polyak floor-polish: f={f_pol_chk:.8f}', flush=True)

    elapsed = time.perf_counter() - t0
    fra     = best['f']
    S_save  = best['S'] if best['S'] is not None else np.zeros((0, 0))

    np.savez(
        out_path,
        framability  = np.array(fra),
        floor        = np.array(floor),
        gap          = np.array(fra - floor),
        D            = np.asarray(best['D']),
        S            = np.asarray(S_save),
        x            = np.asarray(best['x']),
        use_complex  = np.array(best['use_complex']),
        stage        = np.array(best['stage']),
        alpha        = np.array(alpha),
        beta         = np.array(beta),
        gamma        = np.array(gamma),
        p            = np.array(p),
        d_ext_single = np.array(d_ext_single),
        gate_label   = np.array(label),
        batch_idx    = np.array(batch_idx),
        code_version = np.array(OPT_VERSION),
    )
    print(f'[task {args.task_id}] saved {out_path}  fra={fra:.8f}  '
          f'floor={floor:.8f}  gap={fra - floor:.8f}  winner={best["stage"]}  '
          f'elapsed={elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
