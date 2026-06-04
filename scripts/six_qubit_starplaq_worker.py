"""
Per-task worker for the 6-qubit star+plaquette Lindbladian Trotter scan.

Lindbladian:  L = -i[H, .] + D[L_s] + D[L_p]
    H   = h (X_u + X_r) + lam (Z_u + Z_r)         (on u, r only)
    L_s = sqrt(gamma_s) X_u X_r X_d X_l           (star)
    L_p = sqrt(gamma_p) Z_u Z_r Z_ur Z_ru         (plaquette)

Grid:
    h = lam = 1
    gamma_s, gamma_p in [GAMMA_STEP * i for i in range(N_GRID)],
        with GAMMA_STEP = 0.4, N_GRID = 10  (so values 0.0 .. 3.6)
    dt = GAMMA_STEP / 10 = 0.04
    Total: 10*10 = 100 grid points

Parallelisation: 10 jobs.  Each job processes 10 grid points.
    task_id in 0..9
    For task t, the grid indices it owns are start..start+9 where
    start = t * 40 (linear index = ig * N_GRID + igp).

Per-point output: <out_dir>/starplaq_<ig:03d>_<igp:03d>.npz with keys:
    ss_vn_entropy
    neg_urdl_urru, neg_dl_urur            (negativities on the two partitions)
    lpdo_urdl_urru, lpdo_dl_urur          (LPDO bond entropies on the two)
    decay_rate
    otoc_small, otoc_large
    channel_stab_purity
    pauli_fra
    opt_fra_4, opt_fra_6                 (NaN unless --do_fra_opt)
    gamma_s, gamma_p, h, lam, dt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix, eye as sp_eye, hstack as sp_hstack, vstack as sp_vstack
from scipy.sparse.linalg import eigs, expm_multiply
from scipy.optimize import linprog, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from two_d_lindbladian import (
    six_qubit_lindbladian, N_QUBITS_6Q, DIM_6Q, _index_to_string_n,
    four_qubit_lindbladian_star, four_qubit_lindbladian_plaquette,
    I_, X_, Y_, Z_,
)
from lpdo import purification_sqrt, _bond_entropy

# scipy internals for the fast per-column LP path (option 1)
from scipy.optimize._linprog_highs import _linprog_highs
from scipy.optimize._linprog_util import _LPProblem, _clean_inputs

# highspy: build the LP model once and warm-start across columns (~10x faster
# than rebuilding the scipy model per column).  Optional; falls back to the
# scipy per-column path if not installed.
try:
    import highspy
    _HAS_HIGHSPY = True
except ImportError:
    _HAS_HIGHSPY = False


# ── grid ─────────────────────────────────────────────────────────────────────
GAMMA_STEP = 0.4
N_GRID     = int(round(4.0 / GAMMA_STEP))           # 10
DT         = GAMMA_STEP / 10.0                      # 0.04
N_TOTAL    = N_GRID * N_GRID                        # 100

# qubit ordering [u, r, d, l, ur, ru] in the 6-qubit string
_QU, _QR, _QD, _QL, _QUR, _QRU = 0, 1, 2, 3, 4, 5

# Pauli matrices (computational basis)
_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_MATS = [_I2, _SX, _SY, _SZ]


# ── helpers ──────────────────────────────────────────────────────────────────
def _kron_single_pauli(string):
    """6-qubit Pauli operator (64x64) from a length-6 tuple of indices."""
    P = _PAULI_MATS[string[0]]
    for ai in string[1:]:
        P = np.kron(P, _PAULI_MATS[ai])
    return P


def _string_to_index(s):
    idx = 0
    for ai in s:
        idx = idx * 4 + ai
    return idx


def _pauli_vec_to_rho(c, n_qubits=N_QUBITS_6Q):
    """Pauli-basis coefficient vector -> density matrix (dim 2^n)."""
    d = 2 ** n_qubits
    rho = np.zeros((d, d), dtype=complex)
    for idx in range(len(c)):
        ci = c[idx]
        if abs(ci) < 1e-15:
            continue
        rho = rho + ci * _kron_single_pauli(_index_to_string_n(idx, n_qubits))
    return (rho + rho.conj().T) / 2.0


def _permute_qubits(rho, n_qubits, perm):
    """Permute qubits in a density matrix according to perm."""
    d = 2 ** n_qubits
    rho_t = rho.reshape((2,) * (2 * n_qubits))
    new_axes = list(perm) + [n_qubits + p for p in perm]
    return rho_t.transpose(new_axes).reshape(d, d)


def _initial_zero_state_pauli_vec():
    """Pauli-basis coefficient vector of |0>^6 <0|.

    |0><0| = (I + Z)/2 per qubit, so the 6-qubit product has coefficient
    1/2^6 = 1/d for every Pauli string composed only of {I, Z} and 0 else.
    (I has Pauli index 0, Z has index 3.)
    """
    d = 2 ** N_QUBITS_6Q  # 64
    c0 = np.zeros(DIM_6Q, dtype=float)
    for idx in range(DIM_6Q):
        s = _index_to_string_n(idx, N_QUBITS_6Q)
        if all(p in (0, 3) for p in s):
            c0[idx] = 1.0 / d
    return c0


# Cached |0>^6 initial state (independent of L).
_C0_ZERO_STATE = None


# ── steady state and spectral quantities ────────────────────────────────────
def _steady_state_and_decay(L):
    """Return (c_ss, decay_rate).

    The Lindbladian of two Hermitian Pauli-string jumps (XXXX, ZZZZ) has a
    HIGHLY degenerate null space (~128-dim), so "the" steady state is not
    unique -- it depends on the initial condition.  We therefore define the
    steady state as the t->infinity limit of the physical relaxation of
    |0>^6 <0|, computed via expm_multiply (projection onto null(L) along the
    dynamics).  This is well-defined and consistent with the OTOC initial
    state.

    decay_rate: magnitude of the slowest non-zero eigenvalue of L, used to
    pick an evolution time long enough to converge.
    """
    global _C0_ZERO_STATE
    L_sp = csc_matrix(L.astype(float))

    # --- decay rate: spectral gap (slowest non-zero relaxation mode) ---
    # The null space is ~128-dim, so a shift-invert near 0 returns mostly the
    # zero eigenvalues.  Request k well above the null dimension and take the
    # eigenvalue with the smallest non-zero |.| as the gap.
    decay = float('nan')
    try:
        evals, _ = eigs(L_sp.astype(complex), k=160, sigma=-0.01, which='LM',
                        tol=1e-8, maxiter=10000)
        for e in sorted(np.abs(evals)):
            if e > 1e-4:
                decay = float(e)
                break
    except Exception:
        pass

    # --- steady state: relax |0>^6 to t -> infinity ---
    if _C0_ZERO_STATE is None:
        _C0_ZERO_STATE = _initial_zero_state_pauli_vec()
    c0 = _C0_ZERO_STATE

    # Evolve long enough that the slowest non-zero mode is suppressed.
    # e^{-decay * t} < ~1e-9 needs t ~ 20/decay; clamp to [50, 4000].
    if np.isfinite(decay) and decay > 1e-6:
        t_evolve = float(np.clip(40.0 / decay, 50.0, 4000.0))
    else:
        t_evolve = 400.0

    c_ss = expm_multiply(L_sp * t_evolve, c0).real

    # Validate convergence; if not converged, evolve further once.
    residual = float(np.max(np.abs(L_sp @ c_ss)))
    if residual > 1e-6:
        c_ss = expm_multiply(L_sp * (4.0 * t_evolve), c0).real
        residual = float(np.max(np.abs(L_sp @ c_ss)))
        if residual > 1e-4:
            print(f'  [ss] WARNING: not converged, residual={residual:.2e} '
                  f'(t_evolve={4.0 * t_evolve:.0f})', flush=True)

    return c_ss, decay


# ── entanglement quantities ──────────────────────────────────────────────────
def _vn_entropy(rho):
    rho_h = (rho + rho.conj().T) / 2.0
    e = np.linalg.eigvalsh(rho_h)
    e = e[e > 1e-15]
    return float(-np.sum(e * np.log(e)))


def _negativity_bipartition(rho, d_A):
    """Negativity for an (A, B) bipartition with |A| = d_A, |B| = rho.shape[0]/d_A.

    Assumes rho is already ordered so the A subsystem occupies the first d_A
    indices and B the next d_B indices on each side.
    """
    d = rho.shape[0]
    d_B = d // d_A
    if d_A * d_B != d:
        raise ValueError(f'd_A={d_A} does not divide d={d}.')
    rho_pt = (rho.reshape(d_A, d_B, d_A, d_B)
                  .transpose(0, 3, 2, 1)
                  .reshape(d, d))
    rho_pt = (rho_pt + rho_pt.conj().T) / 2.0
    e = np.linalg.eigvalsh(rho_pt)
    return float(np.sum(np.abs(e[e < -1e-15])))


def _lpdo_bond_entropy(rho, d_A):
    """LPDO-style bond entropy across an (A|B) bipartition with |A|=d_A.

    rho must already be permuted so A occupies the first d_A indices on each
    side.  Steps:
        1. rho = X X^dag via purification_sqrt
        2. reshape X (d, d) -> (d_A, d_B, d_A, d_B) treating (s_A, s_B) as
           rows and (a_A, a_B) as ancilla columns
        3. transpose to (s_A, a_A, s_B, a_B), regroup to (d_A^2, d_B^2)
        4. SVD -> bond entropy
    """
    d = rho.shape[0]
    d_B = d // d_A
    if d_A * d_B != d:
        raise ValueError(f'd_A={d_A} does not divide d={d}.')
    X = purification_sqrt(rho)
    T = X.reshape(d_A, d_B, d_A, d_B).transpose(0, 2, 1, 3)
    M = T.reshape(d_A * d_A, d_B * d_B)
    S = np.linalg.svd(M, compute_uv=False)
    return float(_bond_entropy(S))


# ── OTOC ─────────────────────────────────────────────────────────────────────
def _otoc(L, t, V_qubit=_QU, W_qubit=_QR):
    """OTOC at time t with V = X_{V_qubit}, W = X_{W_qubit}, psi_0 = |0>^6.

        OTOC(t) = <0| W(t)^† V^† W(t) V |0>

    Uses expm_multiply on the sparse L to avoid the O(n^3) dense expm.
    E_t^† c = e^{L^† t} c, so we apply expm_multiply(L^† * t, c).
    """
    if t <= 0:
        return float('nan')
    L_sp = csc_matrix(L.astype(complex))

    w_str = [I_] * N_QUBITS_6Q
    w_str[W_qubit] = X_
    c_w0 = np.zeros(DIM_6Q, dtype=float)
    c_w0[_string_to_index(w_str)] = 1.0

    # (E_t)^† c_w0 = e^{L^† t} c_w0
    c_wt = expm_multiply(L_sp.conj().T * t, c_w0).real
    W_t = _pauli_vec_to_rho(c_wt)        # Hermitian linear combo of Paulis

    v_str = [I_] * N_QUBITS_6Q
    v_str[V_qubit] = X_
    V0 = _kron_single_pauli(v_str)

    psi0 = np.zeros(2 ** N_QUBITS_6Q, dtype=complex)
    psi0[0] = 1.0

    op = W_t.conj().T @ V0.conj().T @ W_t @ V0
    return float(np.real(np.vdot(psi0, op @ psi0)))


# ── channel stabilizer purity ────────────────────────────────────────────────
def _channel_stabilizer_purity(gate):
    """log2( d^2 * sum_i E_ii^2 / (d+1) ) for E = gate (4096x4096), d = 64."""
    diag = np.diag(gate).real
    d = 2 ** N_QUBITS_6Q
    total = (d ** 2) * float(np.sum(diag ** 2))
    return float(np.log2(total / (d + 1)))


# ── framability (Pauli + optional optimised) ────────────────────────────────
def _pauli_framability(gate):
    """Row 1-norm of the gate in Pauli basis."""
    return float(np.max(np.sum(np.abs(gate), axis=1)))


def _highspy_per_column_framability(D, gate):
    """Per-column L1-minimisation framability using a single reused HiGHS model.

    Solves, for every column j of Y = gate^T D,
        min ||u||_1  s.t.  D u = Y[:, j]
    via the standard split  min sum(t)  s.t.  D u = y,  u - t <= 0,  -u - t <= 0.

    The constraint matrix is fixed across all columns, so the HiGHS model is
    built once and only the equality-row bounds (the RHS y) change per column.
    HiGHS warm-starts the simplex from the previous optimal basis, giving an
    ~10x speedup over rebuilding the scipy model each call.
    """
    D = np.asarray(D, dtype=float)
    n, d_ext = D.shape
    if np.max(np.abs(gate.imag)) > 1e-12:
        raise ValueError('gate must be real for the LP framability.')
    Y = (gate.real.T @ D)

    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('presolve', 'off')          # warm start is incompatible w/ presolve
    h.setOptionValue('solver', 'simplex')

    nv = 2 * d_ext  # [u (free), t (>=0)]
    cost = np.concatenate([np.zeros(d_ext), np.ones(d_ext)])
    lb = np.concatenate([-inf * np.ones(d_ext), np.zeros(d_ext)])
    ub = inf * np.ones(nv)
    h.addVars(nv, lb, ub)
    h.changeColsCost(nv, np.arange(nv, dtype=np.int32), cost)

    # Rows: n equality (D u = y) + 2*d_ext inequality (|u| <= t).
    I_de = sp_eye(d_ext, format='csr')
    A_eq = sp_hstack([csc_matrix(D), csc_matrix((n, d_ext))]).tocsr()
    A_ub = sp_vstack([sp_hstack([I_de, -I_de]),
                      sp_hstack([-I_de, -I_de])]).tocsr()
    A = sp_vstack([A_eq, A_ub]).tocsr()
    row_lb = np.concatenate([np.zeros(n), -inf * np.ones(2 * d_ext)])
    row_ub = np.concatenate([np.zeros(n), np.zeros(2 * d_ext)])
    h.addRows(A.shape[0], row_lb, row_ub, A.nnz,
              A.indptr.astype(np.int32), A.indices.astype(np.int32), A.data)

    eq_idx = np.arange(n, dtype=np.int32)
    best = 0.0
    for j in range(d_ext):
        y = Y[:, j]
        h.changeRowsBounds(n, eq_idx, y, y)
        h.run()
        x = np.asarray(h.getSolution().col_value[:d_ext])
        best = max(best, float(np.sum(np.abs(x))))
    return best


def _fast_per_column_framability(D, gate):
    """Per-column L1-minimisation framability.

    Dispatches to the reused-model HiGHS path when highspy is available
    (~10x faster), otherwise the scipy per-column fallback below.

    For each column j of Y = gate^T D, solve
        min ||u||_1  s.t.  D u = Y[:, j]
    by encoding u = u_pos - u_neg with the auxiliary t = (u_pos + u_neg) ≥ |u|
    via inequality constraints, and minimising sum(t).  Variables are
    u_pos (signed real) and t (≥0) for a total of 2*d_ext per LP.

    Speedup vs. naive scipy.linprog:
      - Build and _clean_inputs the LP problem ONCE.
      - Use sparse A_ub (the I/-I block matrix).
      - For each column, only swap b_eq and call _linprog_highs directly.
    Eliminates scipy's per-call validation overhead (~50–200 ms per LP),
    which is the dominant cost for d_ext in the hundreds-to-thousands range.
    """
    if _HAS_HIGHSPY:
        return _highspy_per_column_framability(D, gate)

    D = np.asarray(D, dtype=float)
    n, d_ext = D.shape
    if np.max(np.abs(gate.imag)) > 1e-12:
        raise ValueError('gate must be real for the LP framability.')
    g = gate.real
    Y = g.T @ D

    # Variables: [u_0, ..., u_{d_ext-1}, t_0, ..., t_{d_ext-1}], minimise sum(t).
    c = np.concatenate([np.zeros(d_ext), np.ones(d_ext)])

    # A_eq is (n, 2*d_ext): D in the first d_ext columns, zeros in the next.
    A_eq = csc_matrix(np.hstack([D, np.zeros((n, d_ext))]))

    # A_ub:  [I, -I; -I, -I]  →  rows enforce |u_k| <= t_k.
    I_de = sp_eye(d_ext, format='csc')
    A_ub = sp_vstack(
        [sp_hstack([I_de, -I_de]), sp_hstack([-I_de, -I_de])],
        format='csc',
    )
    b_ub = np.zeros(2 * d_ext)
    bounds = [(None, None)] * d_ext + [(0.0, None)] * d_ext

    # Pre-clean once.  For each column, just swap b_eq.
    lp_raw   = _LPProblem(c, A_ub, b_ub, A_eq, np.zeros(n), bounds, None)
    lp_clean = _clean_inputs(lp_raw)

    one_norms = np.empty(d_ext, dtype=float)
    for j in range(d_ext):
        lp_upd = lp_clean._replace(b_eq=Y[:, j].copy())
        res = _linprog_highs(lp_upd, solver=None, presolve=False)
        if res['status'] == 0:
            x = res['x']
            one_norms[j] = float(np.sum(np.abs(x[:d_ext])))
        else:
            one_norms[j] = np.inf
    return float(np.max(one_norms))


# Backwards-compatible alias for code that may still reference the old name.
_heisenberg_framability_generic = _fast_per_column_framability


_FIXED_S_COL = np.array([[1.0], [0.0], [0.0], [0.0]])


def _kron_power(S, n):
    """Compute kron(S, kron(S, ...)) with n copies."""
    D = S
    for _ in range(n - 1):
        D = np.kron(D, S)
    return D


def _project_cols_bloch(M):
    """Project columns of M (shape 4 x k) onto |c_I| + ||(c_X,c_Y,c_Z)||_2 <= 1."""
    c_I   = np.abs(M[0:1, :])
    bloch = np.linalg.norm(M[1:4, :], axis=0, keepdims=True)
    total = c_I + bloch
    return M / np.maximum(total, 1.0)


def _ixyz_xy_init(d_ext_single):
    """Structured S-init: free cols = [X, Y, Z, (X+Y)/√2, (X-Y)/√2, (X+Z)/√2, (Y+Z)/√2].

    Covers Pauli + extended-Pauli directions without zero columns for d ≤ 8.
    """
    b = 1.0 / np.sqrt(2)
    base = np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., b,  b,  b,  0.],
        [0., 1., 0., b, -b,  0., b ],
        [0., 0., 1., 0.,  0., b,  b ],
    ])
    n_free = d_ext_single - 1
    free = np.zeros((4, n_free))
    k = min(n_free, base.shape[1])
    free[:, :k] = base[:, :k]
    return free.ravel()


def _minimax_framability_4q(gate_1, gate_2, d_ext_single,
                              n_restarts=2, maxfev=30, seed=0, verbose=False):
    """Minimise   max( fra(D, gate_1), fra(D, gate_2) )  over D = kron^4(S).

    S = [I | free], free has d_ext_single - 1 columns parameterised by 4×(d-1)
    real numbers projected onto the Bloch ball before exponentiation.  Each LP
    is solved with the pre-cleaned fast path (option 1).
    """
    n_qubits = 4
    n_s      = 4
    n_free   = d_ext_single - 1
    n_params = n_s * n_free

    def params_to_S(params):
        S_free = _project_cols_bloch(params.reshape(n_s, n_free))
        return np.hstack([_FIXED_S_COL, S_free])

    def objective(params):
        S = params_to_S(params)
        D = _kron_power(S, n_qubits)
        f1 = _fast_per_column_framability(D, gate_1)
        f2 = _fast_per_column_framability(D, gate_2)
        return float(max(f1, f2))

    rng = np.random.default_rng(seed)
    seeds = [_ixyz_xy_init(d_ext_single)]
    seeds.extend(rng.standard_normal(n_params) * 0.3
                 for _ in range(max(0, n_restarts - 1)))

    best_val = float('inf')
    for r, x0 in enumerate(seeds):
        if verbose:
            print(f'    restart {r + 1}/{len(seeds)}  start', flush=True)
        res = minimize(objective, x0, method='Powell',
                       options={'maxfev': maxfev, 'maxiter': maxfev,
                                'ftol': 1e-5, 'xtol': 1e-5})
        if res.fun < best_val:
            best_val = float(res.fun)
        if verbose:
            print(f'    restart {r + 1}/{len(seeds)}  f={float(res.fun):.6f}  '
                  f'best={best_val:.6f}', flush=True)
    return best_val


def _optimise_kron_framability(gate, d_ext_single, n_qubits=N_QUBITS_6Q,
                                n_restarts=1, maxfev=10, seed=0, verbose=False):
    """Minimise framability over D = kron^n_qubits(S) with S of shape (4, d_ext_single).

    Only the d_ext_single - 1 free columns of S are parameterised; the first
    column is fixed to (1, 0, 0, 0) (identity component).  Free columns are
    projected onto the Bloch ball |c_I| + ||(c_X,c_Y,c_Z)||_2 <= 1.
    """
    rng = np.random.default_rng(seed)
    n_s = 4
    n_free = d_ext_single - 1
    n_params = n_s * n_free

    def project(S_free):
        c_I   = np.abs(S_free[0:1, :])
        bloch = np.linalg.norm(S_free[1:4, :], axis=0, keepdims=True)
        total = c_I + bloch
        return S_free / np.maximum(total, 1.0)

    def params_to_S(params):
        S_free = params.reshape(n_s, n_free)
        return np.hstack([_FIXED_S_COL, project(S_free)])

    def kron_power(S):
        D = S
        for _ in range(n_qubits - 1):
            D = np.kron(D, S)
        return D

    def objective(params):
        S = params_to_S(params)
        D = kron_power(S)
        return _heisenberg_framability_generic(D, gate)

    # Initial seed: S = [I, X, Y, Z, (X+Y)/sqrt2, (X-Y)/sqrt2, ...]
    base = np.array([
        [0.0, 0.0, 0.0, 0.0,                      0.0],
        [1.0, 0.0, 0.0, 1.0 / np.sqrt(2),         1.0 / np.sqrt(2)],
        [0.0, 1.0, 0.0, 1.0 / np.sqrt(2),        -1.0 / np.sqrt(2)],
        [0.0, 0.0, 1.0, 0.0,                      0.0],
    ])
    x0_struct = np.zeros((n_s, n_free))
    k = min(n_free, base.shape[1])
    x0_struct[:, :k] = base[:, :k]
    seeds = [x0_struct.ravel()] + [rng.standard_normal(n_params) * 0.3
                                    for _ in range(max(0, n_restarts - 1))]

    best_val = float('inf')
    for r, x0 in enumerate(seeds):
        if verbose:
            print(f'    restart {r+1}/{len(seeds)}', flush=True)
        res = minimize(objective, x0, method='Powell',
                       options={'maxfev': maxfev, 'maxiter': maxfev,
                                'ftol': 1e-4, 'xtol': 1e-4})
        if res.fun < best_val:
            best_val = float(res.fun)
    return best_val


# ── per-point computation ────────────────────────────────────────────────────
def _process_point(ig, igp, args):
    gamma_s = GAMMA_STEP * ig
    gamma_p = GAMMA_STEP * igp
    out_path = Path(args.out_dir) / f'starplaq_{ig:03d}_{igp:03d}.npz'
    if out_path.exists():
        print(f'  skip {out_path.name} (exists)', flush=True)
        return

    t_start = time.perf_counter()
    print(f'[ig={ig:02d} igp={igp:02d}]  gamma_s={gamma_s:.2f}  gamma_p={gamma_p:.2f}',
          flush=True)

    # Lindbladian
    t0 = time.perf_counter()
    L = six_qubit_lindbladian(gamma_s=gamma_s, gamma_p=gamma_p,
                              h=args.h, lam=args.lam).real
    print(f'  L built  ({time.perf_counter()-t0:.1f}s)', flush=True)

    # Steady state + decay rate
    t0 = time.perf_counter()
    c_ss, decay = _steady_state_and_decay(L)
    rho_ss = _pauli_vec_to_rho(c_ss)
    print(f'  steady state + decay  decay={decay:.4e}  '
          f'({time.perf_counter()-t0:.1f}s)', flush=True)

    # Von Neumann entropy
    t0 = time.perf_counter()
    ss_vn = _vn_entropy(rho_ss)
    print(f'  VN entropy = {ss_vn:.4f}  ({time.perf_counter()-t0:.1f}s)',
          flush=True)

    # Partitions
    # Partition 1: ([u,r,d,l] | [ur,ru]) -> A = qubits 0,1,2,3, B = 4,5.  Already ordered.
    rho_p1 = rho_ss
    # Partition 2: ([d,l] | [u,r,ur,ru]) -> need permutation [d,l,u,r,ur,ru] = [2,3,0,1,4,5]
    rho_p2 = _permute_qubits(rho_ss, N_QUBITS_6Q, [2, 3, 0, 1, 4, 5])

    t0 = time.perf_counter()
    neg_p1  = _negativity_bipartition(rho_p1, d_A=16)   # |A|=2^4=16, |B|=4
    neg_p2  = _negativity_bipartition(rho_p2, d_A=4)    # |A|=2^2=4,  |B|=16
    lpdo_p1 = _lpdo_bond_entropy(rho_p1, d_A=16)
    lpdo_p2 = _lpdo_bond_entropy(rho_p2, d_A=4)
    print(f'  partitions  neg=({neg_p1:.4f},{neg_p2:.4f})  '
          f'lpdo=({lpdo_p1:.4f},{lpdo_p2:.4f})  '
          f'({time.perf_counter()-t0:.1f}s)', flush=True)

    # Trotter gate (dense)
    t0 = time.perf_counter()
    gate = expm(L * args.dt).real
    print(f'  expm(L*dt)  ({time.perf_counter()-t0:.1f}s)', flush=True)

    # OTOC (small / large t)
    t0 = time.perf_counter()
    t_small = 0.1 * min(gamma_s, gamma_p) if min(gamma_s, gamma_p) > 0 else 0.01
    t_large = 10.0 * max(gamma_s, gamma_p) if max(gamma_s, gamma_p) > 0 else 1.0
    otoc_s = _otoc(L, t_small)
    otoc_l = _otoc(L, t_large)
    print(f'  OTOC small={otoc_s:.4f}  large={otoc_l:.4f}  '
          f'({time.perf_counter()-t0:.1f}s)', flush=True)

    # Channel stabilizer purity
    chan_M = _channel_stabilizer_purity(gate)

    # Pauli framability (cheap)
    pauli_fra = _pauli_framability(gate)
    print(f'  channel_stab={chan_M:.4f}  pauli_fra={pauli_fra:.4f}', flush=True)

    # 4-qubit minimax framability over the two split Lindbladians
    #   L1: star jump on (u, r, d, l) + half-H on (u, r)
    #   L2: plaquette jump on (u, r, ur, ru) + half-H on (u, r)
    # Compute exp(L_k * dt) and minimax-optimise S over both gates jointly.
    t0 = time.perf_counter()
    L1 = four_qubit_lindbladian_star      (gamma_s=gamma_s, h=args.h / 2, lam=args.lam / 2)
    L2 = four_qubit_lindbladian_plaquette (gamma_p=gamma_p, h=args.h / 2, lam=args.lam / 2)
    gate_1 = expm(L1 * args.dt).real
    gate_2 = expm(L2 * args.dt).real
    print(f'  4q L1/L2 gates  ({time.perf_counter()-t0:.1f}s)', flush=True)

    opt_fra_4 = float('nan')
    opt_fra_6 = float('nan')
    for d_ext_single, key in ((4, 'opt_fra_4'), (6, 'opt_fra_6')):
        if d_ext_single == 4 and not args.do_fra_4:
            continue
        if d_ext_single == 6 and not args.do_fra_6:
            continue
        # d_ext_single=6 LPs are ~270x more expensive per eval than d=4, so
        # use a smaller Powell budget for it.
        if d_ext_single == 4:
            n_restarts, maxfev = args.fra_restarts, args.fra_maxfev
        else:
            n_restarts, maxfev = args.fra_restarts_6, args.fra_maxfev_6
        t0 = time.perf_counter()
        val = _minimax_framability_4q(
            gate_1, gate_2,
            d_ext_single=d_ext_single,
            n_restarts=n_restarts,
            maxfev=maxfev,
            seed=args.seed + ig * 100 + igp,
            verbose=True,
        )
        print(f'  minimax fra (d_ext_single={d_ext_single}) = {val:.4f}  '
              f'({time.perf_counter()-t0:.1f}s)', flush=True)
        if key == 'opt_fra_4':
            opt_fra_4 = val
        else:
            opt_fra_6 = val

    elapsed = time.perf_counter() - t_start
    np.savez(out_path,
             ss_vn_entropy      = np.array(ss_vn),
             neg_urdl_urru      = np.array(neg_p1),
             neg_dl_urur        = np.array(neg_p2),
             lpdo_urdl_urru     = np.array(lpdo_p1),
             lpdo_dl_urur       = np.array(lpdo_p2),
             decay_rate         = np.array(decay),
             otoc_small         = np.array(otoc_s),
             otoc_large         = np.array(otoc_l),
             channel_stab_purity= np.array(chan_M),
             pauli_fra          = np.array(pauli_fra),
             opt_fra_4          = np.array(opt_fra_4),
             opt_fra_6          = np.array(opt_fra_6),
             gamma_s            = np.array(gamma_s),
             gamma_p            = np.array(gamma_p),
             h                  = np.array(args.h),
             lam                = np.array(args.lam),
             dt                 = np.array(args.dt),
             elapsed            = np.array(elapsed))
    print(f'  saved {out_path.name}  (total {elapsed:.1f}s)', flush=True)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',  type=int, required=True,
                        help=f'0..n_jobs-1 (N_TOTAL={N_TOTAL} points split across jobs).')
    parser.add_argument('--n_jobs',   type=int, default=100,
                        help='Total number of parallel jobs (default 100 = 1 point/job).')
    parser.add_argument('--out_dir',  type=str, default='results_six_starplaq')
    parser.add_argument('--h',        type=float, default=1.0)
    parser.add_argument('--lam',      type=float, default=1.0)
    parser.add_argument('--dt',       type=float, default=DT)
    parser.add_argument('--do_fra_4', type=int, default=1,
                        help='Compute 4q-minimax framability with d_ext_single=4.')
    parser.add_argument('--do_fra_6', type=int, default=1,
                        help='Compute 4q-minimax framability with d_ext_single=6 '
                             '(this is the slow one; ~hours per point).')
    parser.add_argument('--fra_restarts', type=int, default=2,
                        help='Powell restarts for d_ext_single=4.')
    parser.add_argument('--fra_maxfev',   type=int, default=30,
                        help='Powell maxfev for d_ext_single=4.')
    parser.add_argument('--fra_restarts_6', type=int, default=1,
                        help='Powell restarts for d_ext_single=6 (kept low; '
                             'each eval is ~14 min).')
    parser.add_argument('--fra_maxfev_6',   type=int, default=10,
                        help='Powell maxfev for d_ext_single=6 (kept low; '
                             'each eval is ~14 min).')
    parser.add_argument('--seed',     type=int, default=0)
    args = parser.parse_args()

    if args.n_jobs <= 0 or N_TOTAL % args.n_jobs != 0:
        print(f'ERROR: n_jobs must divide {N_TOTAL}', file=sys.stderr)
        sys.exit(1)
    points_per_job = N_TOTAL // args.n_jobs

    if args.task_id < 0 or args.task_id >= args.n_jobs:
        print(f'ERROR: task_id {args.task_id} out of range [0, {args.n_jobs - 1}]',
              file=sys.stderr)
        sys.exit(1)

    start = args.task_id * points_per_job
    end   = start + points_per_job
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print(f'[task {args.task_id}] processing linear indices {start}..{end - 1} '
          f'({points_per_job} points)', flush=True)

    for tid in range(start, end):
        ig  = tid // N_GRID
        igp = tid %  N_GRID
        try:
            _process_point(ig, igp, args)
        except Exception as e:
            print(f'  !! point (ig={ig}, igp={igp}) failed: {e!r}',
                  file=sys.stderr, flush=True)
            continue


if __name__ == '__main__':
    main()
