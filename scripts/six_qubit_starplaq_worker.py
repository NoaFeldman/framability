"""
Per-task worker for the 6-qubit star+plaquette Lindbladian Trotter scan.

Lindbladian:  L = -i[H, .] + D[L_s] + D[L_p]
    H   = h (X_u + X_r) + lam (Z_u + Z_r)         (on u, r only)
    L_s = sqrt(gamma_s) X_u X_r X_d X_l           (star)
    L_p = sqrt(gamma_p) Z_u Z_r Z_ur Z_ru         (plaquette)

Grid:
    h = lam = 1
    gamma_s, gamma_p in [GAMMA_STEP * i for i in range(N_GRID)],
        with GAMMA_STEP = 0.2, N_GRID = 20  (so values 0.0 .. 3.8)
    dt = GAMMA_STEP / 10 = 0.02
    Total: 20*20 = 400 grid points

Parallelisation: 10 jobs.  Each job processes 40 grid points.
    task_id in 0..9
    For task t, the grid indices it owns are start..start+39 where
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
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix, eye as sp_eye, hstack as sp_hstack, vstack as sp_vstack
from scipy.sparse.linalg import eigs, spsolve
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


# ── grid ─────────────────────────────────────────────────────────────────────
GAMMA_STEP = 0.2
N_GRID     = int(round(4.0 / GAMMA_STEP))           # 20
DT         = GAMMA_STEP / 10.0                      # 0.02
N_TOTAL    = N_GRID * N_GRID                        # 400

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


# ── steady state and spectral quantities ────────────────────────────────────
def _steady_state_and_decay(L):
    """Return (c_ss, decay_rate).

    c_ss: 4096-vec, the Pauli-basis coefficients of the steady state
          normalised so c_ss[0] = 1/64 (unit trace).
    decay_rate: magnitude of the slowest non-zero eigenvalue of L.

    Steady state via direct sparse solve (pin trace); falls back to the
    maximally mixed state when the null space is degenerate (e.g. free
    qubits at gamma_p=0) or when spsolve returns non-finite values.

    Decay rate via shift-invert eigs with sigma=-0.01: (L+0.01I) is
    nonsingular for any Lindbladian (all eigenvalues have Re<=0, so no
    eigenvalue hits -0.01 unless the decay rate is exactly 0.01), whereas
    sigma=0 or sigma=+epsilon makes UMFPACK factor the near-singular matrix.
    """
    n = L.shape[0]
    d = 2 ** N_QUBITS_6Q  # 64

    # --- steady state: sparse row-0 replacement ---
    L_sp = csc_matrix(L.astype(float))
    L_mod = L_sp.tolil()
    L_mod[0, :] = 0.0
    L_mod[0, 0] = 1.0
    L_mod = L_mod.tocsc()
    rhs = np.zeros(n)
    rhs[0] = 1.0 / d
    # Suppress MatrixRankWarning + RuntimeWarning when L_mod is singular
    # (degenerate null space, e.g. free qubits at gamma_p=0 or gamma_s=0):
    # the fallback below handles it.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            c_ss = spsolve(L_mod, rhs)
        except Exception:
            c_ss = np.full(n, np.nan)
    if np.iscomplexobj(c_ss):
        c_ss = c_ss.real

    # Validate: residual ||L c_ss||_inf on rows 1..n-1
    if np.all(np.isfinite(c_ss)):
        residual = float(np.max(np.abs((L_sp @ c_ss)[1:])))
    else:
        residual = np.inf

    if residual > 0.01:
        # Degenerate null space or solve failure: maximally mixed state.
        # Valid because both jump operators (XXXX, ZZZZ) are unitary, so
        # D[L](I/d) = 0 and -i[H, I/d] = 0 for any H.
        print(f'  [ss] spsolve residual={residual:.2e}, using maximally mixed fallback',
              flush=True)
        c_ss = np.zeros(n)
        c_ss[0] = 1.0 / d

    # --- decay rate via shift-invert eigs (sigma = -0.01) ---
    decay = float('nan')
    try:
        evals, _ = eigs(L_sp.astype(complex), k=4, sigma=-0.01, which='LM',
                        tol=1e-8, maxiter=5000)
        for e in sorted(np.abs(evals)):
            if e > 1e-4:
                decay = float(e)
                break
    except Exception:
        pass

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
    """
    if t <= 0:
        return float('nan')
    E_t = expm(L * t)

    w_str = [I_] * N_QUBITS_6Q
    w_str[W_qubit] = X_
    c_w0 = np.zeros(DIM_6Q, dtype=float)
    c_w0[_string_to_index(w_str)] = 1.0

    c_wt = (E_t.conj().T @ c_w0).real
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


def _fast_per_column_framability(D, gate):
    """Per-column L1-minimisation framability with pre-cleaned LP + direct HiGHS call.

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
        t0 = time.perf_counter()
        val = _minimax_framability_4q(
            gate_1, gate_2,
            d_ext_single=d_ext_single,
            n_restarts=args.fra_restarts,
            maxfev=args.fra_maxfev,
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
                        help=f'0..{(N_TOTAL // 40) - 1} (10 chunks of 40 points).')
    parser.add_argument('--n_jobs',   type=int, default=10,
                        help='Total number of parallel jobs (default 10).')
    parser.add_argument('--out_dir',  type=str, default='results_six_starplaq')
    parser.add_argument('--h',        type=float, default=1.0)
    parser.add_argument('--lam',      type=float, default=1.0)
    parser.add_argument('--dt',       type=float, default=DT)
    parser.add_argument('--do_fra_4', type=int, default=1,
                        help='Compute 4q-minimax framability with d_ext_single=4.')
    parser.add_argument('--do_fra_6', type=int, default=1,
                        help='Compute 4q-minimax framability with d_ext_single=6 '
                             '(this is the slow one; ~hours per point).')
    parser.add_argument('--fra_restarts', type=int, default=2)
    parser.add_argument('--fra_maxfev',   type=int, default=30)
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
