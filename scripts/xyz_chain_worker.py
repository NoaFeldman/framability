"""
Per-gamma worker for the boundary-driven anisotropic XYZ chain (N=4).

For one drive rate gamma (= gamma_L = gamma_R) computes every quantity from the
starplaq analysis, plus the sign-problem parameter:

    ss_vn_entropy                Von Neumann entropy of the NESS
    neg_12_34, neg_1_234         negativity, cuts [1,2]|[3,4] and [1]|[2,3,4]
    lpdo_12_34, lpdo_1_234       LPDO bond entropy, same cuts
    decay_rate                   Liouvillian gap (slowest non-zero mode)
    otoc_small, otoc_large       OTOC of X_1, X_N at t = 0.1*gamma / 10*gamma
    channel_stab_purity          stabiliser purity of exp(L*dt)
    pauli_fra                    Pauli framability of exp(L*dt)
    opt_fra_4, opt_fra_6         optimised kron framability (d_ext_single=4,6)
    sign_init, sign_opt          sign problem s=|sum U|/sum|U|, raw / local-opt
    nsign_init, nsign_opt        -log(s):  0 = no sign problem, >0 otherwise

Gate for the operator-level quantities: U = exp(L*dt).

Sweep:  gamma in [0, 10] step 0.5  (units of Jz=1)  -> 21 points.
Couplings: Jx=1.5, Jy=0.5, Jz=1.0  (anisotropic XYZ).

Output: <out_dir>/xyz_<idx:02d>.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm, null_space
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, eye as sp_eye, hstack as sp_hstack, vstack as sp_vstack

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from one_d_xyz_lindbladian import (
    build_lindbladian_pauli, hamiltonian_xyz, index_to_string, string_to_index,
    _pauli_tensor, _pauli_string_op, _PAULI, I_, X_, Y_, Z_,
)
from sign_problem import sign_problem
from lpdo import purification_sqrt, _bond_entropy

# scipy fast LP fallback
from scipy.optimize._linprog_highs import _linprog_highs
from scipy.optimize._linprog_util import _LPProblem, _clean_inputs
try:
    import highspy
    _HAS_HIGHSPY = True
except ImportError:
    _HAS_HIGHSPY = False


# ── sweep / model parameters ──────────────────────────────────────────────────
N_QUBITS   = 4
JX, JY, JZ = 1.5, 0.5, 1.0
GAMMA_STEP = 0.5
GAMMA_MAX  = 10.0
N_POINTS   = int(round(GAMMA_MAX / GAMMA_STEP)) + 1   # 21
DT         = 0.05
DIM        = 4 ** N_QUBITS                            # 256
D_HILBERT  = 2 ** N_QUBITS                            # 16

_FIXED_S_COL = np.array([[1.0], [0.0], [0.0], [0.0]])  # identity Pauli column


# ── steady state + decay rate ────────────────────────────────────────────────
def _steady_state_and_decay(L):
    """NESS (Pauli coeffs, trace 1) and Liouvillian gap.

    For gamma>0 the boundary drive makes the NESS unique (1-dim null space).
    At gamma=0 the chain is closed and the steady state is degenerate; then the
    NESS quantities are returned as NaN.
    """
    ns = null_space(L, rcond=1e-9)          # columns span ker(L)
    decay = float('nan')
    evals = np.linalg.eigvals(L)
    re = np.sort(evals.real)[::-1]          # descending
    for r in re:
        if r < -1e-9:
            decay = float(-r)
            break

    if ns.shape[1] != 1:
        return None, decay                  # degenerate (gamma=0) -> no NESS
    c = ns[:, 0].real
    if abs(c[0]) < 1e-12:
        return None, decay
    c = c / (c[0] * D_HILBERT)              # normalise to unit trace
    return c, decay


def _pauli_vec_to_rho(c, n=N_QUBITS):
    P = _pauli_tensor(n)
    rho = np.einsum('a,aij->ij', c.astype(complex), P)
    return (rho + rho.conj().T) / 2.0


# ── entanglement quantities ──────────────────────────────────────────────────
def _vn_entropy(rho):
    e = np.linalg.eigvalsh((rho + rho.conj().T) / 2.0)
    e = e[e > 1e-15]
    return float(-np.sum(e * np.log(e)))


def _negativity(rho, d_A):
    d = rho.shape[0]
    d_B = d // d_A
    rho_pt = (rho.reshape(d_A, d_B, d_A, d_B)
                 .transpose(0, 3, 2, 1).reshape(d, d))
    e = np.linalg.eigvalsh((rho_pt + rho_pt.conj().T) / 2.0)
    return float(np.sum(np.abs(e[e < -1e-15])))


def _lpdo_bond_entropy(rho, d_A):
    d = rho.shape[0]
    d_B = d // d_A
    X = purification_sqrt(rho)
    T = X.reshape(d_A, d_B, d_A, d_B).transpose(0, 2, 1, 3)
    M = T.reshape(d_A * d_A, d_B * d_B)
    S = np.linalg.svd(M, compute_uv=False)
    return float(_bond_entropy(S))


# ── OTOC ─────────────────────────────────────────────────────────────────────
def _otoc(L, t, n=N_QUBITS, V_site=0, W_site=None):
    """OTOC(t) = <0| W(t)^† V^† W(t) V |0>,  V = X_{V_site}, W = X_{W_site}."""
    if W_site is None:
        W_site = n - 1
    if t <= 0:
        return float('nan')
    E_t = expm(L * t)                       # 256x256 dense, fast

    w_str = [I_] * n
    w_str[W_site] = X_
    c_w0 = np.zeros(DIM)
    c_w0[string_to_index(w_str)] = 1.0
    c_wt = (E_t.T @ c_w0).real              # Heisenberg-evolved W (L real)
    W_t = _pauli_vec_to_rho(c_wt, n)

    v_str = [I_] * n
    v_str[V_site] = X_
    V0 = _pauli_string_op(v_str)

    psi0 = np.zeros(D_HILBERT, dtype=complex)
    psi0[0] = 1.0
    op = W_t.conj().T @ V0.conj().T @ W_t @ V0
    return float(np.real(np.vdot(psi0, op @ psi0)))


# ── channel stabiliser purity + Pauli framability ────────────────────────────
def _channel_stabilizer_purity(gate):
    diag = np.diag(gate).real
    d = D_HILBERT
    return float(np.log2((d ** 2) * float(np.sum(diag ** 2)) / (d + 1)))


def _pauli_framability(gate):
    return float(np.max(np.sum(np.abs(gate), axis=1)))


# ── framability LP (highspy warm-start, scipy fallback) ───────────────────────
def _kron_power(S, n):
    D = S
    for _ in range(n - 1):
        D = np.kron(D, S)
    return D


def _highspy_framability(D, gate):
    D = np.asarray(D, dtype=float)
    n, d_ext = D.shape
    Y = gate.real.T @ D
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    h.setOptionValue('presolve', 'off')
    h.setOptionValue('solver', 'simplex')
    nv = 2 * d_ext
    cost = np.concatenate([np.zeros(d_ext), np.ones(d_ext)])
    h.addVars(nv, np.concatenate([-inf * np.ones(d_ext), np.zeros(d_ext)]),
              inf * np.ones(nv))
    h.changeColsCost(nv, np.arange(nv, dtype=np.int32), cost)
    I_de = sp_eye(d_ext, format='csr')
    A = sp_vstack([sp_hstack([csc_matrix(D), csc_matrix((n, d_ext))]),
                   sp_hstack([I_de, -I_de]),
                   sp_hstack([-I_de, -I_de])]).tocsr()
    row_lb = np.concatenate([np.zeros(n), -inf * np.ones(2 * d_ext)])
    row_ub = np.zeros(n + 2 * d_ext)
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


def _scipy_framability(D, gate):
    D = np.asarray(D, dtype=float)
    n, d_ext = D.shape
    Y = gate.real.T @ D
    c = np.concatenate([np.zeros(d_ext), np.ones(d_ext)])
    A_eq = csc_matrix(np.hstack([D, np.zeros((n, d_ext))]))
    I_de = sp_eye(d_ext, format='csc')
    A_ub = sp_vstack([sp_hstack([I_de, -I_de]), sp_hstack([-I_de, -I_de])],
                     format='csc')
    b_ub = np.zeros(2 * d_ext)
    bounds = [(None, None)] * d_ext + [(0.0, None)] * d_ext
    lp = _clean_inputs(_LPProblem(c, A_ub, b_ub, A_eq, np.zeros(n), bounds, None))
    best = 0.0
    for j in range(d_ext):
        r = _linprog_highs(lp._replace(b_eq=Y[:, j].copy()), solver=None,
                           presolve=False)
        if r['status'] == 0:
            best = max(best, float(np.sum(np.abs(r['x'][:d_ext]))))
    return best


def _framability(D, gate):
    return _highspy_framability(D, gate) if _HAS_HIGHSPY else _scipy_framability(D, gate)


def _project_cols_bloch(M):
    c_I = np.abs(M[0:1, :])
    bloch = np.linalg.norm(M[1:4, :], axis=0, keepdims=True)
    return M / np.maximum(c_I + bloch, 1.0)


def _ixyz_init(d_ext_single):
    b = 1.0 / np.sqrt(2)
    base = np.array([[0., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., b,  b,  b,  0.],
                     [0., 1., 0., b, -b,  0., b ],
                     [0., 0., 1., 0., 0., b,  b ]])
    n_free = d_ext_single - 1
    free = np.zeros((4, n_free))
    k = min(n_free, base.shape[1])
    free[:, :k] = base[:, :k]
    return free.ravel()


def _optimise_framability(gate, d_ext_single, n_restarts=2, maxfev=30, seed=0):
    """Minimise framability over D = kron^N(S),  S = [I | free]  (4 x d_ext)."""
    n_free = d_ext_single - 1
    n_params = 4 * n_free

    def params_to_S(p):
        return np.hstack([_FIXED_S_COL, _project_cols_bloch(p.reshape(4, n_free))])

    def objective(p):
        D = _kron_power(params_to_S(p), N_QUBITS)
        return float(_framability(D, gate))

    rng = np.random.default_rng(seed)
    seeds = [_ixyz_init(d_ext_single)]
    seeds += [rng.standard_normal(n_params) * 0.3 for _ in range(max(0, n_restarts - 1))]
    best = float('inf')
    for x0 in seeds:
        res = minimize(objective, x0, method='Powell',
                       options={'maxfev': maxfev, 'maxiter': maxfev,
                                'ftol': 1e-5, 'xtol': 1e-5})
        best = min(best, float(res.fun))
    return best


# ── sign problem (raw + local-basis optimised) ───────────────────────────────
def _single_qubit_superop_pauli(R):
    M = np.zeros((4, 4), dtype=complex)
    for i, Pi in enumerate(_PAULI):
        for j, Pj in enumerate(_PAULI):
            M[i, j] = np.trace(Pi @ R @ Pj @ R.conj().T) / 2.0
    return M.real if np.allclose(M.imag, 0, atol=1e-12) else M


def _apply_local_rotation(gate, M, n=N_QUBITS):
    """(M^{⊗n}) gate (M^{⊗n})^T via tensor contractions (no dense kron)."""
    d = 4 ** n
    g = gate.reshape((4,) * (2 * n))
    for k in range(n):
        g = np.moveaxis(np.tensordot(M, g, axes=([1], [k])), 0, k)
    for k in range(n):
        ax = n + k
        g = np.moveaxis(np.tensordot(M, g, axes=([1], [ax])), 0, ax)
    return g.reshape(d, d)


def _maximise_sign(gate, n_restarts=10, seed=0):
    """Maximise s over local R^{⊗N} (R = exp(i*pi*(nx X+ny Y+nz Z)))."""
    _sx, _sy, _sz = _PAULI[1], _PAULI[2], _PAULI[3]
    rng = np.random.default_rng(seed)

    def neg_s(p):
        R = expm(1j * np.pi * (p[0] * _sx + p[1] * _sy + p[2] * _sz))
        M = _single_qubit_superop_pauli(R)
        if np.max(np.abs(M.imag)) > 1e-10:
            return 0.0
        return -float(sign_problem(_apply_local_rotation(gate, M.real)))

    best = float(sign_problem(gate))
    for _ in range(n_restarts):
        x0 = rng.standard_normal(3)
        x0 /= max(np.linalg.norm(x0), 1e-14)
        res = minimize(neg_s, x0, method='BFGS')
        best = max(best, float(-res.fun))
    return best


# ── per-point ─────────────────────────────────────────────────────────────────
def _process_point(idx, args):
    gamma = GAMMA_STEP * idx                 # = gamma_L (the swept drive rate)
    gamma_R = gamma * args.gamma_ratio       # gamma_R = ratio * gamma_L
    out = Path(args.out_dir) / f'xyz_{idx:02d}.npz'
    if out.exists():
        print(f'  skip {out.name} (exists)', flush=True)
        return
    t_start = time.perf_counter()
    print(f'[idx={idx:02d}] gamma_L={gamma:.2f} gamma_R={gamma_R:.2f} '
          f'(ratio={args.gamma_ratio})  Jx={JX} Jy={JY} Jz={JZ}', flush=True)

    L = build_lindbladian_pauli(JX, JY, JZ, gamma, gamma_R, N_QUBITS)

    # --- steady state ---
    c_ss, decay = _steady_state_and_decay(L)
    if c_ss is not None:
        rho = _pauli_vec_to_rho(c_ss)
        ss_vn = _vn_entropy(rho)
        neg_12_34  = _negativity(rho, d_A=4)
        neg_1_234  = _negativity(rho, d_A=2)
        lpdo_12_34 = _lpdo_bond_entropy(rho, d_A=4)
        lpdo_1_234 = _lpdo_bond_entropy(rho, d_A=2)
    else:
        ss_vn = neg_12_34 = neg_1_234 = lpdo_12_34 = lpdo_1_234 = float('nan')
    print(f'  NESS: VN={ss_vn:.4f}  neg=({neg_12_34:.4f},{neg_1_234:.4f})  '
          f'decay={decay:.4e}', flush=True)

    # --- gate-level ---
    gate = expm(L * args.dt).real
    t_small = 0.1 * gamma if gamma > 0 else 0.05
    t_large = 10.0 * gamma if gamma > 0 else 1.0
    otoc_s = _otoc(L, t_small)
    otoc_l = _otoc(L, t_large)
    chan = _channel_stabilizer_purity(gate)
    pauli_fra = _pauli_framability(gate)
    print(f'  OTOC=({otoc_s:.4f},{otoc_l:.4f})  chan={chan:.4f}  '
          f'pauli_fra={pauli_fra:.4f}', flush=True)

    opt_fra_4 = opt_fra_6 = float('nan')
    if args.do_fra_4:
        t0 = time.perf_counter()
        opt_fra_4 = _optimise_framability(gate, 4, args.fra_restarts, args.fra_maxfev,
                                          seed=args.seed + idx)
        print(f'  opt_fra_4={opt_fra_4:.4f}  ({time.perf_counter()-t0:.0f}s)', flush=True)
    if args.do_fra_6:
        t0 = time.perf_counter()
        opt_fra_6 = _optimise_framability(gate, 6, args.fra_restarts_6, args.fra_maxfev_6,
                                          seed=args.seed + idx)
        print(f'  opt_fra_6={opt_fra_6:.4f}  ({time.perf_counter()-t0:.0f}s)', flush=True)

    # --- sign problem ---
    sign_init = float(sign_problem(gate))
    sign_opt = _maximise_sign(gate, n_restarts=args.sign_restarts,
                              seed=args.seed + idx)
    # -log(s): 0 = no sign problem (s=1), >0 otherwise.
    nsign_init = float(-np.log(max(sign_init, 1e-300)))
    nsign_opt = float(-np.log(max(sign_opt, 1e-300)))
    print(f'  sign: init={sign_init:.4f} opt={sign_opt:.4f}  '
          f'-log: init={nsign_init:.4f} opt={nsign_opt:.4f}', flush=True)

    np.savez(out,
             gamma=np.array(gamma),
             ss_vn_entropy=np.array(ss_vn),
             neg_12_34=np.array(neg_12_34), neg_1_234=np.array(neg_1_234),
             lpdo_12_34=np.array(lpdo_12_34), lpdo_1_234=np.array(lpdo_1_234),
             decay_rate=np.array(decay),
             otoc_small=np.array(otoc_s), otoc_large=np.array(otoc_l),
             channel_stab_purity=np.array(chan),
             pauli_fra=np.array(pauli_fra),
             opt_fra_4=np.array(opt_fra_4), opt_fra_6=np.array(opt_fra_6),
             sign_init=np.array(sign_init), sign_opt=np.array(sign_opt),
             nsign_init=np.array(nsign_init), nsign_opt=np.array(nsign_opt),
             gamma_R=np.array(gamma_R), gamma_ratio=np.array(args.gamma_ratio),
             Jx=np.array(JX), Jy=np.array(JY), Jz=np.array(JZ), dt=np.array(args.dt))
    print(f'  saved {out.name}  (total {time.perf_counter()-t_start:.0f}s)', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, required=True, help=f'0..{N_POINTS-1}')
    p.add_argument('--n_jobs', type=int, default=N_POINTS)
    p.add_argument('--out_dir', type=str, default='results_xyz_chain')
    p.add_argument('--gamma_ratio', type=float, default=1.0,
                   help='gamma_R / gamma_L. 1.0 = symmetric (default), '
                        '0.5 = half right drain, 0.0 = left pump only.')
    p.add_argument('--dt', type=float, default=DT)
    p.add_argument('--do_fra_4', type=int, default=1)
    p.add_argument('--do_fra_6', type=int, default=1)
    p.add_argument('--fra_restarts', type=int, default=2)
    p.add_argument('--fra_maxfev', type=int, default=30)
    p.add_argument('--fra_restarts_6', type=int, default=1)
    p.add_argument('--fra_maxfev_6', type=int, default=10)
    p.add_argument('--sign_restarts', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    if not (0 <= args.task_id < args.n_jobs):
        print(f'ERROR: task_id out of range [0, {args.n_jobs})', file=sys.stderr)
        sys.exit(1)

    # Split N_POINTS across n_jobs.
    per = (N_POINTS + args.n_jobs - 1) // args.n_jobs
    start = args.task_id * per
    stop = min(N_POINTS, start + per)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f'[task {args.task_id}/{args.n_jobs}] points [{start},{stop})  '
          f'highspy={_HAS_HIGHSPY}', flush=True)
    for idx in range(start, stop):
        try:
            _process_point(idx, args)
        except Exception as e:
            print(f'  !! point idx={idx} failed: {type(e).__name__}({e!r})',
                  flush=True)


if __name__ == '__main__':
    main()
