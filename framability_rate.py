"""
framability_rate.py -- dt-free framability via the l1 log-norm certificate.

For a frame D (16 x d_ext, D = S (x) S) and the generator A = L^T acting on
observables, the polytope P = conv{+-d_j} is invariant under the flow
exp(t A) with decay rate mu whenever there is an H with

        A D = D H      and      mu_1(H) = max_j ( H_jj + sum_{k != j} |H_kj| ) <= mu

(Molchanov-Pyatnitskii / Blanchini polytopic Lyapunov condition, vertex form).
Then exp(t A) D = D exp(t H) and ||exp(t H)||_1 <= exp(t mu), hence

        framability(exp(dt L), D) <= exp(dt * mu)     for every dt,

and conversely mu*(D) = lim_{dt -> 0} (framability(dt, D) - 1) / dt.

The best H for a given D is one LP per column (decoupled, batched here):

        mu_j = min_h  h_j + sum_{k != j} |h_k|   s.t.  D h = A d_j,
        mu*(D) = max_j mu_j.

Frame requirements are exactly those of optimize_framability: column 0 of S is
the identity (1,0,0,0) and stays normalised; every other column has operator
norm <= 1 (columns may shrink; for a Hermitian column this is
|c_I| + ||(c_X, c_Y, c_Z)||_2 <= 1); S must have full rank (D rank 16) or the
certificate is vacuous and mu = +inf.

Complex frames (non-Hermitian elements)
---------------------------------------
S may be complex: a column is then the non-Hermitian operator
c_I I + c_X X + c_Y Y + c_Z Z with complex c (e.g. sigma^+-, clock phases
|0><0| + w|1><1|), and the coefficients h are complex too, with the complex
column log-norm

        mu_j = Re H_jj + sum_{k != j} |H_kj|          (|.| = modulus),

Im H_jj being a free phase rotation of column j.  This is a second-order-cone
program, solved with HiGHS by column generation over coefficient phases
(_complex_lp); every LP along the way is an inner approximation, so the H it
returns is a valid certificate and the reported value an upper bound.  A REAL
frame gains nothing from complex coefficients (Re h is feasible whenever h is,
at no larger cost), so real D always takes the real LP and every real result
is unchanged.  Complex frames are the default of framability_rate_global.

Floor: mu*(D) >= max_i Re lambda_i(A) (the gauge is a norm, so
||exp(tA)|| >= rho(exp(tA))); for a trace-preserving generator that is 0, and
the pinned identity column attains it exactly (A e_II = 0 -> h_0 = 0), so
mu* = max(0, coherence rate) just as framability = max(1, margin).

Public API
----------
generator_log_norm(D, A)            batched LP, mu*(D) [+ per-column values, H]
generator_log_norm_reference(D, A)  independent per-column LP (confirmation)
complex_column_values(D, Y)         exact per-column minima, H, witnesses
                                    (complex D; sum objective)
is_complex_frame(X)                 nonzero imaginary part?
column_opnorm(M), project_frame_columns(M)   operator-norm ball (any dtype)
rate_certificate(D, A)              value, binding column, dual witness
rate_value_and_grad(S, A)           mu*(S (x) S) and its analytic subgradient
polyak_rate_polish(S, A)            Polyak subgradient polish toward the floor
minimize_rate(A, d_ext_single)      alternating certificate optimiser over S
framability_bound(mu, dt)           exp(dt * mu)
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import (csc_matrix, coo_matrix, kron as sp_kron,
                          eye as sp_eye, hstack as sp_hstack)

from optimize_framability import (
    _kron_power, _project_columns_bloch, _FIXED_COLS, N_FIXED_COLS,
    _has_full_support, _build_inits, _solve_kron_factor,
    _require_swap_symmetric,
)
from two_qubit_lindbladian import qubit_d

#   1.0-generator-lognorm : real frames, real coefficients
#   1.1-complex-coeff     : complex frames take the complex-coefficient
#                           column-generation LP (real frames unchanged)
RATE_VERSION = '1.1-complex-coeff'

_N_S = qubit_d ** 2          # 4 rows of S

# Complex-coefficient LP (column generation over coefficient phases).  The
# initial phase set {1, i, -1, -i} is the realified l1 (|Re| + |Im|, at most
# sqrt2 x the modulus); pricing then adds the dual-optimal phase per
# coefficient.  At termination the LP is within COMPLEX_TOL x (off-diagonal
# mass) of the cone optimum; it is always a valid upper bound.
COMPLEX_N_PHASE = int(os.environ.get('RATE_COMPLEX_N_PHASE', '4'))  # initial phase polygon
COMPLEX_MAX_ROUNDS = int(os.environ.get('RATE_COMPLEX_MAX_ROUNDS', '60'))
COMPLEX_TOL = 1e-8          # pricing tolerance (relative to the block weight)
# wall-clock cap (s) of one HiGHS solve; a capped solve counts as failed (+inf)
COMPLEX_LP_TIME_LIMIT = float(os.environ.get('RATE_LP_TIME_LIMIT', '300'))


# ---------------------------------------------------------------------------
#  Complex frames: helpers
# ---------------------------------------------------------------------------
def is_complex_frame(X) -> bool:
    """True iff X has a nonzero imaginary part (a complex dtype holding a real
    frame counts as real, so it keeps the real LP and the real results)."""
    X = np.asarray(X)
    return bool(np.iscomplexobj(X) and np.any(X.imag != 0))


def _as_frame_array(X):
    """X as float if it is real (also a complex dtype with zero imaginary
    part), as complex otherwise -- never drops an imaginary part."""
    X = np.asarray(X)
    if np.iscomplexobj(X):
        return X.astype(complex) if np.any(X.imag != 0) else X.real.astype(float)
    return X.astype(float)


def column_opnorm(M):
    """Operator norm (largest singular value) of every column
    c = (c_I, c_X, c_Y, c_Z) of M read as the 2x2 matrix c_I I + c.sigma.

    For complex c = a + i b:  F^dag F = alpha I + v.sigma with
    alpha = |c_I|^2 + ||c||^2,  v = 2 Re(c_I^* c) - 2 a x b,  so
    ||F|| = sqrt(alpha + |v|).  For real columns this is |c_I| + ||c||_2,
    the Bloch rule of optimize_framability."""
    M = np.asarray(M)
    c0, c = M[0], M[1:4]
    if not np.iscomplexobj(M):
        return np.abs(c0) + np.linalg.norm(c, axis=0)
    a, b = c.real, c.imag
    alpha = np.abs(c0) ** 2 + np.sum(np.abs(c) ** 2, axis=0)
    v = 2.0 * np.real(np.conj(c0)[None, :] * c) - 2.0 * np.cross(a, b, axis=0)
    return np.sqrt(alpha + np.linalg.norm(v, axis=0))


def project_frame_columns(M):
    """Shrink every column of M (4 x k) onto the operator-norm ball
    ||c_I I + c.sigma|| <= 1 (columns inside are untouched).  Real M:
    identical to optimize_framability._project_columns_bloch."""
    M = np.asarray(M)
    if not is_complex_frame(M):
        return _project_columns_bloch(np.real(M).astype(float))
    return M / np.maximum(column_opnorm(M), 1.0)


# ---------------------------------------------------------------------------
#  Floor
# ---------------------------------------------------------------------------
def spectral_abscissa(A) -> float:
    """max Re lambda(A): lower bound on mu*(D) for every frame D."""
    return float(np.max(np.linalg.eigvals(np.asarray(A)).real))


def framability_bound(mu: float, dt: float) -> float:
    """Upper bound exp(dt * mu) on the finite-dt framability of the same frame."""
    return float(np.exp(dt * mu))


# ---------------------------------------------------------------------------
#  Generator log-norm LP (batched)
# ---------------------------------------------------------------------------
def generator_log_norm(D, A, *, return_cols=False, return_H=False,
                       target=None):
    """mu*(D) = max_j min { h_j + sum_{k!=j} |h_k| : D h = A d_j }.

    `target` (n x d_ext) replaces the right-hand sides A d_j column by column
    (A is then ignored and may be None); the moving-frame certificate of
    framability_rate_moving uses it with target = A D - (Sdot(x)S + S(x)Sdot)/z.

    One batched LP.  Variables per target column j: a free diagonal entry
    g_j (cost +1, signed) and h+_j, h-_j >= 0 (cost +1 each, the split
    off-diagonal part), with D (g_j e_j + h+_j - h-_j) = A d_j and
    g_j + sum_k (h+ + h-)_{kj} <= t.  Putting the j-th component into g_j is
    never worse than into h+-, so the optimum has the diagonal unsigned and
    the value is exactly the column log-norm.

    Returns mu (float); optionally the per-column values (d_ext,) and the
    certificate matrix H (d_ext x d_ext) with A D = D H.  A rank-deficient D
    gives +inf (vacuous certificate).

    A complex D (non-Hermitian frame elements) goes to the complex-coefficient
    LP (_complex_log_norm; H is then complex); a real D, or a complex dtype
    with zero imaginary part, keeps this real LP.
    """
    if is_complex_frame(D) or (target is not None and is_complex_frame(target)):
        return _complex_log_norm(D, A, target=target, return_cols=return_cols,
                                 return_H=return_H)
    D = _as_frame_array(D)
    n, m = D.shape
    if not _has_full_support(D):
        out = [np.inf]
        if return_cols:
            out.append(None)
        if return_H:
            out.append(None)
        return out[0] if len(out) == 1 else tuple(out)

    if target is None:
        Y = np.asarray(A, dtype=float) @ D      # targets, column j = A d_j
    else:
        Y = _as_frame_array(target)
        if Y.shape != D.shape:
            raise ValueError(f'target shape {Y.shape} != D shape {D.shape}')
    K = sp_kron(sp_eye(m, format='csc'), csc_matrix(D), format='csc')  # (n m, m m)
    diag_idx = np.arange(m) * m + np.arange(m)  # column-major index of H_jj
    G = K[:, diag_idx]                          # (n m, m)
    n_var = 1 + m + 2 * m * m
    c = np.zeros(n_var)
    c[0] = 1.0
    A_eq = sp_hstack([csc_matrix((n * m, 1)), G, K, -K], format='csc')
    b_eq = Y.ravel(order='F')
    Ssum = sp_kron(sp_eye(m, format='csc'), csc_matrix(np.ones((1, m))),
                   format='csc')               # (m, m m): column sums per block
    A_ub = sp_hstack([-np.ones((m, 1)), sp_eye(m, format='csc'), Ssum, Ssum],
                     format='csc')
    b_ub = np.zeros(m)
    bounds = [(None, None)] * (1 + m) + [(0.0, None)] * (2 * m * m)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')
    if res.status != 0:
        out = [np.inf]
        if return_cols:
            out.append(None)
        if return_H:
            out.append(None)
        return out[0] if len(out) == 1 else tuple(out)

    x = res.x
    g = x[1:1 + m]
    Hp = x[1 + m:1 + m + m * m].reshape(m, m, order='F')
    Hm = x[1 + m + m * m:].reshape(m, m, order='F')
    cols = g + (Hp + Hm).sum(axis=0)
    val = float(np.max(cols))
    if not (return_cols or return_H):
        return val
    out = [val]
    if return_cols:
        out.append(cols)
    if return_H:
        out.append(Hp - Hm + np.diag(g))
    return tuple(out)


def generator_log_norm_reference(D, A, *, return_cols=False, target=None):
    """Independent per-column solve of the same LP (confirmation evaluator).
    `target` as in generator_log_norm.  Complex D: one complex-coefficient
    LP per column, each run to convergence (exact per-column minima)."""
    if is_complex_frame(D) or (target is not None and is_complex_frame(target)):
        D = np.asarray(D, dtype=complex)
        if not _has_full_support(D):
            return (np.inf, None) if return_cols else np.inf
        Y = (np.asarray(A, dtype=float) @ D if target is None
             else np.asarray(target, dtype=complex))
        if Y.shape != D.shape:
            raise ValueError(f'target shape {Y.shape} != D shape {D.shape}')
        cols = np.empty(D.shape[1])
        for j in range(D.shape[1]):
            r = _complex_lp(D, Y, mode='sum', blocks=[j])
            if r is None:
                return (np.inf, None) if return_cols else np.inf
            cols[j] = r['cols'][0]
        val = float(np.max(cols))
        return (val, cols) if return_cols else val
    D = _as_frame_array(D)
    n, m = D.shape
    if not _has_full_support(D):
        return (np.inf, None) if return_cols else np.inf
    if target is None:
        Y = np.asarray(A, dtype=float) @ D
    else:
        Y = _as_frame_array(target)
        if Y.shape != D.shape:
            raise ValueError(f'target shape {Y.shape} != D shape {D.shape}')
    cols = np.empty(m)
    for j in range(m):
        # variables [g, h+ (m), h- (m)]; off-diagonal only in h+-:
        # zero the j-th column of the h+- blocks so the diagonal lives in g.
        P = np.eye(m)
        P[j, j] = 0.0
        DP = D @ P
        A_eq = np.hstack([D[:, [j]], DP, -DP])
        c = np.ones(1 + 2 * m)
        bounds = [(None, None)] + [(0.0, None)] * (2 * m)
        r = linprog(c, A_eq=A_eq, b_eq=Y[:, j], bounds=bounds, method='highs')
        if r.status != 0:
            return (np.inf, None) if return_cols else np.inf
        cols[j] = float(r.fun)
    val = float(np.max(cols))
    return (val, cols) if return_cols else val


# ---------------------------------------------------------------------------
#  Complex-coefficient LP (column generation over coefficient phases)
# ---------------------------------------------------------------------------
def _complex_lp(D, Y, *, mode='max', blocks=None, n_phase=None,
                max_rounds=None, tol=None, time_limit=None):
    """Complex column log-norm problem for the target columns `blocks`:

        min  Re g_j + sum_{k != j} |h_kj|   s.t.  D (g_j e_j + h_j) = Y[:, j]

    over complex g_j, h_j, either jointly as max_j (mode='max', epigraph
    variable t: only the binding columns are minimised, as in
    generator_log_norm) or as independent per-column minima (mode='sum').

    Every off-diagonal coefficient is a nonnegative combination
    h_kj = sum lambda e^{i theta} over a phase set, initially the n_phase-gon
    for every k != j; the LP is real (rows: Re and Im of the equality).  After
    each solve, block j's equality duals y (2n reals; w = y_re + i y_im) and
    its weight u_j (1 for 'sum', the epigraph dual for 'max') price the phase
    theta of coefficient k at  u_j - (alpha cos theta + beta sin theta),
    alpha + i beta = conj(w)^T d_k ... in real form alpha = Dr2_k.y,
    beta = Di2_k.y, minimised at theta = atan2(beta, alpha) with value
    u_j - hypot(alpha, beta).  Columns with a negative reduced cost are added
    until none is left (the LP optimum is then the cone optimum: w is dual
    feasible, |w^H d_k| <= u_j) or max_rounds is reached.  Every intermediate
    H is feasible, so the reported per-column costs (exact moduli of H) are
    always valid upper bounds.

    Returns None (LP failure / time limit) or dict(cols, H, W, rounds,
    converged): cols = exact column costs of the blocks, H (M x len(blocks))
    complex certificate columns, W (n x len(blocks)) complex witnesses
    normalised to w^H d_j = 1 (NaN for a 'max' block of zero weight)."""
    n_phase = COMPLEX_N_PHASE if n_phase is None else int(n_phase)
    max_rounds = COMPLEX_MAX_ROUNDS if max_rounds is None else int(max_rounds)
    tol = COMPLEX_TOL if tol is None else float(tol)
    time_limit = COMPLEX_LP_TIME_LIMIT if time_limit is None else time_limit
    D = np.asarray(D, dtype=complex)
    Y = np.asarray(Y, dtype=complex)
    n, M = D.shape
    blocks = np.arange(M) if blocks is None else np.asarray(blocks, dtype=int)
    nb = blocks.size
    n2 = 2 * n
    Dr2 = np.vstack([D.real, D.imag])           # coefficient 1   -> [Re d; Im d]
    Di2 = np.vstack([-D.imag, D.real])          # coefficient i   -> [Re(i d); Im(i d)]
    b_eq = np.vstack([Y.real, Y.imag])[:, blocks].ravel(order='F')
    is_max = (mode == 'max')
    off = 1 if is_max else 0                    # t first in 'max' mode
    n_g = 2 * nb                                # (Re g, Im g) per block

    # initial phase set: every k != j, the n_phase-gon
    th0 = 2.0 * np.pi * np.arange(n_phase) / n_phase
    lam_b, lam_k, lam_t = [], [], []
    ks_all = np.arange(M)
    for b, j in enumerate(blocks):
        ks = ks_all[ks_all != j]
        lam_b.append(np.full(ks.size * n_phase, b))
        lam_k.append(np.repeat(ks, n_phase))
        lam_t.append(np.tile(th0, ks.size))
    lam_b = np.concatenate(lam_b)
    lam_k = np.concatenate(lam_k)
    lam_t = np.concatenate(lam_t)

    g_rows = (n2 * np.arange(nb))[:, None] + np.arange(n2)[None, :]   # (nb, 2n)
    # solver tolerances below the pricing tolerance, so that dual noise never
    # prices out a column (HiGHS defaults are 1e-7)
    opts = dict(time_limit=time_limit, dual_feasibility_tolerance=1e-9,
                primal_feasibility_tolerance=1e-9)
    res = None
    converged = False
    rounds = 0
    obj_hist = []
    for rounds in range(1, max_rounds + 1):
        sol_b, sol_k, sol_t = lam_b, lam_k, lam_t     # phase set of this solve
        n_lam = lam_b.size
        n_var = off + n_g + n_lam
        # ---- equality matrix (COO) --------------------------------------
        V_g = np.concatenate([Dr2[:, blocks].T, Di2[:, blocks].T], axis=1)  # (nb, 2*2n)
        r_g = np.concatenate([g_rows, g_rows], axis=1)
        c_g = np.concatenate([np.repeat(off + 2 * np.arange(nb), n2).reshape(nb, n2),
                              np.repeat(off + 2 * np.arange(nb) + 1, n2).reshape(nb, n2)],
                             axis=1)
        V_l = (Dr2[:, lam_k] * np.cos(lam_t) + Di2[:, lam_k] * np.sin(lam_t))  # (2n, n_lam)
        r_l = (n2 * lam_b)[None, :] + np.arange(n2)[:, None]
        c_l = np.broadcast_to(off + n_g + np.arange(n_lam), (n2, n_lam))
        A_eq = coo_matrix((np.concatenate([V_g.ravel(), V_l.ravel()]),
                           (np.concatenate([r_g.ravel(), r_l.ravel()]),
                            np.concatenate([c_g.ravel(), c_l.ravel()]))),
                          shape=(n2 * nb, n_var)).tocsc()
        c = np.zeros(n_var)
        if is_max:
            c[0] = 1.0
            ub_r = np.concatenate([np.arange(nb), np.arange(nb), lam_b])
            ub_c = np.concatenate([np.zeros(nb, int), off + 2 * np.arange(nb),
                                   off + n_g + np.arange(n_lam)])
            ub_v = np.concatenate([-np.ones(nb), np.ones(nb), np.ones(n_lam)])
            A_ub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(nb, n_var)).tocsc()
            b_ub = np.zeros(nb)
        else:
            c[off + 2 * np.arange(nb)] = 1.0
            c[off + n_g:] = 1.0
            A_ub = b_ub = None
        bounds = np.zeros((n_var, 2))
        bounds[:off + n_g, 0] = -np.inf
        bounds[:, 1] = np.inf
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs', options=opts)
        if res.status != 0:
            return None
        y = getattr(getattr(res, 'eqlin', None), 'marginals', None)
        if y is None:                                   # no duals: stop here
            break
        Yb = np.asarray(y, float).reshape(nb, n2)
        if is_max:
            u = np.maximum(-np.asarray(res.ineqlin.marginals, float), 0.0)
        else:
            u = np.ones(nb)
        alpha = Yb @ Dr2                                # (nb, M)
        beta = Yb @ Di2
        mag = np.hypot(alpha, beta)
        # reduced cost u_j - mag; the threshold is absolute on the dual scale
        # (sum u = 1 for 'max', u = 1 for 'sum'), so a non-binding 'max' block
        # (u_j = 0, duals 0 up to noise) never prices
        viol = mag - u[:, None] - tol * max(1.0, float(u.max()))
        viol[np.arange(nb), blocks] = -np.inf           # diagonal lives in g
        bb, kk = np.nonzero(viol > 0)
        if bb.size == 0:
            converged = True
            break
        # stall guard: the objective must keep dropping (else dual noise)
        obj_hist.append(float(res.fun))
        if len(obj_hist) >= 3 and \
                obj_hist[-3] - obj_hist[-1] <= 1e-12 * max(1.0, abs(obj_hist[-1])):
            break
        lam_b = np.concatenate([lam_b, bb])
        lam_k = np.concatenate([lam_k, kk])
        lam_t = np.concatenate([lam_t, np.arctan2(beta[bb, kk], alpha[bb, kk])])

    # ---- certificate and exact costs (phase set of the LAST solve) -----
    x = res.x
    H = np.zeros((M, nb), dtype=complex)
    H[blocks, np.arange(nb)] = x[off + 2 * np.arange(nb)] + 1j * x[off + 2 * np.arange(nb) + 1]
    lam = x[off + n_g:off + n_g + sol_b.size]
    np.add.at(H, (sol_k, sol_b), lam * np.exp(1j * sol_t))
    diag = H[blocks, np.arange(nb)]
    absoff = np.abs(H).sum(axis=0) - np.abs(diag)
    cols = diag.real + absoff
    W = np.full((n, nb), np.nan, dtype=complex)
    y = getattr(getattr(res, 'eqlin', None), 'marginals', None)
    if y is not None:
        Yb = np.asarray(y, float).reshape(nb, n2)
        u = (np.maximum(-np.asarray(res.ineqlin.marginals, float), 0.0) if is_max
             else np.ones(nb))
        ok = u > 1e-12
        W[:, ok] = ((Yb[ok, :n] + 1j * Yb[ok, n:]) / u[ok, None]).T
    return dict(cols=cols, H=H, W=W, rounds=rounds, converged=converged)


def _complex_log_norm(D, A, *, target=None, return_cols=False, return_H=False):
    """generator_log_norm for a complex frame (one batched 'max' LP)."""
    D = np.asarray(D, dtype=complex)
    n, m = D.shape

    def fail():
        out = [np.inf]
        if return_cols:
            out.append(None)
        if return_H:
            out.append(None)
        return out[0] if len(out) == 1 else tuple(out)

    if not _has_full_support(D):
        return fail()
    Y = (np.asarray(A, dtype=float) @ D if target is None
         else np.asarray(target, dtype=complex))
    if Y.shape != D.shape:
        raise ValueError(f'target shape {Y.shape} != D shape {D.shape}')
    r = _complex_lp(D, Y, mode='max')
    if r is None:
        return fail()
    val = float(np.max(r['cols']))
    if not (return_cols or return_H):
        return val
    out = [val]
    if return_cols:
        out.append(r['cols'])
    if return_H:
        out.append(r['H'])
    return tuple(out)


def complex_column_values(D, Y, *, blocks=None):
    """Exact per-column minima (sum objective: the columns decouple) of the
    complex-coefficient problem, with H and the normalised witnesses W.
    Returns (cols, H, W) or (None, None, None)."""
    D = np.asarray(D, dtype=complex)
    if not _has_full_support(D):
        return None, None, None
    r = _complex_lp(D, Y, mode='sum', blocks=blocks)
    if r is None:
        return None, None, None
    return r['cols'], r['H'], r['W']


def complex_column_witness(D, A, j):
    """(value, h, w) of column j alone: exact minimum, its certificate column
    and the dual witness w (w^H d_j = 1, |w^H d_k| <= 1).  None on failure."""
    D = np.asarray(D, dtype=complex)
    r = _complex_lp(D, np.asarray(A, float) @ D, mode='sum', blocks=[j])
    if r is None or not np.all(np.isfinite(r['W'])):
        return None
    return float(r['cols'][0]), r['H'][:, 0], r['W'][:, 0]


def complex_grad_from_D(S, G):
    """Chain dmu/dD (complex G: dmu = Re sum conj(G) dD) through D = S (x) S.
    Returns Gamma with dmu = Re sum conj(Gamma) dS, i.e. dmu/dRe S = Re Gamma,
    dmu/dIm S = Im Gamma (real S, real G: the real chain rule)."""
    n_s, m = S.shape
    T = G.reshape(n_s, n_s, m, m)
    Sc = np.conj(S)
    return np.einsum('abij,bj->ai', T, Sc) + np.einsum('baji,bj->ai', T, Sc)


# ---------------------------------------------------------------------------
#  Certificate, subgradient, Polyak polish
# ---------------------------------------------------------------------------
def rate_certificate(D, A):
    """Value, binding column j*, dual witness w* and per-column values.

    Dual of the binding column's LP:  max <w, A d_j*>  s.t.
    (D^T w)_j* = 1  and  |(D^T w)_k| <= 1 for k != j*.
    Complex D: the witness (complex, w^H d_j* = 1) and the column j* of H
    come from one exact per-column solve, so they belong together.
    """
    D = _as_frame_array(D)
    A = np.asarray(A, dtype=float)
    val, cols, H = generator_log_norm(D, A, return_cols=True, return_H=True)
    if not np.isfinite(val):
        return dict(value=np.inf, argmax=-1, witness=None, cols=None, H=None)
    j = int(np.argmax(cols))
    if np.iscomplexobj(D):
        r = complex_column_witness(D, A, j)
        if r is None:
            return dict(value=val, argmax=j, witness=None, cols=cols, H=H)
        H = H.copy()
        H[:, j] = r[1]
        return dict(value=val, argmax=j, witness=r[2], cols=cols, H=H)
    y = A @ D[:, j]
    DT = D.T
    m = D.shape[1]
    res = linprog(-y, A_ub=np.vstack([DT, -DT]), b_ub=np.ones(2 * m),
                  A_eq=DT[j:j + 1], b_eq=np.array([1.0]),
                  bounds=[(None, None)] * D.shape[0], method='highs')
    w = res.x.copy() if res.success else None
    return dict(value=val, argmax=j, witness=w, cols=cols, H=H)


def rate_value_and_grad(S, A):
    """mu*(S (x) S) and an analytic subgradient w.r.t. S (envelope theorem).

    d mu_j* / dD = (A^T w*) e_j*^T - w* h*^T, then chain-ruled through
    D = S (x) S exactly as in optimize_framability.framability_value_and_grad.
    Complex S: G = (A^T w) e_j^T - w h^H and the returned dS is the complex
    gradient Gamma (d mu = Re sum conj(Gamma) dS; complex_grad_from_D).  At a
    REAL frame the imaginary part of the gradient vanishes (mu(S) =
    mu(conj S) for a real generator), so gradient steps never leave the real
    subspace by themselves -- complex searches need complex starts.
    """
    S = _as_frame_array(S)
    A = np.asarray(A, dtype=float)
    n_s, m = S.shape
    D = _kron_power(S, 2)
    cert = rate_certificate(D, A)
    val, j, w, H = cert['value'], cert['argmax'], cert['witness'], cert['H']
    if not np.isfinite(val) or w is None:
        return val, np.zeros_like(S)
    h = H[:, j]
    nrows, d_ext = D.shape
    gD = np.zeros((nrows, d_ext), dtype=np.result_type(w, h, float))
    gD[:, j] += A.T @ w
    gD -= np.outer(w, np.conj(h))
    return val, complex_grad_from_D(S, gD)


def polyak_rate_polish(S, A, *, target=None, n_iter=300, tol=1e-10,
                       stall_patience=20, verbose=False):
    """Projected Polyak subgradient descent on mu*(S) toward `target`
    (default: spectral abscissa of A, i.e. 0 for a TP generator).  Works on
    real and complex frames (complex: steps along the complex gradient,
    projection onto the operator-norm ball)."""
    A = np.asarray(A, dtype=float)
    if target is None:
        target = spectral_abscissa(A)
    S = _as_frame_array(S).copy()
    f_best, S_best = np.inf, S.copy()
    beta, stall = 1.0, 0
    for k in range(n_iter):
        val, dS = rate_value_and_grad(S, A)
        if np.isfinite(val) and val < f_best - 1e-15:
            f_best, S_best, stall = val, S.copy(), 0
        else:
            stall += 1
            if stall >= stall_patience:
                beta, stall = 0.5 * beta, 0
                if beta < 1e-3:
                    break
        if val <= target + tol:
            break
        g = dS[:, N_FIXED_COLS:]
        gnorm_sq = float(np.sum(np.abs(g) ** 2))
        if gnorm_sq < 1e-20:
            break
        alpha = beta * (val - target) / gnorm_sq
        free = S[:, N_FIXED_COLS:] - alpha * g
        S = np.hstack([_FIXED_COLS.astype(free.dtype), project_frame_columns(free)])
        if verbose and (k + 1) % 25 == 0:
            print(f'    polish {k + 1}: mu={val:.3e} best={f_best:.3e} '
                  f'beta={beta:.3f}', flush=True)
    return float(f_best), S_best


# ---------------------------------------------------------------------------
#  Level-set H refinement
# ---------------------------------------------------------------------------
def _lognorm_parts(H):
    m = H.shape[0]
    idx = np.arange(m)
    diag = H[idx, idx].copy()
    absoff = np.abs(H)
    absoff[idx, idx] = 0.0
    return idx, diag, absoff, diag + absoff.sum(axis=0)


def _apply_tau(H, over, ds, tau, idx):
    out = H.copy()
    sub = H[:, over]
    new = np.sign(sub) * np.maximum(np.abs(sub) - tau, 0.0)
    jj = idx[over]
    new[jj, np.arange(jj.size)] = ds - tau
    out[:, over] = new
    return out


def _project_columns_lognorm(H, level):
    """Euclidean projection of each column h of H onto
    { h : h_j + sum_{k != j} |h_k| <= level }  (j = column index).

    KKT with multiplier tau >= 0:  h_j -> h_j - tau,  h_k -> soft(h_k, tau).
    With r off-diagonal entries active (|h_k| > tau) the budget is linear in
    tau, giving tau_r = (h_j + sum_{top r} |h_k| - level) / (r + 1); as in
    the Duchi et al. simplex projection the right r is the largest one with
    s_r > tau_r (s = sorted magnitudes), r = 0 meaning only the diagonal moves.
    """
    H = np.asarray(H, dtype=float)
    m = H.shape[0]
    idx, diag, absoff, val = _lognorm_parts(H)
    over = val > level
    if not np.any(over):
        return H
    ds = diag[over]
    s = -np.sort(-absoff[:, over], axis=0)                 # descending per column
    css = np.cumsum(s, axis=0)
    r = np.arange(1, m + 1)[:, None]
    tau_r = (ds[None, :] + css - level) / (r + 1)
    rho = np.count_nonzero(s > tau_r, axis=0)              # prefix property
    tau = np.where(rho > 0,
                   tau_r[np.maximum(rho - 1, 0), np.arange(ds.size)],
                   ds - level)
    return _apply_tau(H, over, ds, tau, idx)


def _project_columns_lognorm_bisect(H, level, n_iter=60):
    """Reference implementation of _project_columns_lognorm (bisection)."""
    H = np.asarray(H, dtype=float)
    idx, diag, absoff, val = _lognorm_parts(H)
    over = val > level
    if not np.any(over):
        return H
    ds = diag[over]
    Ao = absoff[:, over]
    lo = np.zeros(ds.shape)
    hi = val[over] - level                    # budget(hi) <= level
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f = ds - mid + np.maximum(Ao - mid, 0.0).sum(axis=0)
        big = f > level
        lo = np.where(big, mid, lo)
        hi = np.where(big, hi, mid)
    return _apply_tau(H, over, ds, hi, idx)


def _refine_H_level(D, Y, H0, level, n_iter=150, tol_rel=1e-9):
    """min_H ||D H - Y||_F^2  s.t. column log-norms <= level  (FISTA,
    stopped early once an iterate moves by less than tol_rel * ||H||)."""
    L = 2.0 * np.linalg.norm(D, 2) ** 2
    step = 1.0 / L
    DtD = D.T @ D
    DtY = D.T @ Y
    H = _project_columns_lognorm(H0, level)
    Z = H.copy()
    t_acc = 1.0
    scale = max(float(np.linalg.norm(H)), 1e-12)
    for _ in range(n_iter):
        H_new = _project_columns_lognorm(Z - step * 2.0 * (DtD @ Z - DtY), level)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_acc * t_acc))
        Z = H_new + ((t_acc - 1.0) / t_new) * (H_new - H)
        moved = float(np.linalg.norm(H_new - H))
        H, t_acc = H_new, t_new
        if moved < tol_rel * scale:
            break
    return H


# ---------------------------------------------------------------------------
#  Alternating certificate optimiser over S
# ---------------------------------------------------------------------------
_RATE_SHRINK = 0.9
_RATE_PG_ITERS = 150
_RATE_STALL_PATIENCE = 8


def minimize_rate(A, d_ext_single, *, n_restarts=8, maxfev=3000, tol=1e-9,
                  seed=None, verbose=True, extra_init_S=None,
                  polish_iters=300, check_swap=True):
    """min_S mu*(S (x) S) for the generator A (= L^T on observables).

    Same block structure as optimize_framability._run_alternating with the
    gate LP replaced by the generator log-norm LP and no matrix exponential:
      1. exact batched LP  -> mu, H          (incumbent tracking)
      2. level-set H-step  -> H at budget floor + shrink (mu_best - floor)
      3. S-step: right factor, left factor (projected least squares of the
         linear residual A D - D H), symmetrise.
    Then a Polyak polish of the best frame, and a confirmation with the
    independent per-column LP (reported value == reference of the frame).

    Returns (S_opt, mu_opt, info) with info = dict(floor, mu_search,
    mu_polish, mu_reference, n_sweeps).
    """
    A = np.asarray(A, dtype=float)
    if check_swap:
        _require_swap_symmetric(A)
    rng = np.random.default_rng(seed)
    floor = spectral_abscissa(A)
    m = d_ext_single
    n_free = m - N_FIXED_COLS
    d_ext = m ** 2
    extra = None
    if extra_init_S:
        extra = [np.asarray(S0, float)[:, N_FIXED_COLS:].ravel()
                 for S0 in extra_init_S]
    inits = _build_inits(_N_S, m, d_ext, n_restarts, rng,
                         extra_init_xs=extra, use_complex=False)
    n_sweeps = max(10, maxfev // (max(1, len(inits)) * 5))

    best_val, best_S = np.inf, None
    total_sweeps = 0
    for r, x0 in enumerate(inits):
        free = np.asarray(x0, float)[:_N_S * n_free].reshape(_N_S, n_free)
        S = np.hstack([_FIXED_COLS, _project_columns_bloch(free)])
        loc_val, loc_S, stall = np.inf, S.copy(), 0
        for sweep in range(n_sweeps):
            total_sweeps += 1
            D = _kron_power(S, 2)
            val, cols, H = generator_log_norm(D, A, return_cols=True,
                                              return_H=True)
            if not np.isfinite(val) or H is None:
                break
            if val < loc_val - 1e-12:
                loc_val, loc_S, stall = val, S.copy(), 0
            else:
                stall += 1
                if stall >= _RATE_STALL_PATIENCE:
                    break
            if loc_val <= floor + tol:
                break
            level = floor + _RATE_SHRINK * (loc_val - floor)
            H_lvl = _refine_H_level(D, A @ D, H, level, n_iter=_RATE_PG_ITERS)
            B = _solve_kron_factor(A, H_lvl, S, 'right', m)
            A2 = _solve_kron_factor(A, H_lvl, B, 'left', m)
            S_free = 0.5 * (A2[:, N_FIXED_COLS:] + B[:, N_FIXED_COLS:])
            S = np.hstack([_FIXED_COLS, _project_columns_bloch(S_free)])
        if verbose:
            print(f'  rate restart {r + 1}/{len(inits)}: mu={loc_val:.6e}',
                  flush=True)
        if loc_val < best_val:
            best_val, best_S = loc_val, loc_S.copy()

    if best_S is None:
        free = np.asarray(inits[0], float)[:_N_S * n_free].reshape(_N_S, n_free)
        best_S = np.hstack([_FIXED_COLS, _project_columns_bloch(free)])
        best_val = generator_log_norm(_kron_power(best_S, 2), A)

    mu_search = best_val
    mu_polish = best_val
    if polish_iters > 0 and np.isfinite(best_val) and best_val > floor + tol:
        f_pol, S_pol = polyak_rate_polish(best_S, A, target=floor,
                                          n_iter=polish_iters)
        if f_pol < best_val:
            best_val, best_S, mu_polish = f_pol, S_pol, f_pol

    mu_ref = generator_log_norm_reference(_kron_power(best_S, 2), A)
    if np.isfinite(mu_ref) and mu_ref > best_val:
        best_val = mu_ref
    if verbose:
        print(f'floor = {floor:.3e}  mu_search = {mu_search:.6e}  '
              f'mu_polish = {mu_polish:.6e}  mu_ref = {mu_ref:.6e}',
              flush=True)
    info = dict(floor=floor, mu_search=mu_search, mu_polish=mu_polish,
                mu_reference=mu_ref, n_sweeps=total_sweeps)
    return best_S, float(best_val), info


# ---------------------------------------------------------------------------
#  Neighbour refining (cross-evaluation) of a rate scan
# ---------------------------------------------------------------------------
def grid_neighbors(nx, ny):
    """4-connected neighbour lists of an nx x ny grid, flat index i = ix*ny + iy."""
    nbrs = []
    for ix in range(nx):
        for iy in range(ny):
            lst = []
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny:
                    lst.append(jx * ny + jy)
            nbrs.append(lst)
    return nbrs


def line_neighbors(n, K=1):
    """Neighbour lists of a 1-D line: the K points on either side."""
    return [[j for j in range(max(0, i - K), min(n, i + K + 1)) if j != i]
            for i in range(n)]


def neighbor_refine_rates(gens, frames, mus, neighbors, *, tol=1e-9,
                          max_sweeps=200, refine=False, refine_points=None,
                          n_restarts=4, maxfev=2000, polish_iters=300,
                          seed=0, verbose=True):
    """Cross-evaluation ("neighbour refining") of a rate scan.

    Inputs are per point i: the generator gens[i] (already in the picture the
    frames live in, i.e. L^T for observable frames), the optimised frame
    frames[i] (4 x m, identity column first) and its rate mus[i] (an upper
    bound on the true minimum at that point), plus neighbors[i], the indices
    whose frames may be transferred to i (grid_neighbors / line_neighbors).

    Stage 1, propagation (no optimisation): every neighbour's frame is
    evaluated on point i's own generator with the batched LP.  A finite value
    is a rigorous upper bound, so  new = min(old, best neighbour)  is always
    sound; improvements are confirmed with the independent per-column LP
    before acceptance.  Gauss-Seidel sweeps repeat until nothing changes, so
    a good frame travels as far as it transfers exactly.

    Stage 2, optional re-optimisation (refine=True): for the points listed in
    refine_points (default: every point a neighbour frame improved, plus every
    remaining island — a point above all its neighbours by more than tol)
    minimize_rate is re-run on the point's generator seeded with its own frame
    and all neighbour frames; the result is kept only if it beats the
    incumbent, and stage 1 is repeated afterwards so the new frames spread.

    Returns a dict with mu (new values), S (new frames), src (index of the
    point whose optimiser produced the frame now held at i; -1 = refined
    here), cross (best neighbour value seen at i, the island diagnostic),
    changed (bool per point) and n_evals.
    """
    n = len(gens)
    mu = np.array([float(v) for v in mus], dtype=float)
    S = [np.asarray(f, dtype=float).copy() for f in frames]
    src = np.arange(n)
    cross = np.full(n, np.inf)
    changed = np.zeros(n, dtype=bool)
    n_evals = 0

    def propagate(tag):
        nonlocal n_evals
        for sweep in range(1, max_sweeps + 1):
            n_changed = 0
            for i in range(n):
                A = gens[i]
                best_v, best_j = np.inf, -1
                for j in neighbors[i]:
                    v = generator_log_norm(_kron_power(S[j], 2), A)
                    n_evals += 1
                    if np.isfinite(v) and v < best_v:
                        best_v, best_j = v, j
                cross[i] = min(cross[i], best_v)
                if best_j < 0 or best_v >= mu[i] - tol:
                    continue
                v_ref = generator_log_norm_reference(_kron_power(S[best_j], 2), A)
                if np.isfinite(v_ref) and v_ref < mu[i] - tol:
                    mu[i], S[i], src[i] = v_ref, S[best_j].copy(), src[best_j]
                    changed[i] = True
                    n_changed += 1
            if verbose:
                print(f'  [{tag}] sweep {sweep}: {n_changed} point(s) improved '
                      f'({n_evals} LP evals)', flush=True)
            if n_changed == 0:
                break

    propagate('propagate')

    if refine:
        if refine_points is None:
            islands = [i for i in range(n) if neighbors[i] and
                       mu[i] > max(mu[j] for j in neighbors[i]) + tol]
            refine_points = sorted(set(np.flatnonzero(changed)) | set(islands))
        rng = np.random.default_rng(seed)
        for i in refine_points:
            seeds = [S[i]] + [S[j] for j in neighbors[i]]
            S_new, mu_new, _ = minimize_rate(
                gens[i], S[i].shape[1], n_restarts=n_restarts, maxfev=maxfev,
                seed=int(rng.integers(2 ** 31)), verbose=False,
                extra_init_S=seeds, polish_iters=polish_iters)
            if verbose:
                print(f'  [refine] point {i}: {mu[i]:.6e} -> {mu_new:.6e}',
                      flush=True)
            if np.isfinite(mu_new) and mu_new < mu[i] - tol:
                mu[i], S[i], src[i], changed[i] = mu_new, S_new, -1, True
        propagate('propagate-after-refine')

    return dict(mu=mu, S=S, src=src, cross=cross, changed=changed,
                n_evals=n_evals)


# ---------------------------------------------------------------------------
#  Self-tests
# ---------------------------------------------------------------------------
def _check_projection(seed=0):
    """Projection onto the log-norm set: agreement with the bisection
    reference, feasibility and optimality spot-check."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(300):
        m = int(rng.integers(2, 17))
        Hr = rng.standard_normal((m, m)) * rng.choice([0.01, 1.0, 30.0])
        lev = float(rng.choice([-2.0, -0.1, 0.0, 0.05, 0.5, 3.0]))
        Pa = _project_columns_lognorm(Hr, lev)
        Pb = _project_columns_lognorm_bisect(Hr, lev, n_iter=80)
        worst = max(worst, float(np.max(np.abs(Pa - Pb))))
    assert worst < 1e-8, worst
    print(f'projection vs bisection reference: max |diff| = {worst:.2e}')
    H = rng.standard_normal((6, 6))
    P = _project_columns_lognorm(H, 0.3)
    idx = np.arange(6)
    absoff = np.abs(P)
    absoff[idx, idx] = 0.0
    budget = P[idx, idx] + absoff.sum(axis=0)
    assert np.all(budget <= 0.3 + 1e-9), budget
    # optimality: any feasible perturbation must not be closer to H
    for _ in range(200):
        Q = P + 1e-3 * rng.standard_normal(P.shape)
        ao = np.abs(Q)
        ao[idx, idx] = 0.0
        ok = Q[idx, idx] + ao.sum(axis=0) <= 0.3
        for j in np.where(ok)[0]:
            assert (np.linalg.norm(Q[:, j] - H[:, j])
                    >= np.linalg.norm(P[:, j] - H[:, j]) - 1e-12)
    print('projection: ok')


def _check_limit(seed=0, dts=(1e-1, 1e-2, 1e-3, 1e-4)):
    """(framability(dt) - 1)/dt -> mu* and framability <= exp(dt mu*)."""
    from scipy.linalg import expm
    from dissipative_PT import _framability_lp
    from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,
                                          DIM_DEFAULT)
    H1, H2, j1, j2 = MODELS['model3'].build(3.0, 2.0)
    L = build_bond_lindbladian(H1, H2, j1, j2, DIM_DEFAULT).real
    A = L.T
    rng = np.random.default_rng(seed)
    S = np.hstack([_FIXED_COLS, _project_columns_bloch(rng.standard_normal((4, 3)))])
    D = _kron_power(S, 2)
    mu = generator_log_norm(D, A)
    mu_ref = generator_log_norm_reference(D, A)
    print(f'mu batched = {mu:.10f}   mu reference = {mu_ref:.10f}')
    for dt in dts:
        f = _framability_lp(D, expm(dt * L).real)
        print(f'  dt={dt:.0e}: (fra-1)/dt = {(f - 1) / dt:.8f}   '
              f'fra <= exp(dt mu): {f <= np.exp(dt * mu) + 1e-9}')
    val, dS = rate_value_and_grad(S, A)
    eps = 1e-6
    fd = np.zeros_like(S)
    for a in range(4):
        for i in range(S.shape[1]):
            Sp = S.copy()
            Sp[a, i] += eps
            fd[a, i] = (generator_log_norm(_kron_power(Sp, 2), A) - val) / eps
    err = np.linalg.norm(dS - fd) / max(np.linalg.norm(fd), 1e-12)
    print(f'subgradient vs FD relative error: {err:.3e}')


def _check_complex(seed=0, d_exts=(6, 8, 12)):
    """Complex frames: operator norm vs SVD, LP vs closed form on bases (H is
    unique, so the LP must reach the exact moduli), batched vs reference,
    continuity with the real LP, complex gradient vs finite differences, and
    timings per d_ext (one LP evaluation)."""
    import time
    import warnings
    try:
        from numpy.exceptions import ComplexWarning
    except ImportError:                                     # numpy < 1.25
        ComplexWarning = np.ComplexWarning
    warnings.simplefilter('error', ComplexWarning)
    from trotter_lindbladian_scan import MODELS, build_bond_lindbladian
    spec = MODELS['model4']
    A = build_bond_lindbladian(*spec.build(3.0, 2.0), spec.dim).real.T
    rng = np.random.default_rng(seed)

    def rand_frame(m):
        F = rng.standard_normal((4, m - 1)) + 1j * rng.standard_normal((4, m - 1))
        return np.hstack([_FIXED_COLS.astype(complex), project_frame_columns(F)])

    pa = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
          np.diag([1.0, -1.0])]
    F = rng.standard_normal((4, 6)) + 1j * rng.standard_normal((4, 6))
    on = [np.linalg.norm(sum(F[a, i] * pa[a] for a in range(4)), 2) for i in range(6)]
    print(f'operator norm vs SVD: {np.max(np.abs(np.array(on) - column_opnorm(F))):.2e}')
    worst = 0.0
    for _ in range(4):
        S = rand_frame(4)
        D = _kron_power(S, 2)
        H = np.linalg.solve(D, A @ D)
        dg = np.diag(H)
        cf = float((dg.real + np.abs(H).sum(0) - np.abs(dg)).max())
        worst = max(worst, abs(generator_log_norm(D, A) - cf))
    print(f'basis frames, LP vs closed form: max |diff| = {worst:.2e}')
    S = np.hstack([_FIXED_COLS, _project_columns_bloch(rng.standard_normal((4, 5)))])
    vr = generator_log_norm(_kron_power(S, 2), A)
    Sc = S.astype(complex)
    Sc[1:, 1:] += 1e-7j * rng.standard_normal((3, 5))
    print(f'real LP {vr:.8f} vs complex LP at Im ~ 1e-7: '
          f'{generator_log_norm(_kron_power(Sc, 2), A):.8f}')
    for m in d_exts:
        S = rand_frame(m)
        D = _kron_power(S, 2)
        t0 = time.perf_counter()
        r = _complex_lp(D, A @ D, mode='max')
        t1 = time.perf_counter()
        v = float(np.max(r['cols']))
        msg = (f'd_ext {m:2d}: batched {v:.8f}  ({t1 - t0:.2f} s, {r["rounds"]} '
               f'rounds, converged {r["converged"]}, residual '
               f'{np.abs(D @ r["H"] - A @ D).max():.1e})')
        if m <= 8:
            t0 = time.perf_counter()
            vref = generator_log_norm_reference(D, A)
            msg += f'  reference {vref:.8f} ({time.perf_counter() - t0:.2f} s)'
        print(msg, flush=True)
    S = rand_frame(4)
    val, G = rate_value_and_grad(S, A)
    eps, fd = 1e-6, np.zeros_like(S)
    for a in range(4):
        for i in range(1, S.shape[1]):
            for unit in (1.0, 1j):
                Sp = S.copy()
                Sp[a, i] += eps * unit
                dv = (generator_log_norm(_kron_power(Sp, 2), A) - val) / eps
                fd[a, i] += dv * unit           # d/dRe + i d/dIm = Gamma
    err = np.linalg.norm((G - fd)[:, 1:]) / max(np.linalg.norm(fd[:, 1:]), 1e-12)
    print(f'complex gradient vs FD (basis frame): relative error {err:.3e}')


if __name__ == '__main__':
    _check_projection()
    _check_limit()
    _check_complex()
