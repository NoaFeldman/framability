"""
Product-frame "free local unitary" trick (continuous_simulation.tex, product-state example).

For a frame element rho = Psi_0 (x) Psi_1 and the two-qubit Lindbladian

    L(rho) = -i[H, rho] + gamma  (D[L_0](rho) + D[L_1](rho))
                        + gamma' (D[Z_0](rho) + D[Z_1](rho)),
    H = J Z(x)Z + h (X(x)I + I(x)X),

write L(rho) in the local bases B_i = {|Psi_i>, |Psi_i^perp>}, joint basis B_0 (x) B_1
ordered as (|Psi_0 Psi_1>, |Psi_0 Psi_1^perp>, |Psi_0^perp Psi_1>, |Psi_0^perp Psi_1^perp>).
The starred entries are A[1,0] (and A[0,1] = A[1,0]^*) and A[2,0] (and A[0,2]).

A local generator H_0 (x) I + I (x) H_1 contributes -i kappa_1 at [1,0] and -i kappa_0 at
[2,0], with kappa_i = <Psi_i^perp| H_i |Psi_i>.  Cancelling requires

    kappa_1 = -i A[1,0],     kappa_0 = -i A[2,0],

realised by the minimal Hermitian choice H_i = kappa_i |Psi_i^perp><Psi_i| + h.c.,
and U_i(dt) = exp(-i dt H_i).

Jump operator L_k = |-><+|_k, as in the scan workers (e.g. scripts/transverse_x_worker.py).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
_MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)


def bloch_to_ket(r) -> np.ndarray:
    """Pure-state ket for a unit Bloch vector (x, y, z)."""
    x, y, z = np.asarray(r, dtype=float) / np.linalg.norm(r)
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])


def _as_ket(psi) -> np.ndarray:
    """Accept a 2-component ket or a length-3 Bloch vector; return a normalised ket."""
    psi = np.asarray(psi)
    if psi.shape == (3,) and np.isrealobj(psi):
        return bloch_to_ket(psi)
    psi = psi.astype(complex).reshape(2)
    return psi / np.linalg.norm(psi)


def perp_ket(psi: np.ndarray) -> np.ndarray:
    """|psi^perp> = (-b^*, a^*) for |psi> = (a, b)."""
    a, b = psi
    return np.array([-np.conj(b), np.conj(a)])


def _dissipator(A: np.ndarray, rho: np.ndarray) -> np.ndarray:
    Ad = A.conj().T
    AdA = Ad @ A
    return A @ rho @ Ad - 0.5 * (AdA @ rho + rho @ AdA)


def two_qubit_lindbladian_action(rho: np.ndarray, J: float, gamma: float, h: float,
                                 gamma_p: float) -> np.ndarray:
    """L(rho) for the 4x4 density matrix rho (computational basis)."""
    Ld = np.outer(_MINUS, _PLUS.conj())   # |-><+|
    H = J * np.kron(_Z, _Z) + h * (np.kron(_X, _I2) + np.kron(_I2, _X))
    out = -1j * (H @ rho - rho @ H)
    out += gamma * (_dissipator(np.kron(Ld, _I2), rho) + _dissipator(np.kron(_I2, Ld), rho))
    out += gamma_p * (_dissipator(np.kron(_Z, _I2), rho) + _dissipator(np.kron(_I2, _Z), rho))
    return out


def local_basis_matrix(psi0, psi1, J: float, gamma: float, h: float,
                       gamma_p: float) -> tuple[np.ndarray, np.ndarray]:
    """L(Psi_0 (x) Psi_1) in the joint basis B_0 (x) B_1.  Returns (A, V), A = V^dag L(rho) V."""
    k0, k1 = _as_ket(psi0), _as_ket(psi1)
    B0 = np.column_stack([k0, perp_ket(k0)])
    B1 = np.column_stack([k1, perp_ket(k1)])
    V = np.kron(B0, B1)
    ket = np.kron(k0, k1)
    rho = np.outer(ket, ket.conj())
    A = V.conj().T @ two_qubit_lindbladian_action(rho, J, gamma, h, gamma_p) @ V
    return A, V


def product_frame_trick(psi0, psi1, J: float, gamma: float, h: float, gamma_p: float,
                        dt: float | None = None) -> dict:
    """Local unitaries U_0, U_1 cancelling the starred entries of L(Psi_0 (x) Psi_1).

    psi0, psi1 : 2-component kets or length-3 real Bloch vectors.
    dt         : if given, also return U_i(dt) = exp(-i dt H_i).

    Returns dict with
        kappa0, kappa1 : required <Psi_i^perp|H_i|Psi_i>
        H0, H1         : 2x2 Hermitian local generators (computational basis)
        U0, U1         : callables dt -> exp(-i dt H_i); arrays if dt was given
        A              : L(rho) in the local basis (before cancellation)
        A_cancelled    : A + (-i[H_0 + H_1, rho]) in the local basis; starred entries = 0
    """
    k0, k1 = _as_ket(psi0), _as_ket(psi1)
    A, V = local_basis_matrix(k0, k1, J, gamma, h, gamma_p)

    kappa1 = -1j * A[1, 0]
    kappa0 = -1j * A[2, 0]

    def _gen(k, kappa):
        P = np.outer(perp_ket(k), k.conj())       # |Psi^perp><Psi|
        return kappa * P + np.conj(kappa) * P.conj().T

    H0 = _gen(k0, kappa0)
    H1 = _gen(k1, kappa1)

    ket = np.kron(k0, k1)
    rho = np.outer(ket, ket.conj())
    Hloc = np.kron(H0, _I2) + np.kron(_I2, H1)
    free = V.conj().T @ (-1j * (Hloc @ rho - rho @ Hloc)) @ V

    def U0(t):
        return expm(-1j * t * H0)

    def U1(t):
        return expm(-1j * t * H1)

    return {
        "kappa0": kappa0,
        "kappa1": kappa1,
        "H0": H0,
        "H1": H1,
        "U0": U0(dt) if dt is not None else U0,
        "U1": U1(dt) if dt is not None else U1,
        "A": A,
        "A_cancelled": A + free,
    }


# ── does a new 1-qubit frame element earn its place? ────────────────
#
# The framability of a frame D (dissipative_PT._framability_lp) is
#
#     f(D, G) = max_j  gauge_K(G^T d_j),      K = conv{± d_1, ..., ± d_m},
#
# where gauge_K(y) = min { ||c||_1 : D c = y } is the Minkowski gauge of the
# symmetric (absolute) convex hull of the columns.  A candidate column v is
# therefore *redundant* exactly when
#
#     gauge_K(v) = min { sum_i |c_i| : sum_i c_i d_i = v } <= 1,
#
# i.e. when v is a signed combination of the existing columns with
# sum |c_i| <= 1.  Then (i) K is unchanged, so no target gets cheaper, and
# (ii) the new target G^T v = sum_i c_i (G^T d_i) obeys
# gauge_K(G^T v) <= sum_i |c_i| max_j gauge_K(G^T d_j) <= f(D, G) by
# sublinearity, so it never becomes the binding column: f is unchanged for
# *every* gate G.  If instead gauge_K(v) > 1 the hull grows strictly (a
# separating y with ||D^T y||_inf <= 1 < y.v exists, returned as `witness`),
# so the frame is genuinely different -- though note that a non-redundant
# column is not automatically an improvement: it also adds its own target
# column G^T v to the max, which may itself be the expensive one.
#
# CAREFUL -- gauge_K(v) <= 1 decides *useless*, not *useful*.  The gauge test is
# one-way: it certifies that v cannot change f for any gate, but gauge_K(v) > 1
# says only that the hull grew, and f can then go down (useful), stay put
# (useless) or go *up* (harmful), because v contributes a new target column
# G^T v to the max as well.  Usefulness is a property of the pair (frame, gate)
# and no condition on v and K alone can decide it -- use frame_element_improves,
# which just recomputes f.  The cheap necessary condition, sharper than
# gauge_K(v) > 1, is the dual one: if y* certifies a binding column
# (||D^T y*||_inf <= 1, y*.(G^T d_j*) = f) then v can only relieve that column
# when |y*.v| > 1, and |y*.v| > 1 already implies gauge_K(v) >= |y*.v| > 1.
#
# For the product frames used here (D = S (x) S) single-qubit redundancy
# lifts: if s = sum_i c_i s_i with ||c||_1 <= 1 then s (x) s_j =
# sum_i c_i (s_i (x) s_j) with the same ell_1 weight, and s (x) s carries
# weight ||c||_1^2 <= 1, so a redundant column of S adds nothing to S (x) S
# either.  A redundant column also lies in the span of the existing ones, so
# it can never repair a rank-deficient frame (dissipative_PT._has_full_support).

def _as_pauli_column(a) -> np.ndarray:
    """Pauli-coefficient column (c_I, c_X, c_Y, c_Z) of a 1-qubit frame element.

    Accepts a length-4 real vector of Pauli coefficients (used as is, the
    convention of the frames S in dissipative_PT), a length-3 real Bloch
    vector r -> (0, r_x, r_y, r_z) (traceless, as the free columns of
    dissipative_PT._ixyz_init), or a 2x2 Hermitian matrix A, decomposed as
    A = sum_mu c_mu sigma_mu with c_mu = Tr(sigma_mu A) / 2.
    """
    a = np.asarray(a)
    if a.shape == (2, 2):
        return np.array([0.5 * np.trace(P.conj().T @ a).real
                         for P in (_I2, _X, _Y, _Z)], dtype=float)
    a = a.astype(float).ravel()
    if a.size == 3:
        return np.concatenate([[0.0], a])
    if a.size == 4:
        return a
    raise ValueError(f'expected a length-3/4 vector or a 2x2 matrix, got shape {a.shape}')


def _as_frame_matrix(frame) -> np.ndarray:
    """(4 x m) Pauli-coefficient matrix for an existing 1-qubit frame.

    A 2-D array / nested list with 3 or 4 rows is read *column-wise* (the
    4 x d_ext layout of S in dissipative_PT); a stack of 2x2 matrices or any
    other sequence is read element-wise via _as_pauli_column.
    """
    try:
        arr = np.asarray(frame)
    except (TypeError, ValueError):
        arr = None
    square = arr is not None and arr.dtype != object
    if square and arr.ndim == 2 and arr.shape[0] in (3, 4):
        items = [arr[:, j] for j in range(arr.shape[1])]
    elif square and arr.ndim == 3 and arr.shape[1:] == (2, 2):
        items = list(arr)
    else:
        items = list(frame)
    if not items:
        return np.zeros((4, 0))
    return np.column_stack([_as_pauli_column(c) for c in items])


def _min_l1(M: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray | None,
                                                   np.ndarray | None]:
    """min ||c||_1 subject to M c = y -- the gauge of conv{± columns of M} at y.

    Returns (gauge, c, dual).  gauge is +inf if y is outside the span of M;
    dual is a u with ||M^T u||_inf <= 1 and u.y = gauge (a maximiser over the
    polar polytope M^o), or None if the solver gave no usable / consistent one.
    """
    M = np.asarray(M, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    nrows, m = M.shape
    if m == 0:
        return (0.0, np.zeros(0), None) if not np.any(y) else (float('inf'), None, None)

    # free coefficients c (m) + epigraph variables t >= |c| (m), minimise sum t
    c_obj = np.concatenate([np.zeros(m), np.ones(m)])
    Ide = np.eye(m)
    A_ub = np.block([[Ide, -Ide], [-Ide, -Ide]])
    b_ub = np.zeros(2 * m)
    A_eq = np.hstack([M, np.zeros((nrows, m))])
    bounds = [(None, None)] * m + [(0.0, None)] * m

    # Solver ladder, the same remedy dissipative_PT._framability_lp already
    # carries (_HIGHS_ATTEMPTS): a single 'highs' call reports status 4
    # ("HiGHS Status 15: model_status is Unknown; primal_status is Infeasible")
    # on feasible but ill-conditioned problems, which is a solver failure and
    # NOT the status 2 that means y is genuinely outside span(M).  Product
    # frames with near-parallel columns -- exactly what the small tilt
    # eps ~ 2 sqrt(dt a) of product_frame_grow produces at small dt -- hit this
    # routinely, so retry with interior point, dual simplex and presolve off
    # before giving up.  Only if EVERY attempt fails is the problem reported.
    attempts = (dict(method='highs'),
                dict(method='highs-ipm'),
                dict(method='highs-ds'),
                dict(method='highs', options={'presolve': False}))
    res, failures = None, []
    for kw in attempts:
        cand = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=y,
                       bounds=bounds, **kw)
        if cand.status == 2:       # infeasible: y is not in the span of M
            return float('inf'), None, None
        if cand.success:
            res = cand
            break
        failures.append(f'{kw.get("method")}'
                        f'{"/nopresolve" if kw.get("options") else ""}'
                        f' -> status {cand.status}: {cand.message}')
    if res is None:
        raise RuntimeError('gauge LP failed on every HiGHS configuration; '
                           + ' | '.join(failures))

    c = np.asarray(res.x[:m], dtype=float)
    gauge = float(np.abs(c).sum())
    u = getattr(getattr(res, 'eqlin', None), 'marginals', None)
    if u is not None:
        u = np.asarray(u, dtype=float)
        # keep the dual only if it really certifies the value
        ok = (abs(float(u @ y) - gauge) <= 1e-7 * max(1.0, gauge)
              and float(np.max(np.abs(M.T @ u))) <= 1.0 + 1e-7)
        u = u if ok else None
    return gauge, c, u


def frame_element_gauge(frame, v) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """min ||c||_1 subject to S c = v, for a 1-qubit frame S and element v.

    Thin wrapper of _min_l1 with the 1-qubit input coercion; see _min_l1 for
    the return convention.
    """
    return _min_l1(_as_frame_matrix(frame), _as_pauli_column(v))

def frame_element_is_redundant(frame, v, tol: float = 1e-9) -> dict:
    """Does the 1-qubit element v add anything to the existing 1-qubit frame?

    frame : the existing frame -- a 4 x m Pauli-coefficient matrix (the S of
            dissipative_PT.frame_from_params), a 3 x m Bloch matrix, or a list
            of elements (length-4 / length-3 vectors or 2x2 Hermitian matrices).
    v     : the candidate element, same accepted forms.
    tol   : slack on the gauge test, redundant iff gauge <= 1 + tol.

    Returns dict with
        redundant  : True iff v = sum_i c_i s_i with sum_i |c_i| <= 1, i.e. v
                     is already in conv{± existing elements} and leaves the
                     framability of the frame unchanged for every gate
        gauge      : the minimal sum_i |c_i| (+inf if v is outside span(S));
                     1/gauge is the factor by which v sticks out of the hull
        coeffs     : the minimising coefficients c (None if infeasible)
        witness    : if not redundant, a separating y -- ||S^T y||_inf <= 1 and
                     y.v = gauge > 1 -- certifying that v enlarges the hull
                     (None if the solver gave no usable dual)
        new_support: True iff v is outside span(S) (then it also raises the
                     rank of the frame, not just the hull)
    """
    gauge, coeffs, witness = frame_element_gauge(frame, v)
    redundant = bool(gauge <= 1.0 + tol)
    return {
        'redundant': redundant,
        'gauge': gauge,
        'coeffs': coeffs,
        'witness': None if redundant else witness,
        'new_support': not np.isfinite(gauge),
    }


def frame_framability(frame, gate, n_bond: int | None = None) -> float:
    """Framability of `gate` over the 1-qubit frame, via dissipative_PT._framability_lp.

    n_bond : number of tensor copies, D = S (x) ... (x) S.  Default: inferred
        from the gate size (4x4 -> 1, 16x16 -> 2), so a 2-qubit gate is scored
        on D = kron(S, S) exactly as in optimise_framability.
    Returns +inf for a frame that does not span every Pauli direction.
    """
    from dissipative_PT import _framability_lp, _kron_power

    S = _as_frame_matrix(frame)
    G = np.asarray(gate).real
    n = n_bond if n_bond is not None else int(round(np.log(G.shape[0]) / np.log(4.0)))
    if n < 1 or G.shape[0] != 4 ** n:
        raise ValueError(f'gate is {G.shape[0]}x{G.shape[0]}, incompatible with n_bond={n}')
    return _framability_lp(_kron_power(S, n), G)


def frame_element_improves(frame, v, gate, n_bond: int | None = None,
                           tol: float = 1e-9) -> dict:
    """Useful vs useless: does adding v to the 1-qubit frame lower the framability?

    Unlike frame_element_is_redundant (a gate-independent *uselessness*
    certificate), this is the actual useful/useless decision and it needs the
    gate: it recomputes f before and after.  Because the new element is also a
    new target column, the answer can be 'harmful' as well.

    frame, v : as in frame_element_is_redundant.
    gate     : Pauli-transfer matrix of the gate (4x4 for n_bond=1, 16x16 for
               the D = S (x) S bond of optimise_framability).
    n_bond   : tensor power for D; default inferred from the gate size.

    Returns dict with
        verdict     : 'useful' (f strictly down), 'useless' (f unchanged) or
                      'harmful' (f up -- v is a target the frame cannot cover
                      cheaply); 'useless' is also returned unconditionally when
                      the gauge test already certifies redundancy
        f_before    : framability of the frame as given
        f_after     : framability of the frame with v appended
        delta       : f_after - f_before  (negative = improvement)
        redundant   : the gate-independent gauge verdict for v
        gauge       : gauge of the old hull at v (see frame_element_gauge)
    """
    S = _as_frame_matrix(frame)
    col = _as_pauli_column(v)
    red = frame_element_is_redundant(S, col, tol=tol)

    f_before = frame_framability(S, gate, n_bond=n_bond)
    f_after = frame_framability(np.column_stack([S, col]), gate, n_bond=n_bond)
    delta = f_after - f_before
    if red['redundant'] or abs(delta) <= tol:
        verdict = 'useless'
    else:
        verdict = 'useful' if delta < 0 else 'harmful'
    return {
        'verdict': verdict,
        'f_before': f_before,
        'f_after': f_after,
        'delta': delta,
        'redundant': red['redundant'],
        'gauge': red['gauge'],
    }


# ── the useful-vs-useless criterion (gate known) ─────────────────────────────
#
# Write T = G^T (the repo's targets are T d_j), K = conv{± cols of D},
# ||y||_K = min{||c||_1 : D c = y}, and f = f(D, G) = max_j ||T d_j||_K with
# binding set  J = { j : ||T d_j||_K = f }.
#
# Appending v gives K' = conv(K u {± v}) whose gauge is exactly
#
#     ||y||_{K'} = min_s ( |s| + ||y - s v||_K ),
#
# equivalently, in the dual, K'^o = K^o n {u : |v.u| <= 1}:
#
#     ||y||_{K'} = max { y.u : |d_i.u| <= 1 for all i,  |v.u| <= 1 }.
#
# Adding a frame element therefore does two opposing things: it adds the
# constraint |v.u| <= 1 to the polar (every gauge weakly drops) and it adds one
# target column T v to the max.  Split the new framability accordingly:
#
#     A = max_{j in J} ||T d_j||_{K'}   (<= f always),
#     B = ||T v||_{K'}                  (the only term that can exceed f),
#     f(D', G) = max( max_all_j ||T d_j||_{K'},  B ),   A <= f(D',G) <= max(f, B).
#
#   CRITERION      v is useful  <=>  A < f  and  B < f
#                  v is harmful <=>  B > f          (then f(D', G) = B)
#                  v is useless <=>  otherwise, i.e. max(A, B) = f
#
# Spelled out primally, v is useful iff
#
#     (I)  for every binding j in J:  min_s ( |s| + ||T d_j - s v||_K ) < f,
#     (II) min_s ( |s| + ||T v - s v||_K ) < f.
#
# Non-binding columns need no check: their gauge is already < f and gauges only
# drop.  Dually, (I) says v must cut off *every* dual optimum of *every*
# binding column: if u maximises (T d_j).u over K^o and |v.u| <= 1, then u
# survives in K'^o and column j still costs f.  Hence the cheap screen
#
#     |u_j . v| > 1  for every binding j  (necessary, not sufficient),
#
# which is strictly sharper than the hull test gauge_K(v) > 1, since
# gauge_K(v) = max{v.u : u in K^o} >= |u_j . v|.  Redundancy (gauge_K(v) <= 1)
# gives K' = K, hence A = f: useless, as it must be.
#
# Useful sufficient conditions for (II): B <= |s| + ||T v - s v||_K for any
# trial s, so B <= ||T v||_K (s = 0), and if v is an eigenvector of T with
# T v = lam v then B <= |lam| (s = lam) -- so an invariant direction with
# |lam| < f always satisfies (II) and only (I) has to be checked.
#
# For a tensor frame D = S (x) ... (x) S (n_bond copies) the same split holds
# with "the new column" replaced by the whole block of new tensor columns
# (those whose multi-index uses v at least once): B is the max over that block,
# and the dual screen becomes max_k |u_j . c_k| > 1 over the new columns c_k.

def frame_element_criterion(frame, v, gate, n_bond: int | None = None,
                            tol: float = 1e-9) -> dict:
    """Useful / useless / harmful verdict for adding v to a 1-qubit frame.

    Evaluates the criterion above: |J| + (#new columns) LPs rather than the
    full recomputation done by frame_element_improves.

    frame, v : as in frame_element_is_redundant.
    gate     : Pauli-transfer matrix, 4**n_bond wide.
    n_bond   : tensor power for D = S (x) ... (x) S; default from the gate size.
    tol      : relative slack for the binding set and for the strict compares.

    Returns dict with
        verdict    : 'useful', 'useless' or 'harmful'
        f_before   : f(D, G)
        A          : max_{j in J} gauge of T d_j over the enlarged hull
        B          : max over the new columns of their gauge over that hull
        f_after    : (lo, hi) bracket max(A, B) <= f(D', G) <= max(f_before, B);
                     frame_element_improves gives the exact value
        binding    : indices of the binding columns of D
        dual_cuts  : for each binding column, max_k |u_j . c_k| over the new
                     columns -- the screen value, which must exceed 1 for every
                     binding column (nan where no usable dual was returned)
        redundant  : the gate-independent gauge verdict for v
    """
    from dissipative_PT import _kron_power

    S = _as_frame_matrix(frame)
    col = _as_pauli_column(v)
    G = np.asarray(gate).real
    n = n_bond if n_bond is not None else int(round(np.log(G.shape[0]) / np.log(4.0)))
    if n < 1 or G.shape[0] != 4 ** n:
        raise ValueError(f'gate is {G.shape[0]}x{G.shape[0]}, incompatible with n_bond={n}')

    m = S.shape[1]
    D_old = _kron_power(S, n)
    D_new = _kron_power(np.column_stack([S, col]), n)

    # multi-index digits of the columns of D_new in base m+1; a column is new
    # iff it uses the appended element (digit m) at least once.  Dropping those
    # leaves the columns of D_old in the same (lexicographic) order.
    base = m + 1
    idx = np.arange(base ** n)
    digits = np.stack([(idx // base ** (n - 1 - k)) % base for k in range(n)])
    is_new = np.any(digits == m, axis=0)
    old_map = np.flatnonzero(~is_new)
    new_cols = np.flatnonzero(is_new)

    Y_old = G.T @ D_old
    solved = [_min_l1(D_old, Y_old[:, j]) for j in range(D_old.shape[1])]
    g_old = np.array([r[0] for r in solved])
    duals = [r[2] for r in solved]
    f_before = float(np.max(g_old))
    if not np.isfinite(f_before):
        binding = np.flatnonzero(~np.isfinite(g_old))
    else:
        binding = np.flatnonzero(g_old >= f_before - tol * max(1.0, f_before))

    Y_new = G.T @ D_new
    A = max(_min_l1(D_new, Y_new[:, old_map[j]])[0] for j in binding)
    B = max(_min_l1(D_new, Y_new[:, c])[0] for c in new_cols)

    thr = tol * max(1.0, abs(f_before)) if np.isfinite(f_before) else 0.0
    if np.isfinite(f_before) and B > f_before + thr:
        verdict = 'harmful'
    elif A < f_before - thr and B < f_before - thr:
        verdict = 'useful'
    else:
        verdict = 'useless'

    cuts = []
    for j in binding:
        u = duals[j]
        cuts.append(float('nan') if u is None
                    else float(np.max(np.abs(u @ D_new[:, new_cols]))))

    return {
        'verdict': verdict,
        'f_before': f_before,
        'A': float(A),
        'B': float(B),
        'f_after': (float(max(A, B)),
                    float(max(f_before, B)) if np.isfinite(f_before) else float('inf')),
        'binding': binding,
        'dual_cuts': np.array(cuts),
        'redundant': frame_element_is_redundant(S, col, tol=tol)['redundant'],
    }
