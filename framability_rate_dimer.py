r"""
framability_rate_dimer.py -- Heisenberg framability rate with a TWO-qubit
("dimer") frame S.

The lattice is tiled by dimers (qubit a, qubit b); every dimer carries the
SAME 2-qubit observable frame S (16 x m, 2-qubit Pauli-Liouville rows), so the
lattice frame is S (x) S (x) ... and, on any two dimers, D = S (x) S (256 x m^2).
The bond gate B (the model's 16 x 16 bond Lindbladian, build_bond_lindbladian
with the model's dim, one-site shares 1/(2 dim) included) is applied in two
ways, and the optimisation sees both:

  * INSIDE a dimer   -- frame S,          A_in  = B^T          (16 x 16)
  * ACROSS two dimers -- frame S (x) S,   A_cut = (1 (x) B (x) 1)^T on the
    qubits (b of the first dimer, a of the second)               (256 x 256)

Certificates are the column log-norm LP of framability_rate (A D = D H,
mu_1(H) = max_j H_jj + sum_{k != j} |H_kj|).  The lattice generator is the
sum of its bond generators, every bond has its own certificate, and
mu(sum_b H_b (x) 1) <= sum_b mu(H_b), mu(H (x) 1) = mu(H), so per dimer

      mu_lattice / N_dimers <= mu_in + sum_c w_c mu_c ,   sum_c w_c = 2 dim - 1

(a dimer owns 2 dim bonds: 1 inside, 2 dim - 1 cut).  The reported objective
is the PER-BOND average

      F_split(S) = ( mu_in(S) + sum_c w_c mu_c(S (x) S) ) / (2 dim),

directly comparable with the stored per-bond product-frame rates rate_heis_m
(frame S1 (x) S1 per bond).  Tilings:

  dim 1  'chain'     one cut type 'ba' (b of the left dimer, a of the right)
  dim 2  'brick'     dimers in running-bond order: all 3 cuts per dimer are
                     of type 'ba' (needs a swap-symmetric bond, checked)
         'columnar'  stacked dimers: cuts 'ba' + 'aa' + 'bb'

mode='joint' instead gives every cut its own 4-qubit PATCH
      A_c + kappa (A_in (x) 1 + 1 (x) A_in),  kappa = 1 / (2 (2 dim - 1))
(each intra bond is spread over the 2 (2 dim - 1) cuts of its dimer) with ONE
certificate per patch:  F_joint = sum_c w_c mu(patch_c) / (2 dim) <= F_split.

Guarantees used for seeding / sanity checks
-------------------------------------------
* Pauli frame (S = 1_16):  F_split = pauli_rate(B) exactly.
* Product frame S = S1 (x) S1:  mu_in = mu*(S1 (x) S1; bond), the stored
  single-qubit rate, and mu_cut <= it (certificate 1 (x) H (x) 1), so
  F_split <= the product rate: the dimer search can only improve on d4.
* Marginal (partial-trace) LOWER bound on the cut:  keeping only the rows of
  S (x) S with Pauli I on the two qubits the cut does not touch commutes with
  A_cut, so  A_cut D = D H  implies  B^T (S_b (x) S_a) = (S_b (x) S_a) H  with
  the single-qubit marginal frames S_b = S[0:4] (qubit b: rows I q) and
  S_a = S[0::4] (qubit a: rows p I).  Hence
      mu_cut(S (x) S) >= mu*(S_b (x) S_a; bond)
  a PRODUCT-frame rate (two possibly different factors, m columns each), and
  F_split >= (2 dim - 1)/(2 dim) * min_{A, B} mu*(A (x) B).  The dimer frame
  can only win through mu_in; the cut never beats the product frames of its
  own marginals.  ('joint' patches are not bounded this way: the intra terms
  act on the traced qubits.)

Optimisation (minimize_dimer_rate), m = 16 (a 2-qubit BASIS): D is invertible,
H = D^{-1} A D is unique and every rate is closed form (one 256 x 256 product).
  1. seeds: Pauli, products S1 (x) S1 of the single-qubit seed library and of
     the caller's stored d4 frames, any 2-qubit frames the caller passes;
  2. rotate + weight: S = R_U diag(1, w), U in SU(4).  For fixed U every
     column rate is a sum of exponentials of linear functions of log w, so the
     weights are a CONVEX problem (solved exactly, SLSQP epigraph); only the
     15 parameters of U are searched (multistart Nelder-Mead);
  3. smoothed L-BFGS continuation over the full frame (15 x 16 free
     entries; |x| -> sqrt(x^2 + eps^2), max -> log-sum-exp, closed-form
     gradient), from the best candidates and random kicks of the best, then
     a short exact bundle trust-region step (the bundle alone stalls: sparse
     H at product-like frames puts kinks into every column, see
     smoothed_value_grad);
  4. optional m > 16: pad by duplicated columns (rate cannot increase) and
     polish with per-column LPs (256 x m^2 frames: m^2 LPs per evaluation).
Frame rules as everywhere: column 0 = identity (pinned), every other column an
observable with operator norm <= 1 (projection = eigenvalue clipping).

Public API
----------
DimerProblem(B, dim, tiling, mode)     generators of one lattice point
evaluate(S, prob, mode)                F and per-term rates (closed form / LP)
rotate_weight_search(prob, mode)       stage 2
bundle_polish(S, prob, mode)           stage 3 / 4
minimize_dimer_rate(prob, ...)         the full seeded search
stored_product_frames(dirs, ix, iy)    single-qubit frames + best stored rates
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linprog, minimize

from dissipative_PT import _pauli_tensor
from framability_rate_global import (seed_library, minimize_rate_global,
                                     WEIGHT_FLOOR)

DIMER_RATE_VERSION = '1.0-dimer'

NQ = 16                       # 2-qubit Pauli-Liouville dimension (rows of S)
_COND_MAX_S = 1e6             # cond(S) above this -> singular (cond(S(x)S) = cond^2)
_TOL = 1e-9

# cut types: the two dimers' qubits a/b the bond joins, as sites of S (x) S
CUT_SITES = {'ba': (1, 2), 'ab': (0, 3), 'aa': (0, 2), 'bb': (1, 3)}
TILINGS = {
    1: {'chain': {'ba': 1.0}},
    2: {'brick': {'ba': 3.0}, 'columnar': {'ba': 1.0, 'aa': 1.0, 'bb': 1.0}},
}
DEFAULT_TILING = {1: 'chain', 2: 'brick'}

_P2 = _pauli_tensor(2)        # (16, 4, 4), index 4 p_a + p_b (qubit a first)
_E_II = np.zeros((NQ, 1))
_E_II[0, 0] = 1.0


# ---------------------------------------------------------------------------
#  Pauli-Liouville helpers
# ---------------------------------------------------------------------------
def embed_bond(B, sites, n: int = 4) -> np.ndarray:
    """(4^n x 4^n) Pauli-Liouville matrix of the 2-qubit superoperator B
    (16 x 16, its first qubit on sites[0]) on qubits `sites` of an n-qubit
    register (qubit 0 = most significant, as dissipative_PT._pauli_tensor),
    identity on the others."""
    B = np.asarray(B)
    i, j = sites
    others = [q for q in range(n) if q not in (i, j)]
    full = np.kron(B, np.eye(4 ** (n - 2)))           # acts on positions 0, 1
    order = [i, j] + others                            # position k <-> qubit order[k]
    inv = list(np.argsort(order))                      # qubit q sits at position inv[q]
    T = full.reshape([4] * (2 * n))
    T = T.transpose(inv + [n + k for k in inv])
    return T.reshape(4 ** n, 4 ** n)


def swap_superop() -> np.ndarray:
    """16 x 16 Pauli-Liouville matrix of the qubit swap (a <-> b)."""
    P = np.zeros((NQ, NQ))
    for p in range(4):
        for q in range(4):
            P[4 * q + p, 4 * p + q] = 1.0
    return P


def is_swap_symmetric(B, tol: float = 1e-10) -> bool:
    W = swap_superop()
    return bool(np.max(np.abs(W @ np.asarray(B) @ W - np.asarray(B))) <= tol)


def pauli_liouville_unitary(U) -> np.ndarray:
    """R[a, b] = Tr(P_a U P_b U^dag) / 4: real orthogonal 16 x 16, R e_II = e_II."""
    U = np.asarray(U, complex)
    UPU = np.einsum('ij,bjk,lk->bil', U, _P2, U.conj())
    return np.einsum('aij,bji->ab', _P2, UPU).real / 4.0


def su4(theta) -> np.ndarray:
    """U = exp(-i sum_k theta_k P_k) over the 15 non-identity 2-qubit Paulis."""
    return expm(-1j * np.einsum('k,kij->ij', np.asarray(theta, float), _P2[1:]))


def project_columns_opnorm(C) -> np.ndarray:
    """Euclidean projection of every column (16 Pauli coefficients c, the
    observable O = sum_a c_a P_a) onto the operator-norm ball ||O|| <= 1:
    clip the eigenvalues to [-1, 1] (Frobenius projection; ||O||_F^2 =
    4 ||c||^2).  Columns already inside are returned unchanged."""
    C = np.asarray(C, float).copy()
    if C.size == 0:
        return C
    O = np.einsum('ak,aij->kij', C, _P2)
    ev, V = np.linalg.eigh(O)
    out = np.abs(ev).max(axis=1) > 1.0 + 1e-12
    if np.any(out):
        evc = np.clip(ev[out], -1.0, 1.0)
        Vo = V[out]
        O2 = np.einsum('kij,kj,klj->kil', Vo, evc, Vo.conj())
        C[:, out] = np.einsum('aij,kji->ak', _P2, O2).real / 4.0
    return C


def operator_norms(S) -> np.ndarray:
    O = np.einsum('ak,aij->kij', np.asarray(S, float), _P2)
    return np.abs(np.linalg.eigvalsh(O)).max(axis=1)


def frame_from_free(F) -> np.ndarray:
    """(16, m-1) free columns -> 16 x m frame, identity first, columns projected."""
    return np.hstack([_E_II, project_columns_opnorm(F)])


def product_frame(S1) -> np.ndarray:
    """2-qubit frame S1 (x) S1 (16 x k^2) of a single-qubit frame (4 x k)."""
    S1 = np.asarray(S1, float)
    return np.kron(S1, S1)


def pad_frame(S, m: int) -> np.ndarray:
    """Pad a 2-qubit frame to m columns by duplicating its free columns
    cyclically (the certificate of the original frame still works, so the
    rate cannot increase), or trim it to its first m columns."""
    S = np.asarray(S, float)
    k = S.shape[1]
    if k >= m:
        return S[:, :m].copy()
    extra = [S[:, 1 + (i % (k - 1))] for i in range(m - k)]
    return np.hstack([S, np.array(extra).T])


# ---------------------------------------------------------------------------
#  The lattice point
# ---------------------------------------------------------------------------
class DimerProblem:
    """Generators of one lattice point for the dimer frame.

    B      : 16 x 16 Schroedinger bond Lindbladian (build_bond_lindbladian at
             the lattice's dim; one-site shares 1/(2 dim) included)
    dim    : 1 (chain) or 2 (square lattice)
    tiling : see TILINGS (default: chain / brick)
    mode   : default objective, 'split' (intra + cuts, separate certificates,
             the requested structure) or 'joint' (one patch certificate per cut)
    """

    def __init__(self, B, dim: int, tiling: str | None = None,
                 mode: str = 'split'):
        B = np.asarray(B, float)
        self.B = B
        self.dim = int(dim)
        if self.dim not in TILINGS:
            raise ValueError(f'dim must be one of {sorted(TILINGS)}')
        self.tiling = tiling or DEFAULT_TILING[self.dim]
        if self.tiling not in TILINGS[self.dim]:
            raise ValueError(f'tiling {self.tiling!r} not defined for dim {self.dim}')
        self.symmetric = is_swap_symmetric(B)
        if self.dim >= 2 and not self.symmetric:
            raise ValueError('dim >= 2 tilings orient the vertical bonds '
                             'arbitrarily: they need a swap-symmetric bond')
        if mode not in ('split', 'joint'):
            raise ValueError("mode must be 'split' or 'joint'")
        self.mode = mode
        self.n_bonds = 2 * self.dim                    # bonds per dimer
        self.A_in = B.T.copy()
        self.cut_w = dict(TILINGS[self.dim][self.tiling])
        self.A_cut = {c: embed_bond(B, CUT_SITES[c]).T.copy() for c in self.cut_w}
        kappa = 1.0 / (2.0 * (self.n_bonds - 1))
        I16 = np.eye(NQ)
        pair = np.kron(self.A_in, I16) + np.kron(I16, self.A_in)
        self.A_joint = {c: A + kappa * pair for c, A in self.A_cut.items()}

    def terms(self, mode: str | None = None):
        """[(name, kind, A, omega)]: kind 'S' (D = S) or 'SS' (D = S (x) S);
        the objective is sum omega * mu*(D, A) (per-bond average)."""
        mode = mode or self.mode
        nb = float(self.n_bonds)
        if mode == 'split':
            out = [('in', 'S', self.A_in, 1.0 / nb)]
            out += [(f'cut_{c}', 'SS', self.A_cut[c], w / nb)
                    for c, w in self.cut_w.items()]
            return out
        if mode == 'joint':
            return [(f'joint_{c}', 'SS', self.A_joint[c], w / nb)
                    for c, w in self.cut_w.items()]
        raise ValueError(mode)


# ---------------------------------------------------------------------------
#  Rate evaluation: closed form (basis) and per-column LP (any frame)
# ---------------------------------------------------------------------------
def _closed_cols(H):
    dg = np.diag(H)
    return dg + (np.abs(H).sum(axis=0) - np.abs(dg))


def column_lp(D, y, j):
    """Column j of the certificate LP:
        min g + sum (h+ + h-)  s.t.  D (g e_j + h+ - h-) = y,  h+-_j = 0.
    Returns (value, h, w): h the certificate column (h_j = g), w the equality
    dual (d value / d y), i.e. the witness of the envelope gradient."""
    n, M = D.shape
    A_eq = np.hstack([D[:, [j]], D, -D])
    c = np.ones(1 + 2 * M)
    bounds = [(None, None)] + [(0.0, None)] * (2 * M)
    bounds[1 + j] = (0.0, 0.0)
    bounds[1 + M + j] = (0.0, 0.0)
    r = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method='highs')
    if r.status != 0:
        return np.inf, None, None
    x = r.x
    h = x[1:1 + M] - x[1 + M:]
    h[j] = x[0]
    w = np.asarray(r.eqlin.marginals, float)
    return float(r.fun), h, w


def column_lps(D, A):
    """All columns: (cols, H, W) or (inf-array, None, None) if any fails."""
    D = np.asarray(D, float)
    n, M = D.shape
    if np.linalg.matrix_rank(D) < n:
        return np.full(M, np.inf), None, None
    Y = np.asarray(A, float) @ D
    cols = np.empty(M)
    H = np.empty((M, M))
    W = np.empty((n, M))
    for j in range(M):
        v, h, w = column_lp(D, Y[:, j], j)
        if not np.isfinite(v):
            return np.full(M, np.inf), None, None
        cols[j], H[:, j], W[:, j] = v, h, w
    return cols, H, W


def evaluate(S, prob: DimerProblem, mode: str | None = None, *,
             method: str = 'auto', keep: bool = False):
    """Objective F(S) = sum_t omega_t mu*(D_t(S), A_t).

    method: 'closed' (m = 16 only, D^{-1} A D), 'lp' (per-column LPs, any m),
            'auto' (closed for m = 16).  Returns (F, terms); with keep=True
            each term dict carries what bundle_polish needs for gradients."""
    S = np.asarray(S, float)
    m = S.shape[1]
    closed = method == 'closed' or (method == 'auto' and m == NQ)
    Sinv = None
    if closed:
        if m != NQ:
            raise ValueError('closed form needs a 16-column (basis) frame')
        try:
            if np.linalg.cond(S) > _COND_MAX_S:
                return np.inf, []
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return np.inf, []
    total = 0.0
    terms = []
    for name, kind, A, om in prob.terms(mode):
        D = S if kind == 'S' else np.kron(S, S)
        if closed:
            Dinv = Sinv if kind == 'S' else np.kron(Sinv, Sinv)
            H = Dinv @ (A @ D)
            cols = _closed_cols(H)
            aux = Dinv
        else:
            cols, H, aux = column_lps(D, A)
        val = float(np.max(cols))
        if not np.isfinite(val):
            return np.inf, []
        total += om * val
        t = dict(name=name, kind=kind, omega=om, value=val)
        if keep:
            t.update(cols=cols, H=H, aux=aux, A=A, closed=closed)
        terms.append(t)
    return float(total), terms


def term_parts(terms) -> dict:
    return {t['name']: t['value'] for t in terms}


def column_grad(S, term, j) -> np.ndarray:
    """Envelope (sub)gradient of column j's rate of one term w.r.t. S (16 x m):
    d mu_j / dD = (A^T w) e_j^T - w h_j^T, w = D^{-T} sign(h_j) (closed form)
    or the LP equality dual, chained through D = S or D = S (x) S."""
    S = np.asarray(S, float)
    H, A = term['H'], term['A']
    h = H[:, j]
    if term['closed']:
        s = np.sign(h)
        s[j] = 1.0
        w = term['aux'].T @ s
    else:
        w = term['aux'][:, j]
    gD = -np.outer(w, h)
    gD[:, j] += A.T @ w
    if term['kind'] == 'S':
        return gD
    n_s, m = S.shape
    T = gD.reshape(n_s, n_s, m, m)
    return (np.einsum('abij,bj->ai', T, S) + np.einsum('baji,bj->ai', T, S))


# ---------------------------------------------------------------------------
#  Stage 2: rotate + weight (convex weights for fixed U)
# ---------------------------------------------------------------------------
_M_S = np.vstack([np.zeros((1, NQ - 1)), np.eye(NQ - 1)])           # log v = M u
_M_SS = (np.kron(_M_S, np.ones((NQ, 1))) + np.kron(np.ones((NQ, 1)), _M_S))


class _WeightProblem:
    """min_u sum_t omega_t max_P [A'_PP + sum_Q |A'_QP| v_P / v_Q],
    log v = M_t u, u in [log floor, 0]^15 (convex: max / sum of sums of
    exponentials of linear functions)."""

    def __init__(self, rotated_terms, floor: float = WEIGHT_FLOOR):
        self.lo = float(np.log(floor))
        self.items = []
        for kind, A, om in rotated_terms:
            absA = np.abs(A)
            np.fill_diagonal(absA, 0.0)
            M = _M_S if kind == 'S' else _M_SS
            self.items.append((M, absA, np.diag(A).copy(), float(om)))
        self.omega = np.array([it[3] for it in self.items])

    def cols(self, u, k):
        M, absA, dg, _ = self.items[k]
        lv = M @ u
        R = absA * np.exp(lv[None, :] - lv[:, None])
        return dg + R.sum(axis=0), R, M

    def value(self, u) -> float:
        u = np.clip(u, self.lo, 0.0)
        return float(sum(it[3] * self.cols(u, k)[0].max()
                         for k, it in enumerate(self.items)))

    def solve(self, starts):
        nu, nt = NQ - 1, len(self.items)

        def con(z):
            return np.concatenate([z[nu + k] - self.cols(z[:nu], k)[0]
                                   for k in range(nt)])

        def con_jac(z):
            blocks = []
            for k in range(nt):
                _, R, M = self.cols(z[:nu], k)
                r = R.sum(axis=0)
                dcol = r[:, None] * M - R.T @ M
                E = np.zeros((dcol.shape[0], nt))
                E[:, k] = 1.0
                blocks.append(np.hstack([-dcol, E]))
            return np.vstack(blocks)

        obj = lambda z: float(self.omega @ z[nu:])                   # noqa: E731
        obj_jac = lambda z: np.concatenate([np.zeros(nu), self.omega])  # noqa: E731
        bounds = [(self.lo, 0.0)] * nu + [(None, None)] * nt
        best_v, best_u = np.inf, np.zeros(nu)
        for u0 in starts:
            u0 = np.clip(np.asarray(u0, float), self.lo, 0.0)
            t0 = [self.cols(u0, k)[0].max() for k in range(nt)]
            z0 = np.concatenate([u0, t0])
            try:
                r = minimize(obj, z0, jac=obj_jac, method='SLSQP', bounds=bounds,
                             constraints=[{'type': 'ineq', 'fun': con,
                                           'jac': con_jac}],
                             options=dict(maxiter=300, ftol=1e-12))
                u = np.clip(r.x[:nu], self.lo, 0.0)
            except Exception:                               # noqa: BLE001
                u = u0
            for cand in (u0, u):
                v = self.value(cand)
                if v < best_v:
                    best_v, best_u = v, cand
        return best_v, best_u


def rotate_terms(prob: DimerProblem, mode, R):
    out = []
    RR = None
    for _, kind, A, om in prob.terms(mode):
        if kind == 'S':
            out.append((kind, R.T @ A @ R, om))
        else:
            if RR is None:
                RR = np.kron(R, R)
            out.append((kind, RR.T @ A @ RR, om))
    return out


def rw_frame(theta, u) -> np.ndarray:
    R = pauli_liouville_unitary(su4(theta))
    return R @ np.diag(np.exp(np.concatenate([[0.0], u])))


def rotate_weight_search(prob: DimerProblem, mode=None, *, theta_starts=None,
                         n_random: int = 3, maxfev: int = 400,
                         simplex_step: float = 0.2, u_starts=(),
                         floor: float = WEIGHT_FLOOR, seed: int = 0,
                         verbose: bool = False):
    """Outer Nelder-Mead over U = su4(theta), inner exact convex weights.
    The first start is theta = 0 (U = 1: the 15-weight frame, a convex problem
    solved exactly); then n_random random theta.  Inside a Nelder-Mead run the
    convex solve is warm-started from the best weights so far; the first call
    of every start also tries u = 0, u = log(floor)/2 and u_starts.
    Returns (value, S, info)."""
    rng = np.random.default_rng(seed)
    lo = float(np.log(floor))
    base_u = [np.zeros(NQ - 1), np.full(NQ - 1, lo / 2)] + [np.asarray(u, float)
                                                           for u in u_starts]
    state = dict(u=base_u[0], best=(np.inf, None, None), nfev=0)

    def inner(theta, full=False):
        R = pauli_liouville_unitary(su4(theta))
        wp = _WeightProblem(rotate_terms(prob, mode, R), floor)
        v, u = wp.solve([state['u']] + (base_u if full else []))
        state['nfev'] += 1
        if v < state['best'][0]:
            state['best'] = (v, np.array(theta, float), u)
            state['u'] = u
        return v

    starts = [np.zeros(NQ - 1)] if theta_starts is None else \
        [np.asarray(t, float) for t in theta_starts]
    starts += [rng.normal(0.0, 0.3, NQ - 1) for _ in range(n_random)]
    v_id = inner(starts[0], full=True)
    for k, th0 in enumerate(starts):
        if k > 0:
            inner(th0, full=True)
        if maxfev > 0:
            simplex = np.vstack([th0, th0 + simplex_step * np.eye(NQ - 1)])
            minimize(inner, th0, method='Nelder-Mead',
                     options=dict(maxfev=maxfev, xatol=1e-6, fatol=1e-10,
                                  initial_simplex=simplex))
        if verbose:
            print(f'    rw start {k}: best {state["best"][0]:.6e}', flush=True)
    v, theta, u = state['best']
    S = rw_frame(theta, u)
    return float(v), S, dict(theta=theta, u=u, weights_only=float(v_id),
                             nfev=state['nfev'])


# ---------------------------------------------------------------------------
#  Stages 3 / 4: bundle trust-region polish over the full frame
# ---------------------------------------------------------------------------
def bundle_polish(S, prob: DimerProblem, mode=None, *, method: str = 'auto',
                  n_iter: int = 150, radius: float = 0.05, band: float = 0.05,
                  max_cols: int = 96, shrink: float = 0.5, grow: float = 1.6,
                  radius_max: float = 0.3, tol: float = _TOL,
                  verbose: bool = False):
    """Trust-region bundle descent on F(S) = sum_t omega_t max_j mu_tj(S).

    Every near-binding column of every term (within `band` of that term's
    max, at most max_cols per term) is linearised with its envelope gradient.
    max_cols must cover the TIES: product-like / symmetric frames have tens
    of columns at the maximum (the cut term has 256), and a bundle that leaves
    tied columns out proposes steps that raise them, so nothing is accepted;
    the epigraph LP  min sum_t omega_t t_t  s.t.  mu_tj + <g_tj, dS> <= t_t,
    |dS| <= radius  picks the step; columns are projected back onto the
    operator-norm ball and the step is accepted on the TRUE objective.
    Returns (best_value, best_frame, n_accepted)."""
    S = np.asarray(S, float).copy()
    n_s, m = S.shape
    nfree = m - 1
    nvar = n_s * nfree
    F, terms = evaluate(S, prob, mode, method=method, keep=True)
    if not np.isfinite(F):
        return float(F), S, 0
    best_F, best_S, n_acc = F, S.copy(), 0
    nt = len(terms)
    om = np.array([t['omega'] for t in terms])
    for it in range(n_iter):
        rows, rhs = [], []
        for k, t in enumerate(terms):
            cols = t['cols']
            order = np.argsort(-cols)
            thresh = t['value'] - band * max(abs(t['value']), 1e-3)
            for j in order[:max_cols]:
                if cols[j] < thresh:
                    break
                g = column_grad(S, t, int(j))[:, 1:].ravel()
                e = np.zeros(nt)
                e[k] = -1.0
                rows.append(np.concatenate([g, e]))
                rhs.append(-cols[j])
        c = np.concatenate([np.zeros(nvar), om])
        res = linprog(c, A_ub=np.array(rows), b_ub=np.array(rhs),
                      bounds=[(-radius, radius)] * nvar + [(None, None)] * nt,
                      method='highs')
        if not res.success:
            radius *= shrink
            if radius < 1e-7:
                break
            continue
        delta = res.x[:nvar].reshape(n_s, nfree)
        pred = float(om @ res.x[nvar:])
        S_new = frame_from_free(S[:, 1:] + delta)
        F_new, terms_new = evaluate(S_new, prob, mode, method=method, keep=True)
        if np.isfinite(F_new) and F_new < F - tol and \
                (pred >= F or (F - F_new) >= 0.05 * (F - pred)):
            S, F, terms = S_new, F_new, terms_new
            n_acc += 1
            if F < best_F:
                best_F, best_S = F, S.copy()
            radius = min(radius * grow, radius_max)
        else:
            radius *= shrink
            if radius < 1e-7:
                break
        if verbose and (it + 1) % 25 == 0:
            print(f'    bundle {it + 1}: F={F:.6e} radius={radius:.3g} '
                  f'rows={len(rows)}', flush=True)
        if F <= tol:
            break
    return float(best_F), best_S, n_acc


# ---------------------------------------------------------------------------
#  Stage 3: smoothed L-BFGS continuation (basis frames, closed form)
# ---------------------------------------------------------------------------
#  Why not only the bundle: at product-like frames the certificate H is
#  SPARSE, every zero entry H_kj puts a kink |H_kj| into its column's rate,
#  and the envelope subgradient (sign 0 = 0) cannot see it: the linear model
#  predicts descent while the true rate rises at O(|step|) -- the bundle then
#  accepts nothing.  The smoothed objective
#      F_eps,beta = sum_t omega_t LSE_beta_j ( H_jj + sum_{k != j} sqrt(H_kj^2 + eps^2) )
#  (LSE_beta = (1/beta) log sum exp(beta x) >= max) is smooth, >= F, and -> F;
#  its gradient is closed form:  dF = <Sigma, dH>,  dH = D^{-1}(A dD - dD H)
#      => G_D = A^T Omega - Omega H^T,   Omega = D^{-T} Sigma,
#  Sigma_kj = pi_j * (H_kj / sqrt(H_kj^2 + eps^2), 1 on the diagonal),
#  pi = softmax(beta mu~).  Columns leave the operator-norm ball only through
#  a quadratic penalty rho * sum_k (||O_k|| - 1)_+^2; every iterate is judged
#  by the EXACT objective of its projected frame.
SMOOTH_SCHEDULE = ((20.0, 1e-2), (100.0, 1e-3), (500.0, 1e-4), (3000.0, 1e-5))


def _lse(x, beta):
    xm = x.max()
    e = np.exp(beta * (x - xm))
    s = e.sum()
    return xm + np.log(s) / beta, e / s


def smoothed_value_grad(S, prob: DimerProblem, mode, beta, eps, rho=100.0):
    """(F_eps,beta(S) + penalty, gradient w.r.t. S (16 x 16)); basis frames."""
    S = np.asarray(S, float)
    try:
        if np.linalg.cond(S) > _COND_MAX_S:
            return 1e6, np.zeros_like(S)
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return 1e6, np.zeros_like(S)
    total = 0.0
    G = np.zeros_like(S)
    for _, kind, A, om in prob.terms(mode):
        if kind == 'S':
            D, Dinv = S, Sinv
        else:
            D, Dinv = np.kron(S, S), np.kron(Sinv, Sinv)
        H = Dinv @ (A @ D)
        phi = np.sqrt(H * H + eps * eps)
        dgi = np.diag_indices_from(H)
        phi[dgi] = H[dgi]
        mu = phi.sum(axis=0)
        val, pi = _lse(mu, beta)
        sig = H / np.sqrt(H * H + eps * eps)
        sig[dgi] = 1.0
        Omega = Dinv.T @ (sig * pi[None, :])
        GD = A.T @ Omega - Omega @ H.T
        if kind == 'S':
            GS = GD
        else:
            n_s, m = S.shape
            T = GD.reshape(n_s, n_s, m, m)
            GS = (np.einsum('abij,bj->ai', T, S) + np.einsum('baji,bj->ai', T, S))
        total += om * val
        G += om * GS
    # operator-norm penalty on the free columns
    O = np.einsum('ak,aij->kij', S[:, 1:], _P2)
    ev, V = np.linalg.eigh(O)
    idx = np.abs(ev).argmax(axis=1)
    lam = ev[np.arange(ev.shape[0]), idx]
    over = np.abs(lam) - 1.0
    hit = over > 0
    if np.any(hit):
        total += rho * float(np.sum(over[hit] ** 2))
        v = V[np.arange(V.shape[0]), :, idx]                 # (m-1, 4) top vectors
        dn = np.einsum('ki,aij,kj->ak', v.conj(), _P2, v).real * np.sign(lam)[None, :]
        G[:, 1:] += (2.0 * rho * np.where(hit, over, 0.0))[None, :] * dn
    G[:, 0] = 0.0
    return float(total), G


def smooth_polish(S, prob: DimerProblem, mode=None, *, schedule=SMOOTH_SCHEDULE,
                  maxiter: int = 300, rho: float = 100.0, verbose: bool = False):
    """L-BFGS on the smoothed objective through the (beta, eps) schedule,
    every stage started from the previous one; returns the best EXACT value
    over the start and the projected end frame of every stage:
    (best_value, best_frame, per-stage exact values)."""
    S = np.asarray(S, float).copy()
    if S.shape[1] != NQ:
        raise ValueError('smooth_polish needs a 16-column (basis) frame')
    best_F, _ = evaluate(S, prob, mode)
    best_S = S.copy()
    x = S[:, 1:].ravel().copy()
    stages = []

    for beta, eps in schedule:
        def fg(xv):
            Sx = np.hstack([_E_II, xv.reshape(NQ, NQ - 1)])
            v, g = smoothed_value_grad(Sx, prob, mode, beta, eps, rho)
            return v, g[:, 1:].ravel()
        r = minimize(fg, x, jac=True, method='L-BFGS-B',
                     options=dict(maxiter=maxiter, ftol=1e-13, gtol=1e-10))
        x = r.x
        Sp = frame_from_free(x.reshape(NQ, NQ - 1))
        Fp, _ = evaluate(Sp, prob, mode)
        stages.append(Fp)
        if verbose:
            print(f'    smooth beta={beta:g} eps={eps:g}: smoothed {r.fun:.6e} '
                  f'exact {Fp:.6e} ({r.nit} it)', flush=True)
        if np.isfinite(Fp) and Fp < best_F:
            best_F, best_S = Fp, Sp.copy()
    return float(best_F), best_S, stages


def kicked_frames(S, n: int, scale: float, rng) -> list:
    """n random perturbations of a basis frame: an entangling rotation
    exp(-i theta.P) with |theta| ~ scale, then additive noise of the same
    size on the free entries; columns projected."""
    out = []
    for _ in range(n):
        R = pauli_liouville_unitary(su4(rng.normal(0.0, scale, NQ - 1)))
        F = (R @ S)[:, 1:] + rng.normal(0.0, scale / 4.0, (NQ, NQ - 1))
        out.append(frame_from_free(F))
    return out


# ---------------------------------------------------------------------------
#  Stored single-qubit (product) frames and rates
# ---------------------------------------------------------------------------
_S_KEY = re.compile(r'(?:S_heis_|S_|famS_)(\d+)$')
_R_KEY = re.compile(r'(?:rate_heis_|rate_)(\d+)$')


def stored_product_frames(dirs, ix: int, iy: int):
    """Every single-qubit Heisenberg frame and rate stored for grid point
    (ix, iy) in pt_<ix>_<iy>*.npz under `dirs` (base / gopt / refine files:
    S_heis_<m>, rate_heis_<m>; seeded files: S_<m>, rate_<m>, famS_<m>).
    Returns ({m: [S1, ...]}, {m: best stored rate}).  Missing dirs skipped."""
    frames: dict = {}
    best: dict = {}
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.glob(f'pt_{ix:03d}_{iy:03d}*.npz')):
            if f.name.endswith('.tmp.npz'):
                continue
            try:
                z = np.load(f, allow_pickle=True)
            except Exception:                               # noqa: BLE001
                continue
            for k in z.files:
                ms = _S_KEY.fullmatch(k)
                if ms:
                    m = int(ms.group(1))
                    S1 = np.asarray(z[k], float)
                    if S1.shape == (4, m) and np.all(np.isfinite(S1)) and \
                            np.allclose(S1[:, 0], [1, 0, 0, 0], atol=1e-9):
                        frames.setdefault(m, []).append(S1)
                mr = _R_KEY.fullmatch(k)
                if mr:
                    try:
                        v = float(z[k])
                    except (TypeError, ValueError):
                        continue
                    m = int(mr.group(1))
                    if np.isfinite(v) and v < best.get(m, np.inf):
                        best[m] = v
    return frames, best


def default_seed_dirs(model: str, dim: int, model_dim: int):
    """Stored per-point product-frame folders of the existing pipelines at
    the SAME lattice dimension (frames of another dim are never mixed in)."""
    if dim == model_dim:
        return [f'results_{model}_rate/{model}',
                f'results_{model}_rate/{model}_seeded',
                f'results_{model}_seeded/{model}_seeded',
                f'results_{model}_seeded/{model}_seeded_d12']
    return [f'results_{model}_rate_dim{dim}/{model}_seeded',
            f'results_{model}_rate_dim{dim}/{model}_seeded_d12']


# ---------------------------------------------------------------------------
#  The full seeded search
# ---------------------------------------------------------------------------
def _dedup(cands, tol=1e-9):
    out = []
    for name, v, S in sorted(cands, key=lambda c: c[1]):
        if all(abs(v - w) > tol or S.shape != T.shape or
               np.max(np.abs(S - T)) > 1e-8 for _, w, T in out):
            out.append((name, v, S))
    return out


def minimize_dimer_rate(prob: DimerProblem, *, mode=None, m_ext=None,
                        product_seeds=(), dimer_seeds=(), fresh_d4: bool = True,
                        rw_random: int = 3, rw_maxfev: int = 400,
                        n_polish: int = 3, smooth_iters: int = 300,
                        n_kicks: int = 2, kick_scales=(0.02, 0.1),
                        bundle_iters: int = 40,
                        ext_iters: int = 20, certify: bool = True,
                        seed: int = 0, verbose: bool = False):
    """min_S F(S) over 2-qubit frames, m = 16 (closed form) and optionally
    m = m_ext > 16 (LP).

    product_seeds : single-qubit 4-column frames (stored d4 optima); every
                    S1 enters as S1 (x) S1.  Single-qubit frames with more
                    columns are ignored (they would give m = k^2 > 16).
    dimer_seeds   : 2-qubit frames (16 x k, identity first), fit to 16 / m_ext.
    fresh_d4      : if no product seed is given, run the single-qubit
                    minimize_rate_global(B^T, 4) for one (and its value).
    Returns (S16, F16, info); info['ext'] holds the m_ext result if asked."""
    mode = mode or prob.mode
    t0 = time.perf_counter()
    A_bond = prob.A_in
    info: dict = dict(mode=mode, dim=prob.dim, tiling=prob.tiling)

    # ---- 1. seeds ------------------------------------------------------
    cands = []

    def add(name, S):
        v, _ = evaluate(S, prob, mode)
        if np.isfinite(v):
            cands.append((name, float(v), np.asarray(S, float)))

    add('pauli', np.eye(NQ))
    lib1 = seed_library(4, A_bond)
    for name, S1 in lib1.items():
        add(f'prod:{name}', product_frame(S1))
    prod = [np.asarray(S1, float) for S1 in product_seeds
            if np.asarray(S1).shape == (4, 4)]
    if not prod and fresh_d4:
        S1, mu1, _ = minimize_rate_global(A_bond, 4, seed=seed)
        info['fresh_d4'] = float(mu1)
        prod = [S1]
    for i, S1 in enumerate(prod):
        add(f'prod:stored{i}', product_frame(S1))
    for i, S in enumerate(dimer_seeds):
        S = np.asarray(S, float)
        if S.shape[0] == NQ:
            add(f'dimer{i}', pad_frame(S, NQ))
    cands = _dedup(cands)
    info['mu_seed'] = cands[0][1]
    info['seed_name'] = cands[0][0]
    info['n_seeds'] = len(cands)
    if verbose:
        print(f'  seeds ({len(cands)}): best {cands[0][0]} F={cands[0][1]:.6e}',
              flush=True)

    # ---- 2. rotate + weight ---------------------------------------------
    # the single-qubit rescaled-Pauli optimum, as 15 two-qubit log-weights
    u_starts = []
    if 'weights' in lib1:
        v2 = np.diag(product_frame(lib1['weights']))[1:]
        u_starts.append(np.log(np.clip(v2, WEIGHT_FLOOR, 1.0)))
    v_rw, S_rw, rw_info = rotate_weight_search(
        prob, mode, n_random=rw_random, maxfev=rw_maxfev, u_starts=u_starts,
        seed=seed, verbose=verbose)
    info['mu_weights16'] = rw_info['weights_only']
    info['mu_rw'] = v_rw
    info['rw_theta'] = rw_info['theta']
    info['rw_u'] = rw_info['u']
    cands = _dedup(cands + [('rotate+weight', v_rw, S_rw)])
    if verbose:
        print(f'  rotate+weight: F={v_rw:.6e} (U = 1: {rw_info["weights_only"]:.6e})',
              flush=True)

    # ---- 3. smoothed L-BFGS (+ short exact bundle) from the best candidates
    #         and from random kicks of the best (sparse-H kinks trap descent
    #         methods exactly at the symmetric product-like frames) -----------
    rng = np.random.default_rng(seed + 7)
    best_F, best_S, best_name = cands[0][1], cands[0][2], cands[0][0]
    starts = [(name, S) for name, _, S in cands[:n_polish]]
    for scale in kick_scales:
        starts += [(f'{cands[0][0]}~kick{scale:g}', K)
                   for K in kicked_frames(cands[0][2], n_kicks, scale, rng)]
    info['mu_smooth'] = best_F
    for name, S in starts:
        if best_F <= _TOL:
            break
        v0, _ = evaluate(S, prob, mode)
        vs, Ss = (v0, S)
        if smooth_iters > 0:
            vs, Ss, _ = smooth_polish(S, prob, mode, maxiter=smooth_iters,
                                      verbose=verbose)
        vb, Sb, nacc = (vs, Ss, 0)
        if bundle_iters > 0 and vs > _TOL:
            vb, Sb, nacc = bundle_polish(Ss, prob, mode, n_iter=bundle_iters,
                                         verbose=verbose)
        if verbose:
            print(f'  polish {name}: {v0:.6e} -> smooth {vs:.6e} -> bundle '
                  f'{vb:.6e} ({nacc} steps)', flush=True)
        if vs < info['mu_smooth']:
            info['mu_smooth'] = vs
        if vb < best_F:
            best_F, best_S = vb, Sb
            best_name = f'{name}+smooth' + ('+bundle' if vb < vs else '')
    info['mu_polish'] = best_F
    info['label'] = best_name

    # ---- per-term values, both objectives, certification -----------------
    F_split, t_split = evaluate(best_S, prob, 'split')
    F_joint, t_joint = evaluate(best_S, prob, 'joint')
    info['rate_split'] = F_split
    info['rate_joint'] = F_joint
    info['parts_split'] = term_parts(t_split)
    info['parts_joint'] = term_parts(t_joint)
    if certify:
        F_ref, _ = evaluate(best_S, prob, mode, method='lp')
        info['mu_reference'] = F_ref
        if np.isfinite(F_ref) and F_ref > best_F + 1e-7:
            best_F = F_ref
    info['op_norm_max'] = float(operator_norms(best_S)[1:].max())
    info['time16'] = time.perf_counter() - t0

    # ---- 4. optional m > 16 ---------------------------------------------
    if m_ext is not None and m_ext > NQ:
        t1 = time.perf_counter()
        S_ext = pad_frame(best_S, m_ext)
        F0, _ = evaluate(S_ext, prob, mode, method='lp')
        Fe, Se, nacc = bundle_polish(S_ext, prob, mode, method='lp',
                                     n_iter=ext_iters, verbose=verbose)
        Fs, ts = evaluate(Se, prob, 'split', method='lp')
        Fj, tj = evaluate(Se, prob, 'joint', method='lp')
        info['ext'] = dict(m=m_ext, S=Se, value=Fe, padded=F0, steps=nacc,
                           rate_split=Fs, rate_joint=Fj,
                           parts_split=term_parts(ts), parts_joint=term_parts(tj),
                           time=time.perf_counter() - t1)
        if verbose:
            print(f'  m={m_ext}: padded {F0:.6e} -> {Fe:.6e} ({nacc} steps, '
                  f'{info["ext"]["time"]:.0f}s)', flush=True)
    info['time'] = time.perf_counter() - t0
    return best_S, float(best_F), info
