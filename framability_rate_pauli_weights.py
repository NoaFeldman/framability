r"""
framability_rate_pauli_weights.py -- rate of the REWEIGHTED Pauli frame as a
function of the smallest allowed weight w_min (the log(poly N) relaxation of
the Heisenberg-picture normalisation, paper Sec. "Normalization of
Heisenberg-picture frame elements").

Frame
-----
One-site elements {I, w_X X, w_Y Y, w_Z Z}, w_I = 1, w_min <= w_P <= 1; a
k-site element carries the product of its one-site weights.  In code
language this is S = diag(1, w_X, w_Y, w_Z), which satisfies the column
constraint |c_I| + ||c||_2 <= 1 of optimize_framability (a subfamily of the
d_ext = 4 search).  Several weight CLASSES may be used (one per atomic
species, disordered Dicke); the pair (s, t) then uses S_s (x) S_t.

Rate (closed form, the Pauli strings are a basis so M is unique)
----------------------------------------------------------------
A = L^T acts on observables, A[f, e] = coefficient of P_f in L^dag(P_e).
With log-weights y_e (y = sum of the one-site log-weights of the string),

    mu_e(y) = A_ee + sum_{f != e} |A_fe| exp(y_e - y_f),

and mu = max_e mu_e (the identity column is 0 for trace-preserving L).  Each
mu_e is a sum of exponentials of linear forms of the one-site log-weights x,
hence CONVEX in x; the pair average F = sum_p pw_p max_e mu_pe is convex too.
So

    F*(w_min) = min { F(x) : log w_min <= x <= 0 }

is a convex program (solved here by SLSQP on the epigraph with analytic
Jacobians, warm-started along a decreasing w_min sweep plus a few restarts),
and F*(w_min) is non-increasing and convex in log w_min.

Zero certificate.  F = 0 iff every non-identity column of every pair is <= 0,
i.e. iff the MIN-MAX value

    s*(w_min) = min_x max_{p, e != I} mu_pe(x)

is <= 0.  s* is computed by a second epigraph program; s* < 0 is a certified
zero with margin -s*.  w_star = the largest w_min with s*(w_min) <= 0
(bisection in log w_min), i.e. the smallest weight a zero-rate simulation
needs.  With w_min = N^{-c} it is the system size N = w_star^{-1/c} above
which the reweighted Pauli frame has rate zero at this point.

Limit w_min -> 0.  Terms with y_e < y_f vanish like w^(y_f - y_e) and terms
with y_e > y_f must be absent, so F* approaches its limit F_inf as a power
law in w_min (exponentially in log 1/w_min).  F_inf = 0 with F* > 0 at every
finite w_min needs a non-identity column whose limit is exactly 0 (a
conserved or structurally undamped column); generically the sweep either
hits 0 at a finite w_star or plateaus at F_inf > 0.

Cross-check.  The final weights of every sweep level can be re-evaluated with
the generic per-column LP (framability_rate.generator_log_norm on the frame
D = S_s (x) S_t): reference_rate().

Public API
----------
WeightModel(As, pw, pair_classes, n_cls)
    .F(x), .smax(x), .columns(x)
solve_level(model, log_wmin, starts)      -> (F, x)     epigraph of F
solve_minmax(model, log_wmin, starts)     -> (s, x)     epigraph of s
sweep(model, w_grid, ...)                 -> dict of arrays (F, s, x per w)
w_star(model, w_grid, s_grid, x_grid)     -> largest w with s <= 0
reference_rate(model, x)                  -> F from the generic LP
frames_of(model, x)                       -> list of 4x4 S per class
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

PAULI_WEIGHTS_VERSION = '1.0-pauli-weight-floor'
_NQ = 4                       # one-site Pauli basis I, X, Y, Z
_EXP_CLIP = 700.0


class WeightModel:
    """Pairs of 2-qubit Heisenberg generators with weight classes.

    As            list of 16x16 real generators on OBSERVABLES (A = L^T),
                  2-qubit Pauli index e = 4 i + j (i on qubit a, j on qubit b)
    pw            pair weights (sum 1)
    pair_classes  list of (class of qubit a, class of qubit b)
    n_cls         number of weight classes; x has 3 n_cls entries
                  (log w_X, log w_Y, log w_Z of class 0, then class 1, ...)
    """

    def __init__(self, As, pw, pair_classes, n_cls: int):
        self.As = [np.asarray(A, float) for A in As]
        for A in self.As:
            if A.shape != (16, 16):
                raise ValueError(f'expected 16x16 generators, got {A.shape}')
        self.pw = np.asarray(pw, float)
        self.P = len(self.As)
        if len(pair_classes) != self.P or self.pw.shape != (self.P,):
            raise ValueError('As, pw and pair_classes must have the same length')
        self.n_cls = int(n_cls)
        self.n_x = 3 * self.n_cls
        self.d = [np.diag(A).copy() for A in self.As]
        self.C = []
        for A in self.As:
            C = np.abs(A)
            np.fill_diagonal(C, 0.0)
            self.C.append(C)
        # trace preservation: the identity column must be zero
        for p, A in enumerate(self.As):
            if np.max(np.abs(A[:, 0])) > 1e-10:
                raise ValueError(f'pair {p}: identity column nonzero '
                                 f'({np.max(np.abs(A[:, 0])):.2e}); not trace preserving')
        # y = Y_p x for every pair
        self.Y = []
        for a, b in pair_classes:
            Y = np.zeros((16, self.n_x))
            for i in range(_NQ):
                for j in range(_NQ):
                    e = _NQ * i + j
                    if i > 0:
                        Y[e, 3 * int(a) + i - 1] += 1.0
                    if j > 0:
                        Y[e, 3 * int(b) + j - 1] += 1.0
            self.Y.append(Y)

    # -- columns -----------------------------------------------------------
    def _K(self, p, x):
        y = self.Y[p] @ x
        diff = np.clip(y[None, :] - y[:, None], -_EXP_CLIP, _EXP_CLIP)   # [f, e] = y_e - y_f
        return self.C[p] * np.exp(diff)

    def columns(self, x):
        """(P, 16) column rates mu_pe(x) (column 0 = identity = 0)."""
        x = np.asarray(x, float)
        return np.array([self.d[p] + self._K(p, x).sum(axis=0) for p in range(self.P)])

    def columns_and_grad(self, x):
        """columns (P, 16) and d mu_pe / dx (P, 16, n_x)."""
        x = np.asarray(x, float)
        cols = np.empty((self.P, 16))
        grads = np.empty((self.P, 16, self.n_x))
        for p in range(self.P):
            K = self._K(p, x)
            cs = K.sum(axis=0)
            cols[p] = self.d[p] + cs
            G = np.diag(cs) - K.T                 # d mu_e / d y_f
            grads[p] = G @ self.Y[p]
        return cols, grads

    def F(self, x) -> float:
        """Pair-averaged rate sum_p pw_p max(0, max_e mu_pe) (exact)."""
        cols = self.columns(x)
        return float(self.pw @ np.maximum(cols.max(axis=1), 0.0))

    def smax(self, x) -> float:
        """max over pairs and NON-identity columns (zero certificate if <= 0)."""
        return float(self.columns(x)[:, 1:].max())


def frames_of(model: WeightModel, x) -> list:
    """4x4 single-qubit frames S_c = diag(1, w_X, w_Y, w_Z) per class."""
    x = np.asarray(x, float).reshape(model.n_cls, 3)
    return [np.diag(np.concatenate([[1.0], np.exp(xc)])) for xc in x]


def reference_rate(model: WeightModel, x, pair_classes) -> float:
    """F of the weights x re-evaluated with the generic per-column LP
    (framability_rate.generator_log_norm on D = S_a (x) S_b)."""
    from framability_rate import generator_log_norm
    S = frames_of(model, x)
    vals = [generator_log_norm(np.kron(S[a], S[b]), model.As[p])
            for p, (a, b) in enumerate(pair_classes)]
    return float(model.pw @ np.maximum(np.array(vals), 0.0))


# ---------------------------------------------------------------------------
#  Epigraph programs
# ---------------------------------------------------------------------------
def _clip_x(x, lo):
    return np.clip(np.asarray(x, float), lo, 0.0)


def _slsqp(fun, jac, z0, cons, bounds, maxiter):
    try:
        res = minimize(fun, z0, jac=jac, method='SLSQP', constraints=cons,
                       bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-12})
        return res.x
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None


def solve_level(model: WeightModel, log_wmin: float, starts, *, maxiter: int = 500):
    """min_x F(x) on log_wmin <= x <= 0.  Epigraph: min pw . t,
    t_p >= mu_pe(x) (e != I), t_p >= 0.  Returns (F, x) of the best start
    (F re-evaluated exactly, so a failed solve can only lose, never lie)."""
    n, P = model.n_x, model.P
    best_F, best_x = np.inf, None
    for x0 in starts:
        x0 = _clip_x(x0, log_wmin)
        F0 = model.F(x0)
        if F0 < best_F:
            best_F, best_x = F0, x0
        cols0 = model.columns(x0)
        t0 = np.maximum(cols0[:, 1:].max(axis=1), 0.0)
        z0 = np.concatenate([x0, t0])

        def fun(z):
            return float(model.pw @ z[n:])

        def jac(z):
            g = np.zeros_like(z)
            g[n:] = model.pw
            return g

        def cfun(z):
            cols = model.columns(z[:n])[:, 1:]
            return (z[n:][:, None] - cols).ravel()

        def cjac(z):
            _, grads = model.columns_and_grad(z[:n])
            grads = grads[:, 1:, :]                       # (P, 15, n)
            J = np.zeros((P * 15, n + P))
            for p in range(P):
                J[p * 15:(p + 1) * 15, :n] = -grads[p]
                J[p * 15:(p + 1) * 15, n + p] = 1.0
            return J

        bounds = [(log_wmin, 0.0)] * n + [(0.0, None)] * P
        z = _slsqp(fun, jac, z0, [{'type': 'ineq', 'fun': cfun, 'jac': cjac}],
                   bounds, maxiter)
        if z is None or not np.all(np.isfinite(z)):
            continue
        x = _clip_x(z[:n], log_wmin)
        Fx = model.F(x)
        if Fx < best_F:
            best_F, best_x = Fx, x
    return best_F, best_x


def solve_minmax(model: WeightModel, log_wmin: float, starts, *, maxiter: int = 500):
    """min_x max_{p, e != I} mu_pe(x) on log_wmin <= x <= 0.
    Returns (s, x); s <= 0 certifies F = 0 (margin -s)."""
    n, P = model.n_x, model.P
    best_s, best_x = np.inf, None
    for x0 in starts:
        x0 = _clip_x(x0, log_wmin)
        s0 = model.smax(x0)
        if s0 < best_s:
            best_s, best_x = s0, x0
        z0 = np.concatenate([x0, [s0]])

        def fun(z):
            return float(z[n])

        def jac(z):
            g = np.zeros_like(z)
            g[n] = 1.0
            return g

        def cfun(z):
            return (z[n] - model.columns(z[:n])[:, 1:]).ravel()

        def cjac(z):
            _, grads = model.columns_and_grad(z[:n])
            J = np.zeros((P * 15, n + 1))
            J[:, :n] = -grads[:, 1:, :].reshape(P * 15, n)
            J[:, n] = 1.0
            return J

        bounds = [(log_wmin, 0.0)] * n + [(None, None)]
        z = _slsqp(fun, jac, z0, [{'type': 'ineq', 'fun': cfun, 'jac': cjac}],
                   bounds, maxiter)
        if z is None or not np.all(np.isfinite(z)):
            continue
        x = _clip_x(z[:n], log_wmin)
        sx = model.smax(x)
        if sx < best_s:
            best_s, best_x = sx, x
    return best_s, best_x


def _starts(model, log_wmin, warm, rng, n_random):
    out = [np.zeros(model.n_x)]
    out += [w for w in warm if w is not None]
    lo = max(log_wmin, -6.0)
    out += [rng.uniform(lo, 0.0, model.n_x) for _ in range(n_random)]
    return out


def sweep(model: WeightModel, w_grid, *, n_random: int = 3, seed: int = 0,
          warm_x=None, maxiter: int = 500) -> dict:
    """F*(w) and s*(w) on a DECREASING grid of w_min (w_grid[0] should be 1).
    Each level is warm-started from the previous level's optimum (feasible:
    the box only grows), from the Pauli point x = 0, from warm_x (e.g. the
    shared optimum when the model has species classes) and from n_random
    random points; F* and s* are made monotone (cumulative min) and the
    number of levels where that changed anything is reported."""
    w_grid = np.asarray(w_grid, float)
    if np.any(np.diff(w_grid) > 0):
        raise ValueError('w_grid must be decreasing')
    rng = np.random.default_rng(seed)
    nW = len(w_grid)
    F = np.full(nW, np.inf)
    s = np.full(nW, np.inf)
    xF = np.zeros((nW, model.n_x))
    xs = np.zeros((nW, model.n_x))
    prevF = prevs = None
    for k, w in enumerate(w_grid):
        lw = float(np.log(w))
        wx = warm_x[k] if warm_x is not None else None
        Fk, xk = solve_level(model, lw, _starts(model, lw, [prevF, prevs, wx], rng, n_random),
                             maxiter=maxiter)
        sk, xsk = solve_minmax(model, lw, _starts(model, lw, [prevs, xk, wx], rng, n_random),
                               maxiter=maxiter)
        F[k], xF[k], s[k], xs[k] = Fk, xk, sk, xsk
        prevF, prevs = xk, xsk
    # monotonicity in w (the feasible box only grows as w decreases)
    F_mono, s_mono = np.minimum.accumulate(F), np.minimum.accumulate(s)
    return {'w': w_grid, 'F': F_mono, 's': s_mono, 'F_raw': F, 's_raw': s,
            'x_F': xF, 'x_s': xs,
            'n_nonmono_F': int(np.sum(F_mono < F - 1e-12)),
            'n_nonmono_s': int(np.sum(s_mono < s - 1e-12))}


def w_star(model: WeightModel, res: dict, *, n_bisect: int = 30, n_random: int = 2,
           seed: int = 1) -> float:
    """Largest w_min with s*(w_min) <= 0 (bisection in log w between the last
    grid level with s > 0 and the first with s <= 0).  1.0 if the plain Pauli
    frame already has F = 0; nan if no level of the sweep reaches s <= 0."""
    w, s, xs = res['w'], res['s'], res['x_s']
    hit = np.nonzero(s <= 0.0)[0]
    if hit.size == 0:
        return float('nan')
    k = int(hit[0])
    if k == 0:
        return float(w[0])
    rng = np.random.default_rng(seed)
    lo, hi = np.log(w[k]), np.log(w[k - 1])          # s(lo) <= 0 < s(hi)
    x_ok = xs[k]
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        sm, xm = solve_minmax(model, mid, _starts(model, mid, [x_ok, xs[k - 1]], rng, n_random))
        if sm <= 0.0:
            lo, x_ok = mid, xm
        else:
            hi = mid
    return float(np.exp(lo))


def local_slopes(w, F, *, tiny: float = 1e-12):
    """d log F / d log w between consecutive levels (nan where F <= tiny):
    a constant positive tail = power law F ~ w^p; a tail -> 0 = plateau."""
    w, F = np.asarray(w, float), np.asarray(F, float)
    out = np.full(len(w) - 1, np.nan)
    ok = (F[:-1] > tiny) & (F[1:] > tiny)
    out[ok] = np.log(F[1:][ok] / F[:-1][ok]) / np.log(w[1:][ok] / w[:-1][ok])
    return out
