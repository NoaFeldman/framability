r"""
framability_rate_global.py -- seeded global optimisation of the Heisenberg
framability rate mu*(S (x) S) over single-qubit observable frames S.

Companion of framability_rate.minimize_rate (the alternating certificate
scheme), addressing the three reasons that scheme misses optima:

  1. its S-step fits A D ~ D H in least squares instead of the rate;
  2. its starts are random columns or basis vectors;
  3. its polish is a Polyak subgradient step on the binding column only.

What this module does instead
-----------------------------
* d_ext_single = 4: the frame D = S (x) S is a BASIS, so the certificate is
  unique, H = D^{-1} A D, and the rate is the closed form

        mu*(S) = max_j ( H_jj + sum_{k != j} |H_kj| )

  (one 16x16 solve, no LP).  A global search (differential evolution over
  the 12 free entries, then Nelder-Mead polishes) costs seconds.

* d_ext_single > 4: the rate is the batched LP of framability_rate.  The
  search is seeded (see below), run through a short differential evolution,
  then polished with a BUNDLE / trust-region step: every near-binding column
  gets its own dual witness and envelope gradient, and the small LP

        min t   s.t.  mu_j + <g_j, dS> <= t  for the bundle,  |dS| <= radius

  picks the step, accepted on the true LP value.  This removes the zigzag of
  the single-column subgradient.

* Seeds, all evaluated as fixed frames so the result can never be worse than
  the best of them: the exact optimum over rescaled Pauli frames
  diag(1, sX, sY, sZ) (a convex problem in the log weights,
  product_weight_optimum), projector frames on the pump axis (I +- X)/2,
  Z-projector + XY-polygon frames, the Pauli frame, any stored frames the
  caller passes, and the optimum of the next-smaller frame padded (nesting,
  so rates are monotone in d_ext by construction).

Frame conventions are those of optimize_framability / framability_rate:
column 0 of S is the identity (1,0,0,0); every other column has operator norm
<= 1 (Hermitian columns: |c_I| + ||(c_X, c_Y, c_Z)||_2 <= 1); A = L^T acts on
observable coefficients.

Complex frames are the DEFAULT (COMPLEX_FRAMES = True, version 2.0)
-------------------------------------------------------------------
Frame elements may be non-Hermitian (complex Pauli coefficients) and the
certificate coefficients complex; the rate is the complex column log-norm
(framability_rate).  The search space has 8 real parameters per free column
(real parts, then imaginary parts: x = [Re free, Im free]) instead of 4;
frame_from_x still accepts a 4-per-column real vector, so stored real
frames remain valid seeds (a real frame is a complex frame, and complex
coefficients never make it worse).  The closed form for 4 columns carries
over: mu* = max_j (Re H_jj + sum_{k != j} |H_kj|), H = D^{-1} A D.  Real
frames are stationary in the imaginary directions (mu(S) = mu(conj S)), so
the seed library adds complex starts: clock-ladder frames
{I, sigma^+, sigma^-, clock phases |0><0| + w^k |1><1|} about Z, X and Y.
Pass complex_frames=False for the old Hermitian search.

Public API
----------
basis_rate(S, A)                    closed-form rate for a 4-column frame
frame_rate_value(S, A, reference=)  rate of any frame (closed form / LP)
product_weight_optimum(A)           exact optimum over rescaled Pauli frames
clock_ladder_frame(m, axis)         complex seed: sigma^+- plus clock phases
seed_library(m, A)                  deterministic seed frames with m columns
fit_columns(S, m)                   trim / pad a frame to m columns (monotone)
bundle_rate_polish(S, A)            trust-region bundle descent
minimize_rate_global(A, m, seeds=)  the full seeded global search

    python framability_rate_global.py      # self-test against stored scans
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import differential_evolution, minimize, linprog

from optimize_framability import (_project_columns_bloch, _FIXED_COLS,
                                  N_FIXED_COLS, _kron_power)
from framability_rate import (generator_log_norm, generator_log_norm_reference,
                              spectral_abscissa, is_complex_frame,
                              _as_frame_array, project_frame_columns,
                              complex_column_witness, complex_grad_from_D)

#   1.0-global-seeded   : Hermitian frames, real coefficients
#   2.0-complex-default : non-Hermitian frames + complex coefficients default
RATE_GLOBAL_VERSION = '2.0-complex-default'
COMPLEX_FRAMES = True        # default search space: non-Hermitian elements

WEIGHT_FLOOR = 1e-2          # smallest Pauli rescaling (keeps D well conditioned)
_COND_MAX = 1e12             # basis frames beyond this are treated as singular
_TOL = 1e-9

_I = np.array([1.0, 0.0, 0.0, 0.0])
_X = np.array([0.0, 1.0, 0.0, 0.0])
_Y = np.array([0.0, 0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 0.0, 1.0])


def _cf(complex_frames) -> bool:
    return COMPLEX_FRAMES if complex_frames is None else bool(complex_frames)


# ---------------------------------------------------------------------------
#  Frames <-> parameters
# ---------------------------------------------------------------------------
def n_params(m: int, complex_frames=None) -> int:
    """Real search parameters of an m-column frame: 8 per free column for
    complex frames (default), 4 for Hermitian ones."""
    return (8 if _cf(complex_frames) else 4) * (m - N_FIXED_COLS)


def frame_from_x(x, m: int) -> np.ndarray:
    """Flat free entries -> 4 x m frame, identity first, columns projected
    onto the operator-norm ball.  x is either real, row-major (m-1) x 4
    (4 (m-1) entries, a Hermitian frame), or [Re free, Im free] in that
    layout (8 (m-1) entries, a complex frame)."""
    x = np.asarray(x, float).ravel()
    nf = m - N_FIXED_COLS
    if x.size == 4 * nf:
        free = x.reshape(nf, 4).T
    elif x.size == 8 * nf:
        free = (x[:4 * nf] + 1j * x[4 * nf:]).reshape(nf, 4).T
    else:
        raise ValueError(f'frame_from_x: {x.size} entries fit neither 4 nor 8 '
                         f'x {nf} free columns')
    free = project_frame_columns(free)
    return np.hstack([_FIXED_COLS.astype(free.dtype), free])


def x_from_frame(S, complex_frames=None) -> np.ndarray:
    """Inverse of frame_from_x: [Re free, Im free] for complex_frames (the
    default; a real frame gets zero imaginary parts), the real 4 (m-1)
    vector otherwise (a complex frame then raises)."""
    S = np.asarray(S)
    free = S[:, N_FIXED_COLS:].T.ravel()
    if _cf(complex_frames):
        return np.concatenate([np.real(free), np.imag(free)]).astype(float)
    if is_complex_frame(S):
        raise ValueError('x_from_frame: complex frame with complex_frames=False')
    return np.real(free).astype(float)


def fit_columns(S, m: int) -> np.ndarray:
    """Trim a frame to its first m columns, or pad it to m columns by
    replicating the last free column (as dissipative_PT.embed_frame_params,
    for any dtype), so the padded frame's rate is <= the original's."""
    S = _as_frame_array(S)
    k = S.shape[1]
    if k == m:
        return S.copy()
    if k > m:
        return S[:, :m].copy()
    return np.hstack([S, np.repeat(S[:, -1:], m - k, axis=1)])


# ---------------------------------------------------------------------------
#  Rate evaluation
# ---------------------------------------------------------------------------
def basis_rate(S, A, *, return_H: bool = False):
    """Closed-form mu*(S (x) S) for an invertible 4x4 frame (inf if singular):
    max_j (Re H_jj + sum_{k != j} |H_kj|), H = D^{-1} A D (complex for a
    complex frame)."""
    D = _kron_power(_as_frame_array(S), 2)
    try:
        H = np.linalg.solve(D, np.asarray(A, float) @ D)
    except np.linalg.LinAlgError:
        return (np.inf, None) if return_H else np.inf
    if not np.all(np.isfinite(H)) or np.linalg.cond(D) > _COND_MAX:
        return (np.inf, None) if return_H else np.inf
    diag = np.diag(H)
    cols = diag.real + (np.abs(H).sum(axis=0) - np.abs(diag))
    val = float(cols.max())
    return (val, H) if return_H else val


def frame_rate_value(S, A, *, reference: bool = False) -> float:
    """mu*(S (x) S): closed form for 4 columns, batched LP otherwise;
    reference=True uses the independent per-column LP (certification).
    Real or complex S."""
    S = _as_frame_array(S)
    if S.shape[1] == 4 and not reference:
        return basis_rate(S, A)
    D = _kron_power(S, 2)
    f = generator_log_norm_reference if reference else generator_log_norm
    return float(f(D, np.asarray(A, float)))


# ---------------------------------------------------------------------------
#  Seeds
# ---------------------------------------------------------------------------
def product_weight_optimum(A, floor: float = WEIGHT_FLOOR):
    """Exact optimum of the rate over S = diag(1, sX, sY, sZ), s in [floor, 1].

    The column rate A_PP + sum_Q |A_QP| w_P / w_Q (w = s (x) s) is a sum of
    exponentials of linear functions of u = log s, so max_P is convex in u:
    the SLSQP epigraph solve below converges to the global optimum.
    Returns (value, S)."""
    A = np.asarray(A, float)
    absA = np.abs(A)
    diag = np.diag(A).copy()
    lo = np.log(floor)

    def col_rates(u):
        s = np.exp(np.concatenate([[0.0], np.clip(u, lo, 0.0)]))
        w = np.kron(s, s)
        R = absA * (w[None, :] / w[:, None])
        np.fill_diagonal(R, diag)
        return R.sum(axis=0)

    cons = [{'type': 'ineq', 'fun': lambda z: z[3] - col_rates(z[:3])}]
    best_v, best_u = np.inf, np.zeros(3)
    for u0 in (np.zeros(3), np.full(3, lo / 2), np.array([lo / 4, lo / 2, lo]),
               np.array([lo, lo / 4, lo / 2])):
        z0 = np.concatenate([u0, [col_rates(u0).max()]])
        try:
            r = minimize(lambda z: z[3], z0, method='SLSQP',
                         bounds=[(lo, 0.0)] * 3 + [(None, None)],
                         constraints=cons, options=dict(maxiter=500, ftol=1e-12))
            u = np.clip(r.x[:3], lo, 0.0)
        except Exception:                                   # noqa: BLE001
            u = u0
        # Nelder-Mead touch-up guards against an SLSQP early stop
        r2 = minimize(lambda u_: col_rates(u_).max(), u, method='Nelder-Mead',
                      options=dict(maxfev=4000, xatol=1e-9, fatol=1e-12))
        for cand in (u, np.clip(r2.x, lo, 0.0)):
            v = float(col_rates(cand).max())
            if v < best_v:
                best_v, best_u = v, cand
    s = np.exp(np.concatenate([[0.0], best_u]))
    return best_v, np.diag(s)


def _proj(axis, sign):
    """Pauli coefficients of the projector (I + sign * axis) / 2."""
    return 0.5 * (_I + sign * axis)


# Clifford maps of Pauli coefficient vectors (c_I, c_X, c_Y, c_Z) that carry
# the Z axis onto X (Hadamard: X <-> Z, Y -> -Y) or onto Y (Z -> Y, Y -> -Z).
_AXIS_MAP = {
    'z': np.eye(4),
    'x': np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]], float),
    'y': np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]], float),
}


def clock_ladder_frame(m: int, axis: str = 'z') -> np.ndarray:
    """Complex m-column frame {I, sigma^+, sigma^-, v_k} about `axis`:
    sigma^+- = (X +- iY)/2 (the ladder operators |0><1|, |1><0|) and the
    clock phases v_k = |0><0| + w^k |1><1|, w = exp(2 pi i / (m - 2)),
    k = 1 .. m-3 (v_0 = I is column 0; v_(N/2) = Z leads, so a trim to 4
    columns is {I, sigma^+, sigma^-, Z}).  All elements have operator norm 1.
    Under J ZZ the ladder columns pick up only a phase (free with complex
    coefficients) and the neighbour's clock phase moves by one vertex, the
    finite version of the rate-0 complex product frame of model3."""
    if m < 4:
        raise ValueError('clock_ladder_frame needs m >= 4')
    n_clock = m - 2
    w = np.exp(2j * np.pi / n_clock)
    order = [n_clock // 2] + [k for k in range(1, n_clock) if k != n_clock // 2]
    cols = [np.array([1, 0, 0, 0], complex),
            np.array([0, 0.5, 0.5j, 0]), np.array([0, 0.5, -0.5j, 0])]
    cols += [np.array([(1 + w ** k) / 2, 0, 0, (1 - w ** k) / 2]) for k in order]
    S = np.array(cols, complex).T
    return _AXIS_MAP[axis] @ S


def seed_library(m: int, A=None, complex_frames=None) -> dict:
    """Deterministic seed frames with m columns (identity first), keyed by
    name.  Column orders are chosen so that a trim to 4 columns still spans.
    With A given, the exact rescaled-Pauli optimum is included.  With complex
    frames (the default) the clock-ladder frames about Z, X and Y are added
    (real frames are stationary in the imaginary directions, so a complex
    search needs complex starts)."""
    r2 = 1.0 / np.sqrt(2.0)
    lib = {
        'pauli':     [_X, _Y, _Z],
        'pump+':     [_proj(_X, +1), _Y, _Z],
        'pump-':     [_proj(_X, -1), _Y, _Z],
        'pump+-':    [_proj(_X, +1), _Y, _Z, _proj(_X, -1)],
        'zproj':     [_proj(_Z, +1), _X, _Y, _proj(_Z, -1)],
        'polygon':   [_X, _Y, _proj(_Z, +1), _proj(_Z, -1),
                      r2 * (_X + _Y), r2 * (_X - _Y)],
        'pump_poly': [_proj(_X, +1), _Y, _Z, _proj(_X, -1),
                      r2 * (_Y + _Z), r2 * (_Y - _Z)],
    }
    out = {}
    for name, cols in lib.items():
        S = np.hstack([_FIXED_COLS, np.array(cols, float).T])
        out[name] = fit_columns(S, m)
    if A is not None:
        out['weights'] = fit_columns(product_weight_optimum(A)[1], m)
    if _cf(complex_frames):
        for axis in ('z', 'x', 'y'):
            out[f'ladder_{axis}'] = clock_ladder_frame(m, axis)
    return out


# ---------------------------------------------------------------------------
#  Bundle / trust-region polish (any frame size)
# ---------------------------------------------------------------------------
def column_witness(D, A, j):
    """Dual witness of column j:  max <w, A d_j>  s.t. (D^T w)_j = 1,
    |(D^T w)_k| <= 1 (k != j).  Same LP as framability_rate.rate_certificate.
    Complex D: the complex witness of the exact per-column solve (to pair it
    with its own certificate column use complex_column_witness directly)."""
    if is_complex_frame(D):
        r = complex_column_witness(D, A, j)
        return None if r is None else r[2]
    D = _as_frame_array(D)
    DT = D.T
    m = D.shape[1]
    y = A @ D[:, j]
    res = linprog(-y, A_ub=np.vstack([DT, -DT]), b_ub=np.ones(2 * m),
                  A_eq=DT[j:j + 1], b_eq=np.array([1.0]),
                  bounds=[(None, None)] * D.shape[0], method='highs')
    return res.x.copy() if res.success else None


def column_grad(S, A, j, w, h):
    """Envelope gradient of column j's rate w.r.t. S (as rate_value_and_grad,
    for one column): d mu_j / dD = (A^T w) e_j^T - w h^H, chained through
    D = S (x) S.  Complex S / w / h: the complex gradient Gamma
    (d mu_j = Re sum conj(Gamma) dS); real inputs: the real gradient."""
    S = np.asarray(S)
    n_s, m = S.shape
    gD = np.zeros((n_s * n_s, m * m), dtype=np.result_type(S, w, h, float))
    gD[:, j] += np.asarray(A, float).T @ w
    gD -= np.outer(w, np.conj(h))
    return complex_grad_from_D(S, gD)


def grad_vector(Gamma_free, cplx: bool) -> np.ndarray:
    """Flatten a free-column gradient (n_s x nfree) to the step variables:
    [Re, Im] (row-major) for a complex search, the real part otherwise."""
    if cplx:
        return np.concatenate([np.real(Gamma_free).ravel(),
                               np.imag(Gamma_free).ravel()])
    return np.real(Gamma_free).ravel()


def step_frame(S, delta, cplx: bool) -> np.ndarray:
    """S + delta on the free columns (delta in grad_vector's layout),
    projected back onto the operator-norm ball."""
    n_s, m = S.shape
    nf = m - N_FIXED_COLS
    if cplx:
        h = delta.size // 2
        d = (delta[:h] + 1j * delta[h:]).reshape(n_s, nf)
    else:
        d = np.asarray(delta, float).reshape(n_s, nf)
    free = project_frame_columns(S[:, N_FIXED_COLS:] + d)
    return np.hstack([_FIXED_COLS.astype(free.dtype), free])


def _bundle_grads(S, A, D, cols, H, bundle, cplx):
    """(gradient vectors, column values) of the bundle columns.  Complex D:
    witness, certificate column and value all from one exact per-column
    solve; real D: the batched H with a separate dual witness (as before)."""
    G, mu = [], []
    for j in bundle:
        if is_complex_frame(D):
            r = complex_column_witness(D, A, j)
            if r is None:
                continue
            v, h, w = r
        else:
            w = column_witness(D, A, j)
            if w is None:
                continue
            v, h = cols[j], H[:, j]
        g = column_grad(S, A, j, w, h)[:, N_FIXED_COLS:]
        G.append(grad_vector(g, cplx))
        mu.append(v)
    return G, mu


def bundle_rate_polish(S, A, *, n_iter=60, radius=0.1, band=0.05, max_cols=8,
                       shrink=0.5, grow=1.6, radius_max=0.5, tol=_TOL,
                       verbose=False):
    """Trust-region bundle descent on max_j mu_j(S).

    Each iteration linearises every near-binding column (within `band` of the
    maximum, at most `max_cols`) with its own dual witness, solves the small
    epigraph LP for the step inside an infinity-norm trust region, projects
    the columns back onto the ball, and accepts the step only if the TRUE
    batched-LP rate decreases.  Returns (best_value, best_frame).

    A complex frame is polished over real and imaginary parts (8 variables
    per free column); a real frame stays in the real subspace (its gradient
    has no imaginary part -- complex directions come from DE / NM / kicks)."""
    A = np.asarray(A, float)
    S = _as_frame_array(S).copy()
    cplx = is_complex_frame(S)
    n_s, m = S.shape
    nfree = m - N_FIXED_COLS
    nvar = (2 if cplx else 1) * n_s * nfree
    D = _kron_power(S, 2)
    val, cols, H = generator_log_norm(D, A, return_cols=True, return_H=True)
    if not np.isfinite(val) or H is None:
        return float(val), S
    best_val, best_S = float(val), S.copy()
    for it in range(n_iter):
        order = np.argsort(-cols)
        thresh = val - band * max(abs(val), 1e-3)
        bundle = [int(j) for j in order[:max_cols] if cols[j] >= thresh]
        G, mu = _bundle_grads(S, A, D, cols, H, bundle, cplx)
        if not G:
            break
        G = np.array(G)
        c = np.zeros(nvar + 1)
        c[-1] = 1.0
        res = linprog(c, A_ub=np.hstack([G, -np.ones((len(mu), 1))]),
                      b_ub=-np.array(mu),
                      bounds=[(-radius, radius)] * nvar + [(None, None)],
                      method='highs')
        if not res.success:
            radius *= shrink
            if radius < 1e-6:
                break
            continue
        S_new = step_frame(S, res.x[:nvar], cplx)
        D_new = _kron_power(S_new, 2)
        val_new, cols_new, H_new = generator_log_norm(D_new, A, return_cols=True,
                                                      return_H=True)
        pred = val - float(res.x[-1])
        if np.isfinite(val_new) and H_new is not None and val_new < val - tol \
                and (pred <= 0 or (val - val_new) >= 0.05 * pred):
            S, D, val, cols, H = S_new, D_new, float(val_new), cols_new, H_new
            if val < best_val:
                best_val, best_S = val, S.copy()
            radius = min(radius * grow, radius_max)
        else:
            radius *= shrink
            if radius < 1e-6:
                break
        if verbose and (it + 1) % 10 == 0:
            print(f'    bundle {it + 1}: mu={val:.6e} radius={radius:.3g} '
                  f'bundle={len(mu)}', flush=True)
        if val <= tol:
            break
    return best_val, best_S


# ---------------------------------------------------------------------------
#  The seeded global search
# ---------------------------------------------------------------------------
def _de_init(seed_frames, m, popsize, rng, complex_frames=None):
    cf = _cf(complex_frames)
    xs = [x_from_frame(S, cf) for S in seed_frames
          if cf or not is_complex_frame(S)]
    n = n_params(m, cf)
    need = max(5, popsize * n) - len(xs)
    for _ in range(max(need, 0)):
        xs.append(rng.uniform(-1.0, 1.0, n))
    return np.clip(np.array(xs), -1.0, 1.0)


def minimize_rate_global(A, d_ext_single, *, seeds=(), seed=0,
                         de_popsize=None, de_maxiter=None, bundle_iters=60,
                         nm_maxfev=None, verbose=False, complex_frames=None):
    """min_S mu*(S (x) S) by seeds + differential evolution + bundle polish.

    seeds : iterable of frames (4 x k, identity first, any k, real or
            complex); frames with k != d_ext_single are trimmed / padded by
            fit_columns.
    de_popsize / de_maxiter : differential-evolution budget (multiplier of the
            parameter count / generations).  Defaults: (24, 300) for the
            closed-form basis case, (6, 25) for LP-evaluated frames; 0
            disables the DE stage.
    complex_frames : search non-Hermitian frames with complex coefficients
            (default COMPLEX_FRAMES = True: 8 parameters per free column,
            clock-ladder seeds added); False = the old Hermitian search
            (complex seeds are then skipped).
    Returns (S_opt, mu_opt, info); mu_opt is the certified value
    (independent per-column LP for overcomplete frames)."""
    A = np.asarray(A, float)
    m = int(d_ext_single)
    cf = _cf(complex_frames)
    basis = (m == 4)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    if de_popsize is None:
        de_popsize = 24 if basis else 6
    if de_maxiter is None:
        de_maxiter = 300 if basis else 25
    if nm_maxfev is None:
        nm_maxfev = 20000 if basis else 400

    obj = lambda x: frame_rate_value(frame_from_x(x, m), A)      # noqa: E731

    # ---- 1. seeds, evaluated as fixed frames --------------------------------
    lib = seed_library(m, A, complex_frames=cf)
    for i, S in enumerate(seeds):
        if cf or not is_complex_frame(S):
            lib[f'seed{i}'] = fit_columns(S, m)
    best_val, best_S, best_name = np.inf, None, None
    for name, S in lib.items():
        v = frame_rate_value(S, A)
        if np.isfinite(v) and v < best_val:
            best_val, best_S, best_name = float(v), S.copy(), name
    info = dict(mu_seed=best_val, seed_name=best_name, n_seeds=len(lib))
    if verbose:
        print(f'  seeds ({len(lib)}): best {best_name} mu={best_val:.6e}',
              flush=True)

    # ---- 2. differential evolution --------------------------------------
    info['mu_de'] = best_val
    info['n_fev'] = 0
    if de_maxiter > 0 and de_popsize > 0:
        init = _de_init(list(lib.values()), m, de_popsize, rng, cf)
        de = differential_evolution(obj, bounds=[(-1.0, 1.0)] * n_params(m, cf),
                                    init=init, seed=seed, maxiter=de_maxiter,
                                    popsize=de_popsize, tol=1e-10, atol=0.0,
                                    polish=False)
        info['n_fev'] = int(de.nfev)
        if np.isfinite(de.fun) and de.fun < best_val:
            best_val, best_S = float(de.fun), frame_from_x(de.x, m)
        info['mu_de'] = best_val
        if verbose:
            print(f'  DE: mu={best_val:.6e} ({de.nfev} evaluations)', flush=True)

    # ---- 3. polish: bundle (LP frames) and Nelder-Mead ----------------------
    if best_S is None:
        best_S = lib['pauli']
        best_val = frame_rate_value(best_S, A)
    if not basis and bundle_iters > 0 and best_val > _TOL:
        vb, Sb = bundle_rate_polish(best_S, A, n_iter=bundle_iters,
                                    verbose=verbose)
        if vb < best_val:
            best_val, best_S = float(vb), Sb
    info['mu_bundle'] = best_val
    if nm_maxfev > 0 and best_val > _TOL:
        r = minimize(obj, x_from_frame(best_S, cf), method='Nelder-Mead',
                     options=dict(maxfev=nm_maxfev, xatol=1e-9, fatol=1e-12))
        if np.isfinite(r.fun) and r.fun < best_val:
            best_val, best_S = float(r.fun), frame_from_x(r.x, m)
    info['mu_nm'] = best_val

    # ---- 4. certification -------------------------------------------------
    mu_ref = frame_rate_value(best_S, A, reference=True)
    if np.isfinite(mu_ref) and mu_ref > best_val:
        best_val = float(mu_ref)
    info['mu_reference'] = float(mu_ref)
    info['floor'] = float(spectral_abscissa(A))
    info['time'] = time.perf_counter() - t0
    if verbose:
        print(f'  final mu={best_val:.6e} (ref {mu_ref:.6e}) '
              f'in {info["time"]:.0f}s', flush=True)
    return best_S, max(float(best_val), 0.0) if best_val > -_TOL else float(best_val), info


# ---------------------------------------------------------------------------
#  Self-test against stored scans
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from pathlib import Path
    from trotter_lindbladian_scan import MODELS, build_bond_lindbladian, DIM_DEFAULT

    def L_at(model, g, gp):
        H1, H2, j1, j2 = MODELS[model].build(g, gp)
        return build_bond_lindbladian(H1, H2, j1, j2, DIM_DEFAULT).real

    rng = np.random.default_rng(1)
    A = L_at('model4', 3.0, 2.0).T
    S = frame_from_x(rng.uniform(-1, 1, 12), 4)
    print(f'closed form {basis_rate(S, A):.10f}  LP '
          f'{generator_log_norm(_kron_power(S, 2), A):.10f}')

    pts = [('model4', 0., 4.), ('model4', 2., 6.), ('model4', 10., 0.),
           ('model3', 10., 0.), ('model3', 4., 2.)]
    stored = {}
    f4 = Path('results_model4_rate/model4_rate_panels.npz')
    if f4.exists():
        d = np.load(f4)
        stored['model4'] = (d['rate_heis_4'], d['rate_heis_6'], 0.2)
    print('\nmodel  g   gp  | stored d4  d6 | global d4 | global d6 [s]')
    for model, g, gp in pts:
        A = L_at(model, g, gp).T
        s4 = s6 = np.nan
        if model in stored:
            Z4, Z6, step = stored[model]
            s4, s6 = Z4[int(round(g / step)), int(round(gp / step))], \
                Z6[int(round(g / step)), int(round(gp / step))]
        S4, mu4, i4 = minimize_rate_global(A, 4, seed=0)
        S6, mu6, i6 = minimize_rate_global(A, 6, seeds=[S4], seed=0)
        print(f'{model} {g:4.1f} {gp:4.1f} | {s4:8.4f} {s6:8.4f} | {mu4:9.4f} '
              f'| {mu6:9.4f}  [{i4["time"]:.0f}+{i6["time"]:.0f}s]  '
              f'seed {i4["seed_name"]}/{i6["seed_name"]}')
