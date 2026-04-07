"""
Optimise framability over extended Pauli-like frames D.

Problem
-------
    min_D  get_framability(D, gate)

    s.t.  ||D[:, j]||_2 = 1   for every column j        (unit-norm columns)
          (D^T D)_{jj}  >= 1  for every j                (automatic from above)

Why not a linear programme?
---------------------------
D enters the inner LPs both as the *constraint matrix* and through the
*right-hand side*  Y = gate^T D.  This makes the overall problem a
bi-level, non-convex programme that cannot be cast as a single LP or SDP.
We resort to derivative-free optimisation (Powell / Nelder-Mead /
differential evolution) with random restarts.

Two modes are provided:

  'general'   — optimise every entry of D  (pauli_string_dim × d_ext params)
  'kronecker' — constrain D = kron(S, S) and optimise only S
                (qubit_d² × d_single params, much faster)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.optimize._linprog_highs import _linprog_highs
from scipy.optimize._linprog_util import _LPProblem, _clean_inputs

from two_qubit_lindbladian import pauli_string_dim, qubit_d
from framability import get_framability, extended_pauli_D

# cobyqa was added in scipy 1.11; fall back to Powell on older installs.
try:
    from scipy.optimize import minimize as _minimize_probe
    _minimize_probe(lambda x: x[0] ** 2, [1.0], method='cobyqa',
                    options={'maxfev': 1})
    DEFAULT_METHOD = 'cobyqa'
except (ValueError, ImportError):
    DEFAULT_METHOD = 'Powell'


# ---------------------------------------------------------------------------
#  Cached LP components  (keyed by d_ext)
# ---------------------------------------------------------------------------
# The batched framability LP has the structure:
#
#   min  c^T x
#   s.t. A_ub x <= b_ub          (inequality; fixed across all D)
#        A_eq(D) x  = b_eq(D)    (equality; depends on D via kron(I,D))
#        x >= 0
#
# For a fixed d_ext, A_ub / b_ub / c / bounds never change, and kron(I, D)
# always has the same sparsity pattern (only .data changes with D).
# We precompute a _cleaned_ _LPProblem once (including the COO conversion
# performed inside _clean_inputs) and then update A_eq.data and b_eq
# in-place on every call, bypassing scipy's per-call validation overhead.

_LP_CACHE: dict = {}   # d_ext -> (lp_clean, coo_eq, blk_nnz)


def _get_lp_cache(d_ext: int, n: int):
    """Return (lp_clean, coo_eq, blk_nnz) for the batched LP, cached by d_ext."""
    if d_ext not in _LP_CACHE:
        from scipy.sparse import (
            kron as sp_kron, eye as sp_eye, csc_matrix,
            hstack as sp_hstack,
        )
        n_up = d_ext * d_ext
        n_vars = 1 + 2 * n_up
        n_eq = n * d_ext

        c = np.zeros(n_vars)
        c[0] = 1.0

        sum_blk = np.kron(np.eye(d_ext), np.ones((1, d_ext)))
        A_ub = np.hstack([-np.ones((d_ext, 1)), sum_blk, sum_blk])
        b_ub = np.zeros(d_ext)
        bounds = [(0, None)] * n_vars

        # Build A_eq template with an identity-like D (any dense n×d_ext matrix
        # gives the same sparsity pattern, so use ones).
        D_tmpl = np.ones((n, d_ext))
        blk_tmpl = sp_kron(
            sp_eye(d_ext, format='csc'), csc_matrix(D_tmpl), format='csc'
        )
        blk_nnz = blk_tmpl.nnz
        A_eq_tmpl = csc_matrix(
            sp_hstack([csc_matrix((n_eq, 1)), blk_tmpl, -blk_tmpl], format='csc')
        )

        b_eq_buf = np.zeros(n_eq)
        lp_raw = _LPProblem(c, A_ub, b_ub, A_eq_tmpl, b_eq_buf, bounds, None)
        lp_clean = _clean_inputs(lp_raw)
        # lp_clean.A_eq is a COO array sharing the same data buffer order as
        # A_eq_tmpl (col-major).  Update .data in-place each call.
        coo_eq = lp_clean.A_eq

        _LP_CACHE[d_ext] = (lp_clean, coo_eq, blk_nnz)
    return _LP_CACHE[d_ext]


# ---------------------------------------------------------------------------
#  Fast framability: single batched LP  (one linprog call instead of d_ext)
# ---------------------------------------------------------------------------

def _get_framability_fast(D, gate_real):
    """
    Compute framability in a *single* LP call.

    Primal batched formulation
    --------------------------
    min  t
    s.t. D (u⁺_j − u⁻_j) = Y[:, j]   for j = 0 … d_ext-1
         Σ_k (u⁺_{j,k} + u⁻_{j,k}) ≤ t   for each j
         u⁺, u⁻ ≥ 0,  t ≥ 0

    For d_ext = 36 this is one LP with ~2 600 variables vs 36 separate
    calls, eliminating per-call Python / HiGHS overhead.

    The LP is solved via a direct _linprog_highs call, bypassing scipy's
    per-call input validation.  A_eq is updated in-place (only .data
    changes; sparsity pattern is fixed for any dense D).
    """
    n, d_ext = D.shape

    # Retrieve (or build) the pre-cleaned LP object and its mutable COO view
    lp_clean, coo_eq, blk_nnz = _get_lp_cache(d_ext, n)

    # Update A_eq data in-place:
    # kron(I_{d_ext}, D).data = tile(D.ravel('F'), d_ext)
    d_flat = np.tile(D.ravel(order='F'), d_ext)
    coo_eq.data[:blk_nnz] = d_flat
    coo_eq.data[blk_nnz:] = -d_flat

    # Update b_eq (Y = gate^T D, flattened column-major)
    b_eq = (gate_real.T @ D).ravel(order='F')
    lp_upd = lp_clean._replace(b_eq=b_eq)

    res = _linprog_highs(lp_upd, solver=None, presolve=False)
    return res['x'][0] if res['status'] == 0 else np.inf


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _project_columns(D):
    """Return a copy of D with every column normalised to unit 2-norm."""
    norms = np.linalg.norm(D, axis=0, keepdims=True)
    return D / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
#  Main optimiser
# ---------------------------------------------------------------------------

def minimize_framability(gate, d_ext, *, mode='kronecker', n_restarts=5,
                         method='cobyqa', max_iter=200, maxfev=1000,
                         tol=1e-6, seed=None, verbose=True,
                         extra_init_xs=None, return_x=False):
    """
    Find D of shape (pauli_string_dim, d_ext) with unit-norm columns
    that minimises get_framability(D, gate).

    Parameters
    ----------
    gate : ndarray, shape (pauli_string_dim, pauli_string_dim)
        Real gate / propagator whose framability is to be minimised.
    d_ext : int
        Number of columns of D.  In 'kronecker' mode this must be a
        perfect square; d_single = sqrt(d_ext).
    mode : str
        'kronecker' — D = kron(S, S), optimise only S  (fast, default).
        'general'   — optimise all entries of D directly (slower).
    n_restarts : int
        Random restarts for local methods.  The first restart uses the
        standard extended-Pauli D when *d_ext* matches (36).
    method : str
        'cobyqa' (default), 'Powell', 'Nelder-Mead', or
        'differential_evolution'.  'cobyqa' uses quadratic surrogate
        models and typically converges in fewer function evaluations than
        Powell while also running faster per iteration.
    max_iter : int
        Max iterations per restart (or total for differential evolution).
    maxfev : int
        Max function evaluations per restart.
    tol : float
        Convergence tolerance.
    seed : int | None
        Random seed for reproducibility.
    verbose : bool
        Print per-restart progress.
    extra_init_xs : list of ndarray | None
        Additional flat parameter vectors to use as extra restart seeds,
        appended after the standard n_restarts.  In 'kronecker' mode each
        vector must have length n_s * d_single (same as the optimiser's
        own parameter space, i.e. the raw x returned when return_x=True).
    return_x : bool
        If True, return a 3-tuple (D_opt, f_opt, x_opt) where x_opt is
        the raw flat parameter vector corresponding to D_opt.  Default
        False returns the usual 2-tuple (D_opt, f_opt).

    Returns
    -------
    D_opt : ndarray, shape (pauli_string_dim, d_ext)
        Optimal frame matrix (unit-norm columns).
    f_opt : float
        Minimal framability value found.
    x_opt : ndarray  (only when return_x=True)
        Raw flat parameter vector for D_opt (seed-compatible with
        extra_init_xs of a subsequent call).
    """
    rng = np.random.default_rng(seed)
    n = pauli_string_dim
    gate = np.asarray(gate, dtype=float)

    # --- Kronecker mode: D = kron(S, S), optimise S (qubit_d² × d_single) -
    if mode == 'kronecker':
        d_single = int(round(np.sqrt(d_ext)))
        if d_single * d_single != d_ext:
            raise ValueError(
                f'd_ext={d_ext} is not a perfect square; '
                f'cannot use kronecker mode.'
            )
        n_s = qubit_d ** 2          # rows of S  (4 for two qubits)
        n_params = n_s * d_single   # 4 * d_single

        def objective_kron(params):
            S = _project_columns(params.reshape(n_s, d_single))
            D = np.kron(S, S)
            return _get_framability_fast(D, gate)

        result = _run_restarts(
            objective_kron, n_params, d_ext, n_s, d_single,
            gate, rng, n_restarts, method, max_iter, maxfev, tol, verbose,
            is_kron=True, extra_init_xs=extra_init_xs,
        )
        return result if return_x else result[:2]

    # --- General mode: optimise D directly (n × d_ext) --------------------
    if mode != 'general':
        raise ValueError(f"mode must be 'kronecker' or 'general', got '{mode}'")

    n_params = n * d_ext

    def objective_gen(params):
        D = _project_columns(params.reshape(n, d_ext))
        return _get_framability_fast(D, gate)

    result = _run_restarts(
        objective_gen, n_params, d_ext, n, d_ext,
        gate, rng, n_restarts, method, max_iter, maxfev, tol, verbose,
        is_kron=False, extra_init_xs=extra_init_xs,
    )
    return result if return_x else result[:2]


def _run_restarts(objective, n_params, d_ext, n_rows, n_cols,
                  gate, rng, n_restarts, method, max_iter, maxfev, tol,
                  verbose, *, is_kron, extra_init_xs=None):
    """Shared restart loop for both modes.  Returns (D_opt, f_opt, x_opt)."""

    # --- differential evolution (no manual restarts) -----------------------
    if method == 'differential_evolution':
        bounds = [(-2.0, 2.0)] * n_params
        res = differential_evolution(
            objective, bounds, maxiter=max_iter, tol=tol,
            seed=int(rng.integers(2**31)), polish=True, workers=1,
        )
        D_opt = _params_to_D(res.x, n_rows, n_cols, is_kron)
        f_opt = _get_framability_fast(D_opt, gate)
        if verbose:
            print(f'DE finished:  f = {f_opt:.6f}  (success={res.success})')
        return D_opt, f_opt, res.x

    # --- local methods with random restarts --------------------------------
    best_val = np.inf
    best_D = None
    best_x = None

    inits = _build_inits(n_rows, n_cols, d_ext, n_restarts, rng, is_kron,
                         extra_init_xs=extra_init_xs)

    if method == 'cobyqa':
        opts = {'maxfev': maxfev}
    elif method == 'Powell':
        opts = {'maxiter': max_iter, 'maxfev': maxfev, 'ftol': tol, 'xtol': tol}
    elif method == 'Nelder-Mead':
        opts = {'maxiter': max_iter, 'maxfev': maxfev, 'fatol': tol, 'xatol': tol}
    else:
        opts = {'maxiter': max_iter, 'maxfev': maxfev}

    for i, x0 in enumerate(inits):
        # Always evaluate at the initial point before optimising: this
        # guarantees the baseline (e.g. ext-Pauli) is never discarded even
        # if the optimiser diverges to a worse local minimum.
        f_x0 = objective(x0)
        if f_x0 < best_val:
            best_val = f_x0
            best_D = _params_to_D(x0, n_rows, n_cols, is_kron)
            best_x = x0.copy()

        res = minimize(objective, x0, method=method, options=opts)

        D_cand = _params_to_D(res.x, n_rows, n_cols, is_kron)
        f_cand = _get_framability_fast(D_cand, gate)

        if verbose:
            if i >= n_restarts:
                tag = 'neighbor seed'
            elif i == 0 and d_ext == 36:
                tag = 'ext-Pauli init'
            else:
                tag = 'random init'
            print(f'  restart {i + 1}/{len(inits)} ({tag}):  '
                  f'f_init={f_x0:.6f}  f_opt={f_cand:.6f}  (success={res.success})')

        if f_cand < best_val:
            best_val = f_cand
            best_D = D_cand.copy()
            best_x = res.x.copy()

    return best_D, best_val, best_x


def _params_to_D(params, n_rows, n_cols, is_kron):
    """Decode flat parameter vector into a projected D matrix."""
    S = _project_columns(params.reshape(n_rows, n_cols))
    return np.kron(S, S) if is_kron else S


def _build_inits(n_rows, n_cols, d_ext, n_restarts, rng, is_kron,
                 extra_init_xs=None):
    """Build a list of initial flat parameter vectors.

    The first *n_restarts* entries are the standard seeds (extended-Pauli
    D where compatible, then random).  Any vectors in *extra_init_xs* are
    appended afterwards so the caller can distinguish them by index.
    """
    inits = []

    # First init: extended-Pauli D (if compatible)
    D_pauli = extended_pauli_D()
    if D_pauli.shape[1] == d_ext:
        if is_kron:
            # Build S directly from extended_pauli_D's single-qubit recipe
            a = 1
            S_pauli = np.array(
                [[1, 0, 0, 0, 0,             0],
                 [0, 1, 0, 0, a/np.sqrt(2),  a/np.sqrt(2)],
                 [0, 0, 1, 0, 0,             0],
                 [0, 0, 0, 1, a/np.sqrt(2), -a/np.sqrt(2)]])
            inits.append(_project_columns(S_pauli).ravel())
        else:
            inits.append(_project_columns(D_pauli).ravel())

    while len(inits) < n_restarts:
        M = rng.standard_normal((n_rows, n_cols))
        inits.append(_project_columns(M).ravel())

    # Extra seeds from caller (e.g. a neighbor point's x_opt) — appended
    # after the n_restarts standard seeds so they can be tagged in verbose.
    if extra_init_xs:
        for x in extra_init_xs:
            inits.append(np.asarray(x, dtype=float))

    return inits


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    from two_qubit_lindbladian import numeric_two_qubit_lindbladian
    from scipy.linalg import expm

    J, gamma, gamma_p = 1.0, 0.5, 0.1
    L = numeric_two_qubit_lindbladian(J, gamma, gamma_p)
    dt = 0.01
    gate = expm(dt * L).real

    print(f'Gate built: J={J}, gamma={gamma}, gamma_p={gamma_p}, dt={dt}')

    # Baseline: extended-Pauli framability
    D_pauli = extended_pauli_D()
    d_ext = D_pauli.shape[1]   # 36

    t0 = time.perf_counter()
    f_primal = get_framability(D_pauli, gate)
    t_primal = time.perf_counter() - t0

    t0 = time.perf_counter()
    f_fast = _get_framability_fast(D_pauli, gate)
    t_fast = time.perf_counter() - t0

    print(f'Primal LP framability: {f_primal:.6f}  ({t_primal*1000:.1f} ms)')
    print(f'Batch  LP framability: {f_fast:.6f}  ({t_fast*1000:.1f} ms)')
    print()

    # Optimise with Kronecker mode (24 params instead of 576)
    print(f'Optimising (kronecker mode, d_ext={d_ext}, 3 restarts, '
          f'maxfev=500 each) ...')
    t0 = time.perf_counter()
    D_opt, f_opt = minimize_framability(
        gate, d_ext=d_ext, mode='kronecker', n_restarts=3,
        method='cobyqa', max_iter=100, maxfev=500,
        seed=42, verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f'\nOptimal framability: {f_opt:.6f}  ({elapsed:.1f} s)')
    delta = f_primal - f_opt
    print(f'Improvement over extended Pauli: {delta:.6f}'
          f'  ({100 * delta / f_primal:.1f}%)')

    # Verify constraints
    norms = np.linalg.norm(D_opt, axis=0)
    gram_diag = np.diag(D_opt.T @ D_opt)
    print(f'\nColumn norms:  min={norms.min():.6f}  max={norms.max():.6f}')
    print(f'Gram diagonal: min={gram_diag.min():.6f}  max={gram_diag.max():.6f}')
