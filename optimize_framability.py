"""
Optimise framability over extended Pauli-like frames D.

Problem
-------
    min_D  heisenberg_framability(D, gate)

    s.t.  D = kron^n_qubits(S)         (Kronecker structure)
          |c_I| + ||(c_X, c_Y, c_Z)||_2 <= 1  for every column of S
                                              (Bloch-norm bound)

S may be complex-valued; the LP equality constraint is split into its
real and imaginary parts so the framability LP variables remain real.


Structure
---------
D is constrained to be an n_qubits-fold Kronecker product of the same
single-qubit frame S (shape qubit_d² × d_ext_single):

    D = kron(S, kron(S, ... S ...))  [n_qubits copies]

This reduces the search space from pauli_string_dim × d_ext to
2 × qubit_d² × d_ext_single real parameters (real + imaginary halves),
where d_ext = d_ext_single ** n_qubits.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.optimize._linprog_highs import _linprog_highs
from scipy.optimize._linprog_util import _LPProblem, _clean_inputs

from two_qubit_lindbladian import pauli_string_dim, qubit_d
from framability import heisenberg_framability, extended_pauli_D

# Nelder-Mead is the default: the simplex method makes no smoothness assumption,
# handling the nonsmooth max-of-LP objective well, and converges reliably within
# the moderate evaluation budgets used in the depol sweep.
# Use 'dual_annealing' when a large budget (maxfev >> 5000) is available for
# true global search, or 'subgradient' when the problem is known to be convex.
DEFAULT_METHOD = 'Nelder-Mead'

try:
    import scipy.optimize as _sopt
    _DUAL_ANNEALING_AVAILABLE = hasattr(_sopt, 'dual_annealing')
    del _sopt
except ImportError:
    _DUAL_ANNEALING_AVAILABLE = False


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
    """Return (lp_clean, coo_eq, blk_nnz) for the batched LP, cached by d_ext.

    Each complex equality D u_j = Y[:, j] is split into its real and imaginary
    parts, so n_eq = 2 * n * d_ext and the per-target block is [Re(D); Im(D)]
    stacked vertically (shape 2n × d_ext).
    """
    if d_ext not in _LP_CACHE:
        from scipy.sparse import (
            kron as sp_kron, eye as sp_eye, csc_matrix,
            hstack as sp_hstack,
        )
        n_up = d_ext * d_ext
        n_vars = 1 + 2 * n_up
        n_eq = 2 * n * d_ext

        c = np.zeros(n_vars)
        c[0] = 1.0

        sum_blk = np.kron(np.eye(d_ext), np.ones((1, d_ext)))
        A_ub = np.hstack([-np.ones((d_ext, 1)), sum_blk, sum_blk])
        b_ub = np.zeros(d_ext)
        bounds = [(0, None)] * n_vars

        # Build A_eq template with an identity-like D (any dense 2n×d_ext matrix
        # gives the same sparsity pattern, so use ones).
        D_tmpl = np.ones((2 * n, d_ext))
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

def _get_framability_fast(D, gate):
    """
    Compute framability in a *single* LP call.

    Primal batched formulation
    --------------------------
    min  t
    s.t. D (u⁺_j − u⁻_j) = Y[:, j]   for j = 0 … d_ext-1   (complex eq.)
         Σ_k (u⁺_{j,k} + u⁻_{j,k}) ≤ t   for each j
         u⁺, u⁻ ≥ 0,  t ≥ 0

    D and gate may be complex.  Each complex equality is split into its
    real and imaginary parts so the LP stays real (u⁺, u⁻, t ∈ R).

    The LP is solved via a direct _linprog_highs call, bypassing scipy's
    per-call input validation.  A_eq is updated in-place (only .data
    changes; sparsity pattern is fixed for any dense D).
    """
    n, d_ext = D.shape

    # Retrieve (or build) the pre-cleaned LP object and its mutable COO view
    lp_clean, coo_eq, blk_nnz = _get_lp_cache(d_ext, n)

    # Stack real and imaginary parts of D so each per-target block has
    # shape (2n, d_ext); this matches the LP cache's sparsity pattern.
    D_stacked = np.vstack([D.real, D.imag])
    d_flat = np.tile(D_stacked.ravel(order='F'), d_ext)
    coo_eq.data[:blk_nnz] = d_flat
    coo_eq.data[blk_nnz:] = -d_flat

    # Y = gate^T D (possibly complex); stack Re/Im rows then column-major flat.
    Y = gate.T @ D
    b_eq = np.vstack([Y.real, Y.imag]).ravel(order='F')
    lp_upd = lp_clean._replace(b_eq=b_eq)

    res = _linprog_highs(lp_upd, solver=None, presolve=False)
    return res['x'][0] if res['status'] == 0 else np.inf


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _project_columns_bloch(M):
    """Project columns of M (shape 4xk) onto |c_I| + ||(c_X,c_Y,c_Z)||_2 <= 1.

    Columns already satisfying the constraint are left untouched; columns
    that exceed it are rescaled so that the sum equals exactly 1.  This
    treats the column constraint as an inequality (the natural feasible
    set for a single-qubit Pauli-expansion column representing a sub-
    normalised state) instead of forcing equality.
    """
    c_I   = np.abs(M[0:1, :])                                  # (1, k)
    bloch = np.linalg.norm(M[1:4, :], axis=0, keepdims=True)  # (1, k)
    total = c_I + bloch
    return M / np.maximum(total, 1.0)


# First column of S is always fixed: identity (1,0,0,0)^T.
_FIXED_COLS = np.array([[1], [0], [0], [0]], dtype=float)  # (4, 1)
N_FIXED_COLS = _FIXED_COLS.shape[1]  # 1


def _kron_power(S, n):
    """Compute kron(S, kron(S, ...)) with n copies of S."""
    result = S
    for _ in range(n - 1):
        result = np.kron(result, S)
    return result


# ---------------------------------------------------------------------------
#  Main optimiser
# ---------------------------------------------------------------------------

def minimize_framability(gate, d_ext_single, *, n_restarts=5,
                         method=None, max_iter=500, maxfev=2000,
                         tol=1e-6, seed=None, verbose=True,
                         extra_init_xs=None, return_x=False,
                         use_complex=None):
    """
    Find D = kron^n_qubits(S) with unit-norm columns of S that minimises
    heisenberg_framability(D, gate).

    D is constrained to be an n_qubits-fold Kronecker product of an identical
    single-qubit frame S of shape (qubit_d², d_ext_single), reducing the
    search space from pauli_string_dim × d_ext to qubit_d² × d_ext_single
    parameters.  The full frame has d_ext = d_ext_single ** n_qubits columns,
    where n_qubits is inferred from pauli_string_dim and qubit_d.

    Parameters
    ----------
    gate : ndarray, shape (pauli_string_dim, pauli_string_dim)
        Real gate / propagator whose framability is to be minimised.
    d_ext_single : int
        Number of columns of the single-qubit frame S.
        The full frame D has d_ext = d_ext_single ** n_qubits columns.
    n_restarts : int
        Random restarts for local methods.  The first restart uses the
        standard extended-Pauli S when d_ext_single == 6 (two qubits).
    method : str | None
        Optimisation algorithm.  None (default) falls back to DEFAULT_METHOD
        which is 'dual_annealing'.

        'dual_annealing'      Recommended.  Cauchy-Lorentz simulated annealing
                              with Powell polishing.  Handles the nonsmooth
                              max-of-LP objective without any smoothness
                              assumption and performs global search in one call.
                              Uses maxfev evaluations total; warm-starts from
                              extra_init_xs[0] when provided (useful for sweeps).
        'subgradient'         Projected numerical subgradient with Polyak step
                              size.  Converges to the global optimum when the
                              objective is convex in S (not guaranteed by the
                              Kronecker structure but often holds in practice).
                              Uses n_restarts × budget iterations.
        'basinhopping'        Basin-hopping with Powell local minimiser.
                              Good when a rough global layout is known but
                              individual basins are hard to escape.
        'differential_evolution'  SciPy DE; exhaustive but slow.
        'cobyqa','Powell','Nelder-Mead'  Local methods with n_restarts random
                              restarts (legacy; COBYQA/Powell assume smoothness).
    max_iter : int
        Max iterations per restart for local methods.
    maxfev : int
        Total function evaluation budget (all methods).
    tol : float
        Convergence tolerance.
    seed : int | None
        Random seed for reproducibility.
    verbose : bool
        Print per-restart progress.
    extra_init_xs : list of ndarray | None
        Additional flat parameter vectors to use as extra restart seeds,
        appended after the standard n_restarts.  Each vector must have
        length qubit_d² * d_ext_single (same as the optimiser's own
        parameter space, i.e. the raw x returned when return_x=True).
    return_x : bool
        If True, return a 3-tuple (D_opt, f_opt, x_opt) where x_opt is
        the raw flat parameter vector for S corresponding to D_opt.
        Default False returns the usual 2-tuple (D_opt, f_opt).

    Returns
    -------
    D_opt : ndarray, shape (pauli_string_dim, d_ext)
        Optimal frame matrix D = kron^n_qubits(S_opt).
    f_opt : float
        Minimal framability value found.
    x_opt : ndarray  (only when return_x=True)
        Raw flat parameter vector for S_opt (seed-compatible with
        extra_init_xs of a subsequent call).
    """
    if method is None:
        method = DEFAULT_METHOD
    rng = np.random.default_rng(seed)
    gate = np.asarray(gate, dtype=complex)

    n_s = qubit_d ** 2                                        # rows of S (4 per qubit)
    n_qubits = int(round(np.log(pauli_string_dim) / np.log(n_s)))  # inferred qubit count
    d_ext = d_ext_single ** n_qubits                          # total columns of D

    # Complex S is feasible only when d_ext >= 2*n_rows: the Heisenberg LP
    # seeks real u such that D u = Y_j (complex equation) which becomes a
    # 2*n_rows × d_ext real system.  For d_ext < 2*n_rows it is overdetermined
    # and generically infeasible; restrict S to real in that case.
    # Caller may override via use_complex= (True/False); None -> auto-select.
    if use_complex is None:
        _use_complex = (d_ext >= 2 * pauli_string_dim)
    else:
        _use_complex = bool(use_complex)

    n_free = d_ext_single - N_FIXED_COLS
    n_params = (2 if _use_complex else 1) * n_s * n_free

    _PENALTY = 1e6   # returned when LP is infeasible (rank-deficient D)

    def objective(params):
        D = _params_to_D(params, n_s, d_ext_single, n_qubits,
                         use_complex=_use_complex)
        f = _get_framability_fast(D, gate)
        return f if np.isfinite(f) else _PENALTY

    result = _run_restarts(
        objective, n_params, d_ext, n_s, d_ext_single, n_qubits,
        rng, n_restarts, method, max_iter, maxfev, tol, verbose,
        extra_init_xs=extra_init_xs,
        use_complex=_use_complex,
    )
    return result if return_x else result[:2]


def _project_x(x, n_s, n_free, *, use_complex=True):
    """Round-trip a flat parameter vector through the Bloch-ball projection."""
    half = n_s * n_free
    free_re = x[:half].reshape(n_s, n_free)
    if use_complex:
        free_im = x[half:].reshape(n_s, n_free)
        free_proj = _project_columns_bloch(free_re + 1j * free_im)
        return np.concatenate([free_proj.real.ravel(), free_proj.imag.ravel()])
    else:
        free_proj = _project_columns_bloch(free_re.astype(float))
        return free_proj.real.ravel().astype(float)


def _run_subgradient(objective, n_params, n_s, d_ext_single, n_qubits, d_ext,
                     rng, n_restarts, maxfev, tol, verbose, *,
                     extra_init_xs=None, use_complex=True):
    """Projected numerical-subgradient descent with Polyak step size.

    Exploits convexity of the objective in D (framability = max of LP values
    is convex in D).  The Kronecker constraint D = kron^n(S) breaks convexity
    in S, but empirically the landscape is well-behaved for the depol-sweep
    gates, and the Polyak step drives fast convergence near the optimum.

    Algorithm
    ---------
    For each restart:
      1. Compute the numerical forward-difference gradient g (n LP calls).
      2. Polyak step: alpha = (f - f_lower) / ||g||^2, with f_lower estimated
         as 95 % of the best value seen so far (conservative lower bound).
         A floor of alpha_0/sqrt(k+1) prevents the step from collapsing.
      3. Project the updated x back onto the Bloch ball.
      4. Track the best iterate (subgradient methods are non-monotone).

    Budget: maxfev function evaluations spread across n_restarts restarts,
    each running (maxfev // n_restarts) // (n_params + 1) iterations.
    """
    n_free = d_ext_single - N_FIXED_COLS
    inits = _build_inits(n_s, d_ext_single, d_ext, n_restarts, rng,
                         extra_init_xs=extra_init_xs,
                         use_complex=use_complex)

    budget_per_restart = maxfev // max(1, n_restarts)
    n_iter = max(5, budget_per_restart // (n_params + 1))
    eps_fd = 1e-5  # forward-difference step

    global_best_val = np.inf
    global_best_D = None
    global_best_x = None

    for restart, x0 in enumerate(inits):
        x = _project_x(x0, n_s, n_free, use_complex=use_complex)
        f = objective(x)

        # Track the best across this restart (subgradient can go uphill).
        local_best_val = f
        local_best_x = x.copy()

        if verbose:
            print(f'  subgradient restart {restart+1}/{len(inits)}: '
                  f'f_init={f:.6f}', flush=True)

        # Estimate initial step from first-iteration gradient norm.
        alpha_0 = None

        for k in range(n_iter):
            # Forward-difference subgradient (n_params LP evaluations).
            grad = np.empty(n_params)
            for i in range(n_params):
                xp = x.copy(); xp[i] += eps_fd
                grad[i] = (objective(xp) - f) / eps_fd

            gnorm_sq = float(np.dot(grad, grad))
            if gnorm_sq < 1e-20:
                break  # at a stationary point

            if alpha_0 is None:
                # Calibrate: one Newton-like step should move by ~0.1.
                alpha_0 = 0.1 / np.sqrt(gnorm_sq)

            # Polyak step: drives f toward f_lower.
            f_lower = 0.95 * local_best_val
            alpha_polyak = (f - f_lower) / gnorm_sq
            # Floor: ensures progress even when Polyak estimate is tiny.
            alpha_floor = alpha_0 / np.sqrt(k + 1)
            alpha = max(alpha_polyak, alpha_floor)

            x_new = x - alpha * grad
            x_new = _project_x(x_new, n_s, n_free, use_complex=use_complex)

            f_new = objective(x_new)
            x, f = x_new, f_new

            if f < local_best_val:
                local_best_val = f
                local_best_x = x.copy()

            if local_best_val < tol:
                break

        D_cand = _params_to_D(local_best_x, n_s, d_ext_single, n_qubits,
                              use_complex=use_complex)
        f_cand = objective(local_best_x)
        if verbose:
            print(f'    -> f_final={f_cand:.6f}  ({n_iter} iters)', flush=True)

        if f_cand < global_best_val:
            global_best_val = f_cand
            global_best_D = D_cand.copy()
            global_best_x = local_best_x.copy()

    return global_best_D, global_best_val, global_best_x


def _run_restarts(objective, n_params, d_ext, n_s, d_ext_single, n_qubits,
                  rng, n_restarts, method, max_iter, maxfev, tol,
                  verbose, *, extra_init_xs=None, use_complex=True):
    """Optimisation driver for Kronecker-structured framability.  Returns (D_opt, f_opt, x_opt).

    Methods
    -------
    'dual_annealing'      Cauchy-Lorentz SA with Powell polishing.  No smoothness
                          assumption; good global search.  Recommended default.
    'basinhopping'        Random basin-hopping around deterministic inits with
                          Powell local minimiser.
    'subgradient'         Projected numerical subgradient w/ Polyak step size.
                          Converges to global optimum when f(S) is convex (not
                          guaranteed due to Kronecker structure, but often holds
                          empirically for the depol-sweep gates).
    'differential_evolution'  SciPy DE; thorough but slow.
    'cobyqa','Powell','Nelder-Mead'  Local methods with random restarts (legacy).
    """

    _BOUNDS = [(-1.5, 1.5)] * n_params   # Bloch-ball values live in [-1, 1]

    # ------------------------------------------------------------------
    # dual_annealing  (global, nonsmooth-safe)
    # ------------------------------------------------------------------
    if method == 'dual_annealing':
        from scipy.optimize import dual_annealing
        # warm-start from the best provided seed (e.g. neighbour's x_opt)
        x0_da = extra_init_xs[0] if extra_init_xs else None
        # maxfun caps total evaluations; maxiter controls SA loop count.
        # With n_params <= ~60 and maxfev=2000 a budget of 1000 SA steps
        # is usually sufficient; the Powell polishing uses up to maxfev//5.
        res = dual_annealing(
            objective, _BOUNDS,
            maxfun=maxfev,
            maxiter=max(200, maxfev // 3),
            x0=x0_da,
            seed=int(rng.integers(2**31)),
            minimizer_kwargs={
                'method': 'Powell',
                'options': {'maxfev': max(100, maxfev // 5),
                            'ftol': tol, 'xtol': tol},
            },
        )
        D_opt = _params_to_D(res.x, n_s, d_ext_single, n_qubits,
                             use_complex=use_complex)
        f_opt = objective(res.x)   # use penalty-wrapped value for consistency
        if verbose:
            print(f'Dual annealing: f = {f_opt:.6f}  nfev={res.nfev}  '
                  f'success={res.success}')
        return D_opt, f_opt, res.x

    # ------------------------------------------------------------------
    # basinhopping  (local restarts with random perturbations)
    # ------------------------------------------------------------------
    if method == 'basinhopping':
        from scipy.optimize import basinhopping
        inits = _build_inits(n_s, d_ext_single, d_ext, n_restarts, rng,
                             extra_init_xs=extra_init_xs,
                             use_complex=use_complex)
        best_val, best_D, best_x = np.inf, None, None
        local_opts = {'method': 'Powell',
                      'options': {'maxfev': max(100, maxfev // n_restarts),
                                  'ftol': tol, 'xtol': tol}}
        for i, x0 in enumerate(inits):
            f_x0 = objective(x0)
            if f_x0 < best_val:
                best_val, best_D, best_x = (
                    f_x0,
                    _params_to_D(x0, n_s, d_ext_single, n_qubits,
                                 use_complex=use_complex),
                    x0.copy())
            res = basinhopping(
                objective, x0,
                niter=max(10, maxfev // (max(1, n_restarts) * 30)),
                minimizer_kwargs=local_opts,
                seed=int(rng.integers(2**31)),
            )
            D_cand = _params_to_D(res.x, n_s, d_ext_single, n_qubits,
                                  use_complex=use_complex)
            f_cand = objective(res.x)
            if verbose:
                print(f'  basinhopping restart {i+1}/{len(inits)}: '
                      f'f_init={f_x0:.6f}  f_opt={f_cand:.6f}')
            if f_cand < best_val:
                best_val, best_D, best_x = f_cand, D_cand.copy(), res.x.copy()
        return best_D, best_val, best_x

    # ------------------------------------------------------------------
    # subgradient  (projected numerical subgradient, Polyak step)
    # ------------------------------------------------------------------
    if method == 'subgradient':
        return _run_subgradient(
            objective, n_params, n_s, d_ext_single, n_qubits, d_ext,
            rng, n_restarts, maxfev, tol, verbose,
            extra_init_xs=extra_init_xs,
            use_complex=use_complex,
        )

    # ------------------------------------------------------------------
    # differential_evolution  (legacy global method)
    # ------------------------------------------------------------------
    if method == 'differential_evolution':
        res = differential_evolution(
            objective, _BOUNDS, maxiter=max_iter, tol=tol,
            seed=int(rng.integers(2**31)), polish=True, workers=1,
        )
        D_opt = _params_to_D(res.x, n_s, d_ext_single, n_qubits,
                             use_complex=use_complex)
        f_opt = objective(res.x)
        if verbose:
            print(f'DE finished:  f = {f_opt:.6f}  (success={res.success})')
        return D_opt, f_opt, res.x

    # ------------------------------------------------------------------
    # local methods with random restarts  (cobyqa / Powell / Nelder-Mead)
    # ------------------------------------------------------------------
    best_val = np.inf
    best_D = None
    best_x = None

    inits = _build_inits(n_s, d_ext_single, d_ext, n_restarts, rng,
                         extra_init_xs=extra_init_xs,
                         use_complex=use_complex)

    if method == 'cobyqa':
        opts = {'maxfev': maxfev}
    elif method == 'Powell':
        opts = {'maxiter': max_iter, 'maxfev': maxfev, 'ftol': tol, 'xtol': tol}
    elif method == 'Nelder-Mead':
        opts = {'maxiter': max_iter, 'maxfev': maxfev, 'fatol': tol, 'xatol': tol}
    else:
        opts = {'maxiter': max_iter, 'maxfev': maxfev}

    for i, x0 in enumerate(inits):
        f_x0 = objective(x0)
        if f_x0 < best_val:
            best_val = f_x0
            best_D = _params_to_D(x0, n_s, d_ext_single, n_qubits,
                                  use_complex=use_complex)
            best_x = x0.copy()

        res = minimize(objective, x0, method=method, options=opts)

        D_cand = _params_to_D(res.x, n_s, d_ext_single, n_qubits,
                               use_complex=use_complex)
        f_cand = objective(res.x)

        if verbose:
            n_pauli_inits = 1 if d_ext == 36 else 0
            if i >= n_restarts:
                tag = 'neighbor seed'
            elif d_ext == 36 and i == 0:
                tag = 'ext-Pauli init'
            elif i == n_pauli_inits:
                tag = 'identity init'
            else:
                tag = 'random init'
            print(f'  restart {i + 1}/{len(inits)} ({tag}):  '
                  f'f_init={f_x0:.6f}  f_opt={f_cand:.6f}  (success={res.success})')

        if f_cand < best_val:
            best_val = f_cand
            best_D = D_cand.copy()
            best_x = res.x.copy()

    return best_D, best_val, best_x


def _params_to_D(params, n_s, d_ext_single, n_qubits, *, use_complex=True):
    """Decode flat parameter vector into D = kron^n_qubits(S).

    When use_complex=True (default, requires d_ext >= 2*n_rows):
        params = [Re_flat | Im_flat] of the n_free free columns of S.
    When use_complex=False (for square/undercomplete D):
        params = Re_flat only; imaginary parts are forced to zero so D is real.

    The first N_FIXED_COLS column of S is always fixed to _FIXED_COLS (identity).
    Free columns are normalised to satisfy |c_I| + ||(c_X,c_Y,c_Z)||_2 <= 1.
    """
    n_free = d_ext_single - N_FIXED_COLS
    half = n_s * n_free
    free_re = params[:half].reshape(n_s, n_free)
    if use_complex:
        free_im = params[half:].reshape(n_s, n_free)
        free = _project_columns_bloch(free_re + 1j * free_im)
        S = np.hstack([_FIXED_COLS.astype(complex), free])
    else:
        free = _project_columns_bloch(free_re.astype(float))
        S = np.hstack([_FIXED_COLS, free]).astype(float)
    return _kron_power(S, n_qubits)


def _build_inits(n_s, d_ext_single, d_ext, n_restarts, rng,
                 extra_init_xs=None, use_complex=True):
    """Build a list of initial flat parameter vectors for S (shape n_s × d_ext_single).

    Fixed seeds (always included, in order):
      1. Extended-Pauli S — only when d_ext == 36 (d_ext_single == 6, two qubits).
      2. Cycling-identity S — always included.  Columns of S are standard basis
         vectors cycling through indices 0..n_s-1.

    Random seeds fill slots up to *n_restarts* total.  Any vectors in
    *extra_init_xs* are appended afterwards (e.g. a neighbor's x_opt).

    When use_complex=False, parameter vectors contain only the real part
    (length n_s * n_free instead of 2 * n_s * n_free).
    """
    inits = []
    n_free = d_ext_single - N_FIXED_COLS

    def _pack(M):
        M = np.asarray(M)
        if use_complex:
            M = M.astype(complex)
            return np.concatenate([M.real.ravel(), M.imag.ravel()])
        else:
            return M.real.ravel().astype(float)

    # First init: free columns of extended-Pauli S (when d_ext == 36)
    if d_ext == 36:
        a = 1
        S_pauli = np.array(
            [[1, 0, 0, 0, 0,             0],
             [0, 1, 0, 0, a/np.sqrt(2),  a/np.sqrt(2)],
             [0, 0, 1, 0, 0,             0],
             [0, 0, 0, 1, a/np.sqrt(2), -a/np.sqrt(2)]])
        free_pauli = _project_columns_bloch(S_pauli[:, N_FIXED_COLS:].astype(
            complex if use_complex else float))
        inits.append(_pack(free_pauli))

    # Second init: cycling-identity for the free columns.
    S_id_free = np.zeros((n_s, n_free), dtype=complex if use_complex else float)
    for j in range(n_free):
        S_id_free[(j + N_FIXED_COLS) % n_s, j] = 1.0
    inits.append(_pack(_project_columns_bloch(S_id_free)))

    while len(inits) < n_restarts:
        if use_complex:
            M = (rng.standard_normal((n_s, n_free))
                 + 1j * rng.standard_normal((n_s, n_free)))
        else:
            M = rng.standard_normal((n_s, n_free))
        inits.append(_pack(_project_columns_bloch(M)))

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
    d_ext = D_pauli.shape[1]           # 36
    d_ext_single = int(round(np.sqrt(d_ext)))  # 6

    t0 = time.perf_counter()
    f_primal = heisenberg_framability(D_pauli, gate)
    t_primal = time.perf_counter() - t0

    t0 = time.perf_counter()
    f_fast = _get_framability_fast(D_pauli, gate)
    t_fast = time.perf_counter() - t0

    print(f'Primal LP framability: {f_primal:.6f}  ({t_primal*1000:.1f} ms)')
    print(f'Batch  LP framability: {f_fast:.6f}  ({t_fast*1000:.1f} ms)')
    print()

    # Optimise (Kronecker structure: 24 params instead of 576)
    print(f'Optimising (kron structure, d_ext_single={d_ext_single}, '
          f'method={DEFAULT_METHOD}, maxfev=1000) ...')
    t0 = time.perf_counter()
    D_opt, f_opt = minimize_framability(
        gate, d_ext_single=d_ext_single, n_restarts=3,
        method=DEFAULT_METHOD, max_iter=200, maxfev=1000,
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
