"""
Optimise framability over extended Pauli-like frames D.

Problem (Heisenberg picture, minimize_framability)
--------------------------------------------------
    min_D  heisenberg_framability(D, gate)

    s.t.  D = kron^n_qubits(S)         (Kronecker structure)
          |c_I| + ||(c_X, c_Y, c_Z)||_2 <= 1  for every column of S
                                              (Bloch-norm bound)

Problem (Schrödinger picture, minimize_schroedinger_framability)
----------------------------------------------------------------
    min_D  schroedinger_framability(D, gate)

    s.t.  D = kron^n_qubits(S)         (Kronecker structure)
          S[I, j]  = 1/2                for every column j of S
          ||(S[X,j], S[Y,j], S[Z,j])||_2 <= 1/2
                                        (every column is a legal quantum state)
          max_j |S[a, j]| >= SUPPORT_EPS  for each a in {X, Y, Z}
                                        (every Pauli has non-negligible
                                         support on some frame element)

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
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.optimize._linprog_highs import _linprog_highs
from scipy.optimize._linprog_util import _LPProblem, _clean_inputs

from two_qubit_lindbladian import pauli_string_dim, qubit_d
from framability import heisenberg_framability, extended_pauli_D

# ---------------------------------------------------------------------------
#  Code version stamp
# ---------------------------------------------------------------------------
# Bumped whenever the framability optimisation changes in a way that makes
# previously cached results stale.  Downstream workers store this string in
# their output and re-run the optimisation when the stored stamp differs from
# the current one (see depol_kron_worker, dissipative_PT, scan_worker).
#
#   2.0-dual-floor : dual certificate + spectral-radius floor, analytic
#                    subgradient, smooth conditioning penalty (replaces the
#                    hard 1e6 infeasibility cliff).
OPT_VERSION = '2.0-dual-floor'


def spectral_floor(gate) -> float:
    """Spectral radius of the gate's transfer matrix = floor on framability.

    Framability is an induced operator norm of ``gate`` (the gauge of the
    frame polytope), and every induced norm is bounded below by the spectral
    radius rho(gate) = max_i |lambda_i|.  No frame — of any size or shape —
    can push the framability below this value, so it is the achievable floor
    (reached in the limit by a norm in which the gate is an isometry).

    For a unitary channel rho = 1; for a dissipative (CPTP) channel rho <= 1.
    """
    gate = np.asarray(gate)
    eig = np.linalg.eigvals(gate)
    return float(np.max(np.abs(eig)))


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

_LP_CACHE: dict = {}   # (d_ext, n) -> (lp_clean, coo_eq, blk_nnz)


def _get_lp_cache(d_ext: int, n: int):
    """Return (lp_clean, coo_eq, blk_nnz) for the batched LP, cached by (d_ext, n).

    Each complex equality D u_j = Y[:, j] is split into its real and imaginary
    parts, so n_eq = 2 * n * d_ext and the per-target block is [Re(D); Im(D)]
    stacked vertically (shape 2n × d_ext).
    """
    key = (d_ext, n)
    if key not in _LP_CACHE:
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

        _LP_CACHE[key] = (lp_clean, coo_eq, blk_nnz)
    return _LP_CACHE[key]


# ---------------------------------------------------------------------------
#  Fast framability: single batched LP  (one linprog call instead of d_ext)
# ---------------------------------------------------------------------------

def _get_framability_fast(D, gate, return_norms=False):
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
    if res['status'] != 0:
        if return_norms:
            return np.inf, None
        return np.inf

    val = res['x'][0]
    if not return_norms:
        return val

    # Recover the per-target-column atomic norms s_j = Σ_k (u⁺+u⁻)_{jk}.
    # Variable layout: x = [t, u⁺ (d_ext²), u⁻ (d_ext²)]; the coefficient
    # vector for target column j occupies the contiguous block j*d_ext:(j+1)*d_ext
    # (kron(eye(d_ext), D) is block-diagonal with one D-block per target).
    n_up = d_ext * d_ext
    u_plus  = res['x'][1:1 + n_up]
    u_minus = res['x'][1 + n_up:1 + 2 * n_up]
    col_norms = (u_plus + u_minus).reshape(d_ext, d_ext).sum(axis=1)
    return val, col_norms


def framability_certificate(D, gate):
    """Return (value, gap-free) certificate data for a real frame/gate.

    Returns a dict with:
        value    : the framability  max_j ||gateᵀ d_j||_{A(D)}
        argmax   : index j* of the binding frame column
        witness  : dual vector w* ∈ R^nrows with <w*, gateᵀ d_{j*}> = value
                   and ||Dᵀ w*||_∞ ≤ 1  (certifies the value of column j*)
        col_norms: per-column atomic norms (length d_ext)

    The witness is the Lagrange multiplier of the binding column; its outer
    products give the analytic subgradient of the framability w.r.t. D
    (see framability_value_and_grad).  Real D and real gate only — the case
    used by every production pipeline (gates are real superoperators).

    Cost: one batched primal LP (value + binding column) plus a single small
    dual LP for the witness — far cheaper than the per-parameter
    finite-difference gradient it replaces.
    """
    D = np.asarray(D, dtype=float)
    gate = np.asarray(gate).real
    nrows, d_ext = D.shape

    val, col_norms = _get_framability_fast(D, gate, return_norms=True)
    if not np.isfinite(val) or col_norms is None:
        return dict(value=np.inf, argmax=-1, witness=None, col_norms=None)

    j_star = int(np.argmax(col_norms))
    y = gate.T @ D[:, j_star]            # target = gateᵀ d_{j*}

    # Dual LP for the binding column:  max <w, y>  s.t.  -1 ≤ Dᵀ w ≤ 1.
    # linprog minimises, so minimise -<w, y>.
    DT = D.T
    res = linprog(c=-y, A_ub=np.vstack([DT, -DT]), b_ub=np.ones(2 * d_ext),
                  bounds=[(None, None)] * nrows, method='highs')
    witness = res.x.copy() if res.success else None
    return dict(value=float(val), argmax=j_star, witness=witness,
                col_norms=col_norms)


def framability_value_and_grad(S, gate, n_qubits=2):
    """Framability value and its analytic (sub)gradient w.r.t. the frame S.

    For D = S⊗…⊗S (n_qubits copies) and a real gate, returns (value, dS)
    where dS has the shape of S and is a subgradient of
    Φ(S) = max_j ||gateᵀ (S^{⊗n})_j||_{A(D)} at S.

    Derivation
    ----------
    With the binding column j* (witness w*, primal coefficients u*) the
    envelope theorem gives the subgradient w.r.t. D:

        ∂Φ/∂D = (gate · w*) e_{j*}ᵀ  −  w* (u*)ᵀ            (nrows × d_ext)

    which is then chain-ruled through the Kronecker structure D = S⊗S.

    Only n_qubits == 2 is handled analytically (the production case); other
    values raise NotImplementedError so callers fall back to finite
    differences.
    """
    if n_qubits != 2:
        raise NotImplementedError(
            'analytic gradient implemented only for n_qubits == 2')

    S = np.asarray(S, dtype=float)
    gate = np.asarray(gate).real
    n_s, m = S.shape
    D = _kron_power(S, 2)               # (n_s², m²)

    cert = framability_certificate(D, gate)
    val, j_star, w = cert['value'], cert['argmax'], cert['witness']
    if not np.isfinite(val) or w is None:
        return val, np.zeros_like(S)

    # Recover the binding column's primal coefficients u* (min-1-norm
    # representation of gateᵀ d_{j*} in D) via one atomic-norm LP.
    y = gate.T @ D[:, j_star]
    nrows, d_ext = D.shape
    c = np.concatenate([np.ones(d_ext), np.ones(d_ext)])
    res = linprog(c=c, A_eq=np.hstack([D, -D]), b_eq=y,
                  bounds=[(0, None)] * (2 * d_ext), method='highs')
    if not res.success:
        return val, np.zeros_like(S)
    u = res.x[:d_ext] - res.x[d_ext:]

    # Subgradient w.r.t. D:  (gate·w) e_{j*}ᵀ − w uᵀ
    gD = np.zeros((nrows, d_ext))
    gD[:, j_star] += gate @ w
    gD -= np.outer(w, u)

    # Chain rule through D = S⊗S:  D_{(a,b),(i,j)} = S_{a,i} S_{b,j}.
    T = gD.reshape(n_s, n_s, m, m)     # T[a,b,i,j]
    dS = np.einsum('abij,bj->ai', T, S) + np.einsum('baji,bj->ai', T, S)
    return val, dS


def _check_subgradient_against_fd(seed=0, eps=1e-6):
    """Self-test: compare the analytic subgradient to a finite difference.

    Validates framability_value_and_grad.  Run manually:

        python -c "import optimize_framability as o; o._check_subgradient_against_fd()"

    Note: the framability is nonsmooth where two frame columns tie for the
    binding maximum.  At such a point the analytic value is a valid
    subgradient but a one-sided finite difference measures a different
    directional derivative, so a large error there is expected, not a bug —
    the check reports the tie so it is not misread.  At differentiable points
    (unique binding column) the agreement is ~1e-6.
    """
    rng = np.random.default_rng(seed)
    gate = rng.standard_normal((pauli_string_dim, pauli_string_dim)) * 0.3
    n_s = qubit_d ** 2
    S = _project_columns_bloch(rng.standard_normal((n_s, 5)))
    val, dS = framability_value_and_grad(S, gate, n_qubits=2)

    cert = framability_certificate(_kron_power(S, 2), gate)
    cn = np.sort(cert['col_norms'])[::-1]
    binding_gap = float(cn[0] - cn[1])     # 0 => nonsmooth (tied columns)

    fd = np.zeros_like(S)
    for a in range(S.shape[0]):
        for i in range(S.shape[1]):
            Sp = S.copy(); Sp[a, i] += eps
            vp = _get_framability_fast(_kron_power(Sp, 2), gate)
            fd[a, i] = (vp - val) / eps
    err = np.linalg.norm(dS - fd) / max(np.linalg.norm(fd), 1e-12)
    tag = '  [nonsmooth: tied binding columns]' if binding_gap < 1e-9 else ''
    print(f'analytic vs FD subgradient relative error: {err:.3e}  '
          f'(binding gap {binding_gap:.2e}){tag}')
    return err


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
                         use_complex=None, return_floor=False):
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
        Optimisation algorithm.  None (default) falls back to DEFAULT_METHOD,
        which is 'Nelder-Mead' (a local simplex method run with n_restarts
        random restarts).

        'dual_annealing'      Cauchy-Lorentz simulated annealing with Powell
                              polishing.  Handles the nonsmooth max-of-LP
                              objective without any smoothness assumption and
                              performs global search in one call; use it when a
                              large budget (maxfev >> 5000) is available.
                              Warm-starts from extra_init_xs[0] when provided.
        'subgradient'         Projected subgradient with Polyak step size, using
                              the analytic subgradient from the LP dual witness
                              (framability_value_and_grad) for n_qubits == 2 and
                              a real gate, else a finite-difference fallback.
                              NOTE: Φ(S) is NOT convex — the framability is
                              convex in D, but D appears in both the dictionary
                              and the target (Y = gateᵀD), and D = S⊗S adds
                              further nonconvexity.  This method is therefore a
                              local descent with no global guarantee; rely on
                              restarts and the spectral-radius floor (see
                              return_floor) to judge how close it gets.
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
        If True, append the raw flat parameter vector x_opt for S_opt to the
        returned tuple.  Default False.
    return_floor : bool
        If True, append the spectral-radius floor (see spectral_floor) to the
        returned tuple — the achievable lower bound on framability for *any*
        frame, reported whether or not the optimisation reached it.

    Returns
    -------
    By default a 2-tuple (D_opt, f_opt).  With the flags set, the extra values
    are appended in this order: x_opt (return_x), floor (return_floor).
    E.g. return_x=True, return_floor=True  ->  (D_opt, f_opt, x_opt, floor).

    D_opt : ndarray, shape (pauli_string_dim, d_ext)
        Optimal frame matrix D = kron^n_qubits(S_opt).
    f_opt : float
        Minimal framability value found.
    x_opt : ndarray  (only when return_x=True)
        Raw flat parameter vector for S_opt (seed-compatible with
        extra_init_xs of a subsequent call).
    floor : float  (only when return_floor=True)
        Spectral radius of the gate = framability floor.
    """
    if method is None:
        method = DEFAULT_METHOD
    rng = np.random.default_rng(seed)
    gate = np.asarray(gate, dtype=complex)
    floor = spectral_floor(gate)

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

    # When the frame D is (near) rank-deficient the framability LP is
    # infeasible.  Instead of a flat 1e6 cliff (a discontinuous wall that
    # derivative-free methods stumble on and gradients cannot see across),
    # return a barrier that *grows smoothly* as D loses rank, so the optimiser
    # is pushed back toward well-conditioned frames.
    _PENALTY_BASE = 1e3

    def objective(params):
        D = _params_to_D(params, n_s, d_ext_single, n_qubits,
                         use_complex=_use_complex)
        f = _get_framability_fast(D, gate)
        if np.isfinite(f):
            return f
        # smallest singular value -> 0 as the frame collapses; 1/sigma_min
        # gives a finite-valued, monotone barrier away from the cliff.
        sigma_min = float(np.linalg.svd(D, compute_uv=False)[-1])
        return _PENALTY_BASE * (1.0 + 1.0 / max(sigma_min, 1e-12))

    result = _run_restarts(
        objective, n_params, d_ext, n_s, d_ext_single, n_qubits,
        rng, n_restarts, method, max_iter, maxfev, tol, verbose,
        extra_init_xs=extra_init_xs,
        use_complex=_use_complex, gate=gate,
    )

    if verbose:
        f_opt = result[1]
        print(f'floor (spectral radius) = {floor:.6f}   '
              f'framability = {f_opt:.6f}   gap = {f_opt - floor:.6f}')

    out = list(result) if return_x else list(result[:2])
    if return_floor:
        out.append(floor)
    return tuple(out)


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
                     extra_init_xs=None, use_complex=True, gate=None):
    """Projected subgradient descent with Polyak step size.

    NOT a convex method.  The framability is convex in D, but D enters both
    the dictionary and the target (Y = gateᵀD), and D = S⊗S adds further
    nonconvexity, so Φ(S) is nonconvex with no global guarantee — this is a
    local descent leaning on restarts (and the spectral-radius floor) for
    confidence.

    Gradient source
    ---------------
    For a real gate with n_qubits == 2 the analytic subgradient from the LP
    dual witness (framability_value_and_grad) is used — two LP solves per
    step instead of the n_params solves a finite difference needs.  Otherwise
    (complex S, or n_qubits != 2) it falls back to forward differences.

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

    # Analytic subgradient available only for a real gate on two qubits.
    _use_analytic = (gate is not None and not use_complex and n_qubits == 2
                     and np.max(np.abs(np.asarray(gate).imag)) < 1e-12)
    gate_real = np.asarray(gate).real if _use_analytic else None

    def _analytic_grad(x_flat):
        """Flat parameter-space subgradient from the LP-dual witness."""
        free = x_flat.reshape(n_s, n_free)
        S = np.hstack([_FIXED_COLS, _project_columns_bloch(free)])
        _, dS = framability_value_and_grad(S, gate_real, n_qubits=2)
        return dS[:, N_FIXED_COLS:].ravel()

    budget_per_restart = maxfev // max(1, n_restarts)
    # An FD gradient costs n_params LP solves per step; the analytic gradient
    # costs ~2, so it affords many more steps for the same budget.
    cost_per_step = 2 if _use_analytic else (n_params + 1)
    n_iter = max(5, budget_per_restart // cost_per_step)
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
            if _use_analytic:
                # Analytic subgradient from the LP dual witness (~2 LP solves).
                grad = _analytic_grad(x)
            else:
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
                  verbose, *, extra_init_xs=None, use_complex=True, gate=None):
    """Optimisation driver for Kronecker-structured framability.  Returns (D_opt, f_opt, x_opt).

    Methods
    -------
    'dual_annealing'      Cauchy-Lorentz SA with Powell polishing.  No smoothness
                          assumption; good global search (large budgets).
    'basinhopping'        Random basin-hopping around deterministic inits with
                          Powell local minimiser.
    'subgradient'         Projected subgradient w/ Polyak step size, using the
                          analytic LP-dual gradient for real 2-qubit gates.
                          Φ(S) is nonconvex, so this is local descent with no
                          global guarantee (see _run_subgradient).
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
            use_complex=use_complex, gate=gate,
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
#  Schrödinger-framability optimiser  (state frames)
# ---------------------------------------------------------------------------
# Independent version stamp so cached Heisenberg results keyed on OPT_VERSION
# are not invalidated by changes to the Schrödinger optimiser.
#
#   1.0-state-frame : columns of S are legal quantum states (c_I = 1/2,
#                     Bloch norm <= 1/2), Pauli-support penalty, Nelder-Mead
#                     restarts over octahedron / SIC / random Bloch inits.
SCHRO_OPT_VERSION = '1.0-state-frame'

# State-frame column convention  c_a = Tr(sigma_a rho) / 2  (as used by
# _single_qubit_dyadic_D / make_product_state_D in framability.py):
# a legal single-qubit state has identity coefficient exactly 1/2 and
# Bloch part ||(c_X, c_Y, c_Z)||_2 <= 1/2 (equality <=> pure state).
STATE_C_I = 0.5

# Each Pauli X, Y, Z must have non-negligible support on at least one frame
# element:  max_j |S[a, j]| >= SUPPORT_EPS  for every a in {X, Y, Z}.
# Enforced by a smooth quadratic penalty (weight _SUPPORT_WEIGHT) rather than
# a hard wall, so the simplex is steered back into the feasible region.
SUPPORT_EPS = 1e-2
_SUPPORT_WEIGHT = 1e3


def _project_columns_state(B):
    """Project Bloch columns of B (shape 3 x k) onto the ball ||b||_2 <= 1/2.

    Columns already inside the Bloch ball are left untouched; longer columns
    are rescaled to length exactly 1/2 (pure states on the Bloch sphere).
    """
    norms = np.linalg.norm(B, axis=0, keepdims=True)
    return B / np.maximum(2.0 * norms, 1.0)


def _state_params_to_S(params, d_ext_single):
    """Decode flat Bloch parameters into a state frame S (4 x d_ext_single).

    Row 0 (identity coefficient) is fixed to STATE_C_I = 1/2 for every column;
    rows 1-3 are the Bloch vectors, projected onto ||b||_2 <= 1/2, so every
    column of S is a legal (possibly mixed) single-qubit density matrix.
    """
    B = np.asarray(params, dtype=float).reshape(3, d_ext_single)
    B = _project_columns_state(B)
    return np.vstack([np.full((1, d_ext_single), STATE_C_I), B])


def _pauli_support_penalty(S):
    """Penalty for Paulis with negligible support on every frame element.

    For each a in {X, Y, Z} the support is max_j |S[a, j]|.  Any support
    below SUPPORT_EPS contributes a quadratic deficit term; the penalty is
    zero on the feasible set and grows continuously as a Pauli row collapses,
    so Nelder-Mead sees a slope back toward feasibility instead of a cliff.
    """
    support = np.max(np.abs(S[1:4, :]), axis=1)             # (3,) per Pauli
    deficit = np.maximum(0.0, SUPPORT_EPS - support) / SUPPORT_EPS
    return _SUPPORT_WEIGHT * float(np.sum(deficit ** 2))


def _build_state_inits(d_ext_single, n_restarts, rng, extra_init_xs=None):
    """Build (inits, tags): flat Bloch-parameter vectors for the state frame.

    Deterministic seeds (in order, when d_ext_single allows):
      1. Octahedron / stabilizer states +-X, +-Y, +-Z (needs >= 6 columns;
         any extra columns are filled with random Bloch vectors).
      2. SIC tetrahedron (needs >= 4 columns; extra columns random).

    Random Bloch seeds fill the remaining slots up to n_restarts total; any
    vectors in extra_init_xs (length 3 * d_ext_single each, e.g. a neighbour's
    x_opt) are appended afterwards.
    """
    def _rand_bloch(k):
        return _project_columns_state(0.5 * rng.standard_normal((3, k)))

    def _fill(B_det):
        k_det = B_det.shape[1]
        if k_det < d_ext_single:
            return np.hstack([B_det, _rand_bloch(d_ext_single - k_det)])
        return B_det

    inits, tags = [], []

    octa = 0.5 * np.array([[1., -1., 0.,  0., 0.,  0.],
                           [0.,  0., 1., -1., 0.,  0.],
                           [0.,  0., 0.,  0., 1., -1.]])
    if d_ext_single >= 6:
        inits.append(_fill(octa).ravel())
        tags.append('octahedron init')

    tetra = (0.5 / np.sqrt(3.0)) * np.array([[1.,  1., -1., -1.],
                                             [1., -1.,  1., -1.],
                                             [1., -1., -1.,  1.]])
    inits.append(_fill(tetra).ravel())
    tags.append('SIC-tetrahedron init')

    while len(inits) < n_restarts:
        inits.append(_rand_bloch(d_ext_single).ravel())
        tags.append('random init')

    if extra_init_xs:
        for x in extra_init_xs:
            inits.append(np.asarray(x, dtype=float))
            tags.append('neighbor seed')

    return inits, tags


def minimize_schroedinger_framability(gate, d_ext_single, *, n_restarts=5,
                                      method=None, max_iter=500, maxfev=2000,
                                      tol=1e-6, seed=None, verbose=True,
                                      extra_init_xs=None, return_x=False,
                                      return_floor=False):
    """
    Find a state frame D = kron^n_qubits(S) minimising the Schrödinger
    framability of `gate`: max_j min { ||v||_1 : D v = gate d_j }.

    Constraints on the single-qubit frame S (shape qubit_d² x d_ext_single):
      * every column is a legal quantum state in the Pauli convention
        c_a = Tr(sigma_a rho) / 2 — identity coefficient fixed to 1/2 and
        Bloch part with 2-norm <= 1/2 (STATE_C_I, _project_columns_state);
      * every Pauli X, Y, Z has support >= SUPPORT_EPS on at least one
        column of S (soft quadratic penalty, _pauli_support_penalty).

    The frame is real (state columns are Hermitian operators), so the search
    space is the 3 * d_ext_single free Bloch components.  The Schrödinger LP
    is evaluated through the same batched fast LP as the Heisenberg picture:
    _get_framability_fast computes targets g.T @ D, so it is called with
    g = gate.T, giving targets gate @ D (gate applied forward to the states).

    Parameters
    ----------
    gate : ndarray, shape (pauli_string_dim, pauli_string_dim)
        Real gate / propagator in the Pauli-string basis.
    d_ext_single : int
        Number of state columns of S; must be >= 4 (fewer states cannot span
        the single-qubit operator space, making the LP always infeasible).
        The full frame D has d_ext = d_ext_single ** n_qubits columns.
    n_restarts : int
        Restart count for local methods.  The first restarts use the
        octahedron (stabilizer) and SIC-tetrahedron state frames when
        d_ext_single allows; the rest are random Bloch vectors.
    method : str | None
        None (default) falls back to DEFAULT_METHOD ('Nelder-Mead', the
        simplex method — no smoothness assumption, robust on the nonsmooth
        max-of-LP objective).  'dual_annealing' runs a global search within
        the Bloch box; 'Powell' / 'cobyqa' are accepted as local
        alternatives with the same restart scheme.
    max_iter, maxfev, tol, seed, verbose :
        As in minimize_framability.
    extra_init_xs : list of ndarray | None
        Extra restart seeds, each of length 3 * d_ext_single (the raw x
        returned when return_x=True — seed-compatible across calls).
    return_x : bool
        Append the raw flat Bloch-parameter vector x_opt to the return tuple.
    return_floor : bool
        Append the spectral-radius floor (spectral_floor) to the return tuple.

    Returns
    -------
    (D_opt, f_opt[, x_opt][, floor])  — same convention as
    minimize_framability.  D_opt has shape (pauli_string_dim, d_ext) and
    every column is a product of legal single-qubit states; f_opt is the
    best objective value (equal to the Schrödinger framability whenever the
    Pauli-support constraint is satisfied, since the penalty is then zero).
    """
    if method is None:
        method = DEFAULT_METHOD
    if d_ext_single < 4:
        raise ValueError(
            f'd_ext_single must be >= 4 for a state frame (got {d_ext_single}): '
            f'fewer states cannot span the single-qubit operator space, so '
            f'the Schrödinger LP is always infeasible.')

    gate = np.asarray(gate)
    if np.max(np.abs(gate.imag)) > 1e-12:
        raise ValueError(
            'The gate has a non-negligible imaginary part; the state-frame '
            'Schrödinger LP requires a real Pauli-basis superoperator.')
    gate = gate.real.astype(float)

    rng = np.random.default_rng(seed)
    floor = spectral_floor(gate)

    n_s = qubit_d ** 2
    n_qubits = int(round(np.log(pauli_string_dim) / np.log(n_s)))
    n_params = 3 * d_ext_single

    # _get_framability_fast solves for targets g.T @ D; passing g = gate.T
    # yields targets gate @ D — the Schrödinger picture.
    gate_lp = np.ascontiguousarray(gate.T)

    _PENALTY_BASE = 1e3

    def objective(params):
        S = _state_params_to_S(params, d_ext_single)
        pen = _pauli_support_penalty(S)
        D = _kron_power(S, n_qubits)
        f = _get_framability_fast(D, gate_lp)
        if np.isfinite(f):
            return f + pen
        # Rank-deficient frame: smooth barrier as in minimize_framability.
        sigma_min = float(np.linalg.svd(D, compute_uv=False)[-1])
        return _PENALTY_BASE * (1.0 + 1.0 / max(sigma_min, 1e-12)) + pen

    inits, tags = _build_state_inits(d_ext_single, n_restarts, rng,
                                     extra_init_xs=extra_init_xs)

    # ------------------------------------------------------------------
    # dual_annealing  (global, nonsmooth-safe)
    # ------------------------------------------------------------------
    if method == 'dual_annealing':
        from scipy.optimize import dual_annealing
        # Bloch components live in [-1/2, 1/2]; a slightly wider box lets
        # the projection absorb boundary proposals.
        bounds = [(-0.6, 0.6)] * n_params
        x0_da = np.clip(inits[0], -0.6, 0.6)
        res = dual_annealing(
            objective, bounds,
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
        best_x, best_val = res.x.copy(), objective(res.x)
        if verbose:
            print(f'Dual annealing: f = {best_val:.6f}  nfev={res.nfev}  '
                  f'success={res.success}')
    # ------------------------------------------------------------------
    # local methods with restarts  (Nelder-Mead default / Powell / cobyqa)
    # ------------------------------------------------------------------
    else:
        if method == 'Nelder-Mead':
            opts = {'maxiter': max_iter, 'maxfev': maxfev,
                    'fatol': tol, 'xatol': tol}
        elif method == 'Powell':
            opts = {'maxiter': max_iter, 'maxfev': maxfev,
                    'ftol': tol, 'xtol': tol}
        elif method == 'cobyqa':
            opts = {'maxfev': maxfev}
        else:
            opts = {'maxiter': max_iter, 'maxfev': maxfev}

        best_val, best_x = np.inf, None
        for i, x0 in enumerate(inits):
            f_x0 = objective(x0)
            if f_x0 < best_val:
                best_val, best_x = f_x0, np.asarray(x0, dtype=float).copy()

            res = minimize(objective, x0, method=method, options=opts)
            f_cand = objective(res.x)

            if verbose:
                print(f'  restart {i + 1}/{len(inits)} ({tags[i]}):  '
                      f'f_init={f_x0:.6f}  f_opt={f_cand:.6f}  '
                      f'(success={res.success})')

            if f_cand < best_val:
                best_val, best_x = f_cand, res.x.copy()

    S_opt = _state_params_to_S(best_x, d_ext_single)
    D_opt = _kron_power(S_opt, n_qubits)

    if verbose:
        support = np.max(np.abs(S_opt[1:4, :]), axis=1)
        print(f'floor (spectral radius) = {floor:.6f}   '
              f'schro framability = {best_val:.6f}   '
              f'gap = {best_val - floor:.6f}')
        print(f'Pauli support on frame:  X={support[0]:.4f}  '
              f'Y={support[1]:.4f}  Z={support[2]:.4f}  '
              f'(required >= {SUPPORT_EPS})')

    out = [D_opt, float(best_val)]
    if return_x:
        out.append(best_x)
    if return_floor:
        out.append(floor)
    return tuple(out)


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

    # ------------------------------------------------------------------
    # Schrödinger picture: optimal state frame
    # ------------------------------------------------------------------
    from framability import schroedinger_framability

    print(f'\nOptimising Schrödinger framability (state frame, '
          f'd_ext_single={d_ext_single}, method={DEFAULT_METHOD}, '
          f'maxfev=1000) ...')

    # Baseline: octahedron (stabilizer-state) frame, +-X, +-Y, +-Z.
    octa = 0.5 * np.array([[1., -1., 0.,  0., 0.,  0.],
                           [0.,  0., 1., -1., 0.,  0.],
                           [0.,  0., 0.,  0., 1., -1.]])
    S_octa = np.vstack([np.full((1, 6), STATE_C_I), octa])
    D_octa = _kron_power(S_octa, 2)
    f_octa = schroedinger_framability(D_octa, gate)
    print(f'Octahedron-frame Schrödinger framability: {f_octa:.6f}')

    t0 = time.perf_counter()
    D_schro, f_schro = minimize_schroedinger_framability(
        gate, d_ext_single=d_ext_single, n_restarts=3,
        method=DEFAULT_METHOD, max_iter=200, maxfev=1000,
        seed=42, verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f'\nOptimal Schrödinger framability: {f_schro:.6f}  ({elapsed:.1f} s)')
    f_check = schroedinger_framability(D_schro, gate)
    print(f'Reference LP cross-check:        {f_check:.6f}')
    delta_s = f_octa - f_schro
    print(f'Improvement over octahedron frame: {delta_s:.6f}'
          f'  ({100 * delta_s / f_octa:.1f}%)')
