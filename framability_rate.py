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
the identity (1,0,0,0) and stays normalised; every other column satisfies
|c_I| + ||(c_X, c_Y, c_Z)||_2 <= 1 (operator norm <= 1, so columns may shrink);
S must have full rank (D rank 16) or the certificate is vacuous and mu = +inf.

Floor: mu*(D) >= max_i Re lambda_i(A) (the gauge is a norm, so
||exp(tA)|| >= rho(exp(tA))); for a trace-preserving generator that is 0, and
the pinned identity column attains it exactly (A e_II = 0 -> h_0 = 0), so
mu* = max(0, coherence rate) just as framability = max(1, margin).

Public API
----------
generator_log_norm(D, A)            batched LP, mu*(D) [+ per-column values, H]
generator_log_norm_reference(D, A)  independent per-column LP (confirmation)
rate_certificate(D, A)              value, binding column, dual witness
rate_value_and_grad(S, A)           mu*(S (x) S) and its analytic subgradient
polyak_rate_polish(S, A)            Polyak subgradient polish toward the floor
minimize_rate(A, d_ext_single)      alternating certificate optimiser over S
framability_bound(mu, dt)           exp(dt * mu)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import (csc_matrix, kron as sp_kron, eye as sp_eye,
                          hstack as sp_hstack)

from optimize_framability import (
    _kron_power, _project_columns_bloch, _FIXED_COLS, N_FIXED_COLS,
    _has_full_support, _build_inits, _solve_kron_factor,
    _require_swap_symmetric,
)
from two_qubit_lindbladian import qubit_d

RATE_VERSION = '1.0-generator-lognorm'

_N_S = qubit_d ** 2          # 4 rows of S


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
def generator_log_norm(D, A, *, return_cols=False, return_H=False):
    """mu*(D) = max_j min { h_j + sum_{k!=j} |h_k| : D h = A d_j }.

    One batched LP.  Variables per target column j: a free diagonal entry
    g_j (cost +1, signed) and h+_j, h-_j >= 0 (cost +1 each, the split
    off-diagonal part), with D (g_j e_j + h+_j - h-_j) = A d_j and
    g_j + sum_k (h+ + h-)_{kj} <= t.  Putting the j-th component into g_j is
    never worse than into h+-, so the optimum has the diagonal unsigned and
    the value is exactly the column log-norm.

    Returns mu (float); optionally the per-column values (d_ext,) and the
    certificate matrix H (d_ext x d_ext) with A D = D H.  A rank-deficient D
    gives +inf (vacuous certificate).
    """
    D = np.asarray(D, dtype=float)
    A = np.asarray(A, dtype=float)
    n, m = D.shape
    if not _has_full_support(D):
        out = [np.inf]
        if return_cols:
            out.append(None)
        if return_H:
            out.append(None)
        return out[0] if len(out) == 1 else tuple(out)

    Y = A @ D                                   # targets, column j = A d_j
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


def generator_log_norm_reference(D, A, *, return_cols=False):
    """Independent per-column solve of the same LP (confirmation evaluator)."""
    D = np.asarray(D, dtype=float)
    A = np.asarray(A, dtype=float)
    n, m = D.shape
    if not _has_full_support(D):
        return (np.inf, None) if return_cols else np.inf
    Y = A @ D
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
#  Certificate, subgradient, Polyak polish
# ---------------------------------------------------------------------------
def rate_certificate(D, A):
    """Value, binding column j*, dual witness w* and per-column values.

    Dual of the binding column's LP:  max <w, A d_j*>  s.t.
    (D^T w)_j* = 1  and  |(D^T w)_k| <= 1 for k != j*.
    """
    D = np.asarray(D, dtype=float)
    A = np.asarray(A, dtype=float)
    val, cols, H = generator_log_norm(D, A, return_cols=True, return_H=True)
    if not np.isfinite(val):
        return dict(value=np.inf, argmax=-1, witness=None, cols=None, H=None)
    j = int(np.argmax(cols))
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
    """
    S = np.asarray(S, dtype=float)
    A = np.asarray(A, dtype=float)
    n_s, m = S.shape
    D = _kron_power(S, 2)
    cert = rate_certificate(D, A)
    val, j, w, H = cert['value'], cert['argmax'], cert['witness'], cert['H']
    if not np.isfinite(val) or w is None:
        return val, np.zeros_like(S)
    h = H[:, j]
    nrows, d_ext = D.shape
    gD = np.zeros((nrows, d_ext))
    gD[:, j] += A.T @ w
    gD -= np.outer(w, h)
    T = gD.reshape(n_s, n_s, m, m)
    dS = np.einsum('abij,bj->ai', T, S) + np.einsum('baji,bj->ai', T, S)
    return val, dS


def polyak_rate_polish(S, A, *, target=None, n_iter=300, tol=1e-10,
                       stall_patience=20, verbose=False):
    """Projected Polyak subgradient descent on mu*(S) toward `target`
    (default: spectral abscissa of A, i.e. 0 for a TP generator)."""
    A = np.asarray(A, dtype=float)
    if target is None:
        target = spectral_abscissa(A)
    S = np.asarray(S, dtype=float).copy()
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
        gnorm_sq = float(np.sum(g * g))
        if gnorm_sq < 1e-20:
            break
        alpha = beta * (val - target) / gnorm_sq
        free = S[:, N_FIXED_COLS:] - alpha * g
        S = np.hstack([_FIXED_COLS, _project_columns_bloch(free)])
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


if __name__ == '__main__':
    _check_projection()
    _check_limit()
