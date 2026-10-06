r"""
framability_rate_margin.py -- the MARGIN of an observable frame: how far below
the floor mu* = 0 its non-identity columns sit, and a bundle method that
maximises it.

Why
---
The rate mu*(D) = max_j mu_j is pinned at >= 0 by the identity column
(A e_II = 0, so mu_II = 0).  An optimiser therefore stops the moment the rate
touches 0, and the frame it returns has no slack: some columns sit exactly at
0, so at a neighbouring grid point with slightly less noise the same frame is
slightly positive.  Rate-0 frames found that way cannot spread to their
neighbours by transfer.  Optimising

        F(S) = max_{j != II} mu_j(S (x) S)          (margin = -F)

instead removes the pin: F < 0 means every column decays, and since every
mu_j is a convex, Lipschitz function of the generator (an LP value with the
generator in the right-hand side), a frame with margin delta keeps rate 0 on a
whole neighbourhood of generators.  At a point with positive rate F equals the
rate, so the same method is also a refined rate polish.

Per-column values
-----------------
framability_rate.generator_log_norm minimises only the MAXIMUM over columns
(epigraph variable t); the per-column values it reports for non-binding
columns are not their minima.  column_values solves the same constraints with
the objective sum_j (g_j + sum_k |h_kj|): the columns decouple, so every
column sits at its own minimum, and the equality duals of block j are the
column's dual witness w_j -- the envelope gradient of every column comes out
of one LP.

Public API
----------
column_values(S, A)                 per-column rates, H, dual witnesses
excess(S, A)                        F = max over non-identity columns
excess_reference(S, A)              same with the independent per-column LP
margin_polish(S, A)                 trust-region bundle descent on F

    python framability_rate_margin.py      # self-check on model10
"""

from __future__ import annotations

import os

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import (csc_matrix, kron as sp_kron, eye as sp_eye,
                          hstack as sp_hstack)

from optimize_framability import (_kron_power, _project_columns_bloch,
                                  _FIXED_COLS, N_FIXED_COLS, _has_full_support)
from framability_rate import generator_log_norm_reference
from framability_rate_global import column_witness, column_grad

RATE_MARGIN_VERSION = '1.0-margin-bundle'
IDENTITY_COL = 0          # column (I, I) of D = S (x) S when S[:, 0] = I
_TOL = 1e-9
# Wall-clock cap (s) of every HiGHS solve here.  A normal d_ext = 12 solve
# takes seconds; a solve that hits the cap returns a non-optimal status and the
# frame is treated as failed (+inf), so one pathological frame cannot stall an
# array task for hours.  Override with the env var RATE_LP_TIME_LIMIT.
LP_TIME_LIMIT = float(os.environ.get('RATE_LP_TIME_LIMIT', '300'))
_LP_OPTIONS = dict(time_limit=LP_TIME_LIMIT)


# ---------------------------------------------------------------------------
#  Per-column values
# ---------------------------------------------------------------------------
def column_values(S, A):
    """Per-column minima mu_j of D = S (x) S, the certificate H (A D = D H)
    and the dual witnesses W (column j = witness of column j).

    Returns (cols, H, W); (None, None, None) if D lacks full support or the
    LP fails.  W is None when the solver exposes no equality marginals."""
    S = np.asarray(S, float)
    A = np.asarray(A, float)
    D = _kron_power(S, 2)
    n, M = D.shape
    if not _has_full_support(D):
        return None, None, None
    Y = A @ D
    K = sp_kron(sp_eye(M, format='csc'), csc_matrix(D), format='csc')
    G = K[:, np.arange(M) * M + np.arange(M)]
    A_eq = sp_hstack([G, K, -K], format='csc')
    c = np.ones(M + 2 * M * M)
    bounds = [(None, None)] * M + [(0.0, None)] * (2 * M * M)
    res = linprog(c, A_eq=A_eq, b_eq=Y.ravel(order='F'), bounds=bounds,
                  method='highs', options=_LP_OPTIONS)
    if res.status != 0:
        return None, None, None
    x = res.x
    g = x[:M]
    Hp = x[M:M + M * M].reshape(M, M, order='F')
    Hm = x[M + M * M:].reshape(M, M, order='F')
    cols = g + (Hp + Hm).sum(axis=0)
    H = Hp - Hm + np.diag(g)
    marg = getattr(getattr(res, 'eqlin', None), 'marginals', None)
    W = None if marg is None else np.asarray(marg, float).reshape(n, M, order='F')
    return cols, H, W


def excess(S, A) -> float:
    """F(S) = max_{j != II} mu_j  (+inf if the frame is singular)."""
    cols, _, _ = column_values(S, A)
    if cols is None:
        return np.inf
    return float(np.delete(cols, IDENTITY_COL).max())


def excess_reference(S, A) -> float:
    """F(S) from the independent per-column LP (certification)."""
    D = _kron_power(np.asarray(S, float), 2)
    v, cols = generator_log_norm_reference(D, np.asarray(A, float),
                                           return_cols=True)
    if cols is None or not np.isfinite(v):
        return np.inf
    return float(np.delete(cols, IDENTITY_COL).max())


# ---------------------------------------------------------------------------
#  Bundle descent on F
# ---------------------------------------------------------------------------
def _grads(S, A, D, cols, H, W, bundle):
    out = []
    for j in bundle:
        w = W[:, j] if W is not None else column_witness(D, A, j)
        if w is None:
            continue
        g = column_grad(S, A, j, w, H[:, j])[:, N_FIXED_COLS:].ravel()
        out.append((j, g))
    return out


def margin_polish(S, A, *, n_iter: int = 40, radius: float = 0.05,
                  band_rel: float = 0.05, band_abs: float = 1e-3,
                  max_cols: int = 32, shrink: float = 0.5, grow: float = 1.6,
                  radius_max: float = 0.5, radius_min: float = 1e-6,
                  tol: float = _TOL, verbose: bool = False):
    """Trust-region bundle descent on F(S) = max_{j != II} mu_j.

    Each iteration linearises every non-identity column within the band of
    the maximum (at most max_cols, all gradients from the duals of one LP),
    solves  min tau  s.t.  mu_j + <g_j, dS> <= tau,  |dS|_inf <= radius,
    projects the free columns back onto the operator-norm ball and accepts
    the step only if the TRUE F decreases.  Returns (best_F, best_S, n_lp).
    """
    A = np.asarray(A, float)
    S = np.asarray(S, float).copy()
    n_s, m = S.shape
    nvar = n_s * (m - N_FIXED_COLS)
    cols, H, W = column_values(S, A)
    n_lp = 1
    if cols is None:
        return np.inf, S, n_lp
    F = float(np.delete(cols, IDENTITY_COL).max())
    best_F, best_S = F, S.copy()
    D = _kron_power(S, 2)
    for it in range(n_iter):
        order = [int(j) for j in np.argsort(-cols) if j != IDENTITY_COL]
        thresh = F - max(band_rel * abs(F), band_abs)
        bundle = [j for j in order[:max_cols] if cols[j] >= thresh]
        grads = _grads(S, A, D, cols, H, W, bundle)
        if not grads:
            break
        Gm = np.array([g for _, g in grads])
        mu = np.array([cols[j] for j, _ in grads])
        c = np.zeros(nvar + 1)
        c[-1] = 1.0
        res = linprog(c, A_ub=np.hstack([Gm, -np.ones((len(mu), 1))]), b_ub=-mu,
                      bounds=[(-radius, radius)] * nvar + [(None, None)],
                      method='highs', options=_LP_OPTIONS)
        n_lp += 1
        if not res.success:
            radius *= shrink
            if radius < radius_min:
                break
            continue
        delta = res.x[:nvar].reshape(n_s, m - N_FIXED_COLS)
        S_new = np.hstack([_FIXED_COLS,
                           _project_columns_bloch(S[:, N_FIXED_COLS:] + delta)])
        cols_new, H_new, W_new = column_values(S_new, A)
        n_lp += 1
        pred = F - float(res.x[-1])
        F_new = (np.inf if cols_new is None
                 else float(np.delete(cols_new, IDENTITY_COL).max()))
        if np.isfinite(F_new) and F_new < F - tol and \
                (pred <= 0 or (F - F_new) >= 0.05 * pred):
            S, cols, H, W, F = S_new, cols_new, H_new, W_new, F_new
            D = _kron_power(S, 2)
            if F < best_F:
                best_F, best_S = F, S.copy()
            radius = min(radius * grow, radius_max)
        else:
            radius *= shrink
            if radius < radius_min:
                break
        if verbose and (it + 1) % 10 == 0:
            print(f'    margin {it + 1}: F={F:.6e} radius={radius:.3g} '
                  f'bundle={len(grads)}', flush=True)
    return best_F, best_S, n_lp


# ---------------------------------------------------------------------------
#  Self-check
# ---------------------------------------------------------------------------
def self_check(verbose: bool = True) -> bool:
    """column_values against the independent per-column LP, and a margin
    polish that must not lose the floor, on model10 frames."""
    from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,
                                          MODEL10_J, MODEL10_H)
    from framability_rate_families import model10_frames
    spec = MODELS['model10']
    ok = True

    def check(name, cond, val):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f'  [{"ok" if cond else "FAIL"}] {name}: {val:.3e}', flush=True)

    for d1, d2, m in ((2.5, 0.8, 4), (2.0, 1.5, 8), (1.2, 0.7, 8)):
        A = build_bond_lindbladian(*spec.build(d1, d2), spec.dim).real.T
        S = model10_frames(d1, d2, MODEL10_J, MODEL10_H, m)['B']
        cols, _, W = column_values(S, A)
        D = _kron_power(S, 2)
        _, cref = generator_log_norm_reference(D, A, return_cols=True)
        check(f'per-column values vs reference ({d1},{d2}) m={m}',
              np.max(np.abs(cols - cref)) < 1e-7, np.max(np.abs(cols - cref)))
        check(f'duals returned ({d1},{d2})', W is not None, 0.0)
        F0 = excess(S, A)
        F1, S1, _ = margin_polish(S, A, n_iter=15)
        check(f'margin polish does not increase F ({d1},{d2}): {F0:.3e} ->',
              F1 <= F0 + 1e-12, F1)
        if F0 <= 0:
            check(f'polished frame still at the floor ({d1},{d2})',
                  excess_reference(S1, A) <= 1e-9, excess_reference(S1, A))
    return ok


if __name__ == '__main__':
    print('framability_rate_margin self-check:')
    print('passed' if self_check() else 'FAILED')
