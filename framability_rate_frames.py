r"""
dt-independent framability (framability *rate*) of a FIXED frame, for every
fixed frame used in the scans:

    pauli_rate(L)                 <->  dissipative_PT.pauli_framability(gate)
    stabilizer_3_rate(L)          <->  framability.stabilizer_3_framability(gate)
    product_state_rate(chi, L, D) <->  framability.product_state_framability(chi, gate, D)
    heisenberg_frame_rate(S, L)   <->  dissipative_PT._framability_lp(S(x)S, gate)   (opt_S_4/6)
    schroedinger_frame_rate(S, L) <->  framability.schroedinger_framability(S(x)S, gate) (state frames)

with gate = expm(dt L).  Each function returns

    mu*(D) = lim_{dt -> 0} (framability(dt, D) - 1) / dt

computed without any dt: for a frame D and the generator A acting on the
frame's coefficient vectors (A = L^T in the Heisenberg picture, A = L in the
Schrödinger picture) one solves, column by column,

    mu_j = min_h  h_j + sum_{k != j} |h_k|   s.t.   D h = A d_j ,      mu* = max_j mu_j .

Any feasible h gives A D = D H, hence exp(tA) D = D exp(tH) and
framability(dt, D) <= exp(dt mu*) for EVERY dt (l1 log-norm bound, the
Molchanov-Pyatnitskii / Blanchini vertex-form polytopic Lyapunov condition);
the first-order expansion of the finite-dt LP shows the bound is tight as
dt -> 0.  Frame requirements are the ones of the finite-dt evaluators: the
frames are used exactly as built there (same columns, same picture), so the
rate is directly comparable with the stored finite-dt values, e.g.
fra^(1/dt) ~ exp(mu*).

Pauli frame: D = identity, so H = L^T is unique and mu* is the closed-form
row-wise (l_inf) logarithmic norm of L, max_j (L_jj + sum_{k != j} |L_jk|).

    python framability_rate_frames.py          # self-test (slope -> mu, bound)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._linprog_highs import _linprog_highs
from scipy.optimize._linprog_util import _LPProblem, _clean_inputs

from framability import (_all_stabilizer_D_general, make_product_state_D,
                         pauli_string_dim)
from dissipative_PT import _HIGHS_ATTEMPTS, _has_full_support, _kron_power
from framability_rate import framability_bound, spectral_abscissa   # noqa: F401

RATE_FRAMES_VERSION = '1.0-fixed-frame-rates'


# ---------------------------------------------------------------------------
#  Core: per-column log-norm LP on a fixed frame (memory-light, any d_ext)
# ---------------------------------------------------------------------------
def frame_rate_columns(D, A, *, check_support=True):
    """Per-column rates mu_j of the fixed frame D under generator A.

    Variables h = h+ - h- (both >= 0); the objective counts the column's own
    coefficient h_j WITH its sign (c[j] = +1 on h+_j, c[d+j] = -1 on h-_j) and
    every other entry in absolute value.  The equality block D[h+; h-] is
    shared by all columns, so one pre-cleaned HiGHS template is reused with
    only (c, b_eq) replaced; the presolve ladder of dissipative_PT guards
    against spurious infeasibility.  Returns an array of length d_ext
    (+inf where a column is unreachable / the LP fails).
    """
    D = np.asarray(D, dtype=float)
    A = np.asarray(A, dtype=float)
    n, d_ext = D.shape
    if check_support and not _has_full_support(D):
        return np.full(d_ext, np.inf)
    Y = A @ D
    c0 = np.ones(2 * d_ext)
    A_eq = csc_matrix(np.hstack([D, -D]))
    bounds = [(0.0, None)] * (2 * d_ext)
    lp = _clean_inputs(_LPProblem(c0, None, None, A_eq, Y[:, 0].copy(),
                                  bounds, None))
    mu = np.empty(d_ext)
    for j in range(d_ext):
        c = c0.copy()
        c[d_ext + j] = -1.0
        lp_j = lp._replace(c=c, b_eq=Y[:, j].copy())
        r = None
        for kw in _HIGHS_ATTEMPTS:
            cand = _linprog_highs(lp_j, **kw)
            if cand['status'] == 0:
                r = cand
                break
        mu[j] = float(r['fun']) if r is not None else np.inf
    return mu


def frame_rate(D, L, *, picture='heisenberg', return_cols=False):
    """mu*(D) for a fixed frame D and Lindbladian L (Pauli-basis superoperator).

    picture='heisenberg'   : targets are L^T d_j  (observable frames, e.g. the
                             optimised opt_S frames, the Pauli frame)
    picture='schroedinger' : targets are L d_j    (state frames: stabilizer,
                             product-state, optimised state frames)
    """
    L = np.asarray(L)
    if np.max(np.abs(L.imag)) > 1e-12:
        raise ValueError('L must be real in the Pauli basis.')
    L = L.real
    if picture == 'heisenberg':
        A = L.T
    elif picture == 'schroedinger':
        A = L
    else:
        raise ValueError("picture must be 'heisenberg' or 'schroedinger'")
    cols = frame_rate_columns(D, A)
    val = float(np.max(cols))
    return (val, cols) if return_cols else val


# ---------------------------------------------------------------------------
#  The fixed frames of the scans
# ---------------------------------------------------------------------------
def pauli_rate(L, *, return_cols=False):
    """Rate of the Pauli frame (dt-free pauli_framability): closed form.

    pauli_framability(expm(dt L)) = max_j sum_k |delta_jk + dt L_jk|
                                  = 1 + dt max_j (L_jj + sum_{k != j}|L_jk|) + O(dt^2)
    i.e. the row-wise logarithmic norm of L.  (D = I, H = L^T, no LP needed.)
    """
    L = np.asarray(L).real
    off = np.abs(L).sum(axis=1) - np.abs(np.diag(L))
    cols = np.diag(L) + off
    val = float(np.max(cols))
    return (val, cols) if return_cols else val


def stabilizer_3_rate(L, *, return_cols=False):
    """Rate of the three-qubit stabilizer-state frame (dt-free
    stabilizer_3_framability).  The two-qubit generator is lifted to three
    qubits as kron(I_4, L) exactly as the gate is lifted there; Schrödinger
    picture, 1080 per-column LPs with 64 equality rows."""
    L = np.asarray(L).real
    if L.shape != (16, 16):
        raise ValueError(f'L must have shape (16, 16), got {L.shape}.')
    L3 = np.kron(np.eye(4), L)
    D = _all_stabilizer_D_general(3)
    cols = frame_rate_columns(D, L3)
    val = float(np.max(cols))
    return (val, cols) if return_cols else val


def product_state_rate(chi, L, D=None, mixed=False, rng=None, *,
                       return_cols=False):
    """Rate of the random product-state frame (dt-free
    product_state_framability).  Same signature and the same frame
    construction (make_product_state_D, global-RNG pure draw) so a caller
    that reseeds as the scan does gets the very frame of prod_fra_{chi};
    pass the frame D to evaluate a stored one.  Schrödinger picture."""
    if D is None:
        D = make_product_state_D(chi, mixed=mixed, rng=rng)
    cols = frame_rate_columns(np.asarray(D, dtype=float), np.asarray(L).real)
    val = float(np.max(cols))
    return (val, cols) if return_cols else val


def heisenberg_frame_rate(S, L, *, return_cols=False):
    """Rate of a stored optimised observable frame S (opt_S_4 / opt_S_6,
    4 x d_ext_single, identity column first): D = S(x)S, Heisenberg picture.
    Companion of dissipative_PT._framability_lp(S(x)S, gate)."""
    D = _kron_power(np.asarray(S, dtype=float), 2)
    return frame_rate(D, L, picture='heisenberg', return_cols=return_cols)


def schroedinger_frame_rate(S, L, *, return_cols=False):
    """Rate of a single-qubit state frame S (4 x d_ext_single, identity row
    1/2, Bloch norm <= 1/2, as built by optimize_framability's state-frame
    optimiser): D = S(x)S, Schrödinger picture."""
    D = _kron_power(np.asarray(S, dtype=float), 2)
    return frame_rate(D, L, picture='schroedinger', return_cols=return_cols)


# ---------------------------------------------------------------------------
#  Self-test: rate = slope of the finite-dt evaluators, bound holds
# ---------------------------------------------------------------------------
def _self_test(dts=(1e-2, 1e-3, 1e-4), chi=6):
    from scipy.linalg import expm
    from two_qubit_lindbladian import numeric_two_qubit_lindbladian
    from framability import stabilizer_3_framability, product_state_framability
    from dissipative_PT import pauli_framability, _framability_lp
    from optimize_framability import _project_columns_bloch, _FIXED_COLS

    L = numeric_two_qubit_lindbladian(1.0, 0.5, 0.1).real
    rng = np.random.default_rng(0)

    # Pauli frame: closed form == LP on D = identity.
    mu_p = pauli_rate(L)
    mu_p_lp = frame_rate(np.eye(pauli_string_dim), L, picture='heisenberg')
    print(f'pauli: closed form {mu_p:.10f}   LP on D=I {mu_p_lp:.10f}')
    assert abs(mu_p - mu_p_lp) < 1e-8

    S_obs = np.hstack([_FIXED_COLS, _project_columns_bloch(rng.standard_normal((4, 3)))])
    np.random.seed(12345)
    D_prod = make_product_state_D(chi)

    cases = [
        ('pauli',      mu_p,
         lambda g: pauli_framability(g)),
        ('opt-frame',  heisenberg_frame_rate(S_obs, L),
         lambda g: _framability_lp(_kron_power(S_obs, 2), g)),
        (f'product{chi}', product_state_rate(chi, L, D=D_prod),
         lambda g: product_state_framability(chi, g, D=D_prod)),
        ('stab3',      stabilizer_3_rate(L),
         lambda g: stabilizer_3_framability(g)),
    ]
    for name, mu, fra in cases:
        print(f'{name:>10}: mu = {mu:+.8f}')
        for dt in dts:
            f = fra(expm(dt * L).real)
            slope = (f - 1.0) / dt
            ok = f <= framability_bound(mu, dt) * (1 + 1e-9) + 1e-12
            print(f'            dt={dt:.0e}: (fra-1)/dt = {slope:+.8f}   '
                  f'fra <= exp(dt mu): {ok}')
            assert ok, (name, dt, f, mu)
        assert abs(slope - mu) < 5e-3 * max(1.0, abs(mu)), (name, slope, mu)
    print('fixed-frame rates: ok')


if __name__ == '__main__':
    _self_test()
