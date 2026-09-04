r"""
framability_rate_state.py -- dt-free optimisation of a *state* frame
(Schrodinger picture), the counterpart of framability_rate.minimize_rate.

framability_rate.minimize_rate(A = L^T, d) optimises an OBSERVABLE frame
S (identity column pinned, columns in the operator-norm ball) via the
alternating certificate scheme.  State frames obey a different constraint set
(every column is a density matrix: identity coefficient pinned to
STATE_C_I = 1/2, Bloch part in the ball of radius 1/2, no fixed column) and
optimize_framability's own state-frame optimiser
`minimize_schroedinger_framability` is therefore a direct Nelder-Mead search
over the Bloch parameters rather than the alternating scheme.

This module is that same search with the finite-dt objective replaced by the
dt-free rate:

    minimize_schroedinger_framability :  min_S  framability(S(x)S, expm(dt L))
    minimize_state_rate               :  min_S  mu*(S(x)S)  under A = L

with mu*(D) = lim_{dt->0} (framability(dt, D) - 1)/dt evaluated by
framability_rate_frames.frame_rate(..., picture='schroedinger') -- the very
function the fixed-frame Schrodinger rates use, so an optimised value is
directly comparable with schroedinger_frame_rate(S, L) of any stored frame.

Everything else is reused verbatim from optimize_framability: the parameter
decoding (_state_params_to_S), the Pauli-support penalty
(_pauli_support_penalty), the restart seeds (_build_state_inits: octahedron,
SIC tetrahedron, random Bloch) and the rank-deficiency barrier.

    python framability_rate_state.py        # self-test (rate == slope of the
                                            # finite-dt Schrodinger optimum)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from optimize_framability import (
    _state_params_to_S, _pauli_support_penalty, _build_state_inits,
    _kron_power, SCHRO_DEFAULT_METHOD, STATE_C_I,
)
from two_qubit_lindbladian import pauli_string_dim, qubit_d
from framability_rate import spectral_abscissa
from framability_rate_frames import frame_rate

RATE_STATE_VERSION = '1.0-state-frame-rate'

# Same barrier weight as optimize_framability.minimize_schroedinger_framability.
_PENALTY_BASE = 1e3


def state_rate_objective(params, d_ext_single, L, n_qubits):
    """mu*(S(x)...(x)S) + Pauli-support penalty, for flat Bloch `params`.

    Rank-deficient frames (the rate LP is then vacuous, mu = +inf) are replaced
    by the smooth 1/sigma_min barrier of minimize_schroedinger_framability so
    the simplex is pushed back toward full-rank frames instead of hitting a
    cliff.  Note mu* may be negative (unlike framability >= 1); the barrier is
    strictly positive and far above any attainable rate, so it still dominates.
    """
    S = _state_params_to_S(params, d_ext_single)
    pen = _pauli_support_penalty(S)
    D = _kron_power(S, n_qubits)
    mu = frame_rate(D, L, picture='schroedinger')
    if np.isfinite(mu):
        return mu + pen
    sigma_min = float(np.linalg.svd(D, compute_uv=False)[-1])
    return _PENALTY_BASE * (1.0 + 1.0 / max(sigma_min, 1e-12)) + pen


def minimize_state_rate(L, d_ext_single, *, n_restarts=5, method=None,
                        max_iter=500, maxfev=800, tol=1e-6, seed=None,
                        verbose=False, extra_init_xs=None, return_x=False,
                        return_floor=False):
    """min_S mu*(S (x) S) for a state frame S, Lindbladian L (Pauli basis).

    Parameters mirror optimize_framability.minimize_schroedinger_framability
    (same `method` default, same restart scheme, same `extra_init_xs` seeding
    contract -- an x from either optimiser can seed the other, since both
    parameterise the frame by the same flat 3 x d_ext_single Bloch block).

    Returns (S_opt, mu_opt[, x_opt][, floor]).  Unlike the finite-dt optimiser
    this returns the single-qubit frame S rather than D = S(x)S: the fixed-frame
    evaluator schroedinger_frame_rate(S, L) takes S, and D is one
    _kron_power(S, n_qubits) away.

    floor = spectral_abscissa(L) is the rate analogue of the spectral-radius
    floor: mu*(D) >= max Re lambda(L) for every frame, which is 0 for a
    trace-preserving generator (the steady state pins an eigenvalue at 0), so
    mu_opt >= 0 up to LP tolerance.
    """
    if method is None:
        method = SCHRO_DEFAULT_METHOD
    if d_ext_single < 4:
        raise ValueError(
            f'd_ext_single must be >= 4 for a state frame (got {d_ext_single}): '
            f'fewer states cannot span the single-qubit operator space, so '
            f'the rate LP is always infeasible.')

    L = np.asarray(L)
    if np.max(np.abs(L.imag)) > 1e-12:
        raise ValueError('L must be real in the Pauli basis.')
    L = L.real.astype(float)

    rng = np.random.default_rng(seed)
    floor = spectral_abscissa(L)
    n_qubits = int(round(np.log(pauli_string_dim) / np.log(qubit_d ** 2)))

    def objective(params):
        return state_rate_objective(params, d_ext_single, L, n_qubits)

    inits, tags = _build_state_inits(d_ext_single, n_restarts, rng,
                                     extra_init_xs=extra_init_xs)

    if method == 'Nelder-Mead':
        options = dict(maxiter=max_iter, maxfev=maxfev, xatol=tol, fatol=tol)
    else:
        options = dict(maxiter=max_iter, maxfev=maxfev)

    best_val, best_x = np.inf, None
    for x0, tag in zip(inits, tags):
        res = minimize(objective, np.asarray(x0, float), method=method,
                       options=options)
        if verbose:
            print(f'  state-rate {tag}: mu={res.fun:.6e}', flush=True)
        if res.fun < best_val:
            best_val, best_x = float(res.fun), np.asarray(res.x, float)

    S_opt = _state_params_to_S(best_x, d_ext_single)
    # Report the clean rate of the returned frame, not objective+penalty: the
    # penalty is zero on the feasible set, and a residual penalty would
    # otherwise be silently folded into a physical quantity.
    mu_opt = frame_rate(_kron_power(S_opt, n_qubits), L, picture='schroedinger')
    if verbose:
        print(f'floor = {floor:.3e}  mu_opt = {mu_opt:.6e}  '
              f'(objective {best_val:.6e})', flush=True)

    out = [S_opt, float(mu_opt)]
    if return_x:
        out.append(best_x)
    if return_floor:
        out.append(float(floor))
    return tuple(out)


# ---------------------------------------------------------------------------
#  Self-test: the optimised rate is the dt -> 0 slope of the optimised
#  finite-dt Schrodinger framability, and bounds it at every dt.
# ---------------------------------------------------------------------------
def _self_test(d_ext_single=4, dts=(1e-2, 1e-3, 1e-4)):
    from scipy.linalg import expm
    from two_qubit_lindbladian import numeric_two_qubit_lindbladian
    from framability import schroedinger_framability
    from framability_rate import framability_bound
    from optimize_framability import minimize_schroedinger_framability

    L = numeric_two_qubit_lindbladian(1.0, 0.5, 0.1).real
    S, mu = minimize_state_rate(L, d_ext_single, n_restarts=4, maxfev=600,
                                seed=0, verbose=True)
    D = _kron_power(S, 2)
    print(f'identity row: {S[0]}  (should all be {STATE_C_I})')
    for dt in dts:
        gate = expm(dt * L).real
        f_here = schroedinger_framability(D, gate)
        _, f_opt = minimize_schroedinger_framability(
            gate, d_ext_single, n_restarts=4, maxfev=600, seed=0, verbose=False)
        ok = f_here <= framability_bound(mu, dt) * (1 + 1e-9) + 1e-12
        print(f'  dt={dt:.0e}: (fra-1)/dt this frame = '
              f'{(f_here - 1.0) / dt:+.6f}   dt-optimised = '
              f'{(f_opt - 1.0) / dt:+.6f}   fra <= exp(dt mu): {ok}')
        assert ok, (dt, f_here, mu)
    print(f'mu_opt = {mu:+.8f}   state-frame rate optimiser: ok')


if __name__ == '__main__':
    _self_test()
