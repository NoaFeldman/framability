"""
Framability measures: L1-norm minimisation and extended Pauli basis.
"""

import numpy as np
from scipy.optimize import linprog

from two_qubit_lindbladian import pauli_string_dim


def extended_pauli_D(a=1):
    """Extended Pauli basis isometry (16 x 36) via Kronecker of single-qubit blocks."""
    single_qubit = np.array([[1, 0, 0, 0, 0,            0],
                             [0, 1, 0, 0, a/np.sqrt(2), a/np.sqrt(2)],
                             [0, 0, 1, 0, 0,            0],
                             [0, 0, 0, 1, a/np.sqrt(2), -a/np.sqrt(2)]])
    return np.kron(single_qubit, single_qubit)


def get_framability(D, gate):
    """
    Compute Y = gate.T @ D and solve min ||u||_1 subject to D @ u = v for each
    column v of Y.

    Parameters
    ----------
    D : np.ndarray
        Basis isometry matrix with shape (pauli_string_dim, D_ext).
    gate : np.ndarray
        Gate matrix with shape (pauli_string_dim, pauli_string_dim).

    Returns
    -------
    float
        Maximum optimal 1-norm across all columns.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != pauli_string_dim:
        raise ValueError(f'D must have shape ({pauli_string_dim}, D_ext), got {D.shape}.')

    d_ext = D.shape[1]

    if np.max(np.abs(gate.imag)) > 1e-12:
        raise ValueError(
            'The Lindbladian has a non-negligible imaginary part. '
            'The L1-norm minimisation uses a real-valued linear program '
            'and requires both D and L to be real.'
        )
    gate = gate.real
    Y = gate.T @ D

    # LP: min sum(t) s.t. D u = v, -t <= u <= t, t >= 0
    c = np.concatenate([np.zeros(d_ext), np.ones(d_ext)])
    A_eq = np.hstack([D, np.zeros((pauli_string_dim, d_ext))])
    A_ub = np.vstack([
        np.hstack([np.eye(d_ext), -np.eye(d_ext)]),
        np.hstack([-np.eye(d_ext), -np.eye(d_ext)]),
    ])
    b_ub = np.zeros(2 * d_ext)
    bounds = [(None, None)] * d_ext + [(0.0, None)] * d_ext

    one_norms = np.empty(d_ext, dtype=float)
    for j in range(d_ext):
        b_eq = Y[:, j]
        res = linprog(c=c,
                      A_ub=A_ub,
                      b_ub=b_ub,
                      A_eq=A_eq,
                      b_eq=b_eq,
                      bounds=bounds,
                      method='highs')
        one_norms[j] = np.sum(np.abs(res.x[:d_ext])) if res.success else np.inf

    return np.max(one_norms)
