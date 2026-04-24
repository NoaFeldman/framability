"""
Framability measures: L1-norm minimisation and extended Pauli basis.
"""

import numpy as np
import scipy.linalg
from scipy.optimize import linprog
from scipy.optimize._linprog_highs import _linprog_highs
from scipy.optimize._linprog_util import _LPProblem, _clean_inputs
from scipy.sparse import csc_matrix

from two_qubit_lindbladian import pauli_string_dim


def extended_pauli_D(a=1):
    """Extended Pauli basis isometry (16 x 36) via Kronecker of single-qubit blocks."""
    single_qubit = np.array([[1, 0, 0, 0, 0,            0],
                             [0, 1, 0, 0, a/np.sqrt(2), a/np.sqrt(2)],
                             [0, 0, 1, 0, 0,            0],
                             [0, 0, 0, 1, a/np.sqrt(2), -a/np.sqrt(2)]])
    return np.kron(single_qubit, single_qubit)


def heisenberg_framability(D, gate):
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


def schroedinger_framability(D, gate):
    """
    For each column j of D, find the minimum-1-norm real vector v_j such that
    D @ v_j = gate @ D[:, j], and return the maximum 1-norm over all j.

    D may be real or complex.  When D is complex the equality constraint
    D @ v = b (complex, v real) is split into its real and imaginary parts:

        Re(D) @ v = Re(gate @ D[:, j])
        Im(D) @ v = Im(gate @ D[:, j])

    yielding a real LP whose equality-constraint matrix has twice the row count.

    LP formulation (per column j)
    -----------------------------
    Variables : v in R^{D_ext}, t in R^{D_ext}  (t are slack variables)
    Minimise  : sum(t)                            (proxy for ||v||_1)
    Subject to:
        [Re(D); Im(D)] @ v = [Re(b_j); Im(b_j)]  (equality; 2*pauli_dim rows if D complex)
        v_k - t_k <= 0  for all k                 (|v_k| <= t_k)
       -v_k - t_k <= 0  for all k
        t_k >= 0

    Parameters
    ----------
    D : np.ndarray
        Basis isometry matrix with shape (pauli_string_dim, D_ext).
        May be real or complex.
    gate : np.ndarray
        Real gate matrix with shape (pauli_string_dim, pauli_string_dim).

    Returns
    -------
    float
        Maximum optimal 1-norm across all columns.
    """
    D = np.asarray(D)
    _complex_D = np.iscomplexobj(D)
    if not _complex_D:
        D = D.astype(float)

    if D.ndim != 2 or D.shape[0] != pauli_string_dim:
        raise ValueError(f'D must have shape ({pauli_string_dim}, D_ext), got {D.shape}.')

    d_ext = D.shape[1]

    if np.max(np.abs(gate.imag)) > 1e-12:
        raise ValueError(
            'The gate has a non-negligible imaginary part. '
            'The L1-norm minimisation uses a real-valued linear program '
            'and requires both D and gate to be real.'
        )
    gate = gate.real

    # Equality-constraint sub-matrix (stack Re/Im rows when D is complex)
    if _complex_D:
        A_eq_D = np.vstack([D.real, D.imag])   # (2*pauli_string_dim, d_ext)
        n_eq_rows = 2 * pauli_string_dim
    else:
        A_eq_D = D                              # (pauli_string_dim, d_ext)
        n_eq_rows = pauli_string_dim

    c = np.concatenate([np.zeros(d_ext), np.ones(d_ext)])
    A_eq = np.hstack([A_eq_D, np.zeros((n_eq_rows, d_ext))])
    A_ub = np.vstack([
        np.hstack([np.eye(d_ext), -np.eye(d_ext)]),
        np.hstack([-np.eye(d_ext), -np.eye(d_ext)]),
    ])
    b_ub = np.zeros(2 * d_ext)
    bounds = [(None, None)] * d_ext + [(0.0, None)] * d_ext

    one_norms = np.empty(d_ext, dtype=float)
    for j in range(d_ext):
        col_j = gate @ D[:, j]   # Schrödinger: apply gate forward, no transpose
        if _complex_D:
            b_eq = np.concatenate([col_j.real, col_j.imag])
        else:
            b_eq = col_j.real
        res = linprog(c=c,
                      A_ub=A_ub,
                      b_ub=b_ub,
                      A_eq=A_eq,
                      b_eq=b_eq,
                      bounds=bounds,
                      method='highs')
        one_norms[j] = np.sum(np.abs(res.x[:d_ext])) if res.success else np.inf

    return np.max(one_norms)


def _single_qubit_dyadic_D():
    """
    4 × 21 real matrix: Pauli-basis representations of 21 single-qubit
    Hermitian operators built from the 6 stabilizer states.

    The 6 single-qubit stabilizer states (±Z, ±X, ±Y eigenstates):
        i=0: |0⟩,  i=1: |1⟩,  i=2: |+⟩,  i=3: |−⟩,  i=4: |+Y⟩,  i=5: |−Y⟩

    Operators are enumerated for unordered pairs i ≤ j:
        i == j  :  |s_i⟩⟨s_i|  (projector; max |eigenvalue| = 1)
        i <  j  :  (|s_i⟩⟨s_j| + |s_j⟩⟨s_i|) / max_abs_eigenvalue
                   normalised so the largest absolute eigenvalue is 1.

    Ordering (i,j) with i≤j symmetric ⇒ col(i,j) = col(j,i), so each
    unordered pair is counted exactly once:  6 diagonal + C(6,2)=15 off-diagonal
    = 21 columns in total.  All operators are Hermitian → Pauli vectors are real.

    Pauli component: v_a = Tr(σ_a op) / 2.
    Paulis ordered [I, X, Y, Z] as in numeric_two_qubit_lindbladian.

    The 21 columns span the full 4-dimensional real Hermitian operator space
    (the 6 projectors already span it).
    """
    stab = np.array([
        [1,  0 ],   # |0⟩
        [0,  1 ],   # |1⟩
        [1,  1 ],   # |+⟩  (normalised below)
        [1, -1 ],   # |−⟩  (normalised below)
        [1,  1j],   # |+Y⟩ (normalised below)
        [1, -1j],   # |−Y⟩ (normalised below)
    ], dtype=complex)
    stab[2:] /= np.sqrt(2)

    paulis = [
        np.eye(2, dtype=complex),                        # I
        np.array([[0,  1 ], [1,  0 ]], dtype=complex),  # X
        np.array([[0, -1j], [1j, 0 ]], dtype=complex),  # Y
        np.array([[1,  0 ], [0, -1 ]], dtype=complex),  # Z
    ]

    n_stab = len(stab)                      # 6
    n_cols = n_stab * (n_stab + 1) // 2    # 21
    D = np.zeros((4, n_cols), dtype=float)
    col = 0
    for i in range(n_stab):
        for j in range(i, n_stab):          # i <= j: each unordered pair once
            if i == j:
                op = np.outer(stab[i], stab[i].conj())
            else:
                op = np.outer(stab[i], stab[j].conj()) + np.outer(stab[j], stab[i].conj())
                max_abs_eig = np.max(np.abs(np.linalg.eigvalsh(op)))
                op /= max_abs_eig
            for a, sigma in enumerate(paulis):
                D[a, col] = (np.trace(sigma @ op) / 2).real
            col += 1
    return D


def dyadic_stabilizer_D(n_qubits=2):
    """
    Dyadic stabilizer frame matrix of shape (4**n_qubits, 21**n_qubits), dtype float.

    Each column is the real Pauli-basis representation of a tensor product of
    single-qubit Hermitian operators drawn from the 21-element set
    (see _single_qubit_dyadic_D):

        i == j  :  |s_i⟩⟨s_i|                            (projector)
        i <  j  :  (|s_i⟩⟨s_j| + |s_j⟩⟨s_i|) / √2    (symmetric Hermitian)

    Since the operators are symmetric in i↔j, each unordered pair {i,j} is
    counted exactly once: 6 diagonal + C(6,2)=15 off-diagonal = 21 per qubit.
    All frame elements are Hermitian → D is a real matrix with full row rank
    4**n_qubits.

    For n_qubits = 2 this gives shape (16, 441).

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default 2, compatible with pauli_string_dim = 16).

    Returns
    -------
    D : np.ndarray, shape (4**n_qubits, 6**n_qubits), dtype float
    """
    D = _single_qubit_dyadic_D()   # (4, 36)
    for _ in range(n_qubits - 1):
        D = np.kron(D, _single_qubit_dyadic_D())
    return D


def dyadic_stabilizer_framability(gate, n_qubits=2):
    """
    Schrödinger framability of `gate` w.r.t. the dyadic stabilizer frame.

    The frame consists of all tensor products of single-qubit stabilizer
    projectors |s_{i_k}⟩⟨s_{i_k}|, where each i_k ranges over the 6
    stabilizer states (eigenstates of ±X, ±Y, ±Z).  The frame matrix
    D = dyadic_stabilizer_D(n_qubits) is real with shape
    (4**n_qubits, 6**n_qubits) and full row rank, so the LP equality
    constraint D @ v = gate @ d_j is always feasible for any real gate.

    For n_qubits = 2 the frame has 441 columns and 441 LP calls are needed.

    Implementation (fast primal LP)
    --------------------------------
    To find min ||v_j||_1 s.t. D @ v_j = b_j, split v_j = s⁺ - s⁻ (both ≥ 0):

        min  1ᵀs⁺ + 1ᵀs⁻   s.t.  [D, −D][s⁺; s⁻] = b_j,  s⁺, s⁻ ≥ 0

    This has pauli_dim (16) equality constraints and 2*d_ext (2592) non-negative
    variables and no inequality constraints.  The LP template ([D, −D] and c) is
    pre-built once via scipy's internal _clean_inputs; only b_j = gate @ D[:,j]
    is updated per column, and _linprog_highs is called directly to avoid
    per-call validation overhead.

    Parameters
    ----------
    gate : np.ndarray, shape (pauli_string_dim, pauli_string_dim)
        Real Lindbladian propagator in the Pauli-string basis.
    n_qubits : int
        Number of qubits (default 2, matching pauli_string_dim = 16).

    Returns
    -------
    float
        Maximum optimal 1-norm over all 36**n_qubits frame columns.
    """
    D = dyadic_stabilizer_D(n_qubits)

    if np.max(np.abs(gate.imag)) > 1e-12:
        raise ValueError(
            'The gate has a non-negligible imaginary part. '
            'The L1-norm minimisation requires the gate to be real.'
        )
    gate = np.asarray(gate).real

    if D.shape[0] != gate.shape[0]:
        raise ValueError(
            f'D has {D.shape[0]} rows but gate has shape {gate.shape}.'
        )

    d_ext = D.shape[1]

    B = gate @ D                           # (pauli_dim, d_ext), b_j = B[:,j]

    # Primal equality-only formulation (split v = s⁺ − s⁻, both ≥ 0):
    #   min  1ᵀs⁺ + 1ᵀs⁻
    #   s.t. [D, −D] [s⁺; s⁻] = b_j      (16 equality constraints)
    #        s⁺, s⁻ ≥ 0                   (no inequality constraints)
    #
    # Pre-clean the LP template once (A_eq and c are fixed; only b_eq changes).
    c_primal = np.ones(2 * d_ext)
    A_eq_csc = csc_matrix(np.hstack([D, -D]))         # (pauli_dim, 2*d_ext)
    bounds   = [(0, None)] * (2 * d_ext)

    lp_template = _LPProblem(
        c_primal, None, None, A_eq_csc, B[:, 0].copy(), bounds, None
    )
    lp_clean = _clean_inputs(lp_template)

    one_norms = np.empty(d_ext, dtype=float)
    for j in range(d_ext):
        lp_j = lp_clean._replace(b_eq=B[:, j])
        res  = _linprog_highs(lp_j, solver=None, presolve=False)
        one_norms[j] = res['fun'] if res['status'] == 0 else np.inf

    return np.max(one_norms)


def _all_stabilizer_D(n_qubits=2):
    """
    (4^n_qubits) x n_stabilizer real matrix whose j-th column is the
    Pauli-basis representation of the j-th pure n-qubit stabilizer state:

        D[i, j] = Tr(P_i  |stab_j><stab_j|)

    where P_i ranges over all 4^n_qubits tensor-product Pauli strings
    {I, X, Y, Z}^{n}.

    For n_qubits=2 the result has shape (16, 60).
    Raises NotImplementedError for n_qubits != 2.
    """
    if n_qubits != 2:
        raise NotImplementedError(
            f'Full stabilizer D matrix for n_qubits={n_qubits} is not '
            f'implemented. Only n_qubits=2 is supported.'
        )

    from itertools import product as iproduct
    from functools import reduce

    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
    sy = np.array([[0, -1j], [1j,  0]], dtype=complex)
    sz = np.array([[1,  0 ], [0, -1 ]], dtype=complex)
    paulis_1q = [I2, sx, sy, sz]

    dim = 2 ** n_qubits        # 4 for n_qubits=2
    eye_d = np.eye(dim, dtype=complex)
    pauli_dim = 4 ** n_qubits  # 16 for n_qubits=2

    paulis_nq = [reduce(np.kron, combo)
                 for combo in iproduct(paulis_1q, repeat=n_qubits)]

    # Signed non-identity n-qubit Paulis
    signed = []
    for k in range(1, pauli_dim):
        signed.append((+1, k, paulis_nq[k]))
        signed.append((-1, k, paulis_nq[k]))

    seen = set()
    states = []

    for a in range(len(signed)):
        s1, k1, g1 = signed[a]
        g1s = s1 * g1

        for b in range(a + 1, len(signed)):
            s2, k2, g2 = signed[b]
            if k1 == k2:                    # same unsigned Pauli
                continue
            g2s = s2 * g2

            if not np.allclose(g1s @ g2s, g2s @ g1s):
                continue

            g3 = g1s @ g2s

            if not np.allclose(g3, g3.conj().T, atol=1e-12):
                continue

            rho = (eye_d + g1s + g2s + g3) / dim

            eigs = np.linalg.eigvalsh(rho)
            if np.min(eigs) < -1e-10:
                continue
            if not np.isclose(np.trace(rho @ rho).real, 1.0, atol=1e-8):
                continue

            key = (np.round(rho.real, 8).tobytes(),
                   np.round(rho.imag, 8).tobytes())
            if key not in seen:
                seen.add(key)
                states.append(rho)

    n_states = len(states)   # 60 for n_qubits=2
    D = np.zeros((pauli_dim, n_states), dtype=float)
    for j, rho in enumerate(states):
        for i, P in enumerate(paulis_nq):
            D[i, j] = np.trace(P @ rho).real

    return D


def projector_stabilizer_framability(gate, n_qubits=2):
    """
    Schrödinger framability of `gate` w.r.t. the full n-qubit stabilizer
    projector frame.

    Frame elements: |psi><psi|, where psi ranges over all n-qubit pure
    stabilizer states.  For n_qubits=2 there are 60 such states, giving
    a D matrix of shape (16, 60).  Raises NotImplementedError for
    n_qubits != 2.

    Parameters
    ----------
    gate : np.ndarray, shape (pauli_string_dim, pauli_string_dim)
        Real Lindbladian propagator in the Pauli-string basis.
    n_qubits : int
        Number of qubits (default 2).

    Returns
    -------
    float
        Maximum optimal 1-norm over all 60 frame columns.
    """
    D = _all_stabilizer_D(n_qubits)   # (16, 60) for n_qubits=2

    if np.max(np.abs(gate.imag)) > 1e-12:
        raise ValueError(
            'The gate has a non-negligible imaginary part. '
            'The L1-norm minimisation requires the gate to be real.'
        )
    gate = np.asarray(gate).real

    d_ext = D.shape[1]
    B = gate @ D

    c_primal = np.ones(2 * d_ext)
    A_eq_csc = csc_matrix(np.hstack([D, -D]))
    bounds   = [(0, None)] * (2 * d_ext)

    lp_template = _LPProblem(
        c_primal, None, None, A_eq_csc, B[:, 0].copy(), bounds, None
    )
    lp_clean = _clean_inputs(lp_template)

    one_norms = np.empty(d_ext, dtype=float)
    for j in range(d_ext):
        lp_j = lp_clean._replace(b_eq=B[:, j])
        res  = _linprog_highs(lp_j, solver=None, presolve=False)
        one_norms[j] = res['fun'] if res['status'] == 0 else np.inf

    return np.max(one_norms)


def make_product_state_D(chi):
    """
    Build a two-qubit product-state frame matrix (shape 16 × chi²) from `chi`
    independent Haar-random single-qubit pure states.

    The single-qubit frame matrix D_1 (shape 4 × chi) has columns equal to
    the Pauli-basis representation of each state:

        D_1[a, i] = Tr(σ_a  rho_i) / 2,   σ_a ∈ {I, X, Y, Z}

    consistent with the convention used by _single_qubit_dyadic_D.
    The two-qubit frame matrix is D = kron(D_1, D_1), shape (16, chi²).

    Parameters
    ----------
    chi : int
        Number of random single-qubit states to draw.

    Returns
    -------
    D : np.ndarray, shape (16, chi²), dtype float
    """
    paulis = [
        np.eye(2, dtype=complex),
        np.array([[0,  1 ], [1,  0 ]], dtype=complex),
        np.array([[0, -1j], [1j, 0 ]], dtype=complex),
        np.array([[1,  0 ], [0, -1 ]], dtype=complex),
    ]
    zero_dm = np.array([[1, 0], [0, 0]], dtype=complex)

    D_1 = np.zeros((4, chi), dtype=float)
    for i in range(chi):
        u = haar_measure(2)
        rho = u @ zero_dm @ u.T.conj()
        for a, sigma in enumerate(paulis):
            D_1[a, i] = (np.trace(sigma @ rho) / 2).real

    return np.kron(D_1, D_1)


def product_state_framability(chi, gate, D=None):
    """
    Schrödinger framability of `gate` w.r.t. a product-state frame.

    For each frame column d_j, solves min ||v||_1 s.t. D v = gate @ d_j
    and returns the maximum over all columns (matching paper Section IV.2,
    Eq. 45-46).

    Parameters
    ----------
    chi : int
        Number of random single-qubit states (used only when D is None).
    gate : np.ndarray, shape (16, 16)
        Real Lindbladian propagator in the two-qubit Pauli-string basis.
    D : np.ndarray, shape (16, chi²), optional
        Pre-built frame matrix.  If None, a fresh random D is generated via
        make_product_state_D(chi).  Pass a fixed D to reuse the same random
        states across all data points.

    Returns
    -------
    float
        Schrödinger framability of `gate` w.r.t. the product-state frame.
    """
    if D is None:
        D = make_product_state_D(chi)
    return schroedinger_framability(D, gate)


"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q,ph,q)
    return q