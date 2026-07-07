"""
Framability measures: L1-norm minimisation and extended Pauli basis.
"""

from functools import lru_cache

import numpy as np
import scipy.linalg
from scipy.optimize import linprog
from scipy.optimize._linprog_highs import _linprog_highs
from scipy.optimize._linprog_util import _LPProblem, _clean_inputs
from scipy.sparse import csc_matrix

from two_qubit_lindbladian import pauli_string_dim


def extended_pauli_D(a=1):
    """Extended Pauli basis isometry (16 x 36) via Kronecker of single-qubit blocks.

    Single-qubit block (4 x 6) has columns I, X, Y, Z plus two extra
    columns in the X-Y plane: (X+Y)/sqrt(2) and (X-Y)/sqrt(2).
    """
    single_qubit = np.array([[1, 0,            0,             0, 0, 0],
                             [0, 1,            0,             0, a/np.sqrt(2),  a/np.sqrt(2)],
                             [0, 0,            1,             0, a/np.sqrt(2), -a/np.sqrt(2)],
                             [0, 0,            0,             1, 0, 0]])
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


@lru_cache(maxsize=None)
def _all_stabilizer_D_general(n_qubits):
    """
    (4**n_qubits) x N real matrix whose j-th column is the Pauli-basis
    representation of the j-th pure n-qubit stabilizer state:

        D[i, j] = Tr(P_i  |stab_j><stab_j|),   P_i in {I, X, Y, Z}^{n}

    (same convention and normalisation as _all_stabilizer_D, generalised to
    arbitrary n).  The number of columns is

        N = 2**n * prod_{k=1}^{n} (2**k + 1)  =  6, 60, 1080, ...  (n = 1, 2, 3)

    so n_qubits=3 gives shape (64, 1080).

    Enumeration = (maximal isotropic subspace of F_2^{2n}) x (2**n signs)
    ------------------------------------------------------------------------
    Each nonzero symplectic vector v = (x_1..x_n | z_1..z_n) in F_2^{2n}
    labels an unsigned n-qubit Pauli string.  Two Paulis commute iff their
    symplectic inner product  <a,b> = sum_q (x^a_q z^b_q + z^a_q x^b_q)  is 0
    (mod 2); the unsigned product is the XOR of the vectors.  A stabilizer
    group is an n-dimensional isotropic subspace (all pairs commute); these
    are enumerated by taking every n-tuple of independent, pairwise-commuting
    generators and de-duplicating by their span.  Each of the 2**n sign
    assignments s_i in {+1, -1} on the n generators P_i selects one stabilizer
    state with rank-1 projector

        rho = prod_i (I + s_i P_i) / 2.

    The combinations enumeration is O(C(4**n - 1, n)) and is intended for small
    n (practical through n = 3, where it inspects C(63, 3) = 39711 triples).

    The result is cached (lru_cache) so repeated data points reuse one D build.
    """
    from itertools import product as iproduct, combinations
    from functools import reduce

    n = n_qubits
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0,  1 ], [1,  0 ]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0 ]], dtype=complex)
    sz = np.array([[1,  0 ], [0, -1 ]], dtype=complex)
    paulis_1q = [I2, sx, sy, sz]
    # single-qubit (x, z) symplectic bits -> matrix
    sq = {(0, 0): I2, (1, 0): sx, (1, 1): sy, (0, 1): sz}

    dim = 2 ** n
    pauli_dim = 4 ** n
    eye_d = np.eye(dim, dtype=complex)

    def vec_to_mat(v):
        return reduce(np.kron, [sq[(v[q], v[n + q])] for q in range(n)])

    def symp(a, b):
        s = 0
        for q in range(n):
            s ^= (a[q] & b[n + q]) ^ (a[n + q] & b[q])
        return s

    def xor(a, b):
        return tuple(x ^ y for x, y in zip(a, b))

    def gf2_rank(rows):
        rows = [list(r) for r in rows]
        rank = 0
        for col in range(2 * n):
            piv = next((i for i in range(rank, len(rows)) if rows[i][col]), None)
            if piv is None:
                continue
            rows[rank], rows[piv] = rows[piv], rows[rank]
            for i in range(len(rows)):
                if i != rank and rows[i][col]:
                    rows[i] = [a ^ b for a, b in zip(rows[i], rows[rank])]
            rank += 1
        return rank

    zero = (0,) * (2 * n)

    def span_nonzero(gens):
        elems = set()
        for mask in range(1, 2 ** len(gens)):
            acc = zero
            for b in range(len(gens)):
                if mask & (1 << b):
                    acc = xor(acc, gens[b])
            elems.add(acc)
        return frozenset(elems)

    nonzero = [v for v in iproduct((0, 1), repeat=2 * n) if v != zero]

    # Enumerate maximal isotropic (Lagrangian) subspaces, de-duplicated by span.
    subspaces = {}
    for gens in combinations(nonzero, n):
        if any(symp(gens[a], gens[b])
               for a in range(n) for b in range(a + 1, n)):
            continue
        if gf2_rank(gens) != n:
            continue
        key = span_nonzero(gens)
        if key not in subspaces:
            subspaces[key] = gens

    # Row Pauli strings in the same order as the two_qubit_lindbladian basis
    # (qubit-major iproduct over [I, X, Y, Z]); no normalisation, matching
    # _all_stabilizer_D so projector_stabilizer_framability stays consistent.
    paulis_nq = [reduce(np.kron, combo)
                 for combo in iproduct(paulis_1q, repeat=n)]

    columns = []
    for gens in subspaces.values():
        gmats = [vec_to_mat(g) for g in gens]
        for signs in iproduct((1, -1), repeat=n):
            rho = eye_d
            for s, P in zip(signs, gmats):
                rho = rho @ ((eye_d + s * P) / 2)   # commuting factors
            columns.append([np.trace(P @ rho).real for P in paulis_nq])

    return np.array(columns, dtype=float).T   # (pauli_dim, N)


def stabilizer_3_framability(gate):
    """
    Schrödinger framability of a two-qubit `gate` w.r.t. the full three-qubit
    stabilizer-state frame.

    The frame D = _all_stabilizer_D_general(3) has one column per pure
    three-qubit stabilizer state (shape (64, 1080)).  The two-qubit gate is
    lifted to three qubits as identity ⊗ gate — in the Pauli-string
    coefficient basis this is kron(I_4, gate), shape (64, 64), applying the
    identity channel to the first qubit and `gate` to qubits 2 and 3.  The
    Schrödinger framability of this three-qubit gate is then

        max_j  min_v  ||v||_1   s.t.   D v = (I_4 ⊗ gate) D[:, j].

    The 1080 per-column LPs use the same equality-only split-variable primal
    (v = s⁺ − s⁻, both ≥ 0) as projector_stabilizer_framability.

    Parameters
    ----------
    gate : np.ndarray, shape (16, 16)
        Real two-qubit propagator in the Pauli-string basis (I,X,Y,Z ordering,
        as produced by numeric_two_qubit_lindbladian).

    Returns
    -------
    float
        Maximum optimal 1-norm over all 1080 frame columns.
    """
    gate = np.asarray(gate)
    if gate.shape != (16, 16):
        raise ValueError(f'gate must have shape (16, 16), got {gate.shape}.')
    if np.max(np.abs(gate.imag)) > 1e-12:
        raise ValueError(
            'The gate has a non-negligible imaginary part. '
            'The L1-norm minimisation requires the gate to be real.'
        )
    gate = gate.real

    gate3 = np.kron(np.eye(4), gate)          # identity ⊗ gate, (64, 64)
    D = _all_stabilizer_D_general(3)          # (64, 1080)

    d_ext = D.shape[1]
    B = gate3 @ D                              # b_j = gate3 @ D[:, j]

    c_primal = np.ones(2 * d_ext)
    A_eq_csc = csc_matrix(np.hstack([D, -D]))  # (64, 2*d_ext)
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

    return float(np.max(one_norms))


def _random_single_qubit_frame(chi, mixed, rng):
    """(4 × chi) real Pauli-basis frame of `chi` random single-qubit states.

    Column i is (c_I, c_X, c_Y, c_Z) with c_a = Tr(σ_a ρ_i)/2, so the identity
    coefficient is fixed to 1/2 and the Bloch part (c_X, c_Y, c_Z) obeys the
    legal-state bound ||(c_X, c_Y, c_Z)||_2 <= 1/2.

        mixed=False : Haar-random *pure* states, Bloch norm exactly 1/2
                      (uniform on the Bloch sphere).
        mixed=True  : random *mixed* states, Bloch vector uniform in the
                      Bloch ball of radius 1/2 (norm <= 1/2).

    `rng` is a numpy Generator used for every draw (reproducible).
    """
    dirs = rng.standard_normal((3, chi))                    # isotropic direction
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    if mixed:
        # r = (1/2) U^{1/3} makes the Bloch vectors uniform over the ball volume.
        radii = 0.5 * rng.random(chi) ** (1.0 / 3.0)
    else:
        radii = np.full(chi, 0.5)                           # Bloch sphere (pure)
    bloch = dirs * radii                                    # (3, chi)
    return np.vstack([np.full((1, chi), 0.5), bloch])       # (4, chi)


def make_product_state_D(chi, mixed=False, rng=None):
    """
    Build a two-qubit product-state frame matrix (shape 16 × chi²) from `chi`
    independent random single-qubit states.

    The single-qubit frame matrix D_1 (shape 4 × chi) has columns equal to
    the Pauli-basis representation of each state:

        D_1[a, i] = Tr(σ_a  rho_i) / 2,   σ_a ∈ {I, X, Y, Z}

    consistent with the convention used by _single_qubit_dyadic_D.  Every
    column therefore has identity coefficient 1/2 and Bloch part of 2-norm
    <= 1/2.  The two-qubit frame matrix is D = kron(D_1, D_1), shape (16, chi²).

    Parameters
    ----------
    chi : int
        Number of random single-qubit states to draw.
    mixed : bool
        False (default): Haar-random *pure* states (Bloch norm 1/2), matching
        the original behaviour.  True: random *mixed* states uniform in the
        Bloch ball (norm <= 1/2), i.e. the option requested for
        product-state framability of general single-qubit density matrices.
    rng : np.random.Generator | int | None
        Randomness source used only when ``mixed=True`` (an int is promoted to
        a seeded Generator).  Ignored for the pure branch, which keeps its
        original haar_measure draw off the legacy global RNG so existing
        seed-based pipelines reproduce bit-for-bit.

    Returns
    -------
    D : np.ndarray, shape (16, chi²), dtype float
    """
    if mixed:
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        D_1 = _random_single_qubit_frame(chi, mixed=True, rng=rng)
        return np.kron(D_1, D_1)

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


def _schroedinger_framability_fast(D, gate):
    """Schrödinger framability of `gate` for frame `D`, memory-light per-column LP.

    Identical value to schroedinger_framability(D, gate) for real D, but scales
    to large frames: it uses the equality-only split-variable primal
    (v = s⁺ − s⁻, both ≥ 0) solved column-by-column via a pre-cleaned
    _linprog_highs template — the same formulation as
    dyadic_stabilizer_framability.  Each LP has only pauli_string_dim (16)
    equality rows and 2*d_ext non-negative variables and *no* dense inequality
    block, so peak memory is O(d_ext) rather than the O(d_ext²) dense A_ub the
    generic schroedinger_framability builds.

    D must be real (product-state frames are real Hermitian-operator frames).
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != pauli_string_dim:
        raise ValueError(f'D must have shape ({pauli_string_dim}, D_ext), got {D.shape}.')
    if np.max(np.abs(gate.imag)) > 1e-12:
        raise ValueError(
            'The gate has a non-negligible imaginary part; the L1-norm '
            'minimisation requires a real gate.')
    gate = np.asarray(gate).real

    d_ext = D.shape[1]
    B = gate @ D                                    # b_j = gate @ D[:, j]

    c_primal = np.ones(2 * d_ext)
    A_eq_csc = csc_matrix(np.hstack([D, -D]))       # (16, 2*d_ext)
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

    return float(np.max(one_norms))


def product_state_framability(chi, gate, D=None, mixed=False, rng=None):
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
        make_product_state_D(chi, mixed=mixed, rng=rng).  Pass a fixed D to
        reuse the same random states across all data points.
    mixed : bool
        Forwarded to make_product_state_D when D is None.  False (default)
        draws Haar-random pure single-qubit states; True draws mixed states
        with Bloch norm <= 1/2.
    rng : np.random.Generator | int | None
        Randomness source for the ``mixed=True`` draw (see make_product_state_D).

    Returns
    -------
    float
        Schrödinger framability of `gate` w.r.t. the product-state frame.
    """
    if D is None:
        D = make_product_state_D(chi, mixed=mixed, rng=rng)
    return _schroedinger_framability_fast(D, gate)


"""A Random matrix distributed with Haar measure"""
def haar_measure(n):
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    q,r = scipy.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q,ph,q)
    return q