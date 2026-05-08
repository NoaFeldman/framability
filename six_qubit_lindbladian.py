"""
Six-qubit Lindbladian for a 2x3 rectangular lattice.

System
------
6 qubits arranged on a 2x3 rectangle (2 rows, 3 columns).  Numbering
(row-major):

        0 -- 1 -- 2
        |    |    |
        3 -- 4 -- 5

Nearest-neighbour bonds (7 total):
    horizontal: (0,1), (1,2), (3,4), (4,5)
    vertical:   (0,3), (1,4), (2,5)

Hamiltonian
-----------
    H = J * sum_{<i,j>} Z_i Z_j

Lindblad jump operators (per site i = 0..5):
    L_i^{(-)}  = sqrt(gamma)   * |-><+|_i   (X-basis lowering)
    L_i^{(z)}  = sqrt(gamma')  * Z_i        (dephasing in Z)

Conventions follow `two_qubit_lindbladian.py`:
    rho = sum_a c_a * P_a   with   c_a = Tr(P_a rho) / 2^N
    L_pauli[a, b] = Tr(P_a * L[P_b]) / 2^N

Because 4^N = 4096 for N=6, the routines below use sparse matrices.
"""

from __future__ import annotations

from functools import reduce
from itertools import product as iproduct
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Lattice / qubit-count constants
# ---------------------------------------------------------------------------

LATTICE_SHAPE = (2, 3)               # rows, cols
N_QUBITS      = LATTICE_SHAPE[0] * LATTICE_SHAPE[1]   # 6
HILBERT_DIM   = 2 ** N_QUBITS         # 64
PAULI_DIM     = 4 ** N_QUBITS         # 4096


# Nearest-neighbour edges on the 2x3 lattice.
def lattice_edges(shape: Tuple[int, int] = LATTICE_SHAPE) -> List[Tuple[int, int]]:
    rows, cols = shape
    edges: List[Tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:                      # horizontal
                edges.append((i, r * cols + c + 1))
            if r + 1 < rows:                      # vertical
                edges.append((i, (r + 1) * cols + c))
    return edges


EDGES = lattice_edges()


# ---------------------------------------------------------------------------
# Pauli basics
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]],   dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]],  dtype=complex)
_PAULIS = (_I2, _SX, _SY, _SZ)            # index 0=I, 1=X, 2=Y, 3=Z

# X-basis projectors
_MP = 0.5 * np.array([[1, 1], [-1, -1]], dtype=complex)   # |-><+|
_PM = _MP.conj().T                                         # |+><-|


def _kron_list(mats):
    return reduce(np.kron, mats)


def _site_op(local: np.ndarray, site: int, N: int = N_QUBITS) -> sp.csr_matrix:
    """Embed a 2x2 single-qubit operator at `site` (0=leftmost) on an N-qubit Hilbert space."""
    ops = [local if k == site else _I2 for k in range(N)]
    return sp.csr_matrix(_kron_list(ops))


def _two_site_op(loc_a: np.ndarray, loc_b: np.ndarray, sa: int, sb: int,
                 N: int = N_QUBITS) -> sp.csr_matrix:
    """Embed two single-qubit operators acting on sites sa<sb."""
    ops = []
    for k in range(N):
        if k == sa:
            ops.append(loc_a)
        elif k == sb:
            ops.append(loc_b)
        else:
            ops.append(_I2)
    return sp.csr_matrix(_kron_list(ops))


# ---------------------------------------------------------------------------
# Hamiltonian and jump operators
# ---------------------------------------------------------------------------

def build_hamiltonian(J: float) -> sp.csr_matrix:
    """H = J * sum_<ij> Z_i Z_j  on the 2x3 lattice (sparse, HILBERT_DIM x HILBERT_DIM)."""
    H = sp.csr_matrix((HILBERT_DIM, HILBERT_DIM), dtype=complex)
    for (i, j) in EDGES:
        a, b = (i, j) if i < j else (j, i)
        H = H + J * _two_site_op(_SZ, _SZ, a, b)
    return H.tocsr()


def build_jump_operators(gamma: float, gamma_p: float) -> List[sp.csr_matrix]:
    """List of all jump operators (already pre-multiplied by sqrt(rate))."""
    jumps: List[sp.csr_matrix] = []
    if gamma > 0.0:
        s = np.sqrt(gamma)
        for k in range(N_QUBITS):
            jumps.append(s * _site_op(_MP, k))            # sqrt(gamma) |-><+|_k
    if gamma_p > 0.0:
        s = np.sqrt(gamma_p)
        for k in range(N_QUBITS):
            jumps.append(s * _site_op(_SZ, k))            # sqrt(gamma') Z_k
    return jumps


# ---------------------------------------------------------------------------
# Vectorised superoperator (column-stacking convention: vec(rho)_{ij} -> rho[i,j])
# ---------------------------------------------------------------------------
#  -i[H, rho]                     -> (-i) * (kron(I,H) - kron(H.T, I))  vec(rho)
#  L rho L^dag                    -> kron(L*, L)                        vec(rho)
#  -1/2 {L^dag L, rho}            -> -1/2 (kron(I, L^dag L)
#                                         + kron((L^dag L).T, I)) vec(rho)
# (numpy uses row-major; here we adopt column-stacking conventions consistent
#  with vec(A B C) = (C.T kron A) vec(B).)
# ---------------------------------------------------------------------------

def _superop_commutator(H: sp.csr_matrix) -> sp.csr_matrix:
    d = H.shape[0]
    Id = sp.eye(d, format="csr", dtype=complex)
    return -1j * (sp.kron(Id, H, format="csr") - sp.kron(H.T, Id, format="csr"))


def _superop_dissipator(L: sp.csr_matrix) -> sp.csr_matrix:
    d = L.shape[0]
    Id = sp.eye(d, format="csr", dtype=complex)
    LdL = (L.conj().T @ L).tocsr()
    sandwich = sp.kron(L.conj(), L, format="csr")
    anticom  = -0.5 * (sp.kron(Id, LdL, format="csr")
                        + sp.kron(LdL.T, Id, format="csr"))
    return (sandwich + anticom).tocsr()


def build_lindbladian_comp(J: float, gamma: float, gamma_p: float) -> sp.csr_matrix:
    """
    Build the Lindbladian superoperator in the **computational** basis.

    Returns
    -------
    L_comp : scipy.sparse.csr_matrix, shape (HILBERT_DIM**2, HILBERT_DIM**2)
        Acts on column-stacked vec(rho).
    """
    H = build_hamiltonian(J)
    L_comp = _superop_commutator(H)
    for A in build_jump_operators(gamma, gamma_p):
        L_comp = L_comp + _superop_dissipator(A)
    return L_comp.tocsr()


# ---------------------------------------------------------------------------
# Pauli-string basis transform
# ---------------------------------------------------------------------------

def _build_pauli_basis_matrix(N: int = N_QUBITS) -> sp.csr_matrix:
    """
    Build the change-of-basis matrix V with columns = vec(P_a),
    where P_a runs over all 4^N Pauli strings in lexicographic order.

    vec(rho) = V c  with  c_a = Tr(P_a rho) / 2^N.
    Hence  V^dag V = 2^N * I  =>  V^{-1} = V^dag / 2^N.

    Result has 8^N non-zeros (very sparse).
    """
    dim_h = 2 ** N
    dim_p = 4 ** N

    # Build V column by column with sparse Kronecker products.
    cols = []
    pauli_sparse = [sp.csr_matrix(P) for P in _PAULIS]
    for combo in iproduct(range(4), repeat=N):
        P = pauli_sparse[combo[0]]
        for k in combo[1:]:
            P = sp.kron(P, pauli_sparse[k], format="csr")
        # Column-stacked vec(P).  In SciPy/NumPy default row-major layout,
        # the column-stacked vec is np.asarray(P.T).reshape(-1).
        Pdense = P.toarray()
        col = Pdense.T.reshape(-1)
        cols.append(sp.csr_matrix(col.reshape(-1, 1)))

    V = sp.hstack(cols, format="csr")
    assert V.shape == (dim_h * dim_h, dim_p)
    return V


def comp_to_pauli(L_comp: sp.csr_matrix, V: sp.csr_matrix | None = None,
                  N: int = N_QUBITS) -> np.ndarray:
    """Transform a (column-stacking) computational-basis superoperator to Pauli basis."""
    if V is None:
        V = _build_pauli_basis_matrix(N)
    L_pauli = (V.conj().T @ L_comp @ V) / (2 ** N)
    return np.asarray(L_pauli.todense())


def build_six_qubit_lindbladian_pauli(J: float, gamma: float, gamma_p: float,
                                      V: sp.csr_matrix | None = None) -> np.ndarray:
    """
    Build the 6-qubit Lindbladian in the Pauli-string basis (4096 x 4096).
    """
    L_comp = build_lindbladian_comp(J, gamma, gamma_p)
    return comp_to_pauli(L_comp, V=V, N=N_QUBITS)


# ---------------------------------------------------------------------------
# Steady-state and helpers in computational basis
# ---------------------------------------------------------------------------

def steady_state(L_comp: sp.csr_matrix) -> np.ndarray:
    """
    Find the steady state by computing the right null-vector of L_comp.

    Uses sparse shift-invert eigensolver targeting eigenvalue 0; falls back
    to dense `np.linalg.eig` on failure.  The returned density matrix is
    Hermitian, positive (up to numerical noise) and trace-normalised.
    """
    from scipy.sparse.linalg import eigs

    try:
        vals, vecs = eigs(L_comp, k=1, sigma=0.0, which="LM",
                          maxiter=5000, tol=1e-10)
        idx = int(np.argmin(np.abs(vals)))
        v   = vecs[:, idx]
    except Exception:
        L_dense = L_comp.toarray()
        vals, vecs = np.linalg.eig(L_dense)
        idx = int(np.argmin(np.abs(vals)))
        v   = vecs[:, idx]

    rho = v.reshape(HILBERT_DIM, HILBERT_DIM, order="F")  # column-stacked vec
    rho = (rho + rho.conj().T) / 2.0                       # Hermitise

    tr = np.trace(rho).real
    if abs(tr) < 1e-12:
        raise RuntimeError("Steady-state has zero trace; eigenvalue solver failed.")
    rho = rho / tr                                          # may be sign-flipped
    # Trace must be exactly +1 after this normalisation.
    return rho


def initial_yy_state() -> np.ndarray:
    """Density matrix |+y><+y|^{⊗N} on N=6 qubits."""
    plus_y = (1.0 / np.sqrt(2.0)) * np.array([1.0, 1j], dtype=complex)
    psi    = plus_y
    for _ in range(N_QUBITS - 1):
        psi = np.kron(psi, plus_y)
    return np.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Lattice {LATTICE_SHAPE}, N={N_QUBITS}, edges={EDGES}")
    L = build_lindbladian_comp(J=1.0, gamma=0.3, gamma_p=0.2)
    print(f"L_comp: shape={L.shape}, nnz={L.nnz}")
    rho_ss = steady_state(L)
    print(f"rho_ss trace = {np.trace(rho_ss).real:.6f},  hermiticity = "
          f"{np.linalg.norm(rho_ss - rho_ss.conj().T):.2e}")
