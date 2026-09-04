"""
Generalized N-qubit Lindbladian builder (model3 physics on an arbitrary bond
list), for the item-4/item-6 many-body extensions of trotter_lindbladian_scan's
model3 to N=8 qubits on a ring ("circle") and a 2x4 lattice, and to N=6 qubits
on a ring.

This is the same construction as six_qubit_lindbladian.py (H = J * sum_<i,j>
Z_i Z_j, jumps sqrt(gamma)|-><+|_i and sqrt(gamma')Z_i per site, column-stacking
vec convention, sparse computational-basis superoperator), generalized over N
and the edge list instead of six_qubit_lindbladian.py's hardcoded N=6 / fixed
2x3-lattice module constants, so the same code covers a ring of any size and
reuses dissipative_PT.bonds_2d for open-boundary rectangular lattices.
An optional transverse field h_x * sum_i X_i (keyword-only, default 0)
extends the same builders to trotter_lindbladian_scan's model4, which is
model3 plus h = MODEL4_H = 1.5 on every site.

For N=8 the Liouvillian is 65536x65536: only sparse operations (build_lindbladian_
comp + lindbladian_gap's ARPACK eigs) are tractable -- dense diagonalization
of that size is infeasible (~34GB, O(65536^3) flops).  build_dense_H_jumps is
provided only for small N (N<=6, used by nonequilibrium_phase_characterizers.
spectral_oscillation's H/jump-operator input convention).
"""

from __future__ import annotations

from functools import reduce
from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from analysis import decay_rate, GAP_NOISE_FLOOR_SPARSE

_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_MP = 0.5 * np.array([[1, 1], [-1, -1]], dtype=complex)   # |-><+|  (X-basis lowering)


# ---------------------------------------------------------------------------
#  Edge lists
# ---------------------------------------------------------------------------
def ring_edges(N: int) -> List[Tuple[int, int]]:
    """Periodic-boundary nearest-neighbour edges of an N-site ring ("circle").
    No existing helper builds this in the repo (dissipative_PT.bonds_2d is
    strictly open-boundary), so it is implemented directly."""
    return [(i, (i + 1) % N) for i in range(N)]


# 2D open-boundary lattice edges: reuse dissipative_PT.bonds_2d(Lx, Ly) directly
# (row-major site numbering, horizontal bonds then vertical bonds) -- see the
# item-4 workers for `from dissipative_PT import bonds_2d`.


# ---------------------------------------------------------------------------
#  Local/two-site operator embedding (sparse; six_qubit_lindbladian.py pattern)
# ---------------------------------------------------------------------------
def _kron_list(mats):
    return reduce(np.kron, mats)


def _site_op(local: np.ndarray, site: int, N: int) -> sp.csr_matrix:
    ops = [local if k == site else _I2 for k in range(N)]
    return sp.csr_matrix(_kron_list(ops))


def _two_site_op(loc_a: np.ndarray, loc_b: np.ndarray, sa: int, sb: int,
                 N: int) -> sp.csr_matrix:
    ops = [loc_a if k == sa else (loc_b if k == sb else _I2) for k in range(N)]
    return sp.csr_matrix(_kron_list(ops))


# ---------------------------------------------------------------------------
#  Hamiltonian and jump operators (model3 physics: H=J*sum ZZ, jumps
#  sqrt(gamma)|-><+|_i, sqrt(gamma')Z_i on every site)
# ---------------------------------------------------------------------------
def build_hamiltonian(J: float, N: int, edges: Sequence[Tuple[int, int]], *,
                      h_x: float = 0.0) -> sp.csr_matrix:
    """H = J * sum_<i,j> Z_i Z_j  +  h_x * sum_i X_i   (sparse, 2^N x 2^N).

    h_x = 0 (the default) is model3's field-free Hamiltonian, so every existing
    caller is unchanged; h_x = trotter_lindbladian_scan.MODEL4_H = 1.5 gives
    model4's transverse field.  The field is applied at full strength on every
    site, matching trotter_lindbladian_scan.build_full_lindbladian_model (the
    1/(2*dim) bond share of build_bond_lindbladian is a Trotter-gate
    convention, not part of the physical lattice generator).
    """
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for (i, j) in edges:
        a, b = (i, j) if i < j else (j, i)
        H = H + J * _two_site_op(_SZ, _SZ, a, b, N)
    if h_x != 0.0:
        for k in range(N):
            H = H + h_x * _site_op(_SX, k, N)
    return H.tocsr()


def build_jump_operators(gamma: float, gamma_p: float, N: int) -> List[sp.csr_matrix]:
    """Per-site jump operators (already pre-multiplied by sqrt(rate))."""
    jumps: List[sp.csr_matrix] = []
    if gamma > 0.0:
        s = np.sqrt(gamma)
        jumps += [s * _site_op(_MP, k, N) for k in range(N)]
    if gamma_p > 0.0:
        s = np.sqrt(gamma_p)
        jumps += [s * _site_op(_SZ, k, N) for k in range(N)]
    return jumps


# ---------------------------------------------------------------------------
#  Vectorised superoperator, column-stacking (vec) convention -- identical to
#  six_qubit_lindbladian.py's _superop_commutator/_superop_dissipator.
# ---------------------------------------------------------------------------
def _superop_commutator(H: sp.csr_matrix) -> sp.csr_matrix:
    d = H.shape[0]
    Id = sp.eye(d, format='csr', dtype=complex)
    return -1j * (sp.kron(Id, H, format='csr') - sp.kron(H.T, Id, format='csr'))


def _superop_dissipator(L: sp.csr_matrix) -> sp.csr_matrix:
    d = L.shape[0]
    Id = sp.eye(d, format='csr', dtype=complex)
    LdL = (L.conj().T @ L).tocsr()
    sandwich = sp.kron(L.conj(), L, format='csr')
    anticom = -0.5 * (sp.kron(Id, LdL, format='csr') + sp.kron(LdL.T, Id, format='csr'))
    return (sandwich + anticom).tocsr()


def build_lindbladian_comp(J: float, gamma: float, gamma_p: float, N: int,
                           edges: Sequence[Tuple[int, int]], *,
                           h_x: float = 0.0) -> sp.csr_matrix:
    """Sparse Liouvillian in the computational basis, column-stacking vec
    convention.  Shape (2^N * 2^N, 2^N * 2^N) -- for N=8 this is 65536x65536
    (sparse only; never densify this for N=8).

    h_x = 0 (default) -> model3 physics; h_x = 1.5 -> model4 (see
    build_hamiltonian)."""
    H = build_hamiltonian(J, N, edges, h_x=h_x)
    L = _superop_commutator(H)
    for A in build_jump_operators(gamma, gamma_p, N):
        L = L + _superop_dissipator(A)
    return L.tocsr()


# ---------------------------------------------------------------------------
#  Gap via sparse ARPACK eigs -- thin wrapper over analysis.decay_rate, which
#  holds the one definition of the Liouvillian gap used in this repo.
# ---------------------------------------------------------------------------
def lindbladian_gap(L_comp: sp.csr_matrix, *, k: int = 12, sigma=None,
                    which: str = 'LR', maxiter: int = 10000, tol: float = 0.0,
                    noise_floor: float = GAP_NOISE_FLOOR_SPARSE):
    """Liouvillian gap of the sparse L_comp: the slowest non-zero decay rate.

    Sparse-path alias for analysis.decay_rate -- see there for the definition,
    for why which='LR' with sigma=None (ARPACK regular mode) is the right
    default on an operator this size, and for the RuntimeError raised when no
    mode clears `noise_floor` (k too small).

    Returns (gap, evals) with evals the k eigenvalues found (unsorted, as
    returned by eigs).
    """
    return decay_rate(L_comp, k=k, sigma=sigma, which=which, maxiter=maxiter,
                      tol=tol, noise_floor=noise_floor,
                      return_eigenvalues=True)


# ---------------------------------------------------------------------------
#  Dense H + jump operators, for small N only (item 6: N<=6).  Feeds directly
#  into nonequilibrium_phase_characterizers.spectral_oscillation's (H, c_ops)
#  input convention -- no Pauli-basis transform needed, that function builds L
#  itself in the same vec convention as _superop_commutator/_superop_dissipator
#  above.
# ---------------------------------------------------------------------------
def build_dense_H_jumps(J: float, gamma: float, gamma_p: float, N: int,
                        edges: Sequence[Tuple[int, int]], *, h_x: float = 0.0):
    """(H, [c_1, c_2, ...]) as dense numpy arrays, shape (2^N, 2^N) each.  Only
    for N small enough that 2^N is a manageable dense dimension (N<=6 -> 64).

    h_x = 0 (default) -> model3 physics; h_x = 1.5 -> model4."""
    H_sp = build_hamiltonian(J, N, edges, h_x=h_x)
    jumps_sp = build_jump_operators(gamma, gamma_p, N)
    return H_sp.toarray(), [A.toarray() for A in jumps_sp]
