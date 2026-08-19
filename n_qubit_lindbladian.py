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

For N=8 the Liouvillian is 65536x65536: only sparse operations (build_lindbladian_
comp + lindbladian_gap's shift-invert eigs) are tractable -- dense diagonalization
of that size is infeasible (~34GB, O(65536^3) flops).  build_dense_H_jumps is
provided only for small N (N<=6, used by nonequilibrium_phase_characterizers.
spectral_oscillation's H/jump-operator input convention).
"""

from __future__ import annotations

from functools import reduce
from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

_I2 = np.eye(2, dtype=complex)
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
def build_hamiltonian(J: float, N: int, edges: Sequence[Tuple[int, int]]) -> sp.csr_matrix:
    """H = J * sum_<i,j> Z_i Z_j over `edges` (sparse, 2^N x 2^N)."""
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)
    for (i, j) in edges:
        a, b = (i, j) if i < j else (j, i)
        H = H + J * _two_site_op(_SZ, _SZ, a, b, N)
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
                           edges: Sequence[Tuple[int, int]]) -> sp.csr_matrix:
    """Sparse Liouvillian in the computational basis, column-stacking vec
    convention.  Shape (2^N * 2^N, 2^N * 2^N) -- for N=8 this is 65536x65536
    (sparse only; never densify this for N=8)."""
    H = build_hamiltonian(J, N, edges)
    L = _superop_commutator(H)
    for A in build_jump_operators(gamma, gamma_p, N):
        L = L + _superop_dissipator(A)
    return L.tocsr()


# ---------------------------------------------------------------------------
#  Gap via sparse shift-invert eigs (six_qubit_lindbladian.steady_state /
#  scripts/six_qubit_starplaq_worker._steady_state_and_decay pattern)
# ---------------------------------------------------------------------------
def lindbladian_gap(L_comp: sp.csr_matrix, *, k: int = 6, sigma: float = -1e-3,
                    which: str = 'LM', maxiter: int = 5000, tol: float = 1e-10,
                    noise_floor: float = 1e-6):
    """Gap between the two minimal-|Re(lambda)| eigenvalues of L_comp: the
    steady-state eigenvalue (Re ~ 0) and the slowest-decaying nonzero mode.

    Finds the `k` eigenvalues nearest `sigma` (shift-invert), decay rates
    Gamma_j = -Re(lambda_j); the gap is the smallest Gamma_j above
    `noise_floor` (Gamma_j <= noise_floor is treated as a steady-state mode,
    matching scripts/six_qubit_starplaq_worker.py's decay-rate convention).
    Raises RuntimeError if fewer than 2 modes are resolved or none clear the
    noise floor (increase k or move sigma).

    Returns (gap, evals) with evals the k eigenvalues found (unsorted, as
    returned by eigs).
    """
    from scipy.sparse.linalg import eigs

    vals = eigs(L_comp.astype(complex), k=k, sigma=sigma, which=which,
               maxiter=maxiter, tol=tol, return_eigenvectors=False)
    gammas = np.sort(-vals.real)
    nz = gammas[gammas > noise_floor]
    if nz.size == 0:
        raise RuntimeError(
            f'lindbladian_gap: no eigenvalue cleared noise_floor={noise_floor:g} '
            f'among k={k} modes near sigma={sigma:g}; increase k or move sigma')
    return float(nz[0]), vals


# ---------------------------------------------------------------------------
#  Dense H + jump operators, for small N only (item 6: N<=6).  Feeds directly
#  into nonequilibrium_phase_characterizers.spectral_oscillation's (H, c_ops)
#  input convention -- no Pauli-basis transform needed, that function builds L
#  itself in the same vec convention as _superop_commutator/_superop_dissipator
#  above.
# ---------------------------------------------------------------------------
def build_dense_H_jumps(J: float, gamma: float, gamma_p: float, N: int,
                        edges: Sequence[Tuple[int, int]]):
    """(H, [c_1, c_2, ...]) as dense numpy arrays, shape (2^N, 2^N) each.  Only
    for N small enough that 2^N is a manageable dense dimension (N<=6 -> 64)."""
    H_sp = build_hamiltonian(J, N, edges)
    jumps_sp = build_jump_operators(gamma, gamma_p, N)
    return H_sp.toarray(), [A.toarray() for A in jumps_sp]
