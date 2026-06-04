"""
Boundary-driven anisotropic XYZ Heisenberg chain (1D, open).

Master equation:
    dρ/dt = -i[H, ρ] + D(ρ)

Hamiltonian (nearest-neighbour XYZ exchange):
    H = Σ_{j=1}^{N-1} ( Jx σx_j σx_{j+1} + Jy σy_j σy_{j+1} + Jz σz_j σz_{j+1} )

Boundary dissipators (strong drive -> non-equilibrium steady state with current):
    L_L = σ+_1 = ½(σx_1 + i σy_1)   (spin-up pump on site 1,  rate γ_L)
    L_R = σ-_N = ½(σx_N - i σy_N)   (spin-down drain on site N, rate γ_R)

    D(ρ) = γ_L (L_L ρ L_L† - ½{L_L†L_L, ρ})
         + γ_R (L_R ρ L_R† - ½{L_R†L_R, ρ})

The Liouvillian is returned as a real matrix in the N-qubit Pauli-string basis
    P_a = σ_{a_1} ⊗ ... ⊗ σ_{a_N},   σ ∈ {I=0, X=1, Y=2, Z=3},
    a = a_1·4^{N-1} + ... + a_N,
built directly from  M[a,b] = Tr(P_a · Lind(P_b)) / 2^N.

For N=4 the basis dimension is 4^4 = 256 (the density matrix is 16x16).
"""

from __future__ import annotations

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────
N_DEFAULT = 4
I_, X_, Y_, Z_ = 0, 1, 2, 3
PAULI_LABELS = ['I', 'X', 'Y', 'Z']

_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_SP = 0.5 * (_SX + 1j * _SY)        # σ+  (raising)
_SM = 0.5 * (_SX - 1j * _SY)        # σ-  (lowering)
_PAULI = [_I2, _SX, _SY, _SZ]


# ── index <-> Pauli string ───────────────────────────────────────────────────
def index_to_string(idx: int, n: int = N_DEFAULT) -> tuple[int, ...]:
    s = []
    for _ in range(n):
        s.append(idx % 4)
        idx //= 4
    return tuple(reversed(s))


def string_to_index(s) -> int:
    idx = 0
    for ai in s:
        idx = idx * 4 + ai
    return idx


# ── operator helpers ─────────────────────────────────────────────────────────
def _kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _site_op(op, site, n):
    """Single-site operator op acting on `site` (0-indexed), identity elsewhere."""
    mats = [_I2] * n
    mats[site] = op
    return _kron_list(mats)


def _pauli_string_op(s):
    """N-qubit Pauli operator (2^N x 2^N) from a length-N tuple of indices."""
    return _kron_list([_PAULI[a] for a in s])


def hamiltonian_xyz(Jx, Jy, Jz, n=N_DEFAULT):
    """H = Σ_j Jx XX + Jy YY + Jz ZZ on nearest neighbours of an open chain."""
    d = 2 ** n
    H = np.zeros((d, d), dtype=complex)
    for j in range(n - 1):
        for J, op in ((Jx, _SX), (Jy, _SY), (Jz, _SZ)):
            if J == 0:
                continue
            mats = [_I2] * n
            mats[j] = op
            mats[j + 1] = op
            H = H + J * _kron_list(mats)
    return H


# ── Pauli-string tensor cache ────────────────────────────────────────────────
_P_CACHE: dict[int, np.ndarray] = {}


def _pauli_tensor(n):
    """(4^n, 2^n, 2^n) array of all N-qubit Pauli-string operators."""
    if n not in _P_CACHE:
        dim = 4 ** n
        d = 2 ** n
        P = np.zeros((dim, d, d), dtype=complex)
        for idx in range(dim):
            P[idx] = _pauli_string_op(index_to_string(idx, n))
        _P_CACHE[n] = P
    return _P_CACHE[n]


# ── Liouvillian in the Pauli basis ───────────────────────────────────────────
def build_lindbladian_pauli(Jx, Jy, Jz, gamma_L, gamma_R, n=N_DEFAULT):
    """Real (4^n x 4^n) Liouvillian superoperator in the Pauli-string basis.

    M[a, b] = Tr( P_a · Lind(P_b) ) / 2^n,  where
        Lind(X) = -i[H, X] + Σ_k γ_k ( L_k X L_k† - ½{L_k†L_k, X} ).
    """
    d = 2 ** n
    P = _pauli_tensor(n)
    H = hamiltonian_xyz(Jx, Jy, Jz, n)

    L_L = _site_op(_SP, 0, n)
    L_R = _site_op(_SM, n - 1, n)
    jumps = []
    for gamma, L in ((gamma_L, L_L), (gamma_R, L_R)):
        Ld = L.conj().T
        jumps.append((gamma, L, Ld, Ld @ L))

    dim = 4 ** n
    M = np.zeros((dim, dim), dtype=float)
    for b in range(dim):
        Pb = P[b]
        res = -1j * (H @ Pb - Pb @ H)
        for gamma, L, Ld, LdL in jumps:
            if gamma == 0:
                continue
            res = res + gamma * (L @ Pb @ Ld - 0.5 * (LdL @ Pb + Pb @ LdL))
        # Tr(P_a res) for every a, vectorised.
        M[:, b] = np.einsum('aij,ji->a', P, res).real / d
    return M


# ── quick self-test ──────────────────────────────────────────────────────────
def _self_test(n=4):
    """Trace-preservation + a brute-force cross-check on a random state."""
    Jx, Jy, Jz = 1.5, 0.5, 1.0
    gL = gR = 2.0
    M = build_lindbladian_pauli(Jx, Jy, Jz, gL, gR, n)
    d = 2 ** n

    # Trace preservation: the identity-Pauli row of M must be ~0
    # (d/dt Tr ρ = 0  <=>  column sums onto the identity coefficient vanish).
    id_row = M[0, :]
    print(f'N={n}: max |M[identity row]| = {np.max(np.abs(id_row)):.2e}  '
          f'(should be ~0 for trace preservation)')

    # Brute-force Lindblad on a random Hermitian operator, compare.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    A = A + A.conj().T
    H = hamiltonian_xyz(Jx, Jy, Jz, n)
    L_L = _site_op(_SP, 0, n)
    L_R = _site_op(_SM, n - 1, n)
    bf = -1j * (H @ A - A @ H)
    for gamma, L in ((gL, L_L), (gR, L_R)):
        Ld = L.conj().T
        bf = bf + gamma * (L @ A @ Ld - 0.5 * (Ld @ L @ A + A @ Ld @ L))
    # via M in the Pauli basis
    P = _pauli_tensor(n)
    c = np.einsum('aij,ji->a', P, A).real / d
    c_out = M @ c
    bf_via_M = np.einsum('a,aij->ij', c_out, P)
    print(f'N={n}: max |brute - M| = {np.max(np.abs(bf - bf_via_M)):.2e}')


if __name__ == '__main__':
    _self_test(4)
