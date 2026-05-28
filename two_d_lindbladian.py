"""
Four-qubit Lindbladian for a 2D toric-code-style system.

  H   = h Σ_i X_i + λ Σ_i Z_i           (i = 1..4)
  L_s = sqrt(γ_s) X ⊗ X ⊗ X ⊗ X         (a single "star")
  L_p = sqrt(γ_p) Z ⊗ Z ⊗ Z ⊗ Z         (a single "plaquette")

Dynamics:
  dρ/dt = -i[H, ρ] + D[L_s](ρ) + D[L_p](ρ)
where D[L](ρ) = L ρ L† − ½ {L† L, ρ}.

The superoperators are built in the 4-qubit Pauli-string basis
  P_a = σ_{a_1} ⊗ σ_{a_2} ⊗ σ_{a_3} ⊗ σ_{a_4},
with σ ∈ {I=0, X=1, Y=2, Z=3} and index a = a_1·4³ + a_2·4² + a_3·4 + a_4.
Total dimension: 4⁴ = 256.

Functions:
  hamiltonian_superop(h, λ)         – -i[H, ·]
  star_jump_superop(γ_s)            – D[L_s](·)
  plaquette_jump_superop(γ_p)       – D[L_p](·)
  full_lindbladian(h, λ, γ_s, γ_p)  – sum of the three
  write_tex_symbolic(out_path)      – emit a .tex with the symbolic
                                      sparse representation
The numerical and symbolic backends are auto-selected from the
argument types.
"""

from __future__ import annotations

import numpy as np
import sympy as sp


# ── constants ────────────────────────────────────────────────────────────────
N_QUBITS = 4
DIM = 4 ** N_QUBITS                            # 256
I_, X_, Y_, Z_ = 0, 1, 2, 3
PAULI_LABELS = ['I', 'X', 'Y', 'Z']


# ── index <-> Pauli string ───────────────────────────────────────────────────
def _index_to_string(idx: int) -> tuple[int, ...]:
    s = []
    for _ in range(N_QUBITS):
        s.append(idx % 4)
        idx //= 4
    return tuple(reversed(s))


def _string_to_index(s) -> int:
    idx = 0
    for ai in s:
        idx = idx * 4 + ai
    return idx


def _basis_labels() -> list[str]:
    return [''.join(PAULI_LABELS[ai] for ai in _index_to_string(idx))
            for idx in range(DIM)]


def _is_symbolic(*args) -> bool:
    return any(isinstance(a, sp.Expr) for a in args)


# ── Hamiltonian superoperator -i[H, ρ] ───────────────────────────────────────
def hamiltonian_superop(h, lam):
    """
    -i[H, ρ] with H = h Σ X_i + λ Σ Z_i.

    Using the single-qubit commutators
        -i[X, Y] =  2 Z,   -i[X, Z] = -2 Y,
        -i[Z, X] =  2 Y,   -i[Z, Y] = -2 X,
    the action on each Pauli string is a sum of 4 single-site terms.
    For input string a and qubit k, M[i, j] receives:
        a_k = Y  →  i = (a with Z at k):  +2h   (from X_k term)
        a_k = Z  →  i = (a with Y at k):  -2h
        a_k = X  →  i = (a with Y at k):  +2λ   (from Z_k term)
        a_k = Y  →  i = (a with X at k):  -2λ
    """
    symbolic = _is_symbolic(h, lam)
    if symbolic:
        entries: dict[tuple[int, int], sp.Expr] = {}
    else:
        M = np.zeros((DIM, DIM), dtype=complex)

    def add(i, j, val):
        if symbolic:
            entries[(i, j)] = entries.get((i, j), 0) + val
        else:
            M[i, j] += val

    for j in range(DIM):
        a = _index_to_string(j)
        for k in range(N_QUBITS):
            sk = a[k]
            # -i[X_k, σ_{a_k}]
            if sk == Y_:
                new_a = list(a); new_a[k] = Z_
                add(_string_to_index(new_a), j, 2 * h)
            elif sk == Z_:
                new_a = list(a); new_a[k] = Y_
                add(_string_to_index(new_a), j, -2 * h)
            # -i[Z_k, σ_{a_k}]
            if sk == X_:
                new_a = list(a); new_a[k] = Y_
                add(_string_to_index(new_a), j, 2 * lam)
            elif sk == Y_:
                new_a = list(a); new_a[k] = X_
                add(_string_to_index(new_a), j, -2 * lam)

    if symbolic:
        entries = {ij: v for ij, v in entries.items() if sp.simplify(v) != 0}
        return sp.SparseMatrix(DIM, DIM, entries)
    return M


# ── jump superoperators D[L](ρ) ──────────────────────────────────────────────
def star_jump_superop(gamma_s):
    """
    D[L_s](ρ) with L_s = sqrt(γ_s) X⊗X⊗X⊗X.

    Since (X⊗X⊗X⊗X)² = I, L_s† L_s = γ_s I and the dissipator is
        γ_s (X⊗X⊗X⊗X · ρ · X⊗X⊗X⊗X  −  ρ).
    On a Pauli string P_a the conjugation gives ε_a P_a with
        ε_a = (-1)^(#{k : a_k ∈ {Y, Z}}).
    The superop is therefore diagonal:
        M[a, a] = γ_s (ε_a − 1) = -2γ_s   if  #{Y, Z}  is odd
                                = 0       otherwise.
    """
    symbolic = _is_symbolic(gamma_s)
    if symbolic:
        entries: dict[tuple[int, int], sp.Expr] = {}
    else:
        M = np.zeros((DIM, DIM), dtype=complex)

    for j in range(DIM):
        a = _index_to_string(j)
        n_yz = sum(1 for ai in a if ai in (Y_, Z_))
        if n_yz % 2 == 1:
            val = -2 * gamma_s
            if symbolic:
                entries[(j, j)] = val
            else:
                M[j, j] = val

    if symbolic:
        return sp.SparseMatrix(DIM, DIM, entries)
    return M


def plaquette_jump_superop(gamma_p):
    """
    D[L_p](ρ) with L_p = sqrt(γ_p) Z⊗Z⊗Z⊗Z.

    Same structure with ε_a = (-1)^(#{k : a_k ∈ {X, Y}}).  Diagonal:
        M[a, a] = -2γ_p   if  #{X, Y}  is odd
                = 0       otherwise.
    """
    symbolic = _is_symbolic(gamma_p)
    if symbolic:
        entries: dict[tuple[int, int], sp.Expr] = {}
    else:
        M = np.zeros((DIM, DIM), dtype=complex)

    for j in range(DIM):
        a = _index_to_string(j)
        n_xy = sum(1 for ai in a if ai in (X_, Y_))
        if n_xy % 2 == 1:
            val = -2 * gamma_p
            if symbolic:
                entries[(j, j)] = val
            else:
                M[j, j] = val

    if symbolic:
        return sp.SparseMatrix(DIM, DIM, entries)
    return M


def full_lindbladian(h, lam, gamma_s, gamma_p):
    """Sum of Hamiltonian, star, and plaquette parts."""
    return (hamiltonian_superop(h, lam)
            + star_jump_superop(gamma_s)
            + plaquette_jump_superop(gamma_p))


# ── .tex output ──────────────────────────────────────────────────────────────
def _sparse_matrix_entries(M):
    """Return a sorted list of ((i, j), value) for nonzero entries."""
    if isinstance(M, sp.SparseMatrix):
        items = [((i, j), v) for (i, j), v in M.todok().items() if v != 0]
    else:
        items = [((i, j), M[i, j])
                 for i in range(DIM) for j in range(DIM) if M[i, j] != 0]
    items.sort(key=lambda kv: kv[0])
    return items


def _emit_sparse_align(f, M, labels, chunk: int = 80):
    """Write sparse entries of M to file f, broken into align* chunks."""
    items = _sparse_matrix_entries(M)
    if not items:
        f.write('\\[\\text{(zero matrix)}\\]\n')
        return
    f.write(f'Total nonzero entries: {len(items)}.\n\n')
    for start in range(0, len(items), chunk):
        block = items[start:start + chunk]
        f.write('\\begin{align*}\n')
        for k, ((i, j), val) in enumerate(block):
            sep = ' \\\\' if k < len(block) - 1 else ''
            f.write(f'  ({labels[i]},\\ {labels[j]}) &= '
                    f'{sp.latex(val)}{sep}\n')
        f.write('\\end{align*}\n')


def write_tex_symbolic(out_path: str = 'two_d_Lindbladian.tex') -> str:
    """Generate the .tex file with the symbolic sparse superoperator matrices."""
    h_s   = sp.Symbol('h',           real=True)
    lam_s = sp.Symbol(r'\lambda',    real=True)
    gs_s  = sp.Symbol(r'\gamma_s',   real=True)
    gp_s  = sp.Symbol(r'\gamma_p',   real=True)

    H_sup  = hamiltonian_superop(h_s, lam_s)
    Ls_sup = star_jump_superop(gs_s)
    Lp_sup = plaquette_jump_superop(gp_s)
    labels = _basis_labels()

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\usepackage{amsmath}\n')
        f.write('\\usepackage[margin=2cm]{geometry}\n')
        f.write('\\allowdisplaybreaks\n')
        f.write('\\begin{document}\n')
        f.write('\\section*{Four-qubit toric-code-style Lindbladian}\n')
        f.write('\\begin{align*}\n')
        f.write(r'  H   &= h \sum_{i=1}^{4} X_i + \lambda \sum_{i=1}^{4} Z_i \\' + '\n')
        f.write(r'  L_s &= \sqrt{\gamma_s}\, X\otimes X\otimes X\otimes X \\' + '\n')
        f.write(r'  L_p &= \sqrt{\gamma_p}\, Z\otimes Z\otimes Z\otimes Z' + '\n')
        f.write('\\end{align*}\n')
        f.write('Basis: 256 Pauli strings $P_a = \\sigma_{a_1}\\otimes\\sigma_{a_2}'
                '\\otimes\\sigma_{a_3}\\otimes\\sigma_{a_4}$, $\\sigma \\in \\{I, X, Y, Z\\}$.  '
                'Below each line $(P_i, P_j) = v$ means: the coefficient of $P_i$ in '
                '$\\mathcal{L}(P_j)$ is $v$.\n\n')

        f.write('\\section*{Hamiltonian part $-i[H,\\rho]$}\n')
        _emit_sparse_align(f, H_sup, labels)

        f.write('\\section*{Star jump $\\mathcal{D}[L_s](\\rho)$ (diagonal)}\n')
        _emit_sparse_align(f, Ls_sup, labels)

        f.write('\\section*{Plaquette jump $\\mathcal{D}[L_p](\\rho)$ (diagonal)}\n')
        _emit_sparse_align(f, Lp_sup, labels)

        f.write('\\end{document}\n')
    return out_path


# ── sanity check ─────────────────────────────────────────────────────────────
def _build_basis_numeric():
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]],     dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]],  dtype=complex)
    sz = np.array([[1, 0], [0, -1]],    dtype=complex)
    paulis = [I2, sx, sy, sz]
    basis = []
    for a1 in paulis:
        for a2 in paulis:
            for a3 in paulis:
                for a4 in paulis:
                    basis.append(np.kron(np.kron(a1, a2), np.kron(a3, a4)))
    return basis


def _brute_force_lindbladian(h, lam, gamma_s, gamma_p):
    """Slow direct construction from operator definitions; used to verify."""
    basis = _build_basis_numeric()
    n = len(basis)
    d = basis[0].shape[0]

    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]],    dtype=complex)
    sz = np.array([[1, 0], [0, -1]],   dtype=complex)

    def X_at(k):
        ops = [I2] * N_QUBITS; ops[k] = sx
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out

    def Z_at(k):
        ops = [I2] * N_QUBITS; ops[k] = sz
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out

    H = sum(h * X_at(k) + lam * Z_at(k) for k in range(N_QUBITS))
    Ls = np.sqrt(gamma_s) * (
        np.kron(np.kron(sx, sx), np.kron(sx, sx)))
    Lp = np.sqrt(gamma_p) * (
        np.kron(np.kron(sz, sz), np.kron(sz, sz)))

    L = np.zeros((n, n), dtype=complex)
    for j in range(n):
        rho = basis[j]
        res = -1j * (H @ rho - rho @ H)
        for A in (Ls, Lp):
            Ad = A.conj().T
            res = res + A @ rho @ Ad - 0.5 * (Ad @ A @ rho + rho @ Ad @ A)
        for i in range(n):
            L[i, j] = np.trace(basis[i] @ res) / d
    return L


# ─────────────────────────────────────────────────────────────────────────────
# 6-qubit version: a single star around node v + the adjacent upper-right
# plaquette.  6 qubits ordered as [u, r, d, l, ur, ru]:
#   u, r, d, l  – the four star qubits incident to v (up, right, down, left).
#   ur, ru      – the two extra qubits closing the upper-right plaquette
#                 ("up-then-right" top edge, "right-then-up" right edge).
#
#   H   = h (X_u + X_r) + λ (Z_u + Z_r)        (Hamiltonian acts on u, r only)
#   L_s = √γ_s · X_u X_r X_d X_l               (star jump, 4-body X on u,r,d,l)
#   L_p = √γ_p · Z_u Z_r Z_ur Z_ru             (plaquette jump, 4-body Z)
# ─────────────────────────────────────────────────────────────────────────────
N_QUBITS_6Q = 6
DIM_6Q      = 4 ** N_QUBITS_6Q            # 4096

# qubit positions in the 6-qubit string
_QU, _QR, _QD, _QL, _QUR, _QRU = 0, 1, 2, 3, 4, 5
_HAM_QUBITS_6Q  = (_QU, _QR)                    # u, r
_STAR_QUBITS_6Q = (_QU, _QR, _QD, _QL)          # u, r, d, l
_PLAQ_QUBITS_6Q = (_QU, _QR, _QUR, _QRU)        # u, r, ur, ru


def _index_to_string_n(idx: int, n_qubits: int) -> tuple[int, ...]:
    s = []
    for _ in range(n_qubits):
        s.append(idx % 4)
        idx //= 4
    return tuple(reversed(s))


def six_qubit_lindbladian(gamma_s, gamma_p, h=1.0, lam=1.0):
    """
    Lindbladian superoperator over 6 qubits arranged as one star + the
    adjacent upper-right plaquette.

    Qubit ordering (index 0..5): [u, r, d, l, ur, ru].
        H   = h (X_u + X_r) + λ (Z_u + Z_r)          (on u, r only)
        L_s = √γ_s · X_u X_r X_d X_l                 (star)
        L_p = √γ_p · Z_u Z_r Z_ur Z_ru               (plaquette)

    Returns a (4096, 4096) dense numpy array in the Pauli-string basis.
    Parameters
    ----------
    gamma_s, gamma_p : float
        Required jump rates (no defaults).
    h, lam : float, optional
        Hamiltonian field strengths, default 1.0.
    """
    dim = DIM_6Q
    L = np.zeros((dim, dim), dtype=complex)

    strings = [_index_to_string_n(j, N_QUBITS_6Q) for j in range(dim)]

    # ── Hamiltonian part  -i[H, ρ] (acts only on u, r) ─────────────────────
    for j in range(dim):
        a = strings[j]
        for k in _HAM_QUBITS_6Q:
            sk = a[k]
            if sk == Y_:
                # -i[X_k, Y_k] =  2 Z_k → +2h at (a with Z at k)
                new_a = list(a); new_a[k] = Z_
                L[_string_to_index(new_a), j] += 2 * h
                # -i[Z_k, Y_k] = -2 X_k → -2λ at (a with X at k)
                new_a = list(a); new_a[k] = X_
                L[_string_to_index(new_a), j] += -2 * lam
            elif sk == Z_:
                # -i[X_k, Z_k] = -2 Y_k → -2h at (a with Y at k)
                new_a = list(a); new_a[k] = Y_
                L[_string_to_index(new_a), j] += -2 * h
            elif sk == X_:
                # -i[Z_k, X_k] =  2 Y_k → +2λ at (a with Y at k)
                new_a = list(a); new_a[k] = Y_
                L[_string_to_index(new_a), j] += 2 * lam

    # ── Star jump D[L_s](ρ): diagonal, -2γ_s if odd #{Y,Z} in u,r,d,l ─────
    for j in range(dim):
        a = strings[j]
        n_yz = sum(1 for k in _STAR_QUBITS_6Q if a[k] in (Y_, Z_))
        if n_yz % 2 == 1:
            L[j, j] += -2 * gamma_s

    # ── Plaq. jump D[L_p](ρ): diagonal, -2γ_p if odd #{X,Y} in u,r,ur,ru ─
    for j in range(dim):
        a = strings[j]
        n_xy = sum(1 for k in _PLAQ_QUBITS_6Q if a[k] in (X_, Y_))
        if n_xy % 2 == 1:
            L[j, j] += -2 * gamma_p

    return L


# ─────────────────────────────────────────────────────────────────────────────
# 4-qubit reductions of the 6-qubit star+plaquette Lindbladian.
#
# The 6-qubit dynamics splits into two 4-qubit pieces sharing the qubits u, r:
#   L1 acts on (u, r, d, l):  H/2 on (u, r) + star jump  X^⊗4
#   L2 acts on (u, r, ur, ru):H/2 on (u, r) + plaquette  Z^⊗4
# Each carries half of the original Hamiltonian (h/2, λ/2 by default) so that
# the sum reproduces the full Hamiltonian acting on (u, r).
# Both are real 256×256 Lindbladians in the 4-qubit Pauli basis.
# ─────────────────────────────────────────────────────────────────────────────
_HAM_QUBITS_4Q = (0, 1)   # within the 4-qubit string the Hamiltonian sites are u, r


def _four_qubit_hamiltonian_part(L, h, lam):
    """Accumulate the -i[H, ρ] block of L for H = h(X_0+X_1) + λ(Z_0+Z_1)."""
    dim = 4 ** 4
    for j in range(dim):
        a = _index_to_string_n(j, 4)
        for k in _HAM_QUBITS_4Q:
            sk = a[k]
            if sk == Y_:
                new_a = list(a); new_a[k] = Z_
                L[_string_to_index(new_a), j] += 2 * h
                new_a = list(a); new_a[k] = X_
                L[_string_to_index(new_a), j] += -2 * lam
            elif sk == Z_:
                new_a = list(a); new_a[k] = Y_
                L[_string_to_index(new_a), j] += -2 * h
            elif sk == X_:
                new_a = list(a); new_a[k] = Y_
                L[_string_to_index(new_a), j] += 2 * lam


def four_qubit_lindbladian_star(gamma_s, h=0.5, lam=0.5):
    """4-qubit Lindbladian L1 = -i[h(X_u+X_r)+λ(Z_u+Z_r), ρ] + D[L_s].

    Qubit ordering: [u, r, d, l] (indices 0..3).
    Defaults h=λ=1/2 so that L1 + L2 reproduces the full 6q Hamiltonian.
    """
    dim = 4 ** 4
    L = np.zeros((dim, dim), dtype=float)
    _four_qubit_hamiltonian_part(L, h, lam)
    # Star jump L_s = √γ_s X⊗X⊗X⊗X: diagonal, -2γ_s if odd #{Y,Z}
    for j in range(dim):
        a = _index_to_string_n(j, 4)
        n_yz = sum(1 for ai in a if ai in (Y_, Z_))
        if n_yz % 2 == 1:
            L[j, j] += -2 * gamma_s
    return L


def four_qubit_lindbladian_plaquette(gamma_p, h=0.5, lam=0.5):
    """4-qubit Lindbladian L2 = -i[h(X_u+X_r)+λ(Z_u+Z_r), ρ] + D[L_p].

    Qubit ordering: [u, r, ur, ru] (indices 0..3).
    Defaults h=λ=1/2 so that L1 + L2 reproduces the full 6q Hamiltonian.
    """
    dim = 4 ** 4
    L = np.zeros((dim, dim), dtype=float)
    _four_qubit_hamiltonian_part(L, h, lam)
    # Plaquette jump L_p = √γ_p Z⊗Z⊗Z⊗Z: diagonal, -2γ_p if odd #{X,Y}
    for j in range(dim):
        a = _index_to_string_n(j, 4)
        n_xy = sum(1 for ai in a if ai in (X_, Y_))
        if n_xy % 2 == 1:
            L[j, j] += -2 * gamma_p
    return L


if __name__ == '__main__':
    print('Building symbolic superoperators ...')
    out = write_tex_symbolic('two_d_Lindbladian.tex')
    print(f'Saved {out}')

    print('Sanity check vs. brute-force at (h=0.3, lam=0.2, gs=0.4, gp=0.5) ...')
    L_fast  = full_lindbladian(0.3, 0.2, 0.4, 0.5)
    L_brute = _brute_force_lindbladian(0.3, 0.2, 0.4, 0.5)
    diff = np.max(np.abs(L_fast - L_brute))
    print(f'  max|L_fast - L_brute| = {diff:.3e}')
    assert diff < 1e-10, 'mismatch between fast and brute-force constructions'
    print('  OK -- analytical sparse build matches brute-force.')
