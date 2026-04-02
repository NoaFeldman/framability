"""
Two-qubit Lindbladian superoperator in the Pauli-string basis.

Density matrix:
  rho = sum_{a,b} c_{ab} (sigma_a ⊗ sigma_b),   a,b in {I, X, Y, Z}

Hamiltonian:
  H = J * (Z ⊗ Z)

Lindblad operators:
  sqrt(gamma)   * |−⟩⟨+|_0,  sqrt(gamma)   * |−⟩⟨+|_1,
  sqrt(gamma')  * Z_0,        sqrt(gamma')  * Z_1

The script computes the 16x16 superoperator matrix symbolically
and saves it to two_qubit_Lindbladian.tex.
"""

import numpy as np
import sympy as sp


qubit_d = 2
pauli_string_dim = qubit_d**4


def two_qubit_lindbladian_symbolic():
    J_s = sp.Symbol('J', real=True)
    gamma_s = sp.Symbol(r'\gamma', real=True)
    gamma_p_s = sp.Symbol(r"\gamma'", real=True)

    # Pauli matrices
    I2 = sp.eye(2)
    X = sp.Matrix([[0, 1], [1, 0]])
    Y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    Z = sp.Matrix([[1, 0], [0, -1]])
    paulis = [I2, X, Y, Z]
    pauli_labels = ['I', 'X', 'Y', 'Z']

    # Lowering / raising operators in X basis
    # |+> = (|0>+|1>)/sqrt(2),  |-> = (|0>-|1>)/sqrt(2)
    mp = sp.Rational(1, 2) * sp.Matrix([[1, 1], [-1, -1]])   # |−⟩⟨+|
    mm = sp.Rational(1, 2) * sp.Matrix([[1, -1], [-1, 1]])   # |−⟩⟨−|
    pm = mp.T                                                 # |+⟩⟨−|

    # Hamiltonian
    H = J_s * sp.kronecker_product(Z, Z)

    # Two-qubit operators
    mp0 = sp.kronecker_product(mp, I2)
    pm0 = sp.kronecker_product(pm, I2)
    mp1 = sp.kronecker_product(I2, mp)
    pm1 = sp.kronecker_product(I2, pm)
    pm_mp0 = pm0 * mp0   # |+><-|*|−><+| = |+><+|
    pm_mp1 = pm1 * mp1
    Z0 = sp.kronecker_product(Z, I2)
    Z1 = sp.kronecker_product(I2, Z)

    # Pauli-string basis: {sigma_a ⊗ sigma_b}
    basis = []
    labels = []
    for pi, li in zip(paulis, pauli_labels):
        for pj, lj in zip(paulis, pauli_labels):
            basis.append(sp.kronecker_product(pi, pj))
            labels.append(li + lj)

    n = len(basis)
    L = sp.zeros(n, n)

    for j in range(n):
        rho = basis[j]

        # -i[H, rho]
        res = -sp.I * (H * rho - rho * H)

        # gamma * D[|−⟩⟨+|_0](rho)
        res += gamma_s * (mp0 * rho * pm0
                          - sp.Rational(1, 2) * (pm_mp0 * rho + rho * pm_mp0))

        # gamma * D[|−⟩⟨+|_1](rho)
        res += gamma_s * (mp1 * rho * pm1
                          - sp.Rational(1, 2) * (pm_mp1 * rho + rho * pm_mp1))

        # gamma' * D[Z_0](rho)  (Z†Z = I, so D[Z] = Z rho Z - rho)
        res += gamma_p_s * (Z0 * rho * Z0 - rho)

        # gamma' * D[Z_1](rho)
        res += gamma_p_s * (Z1 * rho * Z1 - rho)

        # Decompose: coefficient of B_i is Tr(B_i * res) / 4
        for i in range(n):
            c = (basis[i] * res).trace() / 4
            c = sp.nsimplify(sp.expand(c))
            if c != 0:
                L[i, j] = c

    L = sp.simplify(L)
    return L, labels


def symbolic_lindbladian_action_on_isometry(D):
    """
    Return the symbolic matrix Y = L.T @ D.

    Parameters
    ----------
    D : sympy.Matrix
        Basis isometry with shape (qubit_d**4, D_ext).
    """
    if not isinstance(D, sp.MatrixBase):
        raise TypeError('D must be a sympy Matrix for symbolic computation.')
    if D.rows != pauli_string_dim:
        raise ValueError(f'D must have {pauli_string_dim} rows, got {D.rows}.')

    L_sym, _ = two_qubit_lindbladian_symbolic()
    return sp.simplify(L_sym.T * D)


def numeric_two_qubit_lindbladian(J, gamma, gamma_p):
    """Build the (qubit_d**4)x(qubit_d**4) Lindbladian in Pauli-string basis."""
    import numpy as np

    # Pauli matrices
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [I2, sx, sy, sz]

    # X-basis operators: |−><+| and |+><−|
    mp = 0.5 * np.array([[1, 1], [-1, -1]], dtype=complex)
    pm = mp.conj().T

    basis = [np.kron(p1, p2) for p1 in paulis for p2 in paulis]
    n = pauli_string_dim

    H = J * np.kron(sz, sz)

    mp0, pm0 = np.kron(mp, I2), np.kron(pm, I2)
    mp1, pm1 = np.kron(I2, mp), np.kron(I2, pm)
    pm_mp0 = pm0 @ mp0
    pm_mp1 = pm1 @ mp1
    Z0 = np.kron(sz, I2)
    Z1 = np.kron(I2, sz)

    L = np.zeros((n, n), dtype=complex)
    for j in range(n):
        rho_j = basis[j]
        res = -1j * (H @ rho_j - rho_j @ H)
        res += gamma * (mp0 @ rho_j @ pm0
                        - 0.5 * (pm_mp0 @ rho_j + rho_j @ pm_mp0))
        res += gamma * (mp1 @ rho_j @ pm1
                        - 0.5 * (pm_mp1 @ rho_j + rho_j @ pm_mp1))
        res += gamma_p * (Z0 @ rho_j @ Z0 - rho_j)
        res += gamma_p * (Z1 @ rho_j @ Z1 - rho_j)
        for i in range(n):
            L[i, j] = np.trace(basis[i] @ res) / 4

    return L


if __name__ == '__main__':
    print('Computing symbolic Lindbladian ...')
    L_sym, labels = two_qubit_lindbladian_symbolic()
    tex = sp.latex(L_sym)

    # Compute row 1-norms of (I + dt*L) symbolically
    dt_s = sp.Symbol(r'\delta t', positive=True)
    J_pos = sp.Symbol('J', positive=True)
    g_pos = sp.Symbol(r'\gamma', positive=True)
    gp_pos = sp.Symbol(r"\gamma'", positive=True)

    subs = {sp.Symbol('J', real=True): J_pos,
            sp.Symbol(r'\gamma', real=True): g_pos,
            sp.Symbol(r"\gamma'", real=True): gp_pos}
    L_pos = L_sym.subs(subs)
    n_sym = L_pos.shape[0]
    M_sym = sp.eye(n_sym) + dt_s * L_pos
    row_norms = []
    for i in range(n_sym):
        norm_i = M_sym[i, i] + sum(sp.Abs(M_sym[i, j])
                                    for j in range(n_sym) if j != i)
        norm_i = sp.simplify(norm_i)
        row_norms.append(norm_i)

    fname = 'two_qubit_Lindbladian.tex'
    with open(fname, 'w', encoding='utf-8') as f:
        f.write('\\documentclass{article}\n')
        f.write('\\usepackage{amsmath}\n')
        f.write('\\usepackage[landscape,margin=0.5cm]{geometry}\n')
        f.write('\\begin{document}\n')
        f.write('\\tiny\n')
        f.write('%% Basis order: ' + ', '.join(labels) + '\n')
        f.write('$$\n')
        f.write(tex + '\n')
        f.write('$$\n')
        f.write('\\end{document}\n')

    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()
    content = content.replace('\\end{document}\n', '')
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)
        f.write('\n\\vspace{1cm}\n')
        f.write('Row 1-norms of $(I + \\delta t \\cdot L)$:\n')
        f.write('\\begin{align*}\n')
        for i in range(n_sym):
            f.write(f'  \\|\\text{{row }}_{{{labels[i]}}}\\|_1 &= '
                    + sp.latex(row_norms[i]) + r' \\' + '\n')
        f.write('\\end{align*}\n')
        f.write('\\end{document}\n')

    print(f'Saved to {fname}')
    print(f'Basis order: {labels}')

