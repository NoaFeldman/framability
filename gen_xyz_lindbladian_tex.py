"""
Emit a LaTeX file with the SYMBOLIC Liouvillian superoperator of the
boundary-driven anisotropic XYZ chain, in the N-qubit Pauli-string basis.

    H   = Σ_j ( J_x XX + J_y YY + J_z ZZ )   (nearest neighbours)
    L_L = σ+_1   (rate γ_L),   L_R = σ-_N   (rate γ_R)
    Lind(A) = -i[H, A] + γ_L D[L_L](A) + γ_R D[L_R](A)

The superoperator entry in the Pauli basis is
    M[a, b] = Tr( P_a · Lind(P_b) ) / 2^N .

Default N=2  -> 16x16 superoperator (the minimal chain: one XYZ bond with a
spin-up pump on site 1 and a spin-down drain on site 2).  The full matrix is
printed (resized to the text width) together with the list of non-zero entries.
For N>=3 the dense matrix is large; the sparse entry list is emitted and the
dense matrix only if --force_matrix is given.

Usage:
    python gen_xyz_lindbladian_tex.py [--n 2] [--out xyz_Lindbladian.tex]
"""

from __future__ import annotations

import argparse

import sympy as sp
from sympy.physics.quantum import TensorProduct

# ── symbolic Pauli matrices ───────────────────────────────────────────────────
_I2 = sp.Matrix([[1, 0], [0, 1]])
_SX = sp.Matrix([[0, 1], [1, 0]])
_SY = sp.Matrix([[0, -sp.I], [sp.I, 0]])
_SZ = sp.Matrix([[1, 0], [0, -1]])
_SP = (_SX + sp.I * _SY) / 2          # σ+
_SM = (_SX - sp.I * _SY) / 2          # σ-
_PAULI = [_I2, _SX, _SY, _SZ]
_LABELS = ['I', 'X', 'Y', 'Z']


def _kron_list(mats):
    out = mats[0]
    for m in mats[1:]:
        out = TensorProduct(out, m)
    return out


def _site_op(op, site, n):
    mats = [_I2] * n
    mats[site] = op
    return _kron_list(mats)


def _index_to_string(idx, n):
    s = []
    for _ in range(n):
        s.append(idx % 4)
        idx //= 4
    return tuple(reversed(s))


def _pauli_op(idxs):
    return _kron_list([_PAULI[a] for a in idxs])


def _labels(n):
    return [''.join(_LABELS[a] for a in _index_to_string(i, n))
            for i in range(4 ** n)]


def build_symbolic_superop(n):
    """Return (M, labels): sympy Matrix (4^n x 4^n) and the Pauli-string labels."""
    Jx, Jy, Jz = sp.symbols('J_x J_y J_z', real=True)
    gL, gR = sp.symbols('gamma_L gamma_R', real=True, nonnegative=True)

    d = 2 ** n
    dim = 4 ** n

    # Hamiltonian
    H = sp.zeros(d, d)
    for j in range(n - 1):
        for J, op in ((Jx, _SX), (Jy, _SY), (Jz, _SZ)):
            mats = [_I2] * n
            mats[j] = op
            mats[j + 1] = op
            H = H + J * _kron_list(mats)

    L_L = _site_op(_SP, 0, n)
    L_R = _site_op(_SM, n - 1, n)
    jumps = []
    for gamma, L in ((gL, L_L), (gR, L_R)):
        Ld = L.H
        jumps.append((gamma, L, Ld, Ld * L))

    # Pauli operators
    P = [_pauli_op(_index_to_string(i, n)) for i in range(dim)]

    M = sp.zeros(dim, dim)
    for b in range(dim):
        Pb = P[b]
        res = -sp.I * (H * Pb - Pb * H)
        for gamma, L, Ld, LdL in jumps:
            res = res + gamma * (L * Pb * Ld - (LdL * Pb + Pb * LdL) / 2)
        for a in range(dim):
            val = sp.expand((P[a] * res).trace() / d)
            if val != 0:
                M[a, b] = sp.nsimplify(sp.simplify(val))
    return M, _labels(n)


# ── LaTeX emission ────────────────────────────────────────────────────────────
def _matrix_latex(M, labels):
    dim = M.rows
    col_spec = 'c|' + 'c' * dim
    lines = ['\\renewcommand{\\arraystretch}{1.2}',
             '\\resizebox{\\textwidth}{!}{$',
             f'\\begin{{array}}{{{col_spec}}}',
             ' & ' + ' & '.join(f'\\mathbf{{{lab}}}' for lab in labels) + ' \\\\ \\hline']
    for i in range(dim):
        row = [f'\\mathbf{{{labels[i]}}}']
        for j in range(dim):
            v = M[i, j]
            row.append('\\cdot' if v == 0 else sp.latex(v))
        lines.append(' & '.join(row) + ' \\\\')
    lines += ['\\end{array}$}']
    return '\n'.join(lines)


def _sparse_latex(M, labels, chunk=60):
    items = [((i, j), M[i, j]) for i in range(M.rows) for j in range(M.cols)
             if M[i, j] != 0]
    items.sort(key=lambda kv: kv[0])
    out = [f'Total non-zero entries: {len(items)}.\n']
    for start in range(0, len(items), chunk):
        out.append('\\begin{align*}')
        block = items[start:start + chunk]
        for k, ((i, j), v) in enumerate(block):
            sep = ' \\\\' if k < len(block) - 1 else ''
            out.append(f'  \\mathcal{{L}}_{{({labels[i]},\\,{labels[j]})}} &= '
                       f'{sp.latex(v)}{sep}')
        out.append('\\end{align*}')
    return '\n'.join(out)


def write_tex(n=2, out_path='xyz_Lindbladian.tex', force_matrix=False):
    M, labels = build_symbolic_superop(n)
    dim = 4 ** n
    show_matrix = force_matrix or dim <= 16

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\\documentclass[11pt]{article}\n')
        f.write('\\usepackage{amsmath,amssymb}\n')
        f.write('\\usepackage{graphicx}\n')
        f.write('\\usepackage[a3paper,landscape,margin=1cm]{geometry}\n')
        f.write('\\begin{document}\n\n')
        f.write('\\section*{Boundary-driven anisotropic XYZ chain: '
                f'Liouvillian superoperator ($N={n}$)}}\n\n')
        f.write('The dynamics $\\dot\\rho = \\mathcal{L}\\rho$ with\n')
        f.write('\\[ \\mathcal{L}\\rho = -i[H,\\rho] '
                '+ \\gamma_L\\!\\left(L_L\\rho L_L^\\dagger '
                '- \\tfrac12\\{L_L^\\dagger L_L,\\rho\\}\\right) '
                '+ \\gamma_R\\!\\left(L_R\\rho L_R^\\dagger '
                '- \\tfrac12\\{L_R^\\dagger L_R,\\rho\\}\\right), \\]\n')
        f.write('\\[ H = \\sum_{j=1}^{N-1} '
                '\\big(J_x\\,\\sigma^x_j\\sigma^x_{j+1} '
                '+ J_y\\,\\sigma^y_j\\sigma^y_{j+1} '
                '+ J_z\\,\\sigma^z_j\\sigma^z_{j+1}\\big), \\quad '
                'L_L = \\sigma^+_1,\\quad L_R = \\sigma^-_N. \\]\n\n')
        f.write('In the Pauli-string basis $P_a = \\sigma_{a_1}\\!\\otimes\\cdots'
                '\\otimes\\sigma_{a_N}$ the superoperator is\n')
        f.write('$M_{ab} = \\tfrac{1}{2^N}\\,\\mathrm{Tr}\\!\\big(P_a\\,'
                '\\mathcal{L}(P_b)\\big)$ (rows $=a$, columns $=b$).\n\n')

        if show_matrix:
            f.write('\\[\nM =\n')
            f.write(_matrix_latex(M, labels))
            f.write('\n\\]\n\n')
        else:
            f.write(f'\\emph{{The dense matrix is {dim}$\\times${dim}; '
                    'only the non-zero entries are listed.}}\n\n')

        f.write('\\subsection*{Non-zero entries}\n')
        f.write(_sparse_latex(M, labels))
        f.write('\n\n\\end{document}\n')

    n_nz = sum(1 for i in range(dim) for j in range(dim) if M[i, j] != 0)
    print(f'Wrote {out_path}  (N={n}, dim={dim}x{dim}, {n_nz} non-zero entries, '
          f'matrix {"shown" if show_matrix else "omitted"})')
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--out', type=str, default='xyz_Lindbladian.tex')
    ap.add_argument('--force_matrix', action='store_true')
    args = ap.parse_args()
    write_tex(args.n, args.out, args.force_matrix)
