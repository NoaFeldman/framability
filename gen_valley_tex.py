"""
Build a LaTeX document containing, for each (gamma, gamma_p) point in
results_valley/, the optimal single-qubit isometry S_opt (4 x d_ext_single)
and the valley_param_size edge matrices S_k.

Since D = kron(S, S) (16 x d_ext_single**2), we print the much smaller S
matrices.  S is reconstructed from x_opt / edge_xs via valley_worker._build_S.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from valley_worker import POINTS, _build_S


def _fmt(x: float, prec: int = 4) -> str:
    if abs(x) < 10 ** (-prec - 1):
        return '0'
    return f'{x:.{prec}f}'


def _matrix_to_pmatrix(M: np.ndarray, prec: int = 4) -> str:
    rows = []
    for r in M:
        rows.append(' & '.join(_fmt(v, prec) for v in r))
    return ('\\begin{pmatrix}\n  '
            + ' \\\\\n  '.join(rows)
            + '\n\\end{pmatrix}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_valley')
    parser.add_argument('--out',     type=str,
                        default='results_valley/valley_matrices.tex')
    parser.add_argument('--d_ext_single', type=int, default=6)
    parser.add_argument('--prec', type=int, default=4)
    parser.add_argument('--tag_suffix', type=str, default='',
                        help='Suffix used by valley_worker (e.g. "long"). '
                             'When set and --out is left at its default, the '
                             'output filename is suffixed automatically.')
    args = parser.parse_args()

    suffix_part = f'_{args.tag_suffix}' if args.tag_suffix else ''
    in_dir = Path(args.in_dir)
    default_out = 'results_valley/valley_matrices.tex'
    if args.tag_suffix and args.out == default_out:
        out_path = Path(f'results_valley/valley_matrices_{args.tag_suffix}.tex')
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(r'\documentclass[10pt]{article}')
    lines.append(r'\usepackage[a4paper,margin=1.5cm,landscape]{geometry}')
    lines.append(r'\usepackage{amsmath,amssymb}')
    lines.append(r'\usepackage{longtable}')
    lines.append(r'\setlength{\arraycolsep}{3pt}')
    lines.append(r'\title{Framability valley matrices}')
    lines.append(r'\author{}\date{}')
    lines.append(r'\begin{document}')
    lines.append(r'\maketitle')
    lines.append('')
    lines.append(r"For each point $(\gamma, \gamma')$, the two-qubit "
                 r"Lindbladian $\mathcal{L}(J, \gamma, \gamma')$ is built "
                 r"with $J = 1$ and the gate is "
                 r"$U(\Delta t)=\exp(\Delta t\,\mathcal{L})$. "
                 r"The Heisenberg-picture frame is "
                 r"$D = S \otimes S$ with $S \in \mathbb{R}^{4 \times d_{\text{ext}}}$. "
                 r"Below we tabulate $S_\text{opt}$ (minimum-framability point) "
                 r"and the matrices $S_k$ found on the edge of the plateau "
                 r"$\{x : f(D(x)) \le f_\text{opt} + \mathrm{tol}\}$.")
    lines.append('')

    for tid, (g, gp) in enumerate(POINTS):
        f = in_dir / f'valley_task{tid:02d}_d{args.d_ext_single}{suffix_part}.npz'
        if not f.exists():
            print(f'[skip] missing {f}')
            continue
        d = np.load(f, allow_pickle=True)
        d_ext_single = int(d['d_ext_single'])
        f_opt   = float(d['f_opt'])
        plateau_tol = float(d['plateau_tol'])
        J       = float(d['J'])
        dt      = float(d['dt'])
        x_opt   = d['x_opt']
        edge_xs = d['edge_xs']
        edge_fs = d['edge_fs']
        edge_alphas = d['edge_alphas']
        edge_step_norms = d['edge_step_norms']
        edge_axis_index = d['edge_axis_index'] if 'edge_axis_index' in d.files else None
        edge_axis_sign  = d['edge_axis_sign']  if 'edge_axis_sign'  in d.files else None

        S_opt = _build_S(x_opt, d_ext_single)

        lines.append(r'\clearpage')
        lines.append(rf"\section*{{Point {tid}: "
                     rf"$\gamma={g}$, $\gamma'={gp}$}}")
        lines.append(r'\noindent')
        lines.append(rf'$J={J}$, $\Delta t={dt}$, '
                     rf'$d_{{\text{{ext}}}}={d_ext_single}$, '
                     rf'$f_\text{{opt}} = {f_opt:.6f}$, '
                     rf'plateau tol $= {plateau_tol:g}$.')
        lines.append('')
        lines.append(r'\paragraph{Optimal $S_\text{opt}$.}')
        lines.append(r'\[')
        lines.append(r'S_\text{opt} = ' + _matrix_to_pmatrix(S_opt, args.prec))
        lines.append(r'\]')
        lines.append('')

        for k, x in enumerate(edge_xs):
            S_k = _build_S(x, d_ext_single)
            fk = float(edge_fs[k])
            ak = float(edge_alphas[k])
            nk = float(edge_step_norms[k])
            axis_info = ''
            if edge_axis_index is not None and edge_axis_sign is not None:
                axis_info = (rf' along axis $e_{{{int(edge_axis_index[k])}}}$ '
                             rf'(sign ${int(edge_axis_sign[k]):+d}$)')
            lines.append(rf'\paragraph{{Edge $S_{{{k}}}$.}} '
                         rf'$\alpha={ak:.4g}${axis_info}, '
                         rf'step$\|\cdot\|={nk:.4g}$, '
                         rf'$f(D)={fk:.6f}$.')
            lines.append(r'\[')
            lines.append(rf'S_{{{k}}} = '
                         + _matrix_to_pmatrix(S_k, args.prec))
            lines.append(r'\]')
            lines.append('')

    lines.append(r'\end{document}')

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[saved] {out_path}')


if __name__ == '__main__':
    main()
