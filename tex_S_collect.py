"""
Collect tex_S worker outputs and regenerate results/one_qubit_S_matrices.tex.

Usage
-----
    python tex_S_collect.py [--out_dir results]
"""

import argparse
import os
import sys
import numpy as np

PARAMS_LIST = [(6.0, 0.0), (7.0, 0.0), (0.0, 0.6), (2.4, 0.4)]
GAMMA_STEP = 0.2


def fmt(x, thr=1e-3):
    if abs(x) < thr:
        return '0'
    return f'{x:.4f}'.rstrip('0').rstrip('.')


def matrix_latex(M):
    rows = []
    for row in M:
        rows.append(' & '.join(fmt(v) for v in row))
    return '\\begin{pmatrix}\n' + ' \\\\\n'.join(rows) + '\n\\end{pmatrix}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='results')
    args = p.parse_args()

    scan_full = np.load(os.path.join(args.out_dir, 'scan_full.npy'))

    S_list, fra_list = [], []
    missing = []
    for i in range(len(PARAMS_LIST)):
        f_path = os.path.join(args.out_dir, f'tex_f_{i:04d}.npy')
        S_path = os.path.join(args.out_dir, f'tex_S_{i:04d}.npy')
        if not os.path.exists(f_path) or not os.path.exists(S_path):
            missing.append(i)
        else:
            S_list.append(np.load(S_path))
            fra_list.append(float(np.load(f_path)[0]))

    if missing:
        print(f'ERROR: missing output files for task_ids: {missing}', file=sys.stderr)
        print('Run tex_S_array.sh first, then re-run this script.', file=sys.stderr)
        sys.exit(1)

    # Report
    for i, (gamma, gp) in enumerate(PARAMS_LIST):
        ig  = int(round(gamma / GAMMA_STEP))
        igp = int(round(gp    / GAMMA_STEP))
        stored = float(scan_full[ig, igp, 3])
        print(f'({gamma}, {gp}): stored={stored:.8f}  found={fra_list[i]:.8f}')

    # Generate LaTeX
    lines = [
        r'\documentclass{article}',
        r'\usepackage{amsmath}',
        r'\usepackage{booktabs}',
        r'\usepackage{geometry}',
        r'\geometry{margin=2cm}',
        r'',
        r'\begin{document}',
        r'',
        r'\section*{Optimal single-qubit frame $S$ for $D = S \otimes S$}',
        r'',
        r'For each $(\gamma,\gamma^\prime)$ the Lindbladian is',
        r'\[',
        r'  \mathcal{L} = i J [Z_1 + Z_2,\,\cdot\,]',
        r'  + \gamma \mathcal{D}[S_1^-] + \gamma \mathcal{D}[S_2^-]',
        r'  + \gamma^\prime \mathcal{D}[S_1^+] + \gamma^\prime \mathcal{D}[S_2^+],',
        r'\]',
        r'with $J=1$.  The gate is $G = e^{\delta t\,\mathcal{L}}$ with',
        r'$\delta t = 0.002$.  The two-qubit frame is $D = S\otimes S$ where',
        r'$S\in\mathbb{R}^{4\times 6}$ has unit-norm columns and the rows',
        r'correspond to $\{I,X,Y,Z\}$.  The framability is',
        r'\[',
        r'  \mathcal{F}(D,G) = \max_j \|G^\top D e_j\|_1\,.',
        r'\]',
        r'',
    ]

    for i, (gamma, gp) in enumerate(PARAMS_LIST):
        ig  = int(round(gamma / GAMMA_STEP))
        igp = int(round(gp    / GAMMA_STEP))
        fra_stored = float(scan_full[ig, igp, 3])
        S = S_list[i]
        lines += [
            r'\subsection*{$(\gamma,\gamma^\prime)='
            + f'({gamma:g},{gp:g})$'
            + r'}',
            r'',
            r'\textbf{Framability:} $\mathcal{F} = '
            + f'{fra_stored:.6f}$',
            r'',
            r'\[',
            r'S = ' + matrix_latex(S),
            r'\]',
            r'',
        ]

    lines += [r'\end{document}', '']

    out_tex = os.path.join(args.out_dir, 'one_qubit_S_matrices.tex')
    with open(out_tex, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f'Written {out_tex}')


if __name__ == '__main__':
    main()
