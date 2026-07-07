"""
Compute and plot the operator bond entropy of exp(dt * L) over the full
(gamma, gamma') grid.

Operator bond entropy:
  gate is 16x16 in the Pauli-string basis.  Treat it as a 4-tensor
  gate[a1, a2, b1, b2]  (shape 4,4,4,4) where a = output, b = input,
  and subscripts 1,2 label the two sites.
  Bipartition: (a1,b1) vs (a2,b2), i.e. reshape to (16,16) after
  transposing gate.reshape(4,4,4,4) to (a1,b1,a2,b2).
  Bond entropy = Shannon entropy of (squared normalised singular values).

Usage
-----
    python _plot_operator_bond_entropy.py [--n_pts 41] [--J 1.0]
                                          [--gamma_step 0.2] [--out_dir results]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian


def _bond_entropy(singular_values):
    p = singular_values ** 2
    total = p.sum()
    if total < 1e-30:
        return 0.0
    p = p / total
    p = p[p > 1e-30]
    return float(-np.sum(p * np.log(p)))


def operator_bond_entropy(gate):
    """
    Operator bond entropy of a 16x16 gate (two-qubit Pauli-string basis).

    Reshape gate to (4,4,4,4) = (out1, out2, in1, in2), then transpose to
    (out1, in1, out2, in2) = (16, 16) and take the SVD.
    """
    T = gate.reshape(4, 4, 4, 4)          # [out1, out2, in1, in2]
    T = T.transpose(0, 2, 1, 3)           # [out1, in1, out2, in2]
    M = T.reshape(16, 16)
    sv = np.linalg.svd(M, compute_uv=False)
    return _bond_entropy(sv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_pts',      type=int,   default=41)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.2)
    p.add_argument('--out_dir',    type=str,   default='results')
    args = p.parse_args()

    n = args.n_pts
    dt = 0.01 * args.gamma_step
    grid = np.zeros((n, n))

    print(f'Computing operator bond entropy on {n}x{n} grid  (dt={dt:.5f})', flush=True)

    for ig in range(n):
        gamma = args.gamma_step * ig
        for igp in range(n):
            gp = args.gamma_step * igp
            L = numeric_two_qubit_lindbladian(J=args.J, gamma=gamma, gamma_p=gp)
            gate = expm(dt * L).real
            grid[ig, igp] = operator_bond_entropy(gate)
        if ig % 5 == 0 or ig == n - 1:
            print(f'  row {ig+1}/{n}  gamma={gamma:.3f}', flush=True)

    # ── Save ──────────────────────────────────────────────────────────────
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, 'operator_bond_entropy.npy')
    np.save(npy_path, grid)
    print(f'Saved {npy_path}  shape={grid.shape}')

    # ── Plot ──────────────────────────────────────────────────────────────
    gammas = args.gamma_step * np.arange(n)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        grid,
        origin='lower',
        extent=[gammas[0], gammas[-1], gammas[0], gammas[-1]],
        aspect='equal',
        cmap='inferno',
    )
    fig.colorbar(im, ax=ax, label='Operator bond entropy')
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r'$\gamma$')
    ax.set_title(r'Operator bond entropy of $e^{dt\,\mathcal{L}}$')
    fig.tight_layout()

    os.makedirs('results_plots', exist_ok=True)
    png_path = os.path.join('results_plots', 'operator_bond_entropy.png')
    fig.savefig(png_path, dpi=150)
    print(f'Saved {png_path}')
    plt.show()


if __name__ == '__main__':
    main()
