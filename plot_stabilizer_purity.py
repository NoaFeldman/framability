"""
Compute and plot the channel Stabilizer Purity (magic measure) for the
two-qubit channel E = exp(L * dt) over the (gamma, gamma') parameter grid.

For a channel E acting on n-qubit density matrices, we generalise the
unitary formula to

    M(E) = log2( sum_P Tr(E(P) P)^2 / (d+1) )

where P runs over all d^2 = 16 unnormalised two-qubit Pauli strings
(satisfying Tr(P_i P_j) = 4 delta_{ij}), and d = 2^2 = 4.

Because the Lindbladian L is stored in the Pauli-string basis with
  L[i,j] = Tr(sigma_i  L(sigma_j)) / 4,
the superoperator E = exp(L*dt) satisfies
  Tr(E(P_j) P_j) = 4 * E[j,j],
so
  sum_P Tr(E(P) P)^2 = 16 * sum_j E[j,j]^2.

Usage
-----
    python plot_stabilizer_purity.py --n_pts 41 --J 1.0 --gamma_step 0.2 \\
                                      --dt 0.002 --out two_qubit_scan.png
"""

import argparse

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

from two_qubit_lindbladian import numeric_two_qubit_lindbladian


def channel_stabilizer_purity(E_mat, d=4):
    """
    Compute the channel Stabilizer Purity:

        M(E) = log2( d^2 * sum_j E[j,j]^2 / (d+1) )

    where E_mat is the d^2 x d^2 superoperator in the unnormalised
    Pauli-string basis with convention  E[i,j] = Tr(sigma_i E(sigma_j)) / d.

    Parameters
    ----------
    E_mat : (d^2, d^2) real array – exp(L * dt) in Pauli-string basis
    d     : int                   – Hilbert-space dimension (default 4)

    Returns
    -------
    float
    """
    diag = np.diag(E_mat).real   # real because L maps Hermitians to Hermitians
    total = (d ** 2) * float(np.sum(diag ** 2))
    return np.log2(total / (d + 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot M(exp(L*dt)) over (gamma, gamma') grid."
    )
    parser.add_argument('--n_pts',      type=int,   default=41)
    parser.add_argument('--J',          type=float, default=1.0)
    parser.add_argument('--gamma_step', type=float, default=0.2)
    parser.add_argument('--dt',         type=float, default=0.002,
                        help='Time step for expm(L * dt). Default: 0.002')
    parser.add_argument('--out',        type=str,   default='two_qubit_scan.png')
    args = parser.parse_args()

    n = args.n_pts
    gammas = args.gamma_step * np.arange(n)
    d = 4  # 2-qubit Hilbert-space dimension

    grid = np.zeros((n, n))

    print(f'Computing M(exp(L*dt)) on {n}x{n} grid  (dt={args.dt}) ...')
    for ig, gamma in enumerate(gammas):
        for igp, gp in enumerate(gammas):
            L = numeric_two_qubit_lindbladian(J=args.J, gamma=gamma, gamma_p=gp)
            E = expm(L * args.dt)
            grid[ig, igp] = channel_stabilizer_purity(E, d=d)

        if ig % 5 == 0:
            print(f'  row {ig}/{n - 1} done')

    print(f'M range: [{grid.min():.4f}, {grid.max():.4f}]')

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    half = args.gamma_step / 2
    extent = [
        gammas[0] - half, gammas[-1] + half,
        gammas[0] - half, gammas[-1] + half,
    ]

    im = ax.imshow(
        grid,
        origin='lower',
        extent=extent,
        aspect='equal',
        cmap='viridis',
    )
    fig.colorbar(im, ax=ax, label=r'$M(e^{L\,\delta t})$')
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r'$\gamma$')
    ax.set_title(
        r'Channel Stabilizer Purity  $M(e^{L\,\delta t})$'
        f'\n(J={args.J}, dt={args.dt})'
    )

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
