r"""
Compute OTOC for an n-qubit Lindbladian and plot two-qubit scans over
(gamma, gamma').

Definitions used:
1) psi_0 from (I + Z)^{\otimes n}, i.e. |0...0> (normalised ket)
2) V_0 = X on qubit 1
3) W_0 = X on qubit n
4) W(t) = E_t^\dagger(W_0), with E_t = exp(L t)
5) OTOC(t) = <psi_0| W(t)^\dagger V_0^\dagger W(t) V_0 |psi_0>

Model (n-qubit extension):
- Hamiltonian: H = J * sum_{i=1}^{n-1} Z_i Z_{i+1}
- Dissipators: gamma * D[|-><+|_i] on each site i
- Dephasing:   gamma' * D[Z_i] on each site i

Usage
-----
python plot_otoc_lindbladian.py --n_qubits 2 --n_pts 41 --gamma_step 0.2 --J 1.0

Outputs
-------
- otoc_two_qubit_tmin.png     with t = 0.1 * min(gamma, gamma')
- otoc_two_qubit_tmax.png     with t = 10  * max(gamma, gamma')
- otoc_tmin.npy, otoc_tmax.npy
"""

import argparse
from functools import reduce
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian


def kron_n(ops):
    return reduce(np.kron, ops)


def pauli_basis_n(n):
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    one_qubit = [I2, X, Y, Z]
    return [kron_n(ops) for ops in product(one_qubit, repeat=n)]


def local_operator(n, site, op):
    I2 = np.eye(2, dtype=complex)
    ops = [I2 for _ in range(n)]
    ops[site] = op
    return kron_n(ops)


def two_site_operator(n, i, op_i, j, op_j):
    I2 = np.eye(2, dtype=complex)
    ops = [I2 for _ in range(n)]
    ops[i] = op_i
    ops[j] = op_j
    return kron_n(ops)


def lindbladian_superop_n_qubits(J, gamma, gamma_p, n):
    """Return L in Pauli basis with convention L[i,j] = Tr(P_i L(P_j)) / d."""
    if n == 2:
        return numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)

    d = 2 ** n
    basis = pauli_basis_n(n)
    m = len(basis)

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    mp = 0.5 * np.array([[1, 1], [-1, -1]], dtype=complex)  # |-><+|
    pm = mp.conj().T

    # H = J * sum_{i=0}^{n-2} Z_i Z_{i+1}
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        H += J * two_site_operator(n, i, Z, i + 1, Z)

    jumps_mp = [local_operator(n, i, mp) for i in range(n)]
    jumps_pm = [local_operator(n, i, pm) for i in range(n)]
    jumps_z = [local_operator(n, i, Z) for i in range(n)]

    L = np.zeros((m, m), dtype=complex)
    for j, Pj in enumerate(basis):
        res = -1j * (H @ Pj - Pj @ H)

        for Ji, Jid in zip(jumps_mp, jumps_pm):
            JdJ = Jid @ Ji
            res += gamma * (Ji @ Pj @ Jid - 0.5 * (JdJ @ Pj + Pj @ JdJ))

        for Zi in jumps_z:
            res += gamma_p * (Zi @ Pj @ Zi - Pj)

        for i, Pi in enumerate(basis):
            L[i, j] = np.trace(Pi @ res) / d

    return L


def operator_to_pauli_coeffs(O, basis, d):
    return np.array([np.trace(P @ O) / d for P in basis], dtype=complex)


def pauli_coeffs_to_operator(coeffs, basis):
    O = np.zeros_like(basis[0], dtype=complex)
    for c, P in zip(coeffs, basis):
        O += c * P
    return O


def heisenberg_evolve_operator(W0, E_t, basis, d):
    """W(t) = E_t^dagger(W0) in Pauli basis."""
    coeffs_w0 = operator_to_pauli_coeffs(W0, basis, d)
    coeffs_wt = E_t.conj().T @ coeffs_w0
    return pauli_coeffs_to_operator(coeffs_wt, basis)


def build_initial_state_ket(n):
    # (I + Z) / 2 = |0><0|, so ket is |0...0>
    d = 2 ** n
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0
    return psi


def otoc_value(Wt, V0, psi0):
    op = Wt.conj().T @ V0.conj().T @ Wt @ V0
    return np.vdot(psi0, op @ psi0)


def compute_two_time_otoc_maps(n_qubits, n_pts, gamma_step, J):
    d = 2 ** n_qubits
    gammas = gamma_step * np.arange(n_pts)

    basis = pauli_basis_n(n_qubits)

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    V0 = local_operator(n_qubits, 0, X)                 # X_1
    W0 = local_operator(n_qubits, n_qubits - 1, X)      # X_n
    psi0 = build_initial_state_ket(n_qubits)

    otoc_tmin = np.zeros((n_pts, n_pts), dtype=float)
    otoc_tmax = np.zeros((n_pts, n_pts), dtype=float)

    for ig, gamma in enumerate(gammas):
        for igp, gamma_p in enumerate(gammas):
            L = lindbladian_superop_n_qubits(J=J, gamma=gamma, gamma_p=gamma_p, n=n_qubits)

            t_min = 0.1 * min(gamma, gamma_p)
            t_max = 10.0 * max(gamma, gamma_p)

            E_min = expm(L * t_min)
            E_max = expm(L * t_max)

            Wt_min = heisenberg_evolve_operator(W0, E_min, basis, d)
            Wt_max = heisenberg_evolve_operator(W0, E_max, basis, d)

            val_min = otoc_value(Wt_min, V0, psi0)
            val_max = otoc_value(Wt_max, V0, psi0)

            # OTOC should be real for this setup; keep real part for plotting.
            otoc_tmin[ig, igp] = float(np.real(val_min))
            otoc_tmax[ig, igp] = float(np.real(val_max))

        if ig % 5 == 0:
            print(f"row {ig}/{n_pts - 1} complete")

    return gammas, otoc_tmin, otoc_tmax


def plot_heatmap(data, gammas, gamma_step, title, out_file):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    half = gamma_step / 2.0
    extent = [
        gammas[0] - half, gammas[-1] + half,
        gammas[0] - half, gammas[-1] + half,
    ]

    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="magma",
    )

    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("OTOC")

    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute OTOC maps for n-qubit Lindbladian evolution.")
    parser.add_argument("--n_qubits", type=int, default=2, help="Number of qubits (default: 2)")
    parser.add_argument("--n_pts", type=int, default=41, help="Grid points per axis (default: 41)")
    parser.add_argument("--gamma_step", type=float, default=0.2, help="Step for gamma and gamma' (default: 0.2)")
    parser.add_argument("--J", type=float, default=1.0, help="ZZ coupling strength (default: 1.0)")
    parser.add_argument("--out_prefix", type=str, default="otoc", help="Output prefix (default: otoc)")
    args = parser.parse_args()

    if args.n_qubits < 2:
        raise ValueError("n_qubits must be at least 2 (V0=X_1 and W0=X_n need distinct sites).")

    print(
        f"Computing OTOC maps for n_qubits={args.n_qubits}, n_pts={args.n_pts}, "
        f"gamma_step={args.gamma_step}, J={args.J}"
    )

    gammas, otoc_tmin, otoc_tmax = compute_two_time_otoc_maps(
        n_qubits=args.n_qubits,
        n_pts=args.n_pts,
        gamma_step=args.gamma_step,
        J=args.J,
    )

    np.save(f"{args.out_prefix}_tmin.npy", otoc_tmin)
    np.save(f"{args.out_prefix}_tmax.npy", otoc_tmax)

    plot_heatmap(
        data=otoc_tmin,
        gammas=gammas,
        gamma_step=args.gamma_step,
        title=r"OTOC at $t=0.1\,\min(\gamma,\gamma')$",
        out_file="otoc_two_qubit_tmin.png",
    )
    plot_heatmap(
        data=otoc_tmax,
        gammas=gammas,
        gamma_step=args.gamma_step,
        title=r"OTOC at $t=10\,\max(\gamma,\gamma')$",
        out_file="otoc_two_qubit_tmax.png",
    )

    print("Saved otoc_two_qubit_tmin.png and otoc_two_qubit_tmax.png")
    print(f"OTOC(t_min) range: [{otoc_tmin.min():.6f}, {otoc_tmin.max():.6f}]")
    print(f"OTOC(t_max) range: [{otoc_tmax.min():.6f}, {otoc_tmax.max():.6f}]")


if __name__ == "__main__":
    main()
