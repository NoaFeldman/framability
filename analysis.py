"""
Steady-state analysis and parameter-scan plotting.

Combines the Lindbladian, LPDO, and framability modules to compute
various properties of the two-qubit steady state.
"""

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import pauli_string_dim, numeric_two_qubit_lindbladian
from lpdo import purification_sqrt, disentangle_ancilla, truncate_and_validate
from framability import get_framability, extended_pauli_D
from optimize_framability import minimize_framability


def decay_rate(L):
    """Spectral gap of the Lindbladian: |Re(lambda)| for the second-smallest eigenvalue."""
    eigenvalues = np.linalg.eigvals(L)
    real_parts = eigenvalues.real

    idx_sorted = np.argsort(np.abs(real_parts))
    lam0 = real_parts[idx_sorted[0]]
    if np.abs(lam0) > 1e-12:
        raise ValueError(
            f'Smallest eigenvalue real part is {lam0}, expected 0 (within 1e-12).'
        )

    lam1 = real_parts[idx_sorted[1]]
    if lam1 > 1e-12:
        raise ValueError(
            f'Second-smallest eigenvalue real part is {lam1}, expected negative.'
        )

    return np.abs(lam1)


def analyze_steady_state(J, gamma, gamma_p, gamma_step=0.01):
    """
    Numerical analysis of the two-qubit Lindbladian steady state.

    Returns (rho_ss, entropy, negativity, magic, pauli_framability,
             ext_framability, min_framability, dec_rate, chi_lpdo):
      rho_ss             – 4x4 steady-state density matrix
      entropy            – von Neumann entropy  -sum lambda*ln(lambda)
      negativity         – sum of |negative eigenvalues| of partial transpose
      magic              – weighted-average stabilizer Rényi entropy
      pauli_framability  – max row 1-norm of expm(dt*L) with dt=0.01*gamma_step
      ext_framability    – get_framability with extended Pauli D
      min_framability    – minimal framability optimised over kronecker frames
      dec_rate           – spectral gap (decay rate) of the Lindbladian
      chi_lpdo           – minimal LPDO bond dimension (fidelity >= 1 - 1e-9)
    """
    # Pauli matrices
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [I2, sx, sy, sz]

    # Two-qubit Pauli basis
    basis = [np.kron(p1, p2) for p1 in paulis for p2 in paulis]
    n = pauli_string_dim
    L = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)

    # 1. Steady state (right eigenvector with eigenvalue 0)
    eigenvalues, eigenvectors = np.linalg.eig(L)
    idx = np.argmin(np.abs(eigenvalues))
    ss_vec = eigenvectors[:, idx].real
    ss_vec = ss_vec / (ss_vec[0] * 4)        # normalise so Tr(rho)=1
    rho_ss = sum(ss_vec[i] * basis[i] for i in range(n))

    # 2. Von Neumann entropy (entanglement with environment)
    evals_rho = np.linalg.eigvalsh(rho_ss)
    evals_pos = evals_rho[evals_rho > 1e-15]
    entropy = -np.sum(evals_pos * np.log(evals_pos))

    # 3. Negativity (entanglement between qubits via partial transpose)
    rho_pt = (rho_ss.reshape([2, 2, 2, 2])
                     .transpose([0, 3, 2, 1])
                     .reshape([4, 4]))
    evals_pt = np.linalg.eigvalsh(rho_pt)
    negativity = np.sum(np.abs(evals_pt[evals_pt < -1e-15]))

    # 4. Magic (weighted-average stabilizer Rényi entropy)
    evals_dm, evecs_dm = np.linalg.eigh(rho_ss)
    magic = 0.0
    for k in range(4):
        lam = evals_dm[k]
        if lam < 1e-15:
            continue
        psi = evecs_dm[:, k]
        rho_k = np.outer(psi, psi.conj())
        coeffs = np.array([np.trace(basis[i] @ rho_k).real
                           for i in range(n)]) / 4
        sre = np.log(np.sum(coeffs**4))
        magic += lam * sre

    # 5. Pauli framability: max row 1-norm of expm(dt*L)
    dt = 0.01 * gamma_step
    M = expm(dt * L)
    if np.max(np.abs(M.imag)) > 1e-12:
        raise ValueError(
            f'expm(dt*L) has non-negligible imaginary part '
            f'(max |imag| = {np.max(np.abs(M.imag)):.2e}). '
            f'The Lindbladian propagator should be real-valued.'
        )
    M = M.real
    pauli_framability = np.max(np.sum(np.abs(M), axis=1))

    # 6. Extended Pauli framability
    D_ext = extended_pauli_D()
    ext_framability = get_framability(D_ext, M)

    # 7. Minimal framability (optimised over Kronecker frames)
    d_ext = D_ext.shape[1]
    _, min_framability = minimize_framability(
        M, d_ext=d_ext, mode='kronecker', n_restarts=2,
        method='cobyqa', max_iter=100, maxfev=500,
        verbose=False,
    )

    # 8. Decay rate
    dec_rate = decay_rate(L)

    # 9. Minimal LPDO bond dimension (2-site partition)
    #    d_site = sqrt(dim(rho)) so that rho lives on d_site^2 x d_site^2
    #    For N qubits split into 2 sites: d_site = 2^(N/2)
    d_site = int(round(np.sqrt(rho_ss.shape[0])))
    X_lp = purification_sqrt(rho_ss)
    A1_lp, A2_lp, chi_dis, _, _ = disentangle_ancilla(X_lp, d_site, maxiter=1000)
    _, _, chi_lpdo, _, _ = truncate_and_validate(rho_ss, A1_lp, A2_lp, d_site)

    return rho_ss, entropy, negativity, magic, pauli_framability, ext_framability, min_framability, dec_rate, chi_lpdo


def scan_and_plot(J=1.0, gamma_step=0.1, n_pts=20):
    """Scan gamma and gamma' on a grid and plot colormaps of steady-state properties."""
    import matplotlib.pyplot as plt

    gammas = [gamma_step * i for i in range(n_pts)]
    gamma_ps = [gamma_step * i for i in range(n_pts)]

    entropy_map = np.zeros((n_pts, n_pts))
    negativity_map = np.zeros((n_pts, n_pts))
    magic_map = np.zeros((n_pts, n_pts))
    pauli_fra_map = np.zeros((n_pts, n_pts))
    ext_fra_map = np.zeros((n_pts, n_pts))
    min_fra_map = np.zeros((n_pts, n_pts))
    decay_map = np.zeros((n_pts, n_pts))
    chi_map = np.zeros((n_pts, n_pts))

    for ig, g in enumerate(gammas):
        if ig % 10 == 0:
            print(f'  row {ig}/{n_pts} (gamma={g:.3f})')
        for igp, gp in enumerate(gamma_ps):
            _, ent, neg, mag, pfra, efra, mfra, dr, chi = analyze_steady_state(J, g, gp, gamma_step)
            entropy_map[ig, igp] = ent
            negativity_map[ig, igp] = neg
            magic_map[ig, igp] = mag
            pauli_fra_map[ig, igp] = pfra
            ext_fra_map[ig, igp] = efra
            min_fra_map[ig, igp] = mfra
            decay_map[ig, igp] = dr
            chi_map[ig, igp] = chi

    extent = [gamma_ps[0], gamma_ps[-1], gammas[0], gammas[-1]]

    fig, axes = plt.subplots(2, 4, figsize=(28, 10))
    axes = axes.ravel()

    titles = ['Von Neumann entropy', 'Negativity', 'Magic (avg SRE)',
              'Pauli framability', 'Extended Pauli framability',
              'Min. framability', 'Decay rate',
              r'LPDO bond dim $\chi$']
    data = [entropy_map, negativity_map, magic_map, pauli_fra_map, ext_fra_map,
            min_fra_map, decay_map, chi_map]

    for ax, d, title in zip(axes, data, titles):
        im = ax.imshow(d, origin='lower', aspect='auto', extent=extent)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    # Hide unused subplot(s)
    for ax in axes[len(data):]:
        ax.set_visible(False)

    fig.suptitle(f'Steady-state properties (J = {J})', fontsize=14)
    plt.tight_layout()
    plt.savefig('two_qubit_scan.png', dpi=150)
    plt.show()
    print('Saved figure to two_qubit_scan.png')


if __name__ == '__main__':
    rho_ss, entropy, negativity, magic, pauli_framability, ext_framability, min_framability, dec_rate, chi_lpdo = analyze_steady_state(
        J=1.0, gamma=0.5, gamma_p=0.1, gamma_step=0.01)
    print(f'--- Steady-state analysis (J=1, gamma=0.5, gamma\'=0.1) ---')
    print(f'Von Neumann entropy : {entropy:.6f}')
    print(f'Negativity          : {negativity:.6f}')
    print(f'Magic (avg SRE)     : {magic:.6f}')
    print(f'Pauli framability   : {pauli_framability:.6f}')
    print(f'Ext. framability    : {ext_framability:.6f}')
    print(f'Min. framability    : {min_framability:.6f}')
    print(f'Decay rate          : {dec_rate:.6f}')
    print(f'LPDO bond dim chi   : {chi_lpdo}')
    print('rho_ss eigenvalues  :', np.sort(np.linalg.eigvalsh(rho_ss))[::-1])

    scan_and_plot(J=1.0, gamma_step=0.1, n_pts=20)
