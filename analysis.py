"""
Steady-state analysis and parameter-scan plotting.

Combines the Lindbladian, LPDO, and framability modules to compute
various properties of the two-qubit steady state.
"""

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import pauli_string_dim, numeric_two_qubit_lindbladian
from lpdo import purification_sqrt, disentangle_ancilla, truncate_and_validate
from framability import heisenberg_framability, extended_pauli_D
from optimize_framability import minimize_framability, DEFAULT_METHOD


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
      ext_framability    – heisenberg_framability with extended Pauli D
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
    ext_framability = heisenberg_framability(D_ext, M)

    # 7. Minimal framability (optimised over Kronecker frames)
    d_ext_single = int(round(np.sqrt(D_ext.shape[1])))
    _, min_framability = minimize_framability(
        M, d_ext_single=d_ext_single, n_restarts=5,
        method=DEFAULT_METHOD, max_iter=200, maxfev=1000,
        verbose=False,
    )
    # Safety clamp: the extended-Pauli frame is a valid Kronecker frame,
    # so the true minimum cannot exceed ext_framability.
    min_framability = min(min_framability, ext_framability)

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


# ── helpers for new properties ──────────────────────────────────────────────

def _nqubit_pauli_basis(N):
    """Build the 4^N  N-qubit Pauli-string matrices (standard order)."""
    from itertools import product as iproduct
    from functools import reduce
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return [reduce(np.kron, combo) for combo in iproduct([I2, sx, sy, sz], repeat=N)]


def _initial_iz_vector(N):
    """
    Normalised Pauli-basis coefficient vector whose non-zero entries
    correspond to all Pauli strings built from I and Z only.

    The resulting density matrix is |0...0><0...0|.
    """
    dim = 4 ** N
    vec = np.zeros(dim)
    for bits in range(2 ** N):
        idx = 0
        for k in range(N):
            pauli_idx = 3 if (bits >> (N - 1 - k)) & 1 else 0
            idx += pauli_idx * (4 ** (N - 1 - k))
        vec[idx] = 1.0
    vec /= 2 ** N
    return vec


def _initial_yy_state_vector(N):
    """
    Normalised Pauli-basis coefficient vector for the state
    (|+Y><+Y|)^{⊗N} = ((I + Y)/2)^{⊗N}.

    Non-zero entries correspond to all Pauli strings built from I and Y only,
    each with coefficient 1/2^N.  For N=2 the state is
    (II + IY + YI + YY) / 4.
    """
    dim = 4 ** N
    vec = np.zeros(dim)
    for bits in range(2 ** N):
        idx = 0
        for k in range(N):
            pauli_idx = 2 if (bits >> (N - 1 - k)) & 1 else 0  # Y=2, I=0
            idx += pauli_idx * (4 ** (N - 1 - k))
        vec[idx] = 1.0
    vec /= 2 ** N
    return vec


def compute_steady_state(J, gamma, gamma_p, N=2):
    """
    Compute the steady-state density matrix and the Lindbladian.

    Returns
    -------
    rho_ss : np.ndarray, shape (2^N, 2^N)
    L : np.ndarray, shape (4^N, 4^N)
    """
    L = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)
    dim = 4 ** N
    basis = _nqubit_pauli_basis(N)
    eigenvalues, eigenvectors = np.linalg.eig(L)
    idx = np.argmin(np.abs(eigenvalues))
    ss_vec = eigenvectors[:, idx].real
    ss_vec /= ss_vec[0] * (2 ** N)
    rho_ss = sum(ss_vec[i] * basis[i] for i in range(dim))
    return rho_ss, L


def compute_magnetization(rho_ss, N=2):
    """Magnetization in Z: sum_i Tr(rho_ss @ Z_i) for i=1..N."""
    from functools import reduce
    I2 = np.eye(2, dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    mag = 0.0
    for i in range(N):
        ops = [sz if k == i else I2 for k in range(N)]
        Z_i = reduce(np.kron, ops)
        mag += np.trace(rho_ss @ Z_i).real
    return mag


def compute_magnetization_x(rho_ss, N=2):
    """Magnetization in X: sum_i Tr(rho_ss @ X_i) for i=1..N."""
    from functools import reduce
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    mag = 0.0
    for i in range(N):
        ops = [sx if k == i else I2 for k in range(N)]
        X_i = reduce(np.kron, ops)
        mag += np.trace(rho_ss @ X_i).real
    return mag


def compute_reduced_pauli_framability(rho_ss, L, gamma_step, N=2, tol=1e-10):
    """Pauli framability restricted to Pauli strings with non-zero steady-state coefficient.

    Removes columns of expm(dt*L) corresponding to Pauli strings whose
    coefficient in rho_ss is below `tol`, then returns the max row 1-norm
    of the reduced matrix.
    """
    dim = 4 ** N
    basis = _nqubit_pauli_basis(N)
    # Pauli coefficients of the steady state: rho = sum_i c_i P_i
    coeffs = np.array([np.trace(basis[i] @ rho_ss).real for i in range(dim)]) / (2 ** N)
    keep = np.abs(coeffs) > tol

    dt = 0.01 * gamma_step
    M = expm(dt * L)
    if np.max(np.abs(M.imag)) < 1e-12:
        M = M.real

    M_reduced = M[:, keep]
    return float(np.max(np.sum(np.abs(M_reduced), axis=1)))


def compute_max_bond_dim(L, rho_ss, gamma_step, N=2, max_steps=200_000,
                         fidelity_threshold=0.9):
    """
    Maximum LPDO bond dimension during Lindbladian time evolution
    from the all-|0> state to the steady state.

    Evolution: v(t+dt) = expm(dt*L) @ v(t) in the Pauli-string basis.
    Stops when Bures fidelity with rho_ss reaches `fidelity_threshold`.

    Uses the direct SVD truncation pipeline (purification -> tensorize ->
    truncate) without the expensive disentangle step, yielding an upper
    bound on the minimal chi at each time step.

    Parameters
    ----------
    L : np.ndarray
        Lindbladian superoperator in the Pauli-string basis.
    rho_ss : np.ndarray
        Steady-state density matrix.
    gamma_step : float
        Grid spacing (dt = 0.01 * gamma_step).
    N : int
        Number of qubits (must be even for the LPDO bipartition).
    max_steps : int
        Safety limit on the number of time steps.
    fidelity_threshold : float
        Stop evolving once Bures fidelity with rho_ss >= this value (default 0.9).

    Returns
    -------
    max_chi : int
        Maximum LPDO bond dimension observed along the trajectory.
    max_entropy : float
        Maximum bond entropy (Shannon entropy of squared normalised
        singular values) observed along the trajectory.
    """
    from lpdo import purification_sqrt, tensorize_to_lpdo, truncate_and_validate, _bures_fidelity, _bond_entropy

    assert N % 2 == 0, "LPDO bipartition requires even N"
    dim = 4 ** N
    d_site = int(round(2 ** (N / 2)))
    dt = 0.01 * gamma_step
    M = expm(dt * L)
    if np.max(np.abs(M.imag)) < 1e-12:
        M = M.real

    basis = _nqubit_pauli_basis(N)
    basis_arr = np.array(basis)          # (dim, 2^N, 2^N)
    v = _initial_yy_state_vector(N)
    max_chi = 0
    max_entropy = 0.0

    for step in range(max_steps):
        # Reconstruct density matrix from Pauli coefficients
        rho = np.einsum('i,ijk->jk', v, basis_arr)
        rho = (rho + rho.conj().T) / 2  # enforce Hermiticity

        # LPDO bond dimension (direct SVD, no disentangle)
        try:
            X_lp = purification_sqrt(rho)
            A1, A2, chi_init = tensorize_to_lpdo(X_lp, d_site)
            _, _, chi, _, info = truncate_and_validate(rho, A1, A2, d_site)
            max_chi = max(max_chi, chi)
            max_entropy = max(max_entropy, _bond_entropy(info['singular_values']))
        except Exception:
            pass

        # Check convergence
        fid = _bures_fidelity(rho, rho_ss)
        if fid >= fidelity_threshold:
            break

        # Evolve
        v_new = M @ v
        if np.allclose(v, v_new, atol=1e-15):
            break
        v = v_new

    # Include the steady state itself (it is the endpoint of the trajectory)
    try:
        X_lp = purification_sqrt(rho_ss)
        A1, A2, chi_init = tensorize_to_lpdo(X_lp, d_site)
        _, _, chi, _, info = truncate_and_validate(rho_ss, A1, A2, d_site)
        max_chi = max(max_chi, chi)
        max_entropy = max(max_entropy, _bond_entropy(info['singular_values']))
    except Exception:
        pass

    return max_chi, max_entropy


def compute_ss_bond_entropy(rho_ss, N=2):
    """Bond entropy of the steady-state LPDO (direct SVD, no disentangling)."""
    from lpdo import purification_sqrt, tensorize_to_lpdo, truncate_and_validate, _bond_entropy
    d_site = int(round(2 ** (N / 2)))
    X_lp = purification_sqrt(rho_ss)
    A1, A2, _ = tensorize_to_lpdo(X_lp, d_site)
    _, _, _, _, info = truncate_and_validate(rho_ss, A1, A2, d_site)
    return _bond_entropy(info['singular_values'])


def _generate_stabilizer_states(N=2):
    """
    Generate all N-qubit pure stabilizer state density matrices.

    Each stabilizer state is the unique +1 eigenstate of a stabilizer group
    generated by N independent, mutually commuting Hermitian Pauli operators.
    For N qubits, the state is  rho = (I + g1 + g2 + g1*g2) / 2^N  (for N=2).

    Currently supports N=1 and N=2.

    Returns
    -------
    list of np.ndarray
        Each element is a (2^N, 2^N) density matrix.
    """
    from itertools import product as iproduct
    from functools import reduce

    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis_1q = [I2, sx, sy, sz]

    if N == 1:
        return [(I2 + s * P) / 2
                for P in [sx, sy, sz] for s in [+1, -1]]

    if N != 2:
        raise NotImplementedError(
            f"Stabilizer state generation for N={N} not implemented."
        )

    dim = 2 ** N
    eye_d = np.eye(dim, dtype=complex)

    # All N-qubit tensor-product Paulis (16 for N=2)
    paulis_nq = [reduce(np.kron, combo)
                 for combo in iproduct(paulis_1q, repeat=N)]

    # Signed non-identity Paulis: (+1, idx, matrix) and (-1, idx, matrix)
    signed = []
    for k in range(1, len(paulis_nq)):
        signed.append((+1, k, paulis_nq[k]))
        signed.append((-1, k, paulis_nq[k]))

    seen = set()
    states = []

    for a in range(len(signed)):
        s1, k1, g1 = signed[a]
        g1s = s1 * g1

        for b in range(a + 1, len(signed)):
            s2, k2, g2 = signed[b]
            if k1 == k2:                    # same unsigned Pauli
                continue
            g2s = s2 * g2

            # Must commute
            if not np.allclose(g1s @ g2s, g2s @ g1s):
                continue

            g3 = g1s @ g2s

            # g3 must be Hermitian (always true for commuting Hermitians,
            # but check as a safety net)
            if not np.allclose(g3, g3.conj().T, atol=1e-12):
                continue

            # Stabilizer state: rho = (I + g1 + g2 + g1*g2) / 2^N
            rho = (eye_d + g1s + g2s + g3) / dim

            # Must be a valid pure state (rank 1, PSD)
            eigs = np.linalg.eigvalsh(rho)
            if np.min(eigs) < -1e-10:
                continue
            if not np.isclose(np.trace(rho @ rho).real, 1.0, atol=1e-8):
                continue

            # Deduplicate via rounded matrix bytes
            key = (np.round(rho.real, 8).tobytes(),
                   np.round(rho.imag, 8).tobytes())
            if key not in seen:
                seen.add(key)
                states.append(rho)

    return states


# Module-level cache for the stabilizer A matrix
_CACHED_A_MATRIX = {}


def _stabilizer_a_matrix(N=2):
    """
    Build the stabilizer A matrix whose columns are Pauli-basis
    representations of all pure stabilizer states.

    A[i, j] = Tr[P_i  |stab_j><stab_j|]

    Shape: (4^N, n_stabilizer_states)   e.g. (16, 60) for N=2.
    """
    if N in _CACHED_A_MATRIX:
        return _CACHED_A_MATRIX[N]

    basis = _nqubit_pauli_basis(N)
    states = _generate_stabilizer_states(N)
    dim = 4 ** N

    A = np.zeros((dim, len(states)))
    for j, rho in enumerate(states):
        for i, P in enumerate(basis):
            A[i, j] = np.trace(P @ rho).real

    _CACHED_A_MATRIX[N] = A
    return A


def compute_rom(rho_ss, N=2):
    """
    Robustness of Magic (RoM) for a density matrix.

    Solves  min ||x||_1  subject to  A x = b
    where A is the stabilizer-state matrix and b_i = Tr[P_i rho_ss].

    Reference
    ---------
    Hamaguchi, Hamada & Yoshioka, "Handbook for Quantifying Robustness
    of Magic", Quantum 8, 1461 (2024).  arXiv:2311.01362

    Parameters
    ----------
    rho_ss : np.ndarray, shape (2^N, 2^N)
    N : int

    Returns
    -------
    rom : float
    """
    from scipy.optimize import linprog

    basis = _nqubit_pauli_basis(N)
    b = np.array([np.trace(P @ rho_ss).real for P in basis])

    A = _stabilizer_a_matrix(N)
    n_stab = A.shape[1]

    # LP reformulation: min 1^T u  s.t. [A, -A] u = b,  u >= 0
    c = np.ones(2 * n_stab)
    A_eq = np.hstack([A, -A])
    res = linprog(c, A_eq=A_eq, b_eq=b, bounds=(0, None), method='highs')

    if not res.success:
        raise RuntimeError(f"RoM LP failed: {res.message}")

    return res.fun


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
