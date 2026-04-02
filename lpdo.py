"""
LPDO (Locally Purified Density Operator) construction pipeline.

Stages:
  1. purification_sqrt   – symmetric square root  X  such that rho = X X†
  2. tensorize_to_lpdo   – reshape X into a 2-site tensor chain via SVD
  3. disentangle_ancilla – variational ancilla unitary to minimise bond entropy
  4. truncate_and_validate – truncate bond dimension to target fidelity
"""

import numpy as np


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def purification_sqrt(rho, tol=1e-10):
    """
    Stage 1 of LPDO construction: global purification via symmetric square root.

    Given a d^2 x d^2 density matrix rho, compute X = V sqrt(Lambda) V^dag
    such that rho = X X^dag.

    Parameters
    ----------
    rho : np.ndarray
        Hermitian positive-semidefinite density matrix of shape (d^2, d^2).
    tol : float
        Tolerance for verification and for clipping small negative eigenvalues.

    Returns
    -------
    X : np.ndarray
        Matrix square root with shape (d^2, d^2), satisfying rho = X @ X.conj().T.
    """
    rho = np.asarray(rho, dtype=complex)
    n = rho.shape[0]
    if rho.shape != (n, n):
        raise ValueError(f'rho must be square, got shape {rho.shape}.')

    # Eigen-decomposition (eigenvalues in ascending order)
    eigenvalues, V = np.linalg.eigh(rho)

    # Clip small negative eigenvalues arising from numerical noise
    if np.any(eigenvalues < -tol):
        raise ValueError(
            f'rho has eigenvalue {eigenvalues.min():.2e}, which is more negative '
            f'than the tolerance {tol:.0e}. The matrix is not positive semidefinite.'
        )
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    # X = V @ diag(sqrt(lambda)) @ V^dag
    sqrt_lam = np.diag(np.sqrt(eigenvalues))
    X = V @ sqrt_lam @ V.conj().T

    # Verify X X^dag ≈ rho
    residual = np.linalg.norm(X @ X.conj().T - rho) / max(np.linalg.norm(rho), 1e-30)
    if residual > tol:
        raise RuntimeError(
            f'Purification verification failed: '
            f'||X X^dag - rho|| / ||rho|| = {residual:.2e} > {tol:.0e}.'
        )

    return X


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def tensorize_to_lpdo(X, d, svd_tol=1e-12):
    """
    Stage 2 of LPDO construction: reshape X into a 2-site LPDO tensor chain.

    X is a (d^2, d^2) matrix satisfying rho = X X^dag.
    Rows are indexed by (s1, s2) — physical indices of the two sites.
    Columns are indexed by (a1, a2) — ancilla (Kraus) indices.

    Steps:
      1. Reshape X -> T[s1, s2, a1, a2]    shape (d, d, d, d)
      2. Permute    -> T[s1, a1, s2, a2]    shape (d, d, d, d)
      3. Group      -> M[(s1,a1), (s2,a2)]  shape (d^2, d^2)
      4. SVD        -> U @ diag(S) @ Vh     truncate to chi_init non-zero values
      5. Return A1[s1, a1, chi] and A2[chi, s2, a2]

    Parameters
    ----------
    X : np.ndarray
        Square root matrix from purification_sqrt, shape (d^2, d^2).
    d : int
        Local Hilbert space dimension per site.
    svd_tol : float
        Singular values below this threshold are discarded.

    Returns
    -------
    A1 : np.ndarray
        Left site tensor of shape (d, d, chi_init)  — indices (s1, a1, bond).
    A2 : np.ndarray
        Right site tensor of shape (chi_init, d, d)  — indices (bond, s2, a2).
    chi_init : int
        Bond dimension after truncation.
    """
    X = np.asarray(X, dtype=complex)
    if X.shape != (d**2, d**2):
        raise ValueError(f'X must have shape ({d**2}, {d**2}), got {X.shape}.')

    # 1. Reshape: rows = (s1, s2), cols = (a1, a2)
    T = X.reshape(d, d, d, d)          # T[s1, s2, a1, a2]

    # 2. Permute to (s1, a1, s2, a2)
    T = T.transpose(0, 2, 1, 3)        # T[s1, a1, s2, a2]

    # 3. Group into matrix M[(s1,a1), (s2,a2)]
    M = T.reshape(d * d, d * d)

    # 4. SVD
    U, S, Vh = np.linalg.svd(M, full_matrices=False)

    # Truncate negligible singular values
    keep = S > svd_tol
    chi_init = int(np.sum(keep))
    if chi_init == 0:
        raise ValueError('All singular values are below svd_tol; X is effectively zero.')

    U = U[:, :chi_init]
    S = S[:chi_init]
    Vh = Vh[:chi_init, :]

    # Absorb singular values into Vh (convention: A1 is left-unitary)
    SVh = np.diag(S) @ Vh              # shape (chi_init, d^2)

    # 5. Reshape into site tensors
    A1 = U.reshape(d, d, chi_init)     # (s1, a1, chi)
    A2 = SVh.reshape(chi_init, d, d)   # (chi, s2, a2)

    return A1, A2, chi_init


# ---------------------------------------------------------------------------
# Stage 3 helpers
# ---------------------------------------------------------------------------

def _params_to_unitary(params, n):
    """Convert n^2 real parameters to an n×n unitary via expm(i·H) with H Hermitian."""
    from scipy.linalg import expm
    H = np.zeros((n, n), dtype=complex)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = params[idx] + 1j * params[idx + n * (n - 1) // 2]
            H[j, i] = params[idx] - 1j * params[idx + n * (n - 1) // 2]
            idx += 1
    diag_start = n * (n - 1)
    for i in range(n):
        H[i, i] = params[diag_start + i]
    return expm(1j * H)


def _bond_entropy(singular_values):
    """Von Neumann entropy of the Schmidt spectrum (from singular values)."""
    p = singular_values ** 2
    total = p.sum()
    if total < 1e-30:
        return 0.0
    p = p / total
    p = p[p > 1e-30]
    return -np.sum(p * np.log(p))


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def disentangle_ancilla(X, d, maxiter=1000, tol=1e-10):
    """
    Stage 3 of LPDO construction: variational disentangling of the ancilla space.

    Apply a unitary U on the ancilla legs (a1, a2) of X to minimise the bond
    entanglement entropy across the (s1,a1)|(s2,a2) partition, thereby
    minimising the bond dimension chi needed for a faithful LPDO.

    Parameters
    ----------
    X : np.ndarray
        Square-root matrix of shape (d^2, d^2) from purification_sqrt.
    d : int
        Local Hilbert space dimension per site.
    maxiter : int
        Maximum optimiser iterations.
    tol : float
        Optimiser convergence tolerance.

    Returns
    -------
    A1 : np.ndarray
        Left site tensor of shape (d, d, chi_opt)  — (s1, a1, bond).
    A2 : np.ndarray
        Right site tensor of shape (chi_opt, d, d)  — (bond, s2, a2).
    chi_opt : int
        Optimised bond dimension.
    U_opt : np.ndarray
        The optimal d^2 × d^2 ancilla unitary.
    info : dict
        Optimisation result from scipy.optimize.minimize.
    """
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=complex)
    n_anc = d ** 2   # ancilla dimension = d^2
    n_params = n_anc ** 2   # number of real parameters for U(n_anc)

    def cost(params):
        U_trial = _params_to_unitary(params, n_anc)
        X_rot = X @ U_trial                       # (d^2, d^2)
        T = X_rot.reshape(d, d, d, d)             # (s1, s2, a1, a2)
        T = T.transpose(0, 2, 1, 3)               # (s1, a1, s2, a2)
        M = T.reshape(d * d, d * d)
        sv = np.linalg.svd(M, compute_uv=False)
        return _bond_entropy(sv)

    # Start from identity (params = 0)
    x0 = np.zeros(n_params)
    result = minimize(cost, x0, method='L-BFGS-B', options={
        'maxiter': maxiter, 'ftol': tol, 'gtol': tol,
    })

    # Reconstruct optimal tensors
    U_opt = _params_to_unitary(result.x, n_anc)
    X_opt = X @ U_opt
    A1, A2, chi_opt = tensorize_to_lpdo(X_opt, d)

    return A1, A2, chi_opt, U_opt, {'fun': result.fun, 'success': result.success,
                                     'nit': result.nit, 'message': result.message}


# ---------------------------------------------------------------------------
# Stage 4 helpers
# ---------------------------------------------------------------------------

def _bures_fidelity(rho, sigma):
    """Bures fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals, 0.0, None)
    sqrt_rho = evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T

    inner = sqrt_rho @ sigma @ sqrt_rho
    evals_inner = np.linalg.eigvalsh(inner)
    evals_inner = np.clip(evals_inner, 0.0, None)
    return float(np.sum(np.sqrt(evals_inner)) ** 2)


def _reconstruct_rho(L1, L2, d):
    """
    Reconstruct density matrix from LPDO tensors L1 and L2.

    L1 has shape (d, d_a1, chi), L2 has shape (chi, d, d_a2).
    rho[s1,s2, s1',s2'] = sum_{a1,a2} (sum_mu L1[s1,a1,mu]*L2[mu,s2,a2])
                                      * conj(sum_mu L1[s1',a1,mu]*L2[mu,s2',a2])
    """
    # Contract bond: M[s1, s2, a1, a2] = sum_mu L1[s1, a1, mu] * L2[mu, s2, a2]
    # L1: (d, d_a1, chi)  L2: (chi, d, d_a2)
    M = np.einsum('iam,mjb->ijab', L1, L2, optimize=True)  # (s1, s2, a1, a2)

    # Flatten to X[row, col] where row = (s1, s2), col = (a1, a2)
    d_a1, d_a2 = L1.shape[1], L2.shape[2]
    X_mat = M.reshape(d * d, d_a1 * d_a2)

    # rho = X @ X^dag
    rho = X_mat @ X_mat.conj().T
    return rho


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------

def truncate_and_validate(rho, A1, A2, d, fidelity_target=1e-9):
    """
    Stage 4 of LPDO construction: truncate bond dimension and validate fidelity.

    Given the LPDO tensors from Stage 3, perform a final SVD across the bond,
    truncate chi to the smallest value achieving Bures fidelity >= 1 - fidelity_target,
    and return the final LPDO tensors L1, L2.

    Parameters
    ----------
    rho : np.ndarray
        Original density matrix, shape (d^2, d^2).
    A1 : np.ndarray
        Left site tensor of shape (d, d_a1, chi) — (s1, a1, bond).
    A2 : np.ndarray
        Right site tensor of shape (chi, d, d_a2) — (bond, s2, a2).
    d : int
        Local Hilbert space dimension per site.
    fidelity_target : float
        Maximum allowed infidelity 1 - F.  Default 1e-9.

    Returns
    -------
    L1 : np.ndarray
        Truncated left tensor of shape (d, d_a1, chi_trunc) — (s1, a1, bond).
    L2 : np.ndarray
        Truncated right tensor of shape (chi_trunc, d, d_a2) — (bond, s2, a2).
    chi_trunc : int
        Truncated bond dimension.
    fidelity : float
        Bures fidelity F(rho, rho_reconstructed).
    info : dict
        Diagnostic information.
    """
    rho = np.asarray(rho, dtype=complex)
    A1 = np.asarray(A1, dtype=complex)
    A2 = np.asarray(A2, dtype=complex)

    d_a1 = A1.shape[1]
    d_a2 = A2.shape[2]
    chi_full = A1.shape[2]

    # Reshape to matrix M[(s1,a1), (s2,a2)] for SVD across the bond
    M = np.einsum('iam,mjb->ijab', A1, A2, optimize=True)  # (s1, s2, a1, a2)
    # Permute to (s1, a1, s2, a2) then group
    M = M.transpose(0, 2, 1, 3)  # (s1, a1, s2, a2)
    M_mat = M.reshape(d * d_a1, d * d_a2)

    U, S, Vh = np.linalg.svd(M_mat, full_matrices=False)

    # Find smallest chi that achieves the fidelity target
    threshold = 1.0 - fidelity_target
    chi_trunc = chi_full
    best_fidelity = None

    for chi_try in range(1, len(S) + 1):
        U_t = U[:, :chi_try]
        S_t = S[:chi_try]
        Vh_t = Vh[:chi_try, :]

        L1_try = U_t.reshape(d, d_a1, chi_try)
        SVh = np.diag(S_t) @ Vh_t
        L2_try = SVh.reshape(chi_try, d, d_a2)

        rho_recon = _reconstruct_rho(L1_try, L2_try, d)
        fid = _bures_fidelity(rho, rho_recon)

        if fid >= threshold:
            chi_trunc = chi_try
            best_fidelity = fid
            break
    else:
        # Full bond dimension used
        chi_trunc = len(S)
        L1_try = U.reshape(d, d_a1, chi_trunc)
        SVh = np.diag(S) @ Vh
        L2_try = SVh.reshape(chi_trunc, d, d_a2)
        rho_recon = _reconstruct_rho(L1_try, L2_try, d)
        best_fidelity = _bures_fidelity(rho, rho_recon)

    L1 = U[:, :chi_trunc].reshape(d, d_a1, chi_trunc)
    SVh = np.diag(S[:chi_trunc]) @ Vh[:chi_trunc, :]
    L2 = SVh.reshape(chi_trunc, d, d_a2)

    info = {
        'chi_full': chi_full,
        'chi_trunc': chi_trunc,
        'singular_values': S,
        'fidelity': best_fidelity,
        'fidelity_target': 1.0 - fidelity_target,
    }

    return L1, L2, chi_trunc, best_fidelity, info
