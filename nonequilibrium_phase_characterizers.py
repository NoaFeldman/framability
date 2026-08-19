"""
Spectral-oscillation characterizers for a Lindbladian small enough to
diagonalize exactly.

Vec convention (fixes which of l_k / r_k carries the initial state)
---------------------------------------------------------------------
Only column-stacking is implemented (`vec_convention='column'`, the only value
currently accepted -- kept as an explicit keyword rather than hardwired so the
convention is documented and callers can't silently assume the wrong one):

    vec(rho) = rho.reshape(-1, order='F')          # vec(rho)[i + j*d] = rho[i, j]

Built from a Hamiltonian H and jump operators c_i in that convention,

    L = -i (I kron H - H^T kron I)
        + sum_i [ conj(c_i) kron c_i
                  - 1/2 (I kron c_i^dag c_i + (c_i^dag c_i)^T kron I) ]

acts on vec(rho) as  d/dt vec(rho) = L @ vec(rho)  (this is exactly the sparse
construction in n_qubit_lindbladian.py / six_qubit_lindbladian.py, generalized
to dense small-d matrices here).  Consequently:

  * a RIGHT eigenvector r_k of L, reshaped (d, d, order='F'), is the decay MODE
    itself: rho(t) = rho_ss + sum_k a_k(0) e^{lambda_k t} r_k for some
    (generally state-dependent) amplitudes a_k(t=0).
  * a LEFT eigenvector l_k defines the amplitude functional: after
    biorthonormalization (Tr[l_j^dag r_k] = delta_jk), the mode amplitude
    excited by a given initial state rho0 is a_k(0) = Tr[l_k^dag rho0] -- i.e.
    l_k, not r_k, is what "carries" rho0 in step 5's c_k = Tr[O r_k] Tr[l_k^dag
    rho0].  (Swap the convention to row-stacking and this assignment flips.)

Dependency-light: numpy only is required; scipy.linalg.eig(..., left=True,
right=True) is used when available for a robust simultaneous left/right
diagonalization (index-matched eigenvalues/eigenvectors by construction) and
falls back to a numpy-only eig(L) / eig(L^dagger) pair matched by nearest
conjugate eigenvalue (best-effort; a large post-hoc biorthogonality residual --
flagged via `tol_biorth` -- is the tell that this matching broke down, most
likely from spectral degeneracy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

try:
    from scipy.linalg import eig as _scipy_eig
    _HAVE_SCIPY = True
except ImportError:                       # pragma: no cover - scipy optional
    _HAVE_SCIPY = False


@dataclass
class SpectralOscillationResult:
    """See spectral_oscillation()'s docstring for field definitions."""
    lam: np.ndarray             # nonzero-mode eigenvalues, sorted by increasing Gamma
    Gamma: np.ndarray
    omega: np.ndarray
    Q: np.ndarray
    N: np.ndarray
    gap: float                  # Delta = min_k Gamma_k over nonzero modes
    omega1: float                # dominant (slowest-decaying nonzero) mode's |omega|
    Q1: float
    N1: float                   # = omega1 / (2*pi*gap)
    steady_idx: np.ndarray      # indices, in the ORIGINAL full spectrum, of steady modes
    c_k: dict | None = None       # {obs index: array over nonzero modes, sorted as above}
    omega_bar: dict | None = None # {obs index: amplitude-weighted mean |omega|}
    Q_bar: dict | None = None     # {obs index: amplitude-weighted mean Q}
    warnings: list = field(default_factory=list)


def _biorthonormalize(lam: np.ndarray, vl: np.ndarray, vr: np.ndarray, *,
                      tol_biorth: float, warnings: list):
    """In-place-equivalent biorthonormalization Tr[l_j^dag r_k] = delta_jk.

    Handles near-degeneracies (eigenvalues within tol_biorth of each other) by
    biorthogonalizing within each cluster jointly -- dividing a single pair by
    a near-zero overlap is what fails at an exceptional point / degeneracy;
    inverting the (generically well-conditioned) cluster overlap matrix does
    not."""
    n = len(lam)
    order = np.argsort(lam.real)           # cluster by nearby Re(lambda) first...
    # group indices whose eigenvalues are mutually within tol_biorth (chain rule:
    # if |lam_i - lam_{i+1}| <= tol, they're in the same cluster)
    clusters = []
    cur = [order[0]]
    for idx in order[1:]:
        if abs(lam[idx] - lam[cur[-1]]) <= tol_biorth:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
    clusters.append(cur)

    vl_new = vl.copy()
    max_resid = 0.0
    for cluster in clusters:
        idx = np.array(cluster)
        R = vr[:, idx]                     # (d, m)
        Lft = vl[:, idx]                    # (d, m)
        M = Lft.conj().T @ R               # (m, m) overlap
        try:
            Minv = np.linalg.inv(M)
        except np.linalg.LinAlgError:
            warnings.append(
                f'biorthonormalize: singular overlap matrix in cluster near '
                f'lambda={lam[idx[0]]:.4g} (size {len(idx)}); left eigenvectors '
                f'for this cluster are unreliable.')
            continue
        Lft_new = Lft @ Minv.conj().T      # Lft_new^dag @ R = I_m
        vl_new[:, idx] = Lft_new
        resid = float(np.max(np.abs(Lft_new.conj().T @ R - np.eye(len(idx)))))
        max_resid = max(max_resid, resid)

    if max_resid > tol_biorth:
        warnings.append(
            f'biorthonormalize: residual {max_resid:.3g} exceeds tol_biorth='
            f'{tol_biorth:.3g}; possible exceptional point / defective Liouvillian.')
    return vl_new, max_resid


def _diagonalize(L: np.ndarray, *, tol_biorth: float, warnings: list):
    """Return (lam, vl, vr) with vl/vr biorthonormalized and index-matched to
    lam (vr[:, k] the right eigenvector, vl[:, k] the left eigenvector, both
    for eigenvalue lam[k])."""
    d2 = L.shape[0]
    if _HAVE_SCIPY:
        lam, vl, vr = _scipy_eig(L, left=True, right=True)
    else:
        lam, vr = np.linalg.eig(L)
        lam_dag, w = np.linalg.eig(L.conj().T)
        # left eigenvector of L for eigenvalue lam[k] is w[:, j] for the j with
        # lam_dag[j] ~ conj(lam[k]) (see module docstring derivation); match by
        # nearest remaining eigenvalue (best-effort without scipy).
        used = np.zeros(d2, dtype=bool)
        vl = np.empty_like(vr)
        max_match_err = 0.0
        for k in range(d2):
            target = np.conj(lam[k])
            dists = np.abs(lam_dag - target)
            dists[used] = np.inf
            j = int(np.argmin(dists))
            used[j] = True
            max_match_err = max(max_match_err, float(dists[j]))
            vl[:, k] = w[:, j]
        if max_match_err > tol_biorth:
            warnings.append(
                f'numpy-only left/right eigenvalue matching residual '
                f'{max_match_err:.3g} exceeds tol_biorth={tol_biorth:.3g} '
                f'(install scipy for a robust simultaneous left/right eig); '
                f'possible exceptional point / defective Liouvillian.')

    vl, _ = _biorthonormalize(lam, vl, vr, tol_biorth=tol_biorth, warnings=warnings)
    return lam, vl, vr


def _build_liouvillian(H: np.ndarray, c_ops: Sequence[np.ndarray]) -> np.ndarray:
    d = H.shape[0]
    Id = np.eye(d, dtype=complex)
    L = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    for c in c_ops:
        c = np.asarray(c, dtype=complex)
        cdc = c.conj().T @ c
        L = L + np.kron(c.conj(), c) - 0.5 * (np.kron(Id, cdc) + np.kron(cdc.T, Id))
    return L


def spectral_oscillation(L: np.ndarray | None = None, *,
                         H: np.ndarray | None = None,
                         c_ops: Sequence[np.ndarray] | None = None,
                         rho0: np.ndarray | None = None,
                         observables: Sequence[np.ndarray] | None = None,
                         vec_convention: str = 'column',
                         tol_steady: float = 1e-8,
                         tol_biorth: float = 1e-6,
                         tol_instability: float = 1e-6,
                         gamma_floor: float = 1e-12,
                         weight_rel_cutoff: float = 1e-3,
                         ) -> SpectralOscillationResult:
    """Spectral-oscillation measures of a Liouvillian small enough to
    diagonalize exactly (dense (d^2, d^2)).

    Parameters
    ----------
    L : dense Liouvillian superoperator, shape (d^2, d^2), OR
    H, c_ops : Hamiltonian (d, d) and list of jump operators (each (d, d)),
        from which L is built in the vec convention documented at module
        level.  Exactly one of `L` or `H` (with `c_ops`, possibly empty for
        purely unitary evolution) must be given.
    rho0 : optional initial density matrix (d, d).
    observables : optional list of observables, each (d, d).
    vec_convention : only 'column' is implemented (see module docstring).
    tol_steady : steady-state modes are those with |lambda_k| <=
        tol_steady * ||L||_2 (spectral norm).
    tol_biorth : biorthogonality-residual / near-degeneracy-clustering
        tolerance (see _biorthonormalize) and left/right eigenvalue-matching
        tolerance in the numpy-only fallback path.
    tol_instability : a nonzero eigenvalue with Re(lambda) > tol_instability
        triggers a numerical-instability warning.
    gamma_floor : decay rates below this are clamped before dividing (Q_k,
        N_k), guarding underflow; the affected modes are still returned (with
        very large |Q_k|/|N_k|) but a warning is recorded.
    weight_rel_cutoff : in the amplitude-weighted summaries (step 5), modes
        with weight w_k < weight_rel_cutoff * max_j(w_j) are dropped so an
        unexcited mode does not pollute omega_bar/Q_bar.

    Returns
    -------
    SpectralOscillationResult
    """
    if vec_convention != 'column':
        raise ValueError(f"vec_convention={vec_convention!r} not implemented; "
                         f"only 'column' is supported (see module docstring).")

    warnings: list[str] = []

    if L is not None:
        if H is not None or c_ops is not None:
            raise ValueError('pass either L, or H (with c_ops), not both')
        L = np.asarray(L, dtype=complex)
    else:
        if H is None:
            raise ValueError('must pass either L or H (with c_ops)')
        L = _build_liouvillian(np.asarray(H, dtype=complex), c_ops or [])

    d2 = L.shape[0]
    if L.shape != (d2, d2):
        raise ValueError(f'L must be square, got shape {L.shape}')
    d = int(round(np.sqrt(d2)))
    if d * d != d2:
        raise ValueError(f'L has shape {L.shape}; d^2 x d^2 with integer d expected')

    # ---- step 1: diagonalize + biorthonormalize -----------------------------
    lam, vl, vr = _diagonalize(L, tol_biorth=tol_biorth, warnings=warnings)

    # ---- step 2: identify steady-state mode(s) -------------------------------
    L_norm = float(np.linalg.norm(L, ord=2))
    steady_mask = np.abs(lam) <= tol_steady * max(L_norm, 1e-300)
    steady_idx = np.where(steady_mask)[0]
    if steady_idx.size > 1:
        warnings.append(
            f'{steady_idx.size} steady-state modes found (degenerate steady-state '
            f'manifold): lambda={lam[steady_idx]}')
    elif steady_idx.size == 0:
        warnings.append('no steady-state mode found within tol_steady; '
                        'the Liouvillian may not conserve trace/positivity here.')

    nz_idx = np.where(~steady_mask)[0]
    unstable = nz_idx[lam[nz_idx].real > tol_instability]
    if unstable.size > 0:
        warnings.append(
            f'{unstable.size} nonzero eigenvalue(s) with Re(lambda) > '
            f'tol_instability={tol_instability:.3g} (numerical instability?): '
            f'lambda={lam[unstable]}')

    # ---- step 3: per-mode Gamma, omega, Q, N (nonzero modes only) -----------
    lam_nz = lam[nz_idx]
    Gamma = -lam_nz.real
    omega = lam_nz.imag
    underflow = Gamma < gamma_floor
    if np.any(underflow):
        warnings.append(
            f'{int(np.sum(underflow))} nonzero mode(s) have Gamma_k < '
            f'gamma_floor={gamma_floor:.3g}; Q_k/N_k clamped to avoid division '
            f'by (near-)zero.')
    Gamma_safe = np.where(underflow, gamma_floor, Gamma)
    Q = omega / (2.0 * Gamma_safe)
    N = omega / (2.0 * np.pi * Gamma_safe)

    order = np.argsort(Gamma)              # ascending Gamma (slowest decay first)
    lam_nz, Gamma, omega, Q, N = (a[order] for a in (lam_nz, Gamma, omega, Q, N))
    nz_idx_sorted = nz_idx[order]           # for c_k bookkeeping below

    # ---- step 4: gap + dominant (slowest-decaying nonzero) mode -------------
    if Gamma.size == 0:
        raise RuntimeError('no nonzero modes found; cannot compute gap')
    gap = float(Gamma[0])
    dom = 0
    omega1 = abs(float(omega[dom]))
    # if the dominant mode is one of a complex-conjugate pair, its |omega| is
    # already shared with its partner (Im(lambda) and Im(conj(lambda)) have
    # equal magnitude), so no extra handling is needed beyond abs().
    Q1 = omega1 / (2.0 * gap) if gap > 0 else float('inf')
    N1 = omega1 / (2.0 * np.pi * gap) if gap > 0 else float('inf')

    # ---- step 5: amplitude-weighted summaries (optional) ---------------------
    c_k = omega_bar = Q_bar = None
    if rho0 is not None and observables:
        rho0 = np.asarray(rho0, dtype=complex)
        r_modes = vr[:, nz_idx_sorted].reshape(d, d, len(nz_idx_sorted), order='F')
        l_modes = vl[:, nz_idx_sorted].reshape(d, d, len(nz_idx_sorted), order='F')
        # Tr[l_k^dag rho0] for every mode (shared across all observables)
        overlap_rho0 = np.einsum('ijk,ij->k', l_modes.conj(), rho0)

        c_k, omega_bar, Q_bar = {}, {}, {}
        for oi, O in enumerate(observables):
            O = np.asarray(O, dtype=complex)
            tr_O_r = np.einsum('ij,jik->k', O, r_modes)   # Tr[O r_k] per mode
            c = tr_O_r * overlap_rho0
            c_k[oi] = c

            w = np.abs(c) / Gamma_safe[order]
            if w.size == 0 or np.all(w == 0):
                omega_bar[oi] = float('nan')
                Q_bar[oi] = float('nan')
                continue
            keep = w >= weight_rel_cutoff * np.max(w)
            wsum = float(np.sum(w[keep]))
            if wsum <= 0:
                omega_bar[oi] = float('nan')
                Q_bar[oi] = float('nan')
                continue
            omega_bar[oi] = float(np.sum(w[keep] * np.abs(omega[keep])) / wsum)
            Q_bar[oi] = float(np.sum(w[keep] * Q[keep]) / wsum)

    return SpectralOscillationResult(
        lam=lam_nz, Gamma=Gamma, omega=omega, Q=Q, N=N,
        gap=gap, omega1=omega1, Q1=Q1, N1=N1,
        steady_idx=steady_idx,
        c_k=c_k, omega_bar=omega_bar, Q_bar=Q_bar,
        warnings=warnings,
    )
