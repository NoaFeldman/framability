"""
Checks of the open-Dicke dimer patch of trotter_lindbladian_scan._make_model_dicke
(dicke_lindbladian.md):

 1. plumbing: build_bond_lindbladian(*build(...), dim=0.5) (Pauli basis, with
    SignedJump) equals the dimer superoperator built directly from H / jumps;
 2. pair split: for N = 2..5, sum_{i<j} dimer_ij / (N-1) acting on random
    matrices equals the collective atom-only Liouvillian
        -i[omega0 J_z + (g^2/4)(2 Im A J_x^2 + Re B J_z + Im B {J_x,J_y}), .]
        + kappa D_note[L] + gamma_phi sum D_note[Z]
    built from J operators (D_note = 2 x the repo's D);
 3. cavity elimination: the slow Liouvillian eigenvalues of the full
    cavity + N atoms model (truncated Fock space) against the atom-only model
    at small g: the O(g^2) shifts away from the bare atomic eigenvalues must
    agree (tests the Lamb-shift / XX / XY coefficients of the note, incl. signs).

    python scripts/dicke_patch_check.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (_make_model_dicke, dicke_coefficients,    # noqa: E402
                                      build_bond_lindbladian, SignedJump,
                                      MODEL_DICKE_DIM, _I2, _SX, _SY, _SZ)
from dissipative_PT import _build_lindbladian                                   # noqa: E402

RNG = np.random.default_rng(1)


def site(op, i, N):
    out = np.array([[1.0 + 0j]])
    for k in range(N):
        out = np.kron(out, op if k == i else _I2)
    return out


def pair(op4, i, j, N):
    """Embed a 4x4 operator on sites (i, j) via its Pauli decomposition."""
    P = [_I2, _SX, _SY, _SZ]
    out = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for a in range(4):
        for b in range(4):
            c = np.trace(np.kron(P[a], P[b]).conj().T @ op4) / 4
            if abs(c) > 1e-15:
                out = out + c * site(P[a], i, N) @ site(P[b], j, N)
    return out


def D_std(L, rho):
    Ld = L.conj().T
    return L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)


def apply_terms(H, terms, rho):
    """terms: list of (weight, L)."""
    out = -1j * (H @ rho - rho @ H)
    for w, L in terms:
        out = out + w * D_std(L, rho)
    return out


def patch_terms(build_out):
    H1, H2, j1, j2 = build_out
    H = np.asarray(H2) + np.kron(H1, _I2) + np.kron(_I2, H1)     # dim 1/2: share 1
    terms = []
    for L in j1:
        terms += [(1.0, np.kron(L, _I2)), (1.0, np.kron(_I2, L))]
    for L in j2:
        terms.append((L.weight, L.L) if isinstance(L, SignedJump) else (1.0, np.asarray(L)))
    return H, terms


def check_plumbing(N, lam_ratio, gphi):
    out = _make_model_dicke(N)(lam_ratio, gphi)
    M = build_bond_lindbladian(*out, MODEL_DICKE_DIM)
    H, terms = patch_terms(out)
    Mref = _build_lindbladian(H, [(w, L, L.conj().T, L.conj().T @ L) for w, L in terms], n=2)
    return float(np.max(np.abs(M - Mref)))


def collective(N, lam, gphi, co):
    Jx = sum(site(_SX, i, N) for i in range(N)) / 2
    Jy = sum(site(_SY, i, N) for i in range(N)) / 2
    Jz = sum(site(_SZ, i, N) for i in range(N)) / 2
    Jm, Jp = Jx - 1j * Jy, Jx + 1j * Jy
    g2_4 = lam ** 2 / N
    A, B = co['A'], co['B']
    H = co['omega0'] * Jz + g2_4 * (2 * A.imag * Jx @ Jx + B.real * Jz
                                    + B.imag * (Jx @ Jy + Jy @ Jx))
    L = np.sqrt(g2_4) * (co['c_m'] * Jm + co['c_p'] * Jp)
    terms = [(2 * co['kappa'], L)] + [(2 * gphi, site(_SZ, i, N)) for i in range(N)]
    return H, terms


def check_split(N, lam_ratio, gphi):
    co = dicke_coefficients()
    lam = lam_ratio * co['lam_c']
    Hc, tc = collective(N, lam, gphi, co)
    Hp, tp = patch_terms(_make_model_dicke(N)(lam_ratio, gphi))
    d = 2 ** N
    err = 0.0
    for _ in range(3):
        X = RNG.normal(size=(d, d)) + 1j * RNG.normal(size=(d, d))
        ref = apply_terms(Hc, tc, X)
        tot = 0
        for i, j in combinations(range(N), 2):
            Hij = pair(Hp, i, j, N)
            tij = [(w, pair(L, i, j, N)) for w, L in tp]
            tot = tot + apply_terms(Hij, tij, X) / (N - 1)
        err = max(err, float(np.max(np.abs(tot - ref)) / np.max(np.abs(ref))))
    return err


def liouvillian(H, terms):
    """Column-stacking vec superoperator."""
    d = H.shape[0]
    I = np.eye(d)
    S = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for w, L in terms:
        Ld = L.conj().T
        LdL = Ld @ L
        S = S + w * (np.kron(L.conj(), L) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I))
    return S


def check_elimination(N=2, g=0.04, omega=1.0, omega0=0.3, kappa=1.0, nmax=4):
    """Full cavity model vs atom-only model: O(g^2) eigenvalue shifts."""
    co = dicke_coefficients(omega, omega0, kappa)
    lam = g * np.sqrt(N) / 2
    da = 2 ** N
    Jx = sum(site(_SX, i, N) for i in range(N)) / 2
    Jz = sum(site(_SZ, i, N) for i in range(N)) / 2
    a = np.diag(np.sqrt(np.arange(1, nmax + 1)), 1).astype(complex)
    Ic, Ia = np.eye(nmax + 1), np.eye(da)
    A_ = np.kron(a, Ia)
    Hf = (omega * np.kron(a.conj().T @ a, Ia) + omega0 * np.kron(Ic, Jz)
          + g * (A_ + A_.conj().T) @ np.kron(Ic, Jx))
    ev_full = np.linalg.eigvals(liouvillian(Hf, [(2 * kappa, A_)]))
    # atom-only model, and the bare atoms (g = 0)
    Ha, ta = collective(N, lam, 0.0, co)
    ev_at = np.linalg.eigvals(liouvillian(Ha, ta))
    ev_0 = np.linalg.eigvals(liouvillian(omega0 * Jz, []))
    # slow full-model eigenvalues: those near the bare atomic ones
    slow = ev_full[np.argsort(-ev_full.real)][:da * da]
    worst = 0.0
    for e in ev_at:
        e_full = slow[np.argmin(np.abs(slow - e))]
        e_bare = ev_0[np.argmin(np.abs(ev_0 - e))]
        shift_at, shift_full = e - e_bare, e_full - e_bare
        if abs(shift_at) > 1e-12:
            worst = max(worst, abs(shift_full - shift_at) / abs(shift_at))
    # sign-flipped Hamiltonian correction as a control: must fail
    Hbad = 2 * omega0 * Jz - Ha
    ev_bad = np.linalg.eigvals(liouvillian(Hbad, ta))
    worst_bad = 0.0
    for e in ev_bad:
        e_full = slow[np.argmin(np.abs(slow - e))]
        e_bare = ev_0[np.argmin(np.abs(ev_0 - e))]
        if abs(e - e_bare) > 1e-12:
            worst_bad = max(worst_bad, abs((e_full - e_bare) - (e - e_bare)) / abs(e - e_bare))
    return worst, worst_bad


def main():
    ok = True
    print('1. plumbing (Pauli-basis bond Lindbladian vs direct), max |diff|:')
    for N in (2, 3, 10, 1000):
        for lr, gp in ((0.0, 0.0), (1.0, 0.0), (1.7, 0.4)):
            e = check_plumbing(N, lr, gp)
            ok &= e < 1e-12
            print(f'   N={N:<5d} lam/lam_c={lr:<4} gamma_phi={gp:<4} {e:.2e}')
    print('2. pair split vs collective Liouvillian, relative max error:')
    for N in (2, 3, 4, 5):
        for lr, gp in ((1.0, 0.0), (1.7, 0.4)):
            e = check_split(N, lr, gp)
            ok &= e < 1e-12
            print(f'   N={N} lam/lam_c={lr:<4} gamma_phi={gp:<4} {e:.2e}')
    print('3. cavity elimination, worst relative mismatch of O(g^2) shifts:')
    for g in (0.04, 0.02):
        w, wb = check_elimination(g=g)
        print(f'   N=2 g={g}: note H {w:.3e}   (control, flipped H correction: {wb:.3e})')
    print('ALL EXACT CHECKS PASSED' if ok else 'EXACT CHECK FAILED')


if __name__ == '__main__':
    main()
