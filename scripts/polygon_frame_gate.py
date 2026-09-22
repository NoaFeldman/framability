r"""
Polygon-frame construction of small entangling 2-qubit CPTP gates, and the
canonical (unique) extraction of their coherent / dephasing / relaxation terms.

Frame (Heisenberg picture, D = S (x) S):
    S_n = { |0><0|, |1><1| } U { d_k = cos(pi k/n) X + sin(pi k/n) Y : k < n }
so conv{+-d_k} is a regular 2n-gon in the XY plane and the two projectors are
the control axis.  Under H = J Z Z the column d_k (x) |b><b| is rotated INSIDE
the polygon (conditional rotation by +-2J t), which costs 2J tan(pi/2n) per
unit time -- the gauge of a rotated vertex of a regular 2n-gon.  Z dephasing
at rate gamma damps the polygon at 2 gamma and leaves the projectors alone, so

    mu*(S_n; J, gamma) = max(0, 2 J tan(pi/2n) - 2 gamma).

Canonical form: L(rho) = -i[H, rho] + sum_{a,b>=1} c_ab (P_a rho P_b - {P_b P_a, rho}/2)
with traceless jumps; H (traceless) and the Kossakowski matrix c are unique.
c = c_R + i c_I : c_R (real symmetric, PSD) = Hermitian jumps = dephasing,
c_I (real antisymmetric) = relaxation / pumping content.

    python scripts/polygon_frame_gate.py      # self-test + threshold table
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm, logm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dissipative_PT import _build_lindbladian, _pauli_tensor, _kron_power   # noqa: E402
from framability_rate_frames import frame_rate, pauli_rate                  # noqa: E402

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1., -1.]).astype(complex)


# ---------------------------------------------------------------------------
#  Frame
# ---------------------------------------------------------------------------
def projector_polygon_frame(n: int) -> np.ndarray:
    """4 x (n+2) Pauli-coefficient matrix (rows I, X, Y, Z) of S_n."""
    cols = [[0.5, 0, 0, 0.5], [0.5, 0, 0, -0.5]]
    for k in range(n):
        th = np.pi * k / n
        cols.append([0, np.cos(th), np.sin(th), 0])
    return np.array(cols, float).T


def polygon_rate_prediction(n: int, J: float, gamma: float) -> float:
    return max(0.0, 2 * J * np.tan(np.pi / (2 * n)) - 2 * gamma)


# ---------------------------------------------------------------------------
#  Generator:  H = J ZZ (+ hz (Z1+Z2) + hx (X1+X2)),  Z dephasing gamma,
#  optional amplitude damping |1><0| at rate g_relax on each qubit.
# ---------------------------------------------------------------------------
def zz_dephasing_lindbladian(J, gamma, hz=0.0, hx=0.0, g_relax=0.0):
    H = J * np.kron(SZ, SZ) + hz * (np.kron(SZ, I2) + np.kron(I2, SZ)) \
        + hx * (np.kron(SX, I2) + np.kron(I2, SX))
    jumps = []
    for A, rate in ((SZ, gamma), (np.array([[0, 0], [1, 0]], complex), g_relax)):
        for Lf in (np.kron(A, I2), np.kron(I2, A)):
            Ld = Lf.conj().T
            jumps.append((rate, Lf, Ld, Ld @ Lf))
    return _build_lindbladian(H, jumps, n=2)


# ---------------------------------------------------------------------------
#  Canonical form extraction from the Pauli-basis superoperator L
#  (L[a,b] = Tr(P_a L(P_b)) / d, i.e. L(P_b) = sum_a L[a,b] P_a)
# ---------------------------------------------------------------------------
def canonical_form(L: np.ndarray, n: int = 2):
    """Return H (d x d Hermitian, traceless), c (Kossakowski, (d^2-1)^2),
    c_R, c_I with c = c_R + i c_I.  Unique for the given L."""
    P = _pauli_tensor(n)
    d = 2 ** n
    L = np.asarray(L, complex)
    # Choi matrix of L:  C = (1/d) sum_ab L_ab  P_a (x) P_b^T
    C = sum(L[a, b] * np.kron(P[a], P[b].T) for a in range(d * d) for b in range(d * d)) / d
    omega = np.eye(d).reshape(-1)                       # |Omega> = sum_i |ii>
    V = np.array([np.kron(P[a], np.eye(d)) @ omega for a in range(d * d)])   # |P_a>>
    G = V.conj() @ C @ V.T / d ** 2                     # G_ab = <<P_a|C|P_b>> / d^2
    H = -sum(G[a, 0].imag * P[a] for a in range(1, d * d))
    c = G[1:, 1:]
    return H, c, c.real, c.imag


def gate_to_generator(gate: np.ndarray, dt: float) -> np.ndarray:
    """Principal logarithm; exact when gate = expm(dt L) with dt small."""
    return logm(gate).real / dt


# ---------------------------------------------------------------------------
#  Entangling witness: first-order partial-transpose eigenvalue on |++>
# ---------------------------------------------------------------------------
def negativity_plus_plus(L, dt):
    P = _pauli_tensor(2)
    plus = np.array([1, 1]) / np.sqrt(2)
    rho0 = np.kron(np.outer(plus, plus), np.outer(plus, plus))
    r = np.array([np.trace(P[a] @ rho0).real for a in range(16)]) / 4
    r_t = expm(dt * np.asarray(L, float)) @ r
    rho = sum(r_t[a] * P[a] for a in range(16))
    pt = rho.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)
    ev = np.linalg.eigvalsh(pt)
    return float(-ev[ev < 0].sum())


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True, linewidth=160)
    J = 1.0
    # --- canonical extraction round trip on a known generator ----------------
    L = zz_dephasing_lindbladian(J, 0.3, hz=0.1, g_relax=0.2)
    H, c, cR, cI = canonical_form(L)
    P = _pauli_tensor(2); labs = [a + b for a in 'IXYZ' for b in 'IXYZ']
    hcoef = {labs[a]: np.trace(P[a] @ H).real / 4 for a in range(16)}
    print('H coefficients (nonzero):', {k: round(v, 4) for k, v in hcoef.items() if abs(v) > 1e-9})
    print('c eigenvalues (>=0 iff CP):', np.round(np.linalg.eigvalsh(c), 4))
    print('||c_I||:', round(np.abs(cI).max(), 4), '  (0 without the relaxation jump)')
    dt = 1e-3
    L_back = gate_to_generator(expm(dt * L), dt)
    print('log(expm(dt L))/dt round trip error:', np.abs(L_back - L).max())
    # --- polygon-frame thresholds ---------------------------------------------
    print('\nmu*(S_n) for H = J ZZ + Z dephasing gamma  (LP vs 2J tan(pi/2n) - 2gamma)')
    print('gamma/J  Pauli  ' + '  '.join(f'n={n}(d={n+2})' for n in (2, 3, 4, 6, 8)))
    for g in (0.15, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2):
        L = zz_dephasing_lindbladian(J, g)
        row = [pauli_rate(L)]
        for n in (2, 3, 4, 6, 8):
            D = _kron_power(projector_polygon_frame(n), 2)
            mu = frame_rate(D, L, picture='heisenberg')
            pred = polygon_rate_prediction(n, J, g)
            row.append(mu if abs(mu - pred) < 1e-6 else float('nan'))
        print(f'{g:6.2f}   ' + '  '.join(f'{v:9.4f}' for v in row) +
              f'   negativity(|++>, dt=1e-2) = {negativity_plus_plus(L, 1e-2):.2e}')
    print('(nan = LP disagrees with the tan(pi/2n) prediction)')
