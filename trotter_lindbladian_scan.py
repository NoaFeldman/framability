"""
Generic two-qubit Trotter-step pipeline for Lindbladian models.

Given, for a translationally-invariant lattice model,

  * a one-qubit Hamiltonian term H1            (2x2, or None)
  * a two-qubit Hamiltonian term H2            (4x4, or None)
  * a list of one-qubit jump operators         (each 2x2, amplitude includes
                                                sqrt(rate), or empty)
  * a list of two-qubit jump operators         (each 4x4, amplitude includes
                                                sqrt(rate), or empty)
  * the physical dimension d in {1, 2, 3}
  * a Trotter time step dt

this module builds the two-qubit *bond* Trotter-step gate.  Each qubit sits on
2*d nearest-neighbour bonds, so every one-qubit term is split across the bonds
it participates in: the one-qubit Hamiltonian/jump coefficients are divided by
2*d while the two-qubit terms are brought in as is.

    H_bond = H2 + 1/(2d) * (H1 (x) I + I (x) H1)
    L_bond(rho) = -i[H_bond, rho]
                + 1/(2d) * sum_k [ D(L1_k (x) I)(rho) + D(I (x) L1_k)(rho) ]
                +          sum_m   D(L2_m)(rho)
    U_bond = expm(L_bond * dt)         (16x16 real Pauli-transfer matrix)

The sign-problem and framability quantities are properties of this bond gate
(d defaults to 2, a 2D lattice).  The steady-state quantities, by contrast, are
evaluated on the *full* LATTICE_LX x LATTICE_LY open-boundary lattice: the per-
site terms (H1, jumps1) are placed on every site and the per-bond terms
(H2, jumps2) on every nearest-neighbour bond, at full coupling strength (no
1/2d bond share).  The module computes (grouped by the figure line they share):

  a1  minimal sign problem (s maximised over translation-invariant local
      rotations; s = 1 means no sign problem)            -- bond gate
  a2  NESS LPDO bond entropy (single-site cut)           -- full lattice
  a4  NESS LPDO bond entropy maximised over bipartitions -- full lattice
  a3  Lindbladian rate (Liouvillian gap = slowest non-zero decay) -- full lattice
  b1  site-averaged Z- and X-magnetisation of the NESS   -- full lattice
  b2  von Neumann entropy of the NESS                    -- full lattice
  b3  half-half entanglement negativity of the NESS      -- full lattice
  c1  dyadic stabilizer-frame framability                -- bond gate
  c2  Pauli-frame framability                            -- bond gate
  d1  optimised framability, d_ext_single = 4            -- bond gate
  d2  optimised framability, d_ext_single = 6            -- bond gate
  d3  gamma_{CH_1} maximised over the single-qubit product frame -- bond gate

All heavy primitives are reused from dissipative_PT / framability /
gamma_ch1_sphere so this stays a thin model-and-orchestration layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

# Reuse the validated primitives from the dissipative-PT pipeline.
from dissipative_PT import (
    _I2, _SX, _SY, _SZ, _PAULI, S_MINUS, _build_lindbladian, _site_op, bonds_2d,
    pauli_to_rho, steady_state_and_decay, vn_entropy, negativity,
    site_magnetization, lpdo_bond_entropy, max_lpdo_bond_entropy,
    pauli_framability, optimise_framability,
    frame_from_params, params_from_frame, embed_frame_params,
    sign_problem_results, spectral_floor,
)
from framability import dyadic_stabilizer_framability
from gamma_ch1_sphere import gamma_CH1, frame_op_1q, pauli_coeffs

# Version stamp for cached results.  Bump when the set of stored quantities or
# the computation of any of them changes, so workers re-run stale points.
# 2.0: NESS quantities now come from the full LATTICE_LX x LATTICE_LY lattice
#      (not the single bond), the framability/sign gate defaults to dim=2, and
#      the maximal LPDO bond entropy (lpdo_max) is added.
# 2.1: model1's decay jump changed from S^- = |0><1| to |-><+|.
TLS_VERSION = '2.1'

DT_DEFAULT = 0.1
DIM_DEFAULT = 2          # framability/sign Trotter gate defaults to a 2D lattice

# Steady-state quantities (Liouvillian gap, magnetisations, entropies, LPDO,
# negativity) are evaluated on the full open-boundary lattice of this size.
LATTICE_LX = 2
LATTICE_LY = 2

# model4's dephasing jump sqrt(gamma)(ZI+IZ) and H=J(XX+YY+Delta ZZ) both commute
# with total S_z, a strong U(1) symmetry that leaves the NESS non-unique (the
# Liouvillian null space is 4-dimensional), so every steady-state quantity comes
# out NaN.  A small on-site decay sqrt(MODEL4_DECAY) S^- breaks that symmetry and
# restores a unique NESS; keep it small so the dephasing physics is barely shifted.
MODEL4_DECAY = 0.05


# ---------------------------------------------------------------------------
#  Bond Trotter gate
# ---------------------------------------------------------------------------
def build_bond_lindbladian(H1, H2, jumps1, jumps2, dim: int) -> np.ndarray:
    """16x16 real bond Lindbladian in the two-qubit Pauli basis.

    H1 : 2x2 one-qubit Hamiltonian term, or None.
    H2 : 4x4 two-qubit Hamiltonian term, or None.
    jumps1 : iterable of 2x2 one-qubit jump operators (sqrt(rate) included).
    jumps2 : iterable of 4x4 two-qubit jump operators (sqrt(rate) included).
    dim : physical lattice dimension (each qubit sits on 2*dim bonds).
    """
    f1 = 1.0 / (2.0 * dim)            # share of a one-qubit term per bond

    H = np.zeros((4, 4), dtype=complex)
    if H2 is not None:
        H = H + np.asarray(H2, dtype=complex)
    if H1 is not None:
        H1 = np.asarray(H1, dtype=complex)
        H = H + f1 * (np.kron(H1, _I2) + np.kron(_I2, H1))

    jumps = []
    for L1 in (jumps1 or []):
        L1 = np.asarray(L1, dtype=complex)
        # 1/(2d) scales the dissipator (rate), i.e. L -> L/sqrt(2d) on each site.
        for Lful in (np.kron(L1, _I2), np.kron(_I2, L1)):
            Ld = Lful.conj().T
            jumps.append((f1, Lful, Ld, Ld @ Lful))
    for L2 in (jumps2 or []):
        L2 = np.asarray(L2, dtype=complex)
        Ld = L2.conj().T
        jumps.append((1.0, L2, Ld, Ld @ L2))

    return _build_lindbladian(H, jumps, n=2)


def bond_trotter_gate(H1, H2, jumps1, jumps2, dim: int, dt: float) -> np.ndarray:
    """16x16 real Trotter gate expm(L_bond * dt)."""
    return expm(build_bond_lindbladian(H1, H2, jumps1, jumps2, dim) * dt).real


# ---------------------------------------------------------------------------
#  Full lattice Lindbladian (for the NESS / steady-state quantities)
# ---------------------------------------------------------------------------
def _embed_two_site(O4: np.ndarray, i: int, j: int, N: int) -> np.ndarray:
    """Embed a 4x4 two-qubit operator (qubit a (x) qubit b) onto sites (i, j) of
    an N-qubit register, via its 2-qubit Pauli decomposition."""
    O4 = np.asarray(O4, dtype=complex)
    d = 2 ** N
    out = np.zeros((d, d), dtype=complex)
    for sp in _PAULI:
        for sq in _PAULI:
            c = np.trace(np.kron(sp, sq).conj().T @ O4) / 4.0
            if abs(c) < 1e-14:
                continue
            out += c * (_site_op(sp, i, N) @ _site_op(sq, j, N))
    return out


def build_full_lindbladian_model(H1, H2, jumps1, jumps2,
                                 Lx: int = LATTICE_LX, Ly: int = LATTICE_LY) -> np.ndarray:
    """(4^N x 4^N) full Lindbladian for the Lx x Ly open-boundary lattice.

    The model's per-site terms (H1, jumps1) are placed on every site and its
    per-bond terms (H2, jumps2) on every nearest-neighbour bond, all at full
    coupling strength -- the 1/(2d) bond share in build_bond_lindbladian is a
    Trotter-gate convention, not part of the physical lattice generator.
    """
    N = Lx * Ly
    d = 2 ** N
    bonds = bonds_2d(Lx, Ly)

    H = np.zeros((d, d), dtype=complex)
    if H1 is not None:
        H1 = np.asarray(H1, dtype=complex)
        for s in range(N):
            H = H + _site_op(H1, s, N)
    if H2 is not None:
        H2 = np.asarray(H2, dtype=complex)
        for (i, j) in bonds:
            H = H + _embed_two_site(H2, i, j, N)

    jumps = []
    for L1 in (jumps1 or []):
        L1 = np.asarray(L1, dtype=complex)
        for s in range(N):
            Lf = _site_op(L1, s, N)
            Ld = Lf.conj().T
            jumps.append((1.0, Lf, Ld, Ld @ Lf))
    for L2 in (jumps2 or []):
        L2 = np.asarray(L2, dtype=complex)
        for (i, j) in bonds:
            Lf = _embed_two_site(L2, i, j, N)
            Ld = Lf.conj().T
            jumps.append((1.0, Lf, Ld, Ld @ Lf))

    return _build_lindbladian(H, jumps, n=N)


# ---------------------------------------------------------------------------
#  Steady-state observables
# ---------------------------------------------------------------------------
def _site_mean(rho: np.ndarray, op: np.ndarray, N: int) -> float:
    """Site-averaged single-qubit expectation (1/N) sum_i <op_i>."""
    return float(np.mean([np.trace(_site_op(op, i, N) @ rho).real for i in range(N)]))


# ---------------------------------------------------------------------------
#  gamma_{CH_1} over the single-qubit product frame
# ---------------------------------------------------------------------------
# Extreme points of the single-qubit operator-norm unit ball live at |b| = 1
# (pure traceless Pauli directions); b = 0 recovers the identity element.  These
# seven seeds cover the corners gamma_CH1(gate . .) is maximised over before a
# local Powell polish.
_CH1_SEEDS = [np.array(v, dtype=float) for v in
              ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
               (0, 0, 1), (0, 0, -1), (0, 0, 0))]


def gamma_ch1_framability(gate: np.ndarray, n_restarts: int = 15,
                          seed: int = 0) -> float:
    """max over product-frame elements rho1 (x) rho2 of gamma_{CH_1}(gate . rho).

    gamma_{CH_1} is the product-frame gauge (gamma_ch1_sphere.gamma_CH1); its
    maximum over the single-qubit product frame is the gate's product-frame
    framability.  rho_i = frame_op_1q(b_i) sweeps the operator-norm unit ball
    (|b_i| <= 1).  gamma_CH1(gate . .) is convex in the input, so the maximum
    sits at the boundary |b_i| = 1 — found from axis seeds plus boundary-seeded
    Powell restarts.
    """
    gate = np.asarray(gate, dtype=float)

    def val(b1: np.ndarray, b2: np.ndarray) -> float:
        op = np.kron(frame_op_1q(b1), frame_op_1q(b2))
        return gamma_CH1(gate @ pauli_coeffs(op))

    best = 0.0
    for b1 in _CH1_SEEDS:
        for b2 in _CH1_SEEDS:
            best = max(best, val(b1, b2))

    def neg(p: np.ndarray) -> float:
        b1, b2 = p[:3], p[3:]
        r1, r2 = np.linalg.norm(b1), np.linalg.norm(b2)
        if r1 > 1.0:
            b1 = b1 / r1
        if r2 > 1.0:
            b2 = b2 / r2
        return -val(b1, b2)

    rng = np.random.default_rng(seed)
    for _ in range(n_restarts):
        x0 = rng.standard_normal(6)
        x0[:3] /= max(np.linalg.norm(x0[:3]), 1e-9)   # start on the boundary
        x0[3:] /= max(np.linalg.norm(x0[3:]), 1e-9)
        res = minimize(neg, x0, method='Powell',
                       options={'maxiter': 200, 'maxfev': 400})
        best = max(best, float(-res.fun))
    return float(best)


# ---------------------------------------------------------------------------
#  Per-point computation
# ---------------------------------------------------------------------------
def compute_point(model: 'ModelSpec', p1: float, p2: float, *,
                  dim: int = DIM_DEFAULT, dt: float = DT_DEFAULT,
                  fra_restarts: int = 5, fra_maxfev_4: int = 1000,
                  fra_maxfev_6: int = 500, sign_restarts: int = 10,
                  ch1_restarts: int = 15, seed: int = 0,
                  verbose: bool = False) -> dict:
    """All scan quantities for one (p1, p2) point of `model`.

    The framability / sign-problem quantities use the two-qubit bond Trotter
    gate (dim-dependent).  The steady-state quantities (Liouvillian gap,
    magnetisations, entropies, LPDO, negativity) use the full LATTICE_LX x
    LATTICE_LY lattice Lindbladian.
    """
    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    gate = bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)

    out: dict = dict(p1=p1, p2=p2, dim=dim, dt=dt)

    # ── group a: gate / Lindbladian ──────────────────────────────────────────
    out['sign_init'], out['sign_opt'] = sign_problem_results(gate, sign_restarts, seed)
    out['floor'] = spectral_floor(gate)

    # ── full LATTICE_LX x LATTICE_LY lattice NESS (groups a/b) ────────────────
    N = LATTICE_LX * LATTICE_LY
    L_full = build_full_lindbladian_model(H1, H2, jumps1, jumps2)
    c_ss, decay = steady_state_and_decay(L_full, N=N)
    out['lind_rate'] = decay                              # a3 Liouvillian gap

    if c_ss is not None:
        rho = pauli_to_rho(c_ss, N=N)
        try:
            out['lpdo'] = lpdo_bond_entropy(rho, d_A=2)   # a2 NESS LPDO entropy
        except Exception:
            out['lpdo'] = float('nan')
        try:
            out['lpdo_max'] = max_lpdo_bond_entropy(rho, N)  # a4 max over cuts
        except Exception:
            out['lpdo_max'] = float('nan')
        out['ss_vn'] = vn_entropy(rho)                    # b2
        out['mag_z'] = _site_mean(rho, _SZ, N)            # b1
        out['mag_x'] = _site_mean(rho, _SX, N)            # b1
        out['neg'] = negativity(rho, d_A=2 ** (N // 2))   # b3 half-half cut
    else:
        out['lpdo'] = out['lpdo_max'] = out['ss_vn'] = out['mag_z'] = \
            out['mag_x'] = out['neg'] = float('nan')

    if verbose:
        print(f'  sign_opt={out["sign_opt"]:.4f}  rate={out["lind_rate"]:.4e}  '
              f'vn={out["ss_vn"]:.4f}', flush=True)

    # ── group c: fixed-frame framabilities ───────────────────────────────────
    out['stab_fra'] = dyadic_stabilizer_framability(gate)   # c1
    out['pauli_fra'] = pauli_framability(gate)              # c2

    # ── group d: optimised framabilities + gamma_CH1 ─────────────────────────
    out['opt_fra_4'], x4 = optimise_framability(
        gate, 4, fra_restarts, fra_maxfev_4, seed, return_x=True)
    out['opt_fra_6'], x6 = optimise_framability(
        gate, 6, fra_restarts, fra_maxfev_6, seed + 1, return_x=True)
    out['opt_S_4'] = frame_from_params(x4, 4)
    out['opt_S_6'] = frame_from_params(x6, 6)
    out['gamma_ch1'] = gamma_ch1_framability(gate, ch1_restarts, seed)   # d3

    if verbose:
        print(f'  stab={out["stab_fra"]:.4f}  pauli={out["pauli_fra"]:.4f}  '
              f'd4={out["opt_fra_4"]:.4f}  d6={out["opt_fra_6"]:.4f}  '
              f'ch1={out["gamma_ch1"]:.4f}', flush=True)

    return out


# ---------------------------------------------------------------------------
#  Model registry
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """One lattice model and its two-parameter scan grid.

    build(p1, p2) -> (H1, H2, jumps1, jumps2) with p1 the x-axis parameter and
    p2 the y-axis parameter.  p1_vals / p2_vals are the (already rounded) grid
    centres; the worker maps point_id = ix * N_Y + iy.
    """
    name: str
    title: str
    p1_name: str
    p1_label: str
    p1_vals: np.ndarray
    p2_name: str
    p2_label: str
    p2_vals: np.ndarray
    build: Callable[[float, float], tuple]
    dim: int = DIM_DEFAULT
    dt: float = DT_DEFAULT

    @property
    def N_X(self) -> int:
        return len(self.p1_vals)

    @property
    def N_Y(self) -> int:
        return len(self.p2_vals)

    @property
    def N_TOTAL(self) -> int:
        return self.N_X * self.N_Y


def _arange(lo: float, hi: float, step: float) -> np.ndarray:
    """Inclusive rounded grid lo, lo+step, ..., hi (avoids float drift)."""
    n = int(round((hi - lo) / step)) + 1
    return np.array([round(lo + step * i, 10) for i in range(n)])


# Single-qubit S^- = |0><1| (decay toward the ground state |0>), as in
# dissipative_PT (S_MINUS).  |-><+| is the X-basis analogue: lowering toward
# |-> = (|0> - |1>)/sqrt(2).  Two-qubit Pauli helpers for the builders:
_KET_MINUS = np.array([1.0, -1.0]) / np.sqrt(2.0)
_KET_PLUS  = np.array([1.0,  1.0]) / np.sqrt(2.0)
MINUS_PLUS = np.outer(_KET_MINUS, _KET_PLUS).astype(complex)   # |-><+|


def _ZZ():
    return np.kron(_SZ, _SZ)


def _build_model1(gamma: float, gamma_p: float):
    # H: J ZZ ; jumps: sqrt(gamma) |-><+|, sqrt(gamma') Z (one-qubit)
    J = 1.0
    H2 = J * _ZZ()
    jumps1 = [np.sqrt(gamma) * MINUS_PLUS, np.sqrt(gamma_p) * _SZ]
    return None, H2, jumps1, []


def _build_model2(h: float, gamma: float):
    # H1: h X ; H2: J ZZ ; jumps: sqrt(gamma) S^- (one-qubit)
    J = 1.0
    H1 = h * _SX
    H2 = J * _ZZ()
    jumps1 = [np.sqrt(gamma) * S_MINUS]
    return H1, H2, jumps1, []


def _build_model3(J_y: float, gamma: float):
    # H2: J_x XX + J_y YY + J_z ZZ ; jumps: sqrt(gamma) S^- (one-qubit)
    J_x, J_z = 0.9, 1.0
    H2 = (J_x * np.kron(_SX, _SX)
          + J_y * np.kron(_SY, _SY)
          + J_z * np.kron(_SZ, _SZ))
    jumps1 = [np.sqrt(gamma) * S_MINUS]
    return None, H2, jumps1, []


def _build_model4(Delta: float, gamma: float):
    # H2: J(XX + YY + Delta ZZ) ; jumps: sqrt(gamma)(Z(x)I + I(x)Z) (two-qubit)
    # plus a small on-site decay sqrt(MODEL4_DECAY) S^- (one-qubit) to break the
    # strong S_z symmetry and make the NESS unique (see MODEL4_DECAY).
    J = 1.0
    H2 = (J * np.kron(_SX, _SX)
          + J * np.kron(_SY, _SY)
          + J * Delta * np.kron(_SZ, _SZ))
    L2 = np.sqrt(gamma) * (np.kron(_SZ, _I2) + np.kron(_I2, _SZ))
    jumps1 = [np.sqrt(MODEL4_DECAY) * S_MINUS]
    return None, H2, jumps1, [L2]


MODELS: dict[str, ModelSpec] = {
    'model1': ModelSpec(
        name='model1',
        title=r"$H=J\,ZZ$,  jumps $\sqrt{\gamma}\,|{-}\rangle\langle{+}|,\ "
              r"\sqrt{\gamma'}\,Z$  (J=1)",
        p1_name='gamma',   p1_label=r'$\gamma$',   p1_vals=_arange(0, 8, 0.2),
        p2_name='gamma_p', p2_label=r"$\gamma'$",  p2_vals=_arange(0, 8, 0.2),
        build=_build_model1),
    'model2': ModelSpec(
        name='model2',
        title=r'$H=h\,X + J\,ZZ$,  jump $\sqrt{\gamma}\,S^-$  (J=1)',
        p1_name='h',     p1_label=r'$h$',       p1_vals=_arange(-2, 2, 0.2),
        p2_name='gamma', p2_label=r'$\gamma$',  p2_vals=_arange(0, 10, 0.2),
        build=_build_model2),
    'model3': ModelSpec(
        name='model3',
        title=r'$H=J_x XX + J_y YY + J_z ZZ$,  jump $\sqrt{\gamma}\,S^-$  '
              r'($J_z=1,\ J_x=0.9$)',
        p1_name='J_y',   p1_label=r'$J_y$',     p1_vals=_arange(0, 4, 0.2),
        p2_name='gamma', p2_label=r'$\gamma$',  p2_vals=_arange(0, 2, 0.2),
        build=_build_model3),
    'model4': ModelSpec(
        name='model4',
        title=r'$H=J(XX+YY+\Delta\,ZZ)$,  jumps $\sqrt{\gamma}(ZI+IZ),\ '
              r'\sqrt{\epsilon}\,S^-$  ($J=1,\ \epsilon=%.2g$)' % MODEL4_DECAY,
        p1_name='Delta', p1_label=r'$\Delta$',  p1_vals=_arange(-3, 3, 0.2),
        p2_name='gamma', p2_label=r'$\gamma$',  p2_vals=_arange(0, 3, 0.2),
        build=_build_model4),
}


# ---------------------------------------------------------------------------
#  Stored quantities and figure layout (shared by collect / refine_collect)
# ---------------------------------------------------------------------------
# (key, label, group letter, is_framability)
QUANTITIES = [
    ('sign_opt',  'Sign problem (min)',         'a', False),
    ('lpdo',      'NESS LPDO bond entropy',      'a', False),
    ('lpdo_max',  'NESS LPDO bond entropy (max)','a', False),
    ('lind_rate', 'Liouvillian gap',             'a', False),
    ('mag_z',     r'$\langle Z\rangle$',         'b', False),
    ('mag_x',     r'$\langle X\rangle$',        'b', False),
    ('ss_vn',     'NESS VN entropy',            'b', False),
    ('neg',       'NESS negativity',            'b', False),
    ('stab_fra',  'Stabilizer framability',     'c', True),
    ('pauli_fra', 'Pauli framability',          'c', True),
    ('opt_fra_4', 'Opt framability (d=4)',      'd', True),
    ('opt_fra_6', 'Opt framability (d=6)',      'd', True),
    ('gamma_ch1', 'max Janek',                  'd', True),
]

# opt_fra_4 / opt_fra_6 are the only quantities refined (neighbour-seeded).
FRA_REFINE_KEYS = {'opt_fra_4': 'opt_S_4', 'opt_fra_6': 'opt_S_6'}


if __name__ == '__main__':
    import time
    m = MODELS['model1']
    p1, p2 = 2.0, 1.0
    print(f'[self-test] {m.name}: {m.p1_name}={p1} {m.p2_name}={p2}  '
          f'grid {m.N_X}x{m.N_Y}={m.N_TOTAL}', flush=True)
    t0 = time.perf_counter()
    res = compute_point(m, p1, p2, fra_restarts=3, fra_maxfev_4=300,
                        fra_maxfev_6=150, sign_restarts=5, ch1_restarts=8,
                        verbose=True)
    print(f'{"-"*50}')
    for k, _, _, _ in QUANTITIES:
        print(f'  {k:12s} = {res[k]:.6f}')
    print(f'  elapsed = {time.perf_counter() - t0:.1f}s')
