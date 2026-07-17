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
  * a Trotter time step dt (chosen per point by choose_dt from the model
    parameters unless the model pins a fixed value)

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
  a4  max LPDO bond entropy along the |0>^N -> NESS path  -- full lattice
  a3  Lindbladian rate (Liouvillian gap = slowest non-zero decay) -- full lattice
  b1  site-averaged Z- and X-magnetisation of the NESS   -- full lattice
  b2  von Neumann entropy of the NESS                    -- full lattice
  b3  half-half entanglement negativity of the NESS      -- full lattice
  c1  stabilizer-3 framability (Schrodinger, 3-qubit stabilizer frame) -- bond gate
  c2  Pauli-frame framability                            -- bond gate
  d1  optimised Heisenberg framability, d_ext_single = 4 -- bond gate
  d2  optimised Heisenberg framability, d_ext_single = 6 -- bond gate
  d3  gamma_{CH_1} maximised over the single-qubit product frame -- bond gate
  e   product-state framability, chi in {10, 20, 40}     -- bond gate
  f   optimised Schrodinger state framability, d_ext_single in {4, 6, 8} -- bond gate
  g   dephasing-augmented optimised framability (Heisenberg & Schrodinger,
      d_ext_single = 4): only for models with no dephasing jump, a small
      pure-dephasing channel sqrt(10 dt) Z is added to the gate       -- bond gate

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
    site_magnetization, lpdo_bond_entropy,
    pauli_framability, optimise_framability,
    frame_from_params, params_from_frame, embed_frame_params,
    sign_problem_results, spectral_floor,
)
from analysis import compute_max_bond_dim, _initial_iz_vector, _initial_ix_vector
from framability import stabilizer_3_framability, product_state_framability
from optimize_framability import minimize_schroedinger_framability
from gamma_ch1_sphere import gamma_CH1, frame_op_1q, pauli_coeffs

# Version stamp for cached results.  Bump when the set of stored quantities or
# the computation of any of them changes, so workers re-run stale points.
# 2.0: NESS quantities now come from the full LATTICE_LX x LATTICE_LY lattice
#      (not the single bond), the framability/sign gate defaults to dim=2, and
#      the maximal LPDO bond entropy (lpdo_max) is added.
# 2.1: model1's decay jump changed from S^- = |0><1| to |-><+|.
# 2.2: lpdo_max is now the maximum bond entropy *along the |0>^N -> NESS
#      relaxation path* (analysis.compute_max_bond_dim), not the steady state.
# 2.3: the lpdo_max relaxation path start state is now per-model
#      (ModelSpec.lpdo_init): model2 and model4 relax from |+>^N instead of
#      |0>^N (all other quantities are unchanged -- 2.2 files are upgraded in
#      place by scripts/trotter_scan_patch_worker.py without re-running the
#      framability optimisation).  model3's gamma grid extends to 10 and
#      model5 (the dissipative-PT model, imported from results_dpt) is added.
# 2.4: the bond Trotter gate's dt is now chosen per point (choose_dt):
#      dt = DT_BASE / max(||H||_1, max_k gamma_k) -- DT_BASE times the fastest
#      timescale, with ||H||_1 the Pauli 1-norm of the Hamiltonian terms and
#      gamma_k the jump rates.  model5 keeps its fixed dt=0.05 so the
#      dissipative-PT import stays exact.
# 2.5: the fixed-frame framability (group c) is now the 3-qubit *stabilizer-3*
#      framability (framability.stabilizer_3_framability) instead of the dyadic
#      stabilizer framability.  New quantities: product-state framability at
#      chi in {10, 20, 40} (group e), optimised Schrodinger state framability at
#      d_ext_single in {4, 6, 8} (group f), and -- for models whose jumps do not
#      already include a pure-dephasing operator -- the optimised Heisenberg and
#      Schrodinger framabilities (d_ext_single=4) of the gate with a small added
#      dephasing channel sqrt(DEPHASING_DT_FACTOR * dt) Z (group g).
# 3.0: the whole model set is redefined (models 1-4 now share H = J ZZ + h X,
#      J=1, no two-qubit jumps):
#        model1  jump sqrt(gamma) S^-,          scan h in [-2,2], gamma in [0,10]
#        model2  jump sqrt(gamma) |-><+|,       scan h in [-2,2], gamma in [0,10]
#        model3  h=0,  jumps sqrt(gamma)|-><+|, sqrt(gamma')Z,  scan gamma, gamma' in [0,10]
#        model4  h=1.5,jumps sqrt(gamma)|-><+|, sqrt(gamma')Z,  scan gamma, gamma' in [0,10]
#        model5  = the old (2.x) model3 Heisenberg XX+YY+ZZ + sqrt(gamma)S^-,
#                 gamma extended to [0,20] (no longer the dissipative-PT import;
#                 dim=2 and adaptive choose_dt like the rest).
#      Everything is re-run from scratch: the default output directory moved to
#      results_trotter_v3 so the earlier results_trotter data is kept untouched
#      but ignored by this code, and the version bump forces a recompute of any
#      point that happens to already exist.
TLS_VERSION = '3.0'

DT_DEFAULT = 0.1         # fallback only (zero generator / legacy files)
DT_BASE = 0.01           # choose_dt: dt = DT_BASE / max(||H||_1, {gamma_k})
DIM_DEFAULT = 2          # framability/sign Trotter gate defaults to a 2D lattice

# Steady-state quantities (Liouvillian gap, magnetisations, entropies, LPDO,
# negativity) are evaluated on the full open-boundary lattice of this size.
LATTICE_LX = 2
LATTICE_LY = 2

# lpdo_max: maximum LPDO bond entropy along the relaxation path from |0>^N to the
# NESS under the full-lattice Lindbladian (analysis.compute_max_bond_dim).
LPDO_PATH_DT = 0.002          # integration step (dt-converged for these models)
LPDO_PATH_FIDELITY = 0.9      # stop once Bures fidelity with the NESS reaches this

# group e: product-state framability uses a random two-qubit *product* frame of
# chi^2 columns (chi single-qubit states per site).  The frame is rebuilt from
# PROD_FRAME_SEED at every point so the identical set of random states is used
# across the whole scan and the value varies smoothly over the grid.
PROD_CHIS = (10, 20, 40)
PROD_FRAME_SEED = 12345

# group f: optimised Schrodinger state-frame framability, single-qubit frame
# sizes d_ext_single (the full product frame has d_ext_single**2 columns).
SCH_DEXTS = (4, 6, 8)

# group g: for a model with no dephasing jump, a pure-dephasing channel
# sqrt(gamma_deph) Z with gamma_deph = DEPHASING_DT_FACTOR * dt is added to the
# bond gate before its optimised (Heisenberg & Schrodinger) framability is taken.
DEPHASING_DT_FACTOR = 10.0


# ---------------------------------------------------------------------------
#  Bond Trotter gate
# ---------------------------------------------------------------------------
def _pauli_1norm(H) -> float:
    """Sum of |Pauli coefficients| of a 2x2 or 4x4 operator, identity component
    excluded (it only contributes a global phase to the dynamics)."""
    H = np.asarray(H, dtype=complex)
    n = H.shape[0]
    basis = _PAULI if n == 2 else [np.kron(p, q) for p in _PAULI for q in _PAULI]
    return float(sum(abs(np.trace(P.conj().T @ H)) / n for P in basis[1:]))


def choose_dt(H1, H2, jumps1, jumps2, base: float = DT_BASE) -> float:
    """Per-point Trotter step matched to the model's fastest rate:

        dt = base / max(||H||_1, max_k gamma_k)

    i.e. `base` times the shortest timescale of the generator.  ||H||_1 is the
    Pauli 1-norm of the Hamiltonian terms (H1 and H2 combined, identity part
    excluded) and gamma_k = ||L_k||^2 (largest singular value squared) is the
    rate carried by each jump operator (amplitudes include sqrt(rate)).  The
    raw model terms are used, before the 1/(2d) bond share, so dt depends on
    the physical parameters only.  Falls back to DT_DEFAULT when every scale
    vanishes (zero generator: the gate is the identity at any dt).
    """
    hnorm = sum(_pauli_1norm(H) for H in (H1, H2) if H is not None)
    rates = []
    for L in list(jumps1 or []) + list(jumps2 or []):
        L = np.asarray(L, dtype=complex)
        if np.linalg.norm(L) < 1e-12:          # zero operator (e.g. gamma=0): skip
            continue
        rates.append(float(np.linalg.norm(L, 2) ** 2))
    scale = max([hnorm] + rates)
    if scale <= 1e-12:
        return DT_DEFAULT
    return base / scale


def _has_dephasing(jumps1, jumps2) -> bool:
    """True if any jump operator is a pure-dephasing operator, i.e. Hermitian and
    diagonal in the computational (Z) basis with a non-constant diagonal (e.g. Z,
    ZI+IZ, ZZ).  Off-diagonal or non-Hermitian jumps such as S^- = |0><1| are not
    dephasing.  Zero operators (padding) are ignored."""
    for L in list(jumps1 or []) + list(jumps2 or []):
        L = np.asarray(L, dtype=complex)
        if np.linalg.norm(L) < 1e-12:
            continue
        if np.max(np.abs(L - L.conj().T)) > 1e-9:                 # not Hermitian
            continue
        if np.max(np.abs(L - np.diag(np.diag(L)))) > 1e-9:        # not Z-diagonal
            continue
        diag = np.diag(L).real
        if np.max(np.abs(diag - diag.mean())) > 1e-9:             # non-constant
            return True
    return False


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
def lpdo_init_vector(model: 'ModelSpec', N: int) -> np.ndarray:
    """Pauli-coefficient vector of the model's lpdo_max path start state."""
    if model.lpdo_init == 'plus':
        return _initial_ix_vector(N)
    return _initial_iz_vector(N)


def compute_point(model: 'ModelSpec', p1: float, p2: float, *,
                  dim: int = DIM_DEFAULT, dt: float | None = None,
                  fra_restarts: int = 5, fra_maxfev_4: int = 1000,
                  fra_maxfev_6: int = 500, sign_restarts: int = 10,
                  ch1_restarts: int = 15, sch_restarts: int = 5,
                  sch_maxfev: int = 800, seed: int = 0,
                  verbose: bool = False) -> dict:
    """All scan quantities for one (p1, p2) point of `model`.

    The framability / sign-problem quantities use the two-qubit bond Trotter
    gate (dim-dependent).  dt=None resolves to the model's own dt if it pins
    one (model5), else to the per-point adaptive step choose_dt.  The resolved
    value is returned in the 'dt' entry.  The steady-state quantities
    (Liouvillian gap, magnetisations, entropies, LPDO, negativity) use the
    full LATTICE_LX x LATTICE_LY lattice Lindbladian.
    """
    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    if dt is None:
        dt = model.dt if model.dt is not None else choose_dt(H1, H2, jumps1, jumps2)
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
            # a4: max LPDO bond entropy along the relaxation path to the NESS,
            # starting from the model's lpdo_init state (|0>^N or |+>^N)
            _, out['lpdo_max'] = compute_max_bond_dim(
                L_full, rho, None, N=N, init=lpdo_init_vector(model, N),
                dt=LPDO_PATH_DT, fidelity_threshold=LPDO_PATH_FIDELITY)
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
    out['stab_fra'] = stabilizer_3_framability(gate)        # c1 stabilizer-3
    out['pauli_fra'] = pauli_framability(gate)              # c2

    # ── group d: optimised (Heisenberg) framabilities + gamma_CH1 ────────────
    out['opt_fra_4'], x4 = optimise_framability(
        gate, 4, fra_restarts, fra_maxfev_4, seed, return_x=True)
    out['opt_fra_6'], x6 = optimise_framability(
        gate, 6, fra_restarts, fra_maxfev_6, seed + 1, return_x=True)
    out['opt_S_4'] = frame_from_params(x4, 4)
    out['opt_S_6'] = frame_from_params(x6, 6)
    out['gamma_ch1'] = gamma_ch1_framability(gate, ch1_restarts, seed)   # d3

    # ── group e: product-state framability (fixed random product frame) ──────
    # Reseed the global RNG (haar_measure draws off it) so the same product
    # states are used at every scan point, giving a smooth grid.
    np.random.seed(PROD_FRAME_SEED)
    for chi in PROD_CHIS:
        out[f'prod_fra_{chi}'] = product_state_framability(chi, gate)

    # ── group f: optimised Schrodinger state framability ─────────────────────
    for de in SCH_DEXTS:
        _, out[f'sch_fra_{de}'] = minimize_schroedinger_framability(
            gate, de, n_restarts=sch_restarts, maxfev=sch_maxfev,
            seed=seed, verbose=False)

    # ── group g: dephasing-augmented optimised framability (d_ext_single=4) ──
    # Only for models with no dephasing jump: add a small pure-dephasing channel
    # sqrt(10 dt) Z on each site and report the optimised framability of the
    # resulting gate in the Heisenberg and Schrodinger pictures.
    if _has_dephasing(jumps1, jumps2):
        out['deph_heis_fra_4'] = out['deph_schro_fra_4'] = float('nan')
    else:
        jumps1_deph = list(jumps1 or []) + [np.sqrt(DEPHASING_DT_FACTOR * dt) * _SZ]
        gate_deph = bond_trotter_gate(H1, H2, jumps1_deph, jumps2, dim, dt)
        out['deph_heis_fra_4'] = optimise_framability(
            gate_deph, 4, fra_restarts, fra_maxfev_4, seed)
        _, out['deph_schro_fra_4'] = minimize_schroedinger_framability(
            gate_deph, 4, n_restarts=sch_restarts, maxfev=sch_maxfev,
            seed=seed, verbose=False)

    if verbose:
        print(f'  stab3={out["stab_fra"]:.4f}  pauli={out["pauli_fra"]:.4f}  '
              f'd4={out["opt_fra_4"]:.4f}  d6={out["opt_fra_6"]:.4f}  '
              f'ch1={out["gamma_ch1"]:.4f}', flush=True)
        print(f'  prod10={out["prod_fra_10"]:.4f}  prod40={out["prod_fra_40"]:.4f}  '
              f'sch4={out["sch_fra_4"]:.4f}  sch8={out["sch_fra_8"]:.4f}  '
              f'deph_H={out["deph_heis_fra_4"]:.4f}  '
              f'deph_S={out["deph_schro_fra_4"]:.4f}', flush=True)

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
    # Trotter step of the bond gate.  None (the default, used by every current
    # model) -> chosen per point by choose_dt from the point's own parameters;
    # a float would pin dt for the whole scan.
    dt: float | None = None
    # start state of the lpdo_max relaxation path: 'zero' -> |0>^N, 'plus' -> |+>^N
    lpdo_init: str = 'zero'

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


# Models 1-4 share the Hamiltonian H = J ZZ + h X (J = 1) with only one-qubit
# jump operators (no two-qubit jumps).  h is a scan axis for models 1-2 and a
# fixed constant for models 3-4.
MODEL4_H = 1.5


def _build_model1(h: float, gamma: float):
    # H = J ZZ + h X ; jump sqrt(gamma) S^- (one-qubit)
    J = 1.0
    H1 = h * _SX
    H2 = J * _ZZ()
    jumps1 = [np.sqrt(gamma) * S_MINUS]
    return H1, H2, jumps1, []


def _build_model2(h: float, gamma: float):
    # H = J ZZ + h X ; jump sqrt(gamma) |-><+| (one-qubit)
    J = 1.0
    H1 = h * _SX
    H2 = J * _ZZ()
    jumps1 = [np.sqrt(gamma) * MINUS_PLUS]
    return H1, H2, jumps1, []


def _build_model3(gamma: float, gamma_p: float):
    # H = J ZZ (h = 0) ; jumps sqrt(gamma) |-><+|, sqrt(gamma') Z (one-qubit)
    J = 1.0
    H2 = J * _ZZ()
    jumps1 = [np.sqrt(gamma) * MINUS_PLUS, np.sqrt(gamma_p) * _SZ]
    return None, H2, jumps1, []


def _build_model4(gamma: float, gamma_p: float):
    # H = J ZZ + h X (h = MODEL4_H = 1.5) ; jumps sqrt(gamma) |-><+|,
    # sqrt(gamma') Z (one-qubit)
    J = 1.0
    H1 = MODEL4_H * _SX
    H2 = J * _ZZ()
    jumps1 = [np.sqrt(gamma) * MINUS_PLUS, np.sqrt(gamma_p) * _SZ]
    return H1, H2, jumps1, []


def _build_model5(J_y: float, gamma: float):
    # (was the 2.x model3.)  Heisenberg bond H2 = J_x XX + J_y YY + J_z ZZ with
    # J_x = 0.9, J_z = 1 ; jump sqrt(gamma) S^- (one-qubit).  Same model and grid
    # as the old model3 except gamma now runs to 20.
    J_x, J_z = 0.9, 1.0
    H2 = (J_x * np.kron(_SX, _SX)
          + J_y * np.kron(_SY, _SY)
          + J_z * np.kron(_SZ, _SZ))
    jumps1 = [np.sqrt(gamma) * S_MINUS]
    return None, H2, jumps1, []


# model6: decay rate is fixed (gamma = 1); the scan axes are the two Bloch-ball
# rotation angles of the jump operator.
MODEL6_GAMMA = 1.0


def rotated_s_minus(rho: float, sigma: float) -> np.ndarray:
    """S^-(rho, sigma) = U S^- U^dag with U = Rz(sigma) Ry(rho): the lowering
    operator rotated around the Bloch ball, i.e. decay toward the pure state at
    Bloch direction n = (sin rho cos sigma, sin rho sin sigma, cos rho).
    (rho, sigma) = (0, 0) recovers S^- = |0><1|."""
    U = expm(-1j * sigma / 2 * _SZ) @ expm(-1j * rho / 2 * _SY)
    return U @ S_MINUS @ U.conj().T


def _build_model6(rho: float, sigma: float):
    # H = J ZZ ; jump sqrt(gamma) S^-(rho, sigma) (one-qubit), J = 1, gamma = 1
    J = 1.0
    H2 = J * _ZZ()
    jumps1 = [np.sqrt(MODEL6_GAMMA) * rotated_s_minus(rho, sigma)]
    return None, H2, jumps1, []


# All models use dim=2 and the per-point adaptive Trotter step (dt=None ->
# choose_dt).  Models 1-4 share H = J ZZ + h X; models 1-2 relax the lpdo_max
# path from |+>^N (the transverse-field / X-basis dynamics sit far from a |0>^N
# start), as do the |-><+| models 3-4.  model5 keeps the |0>^N start of the old
# Heisenberg model3.
MODELS: dict[str, ModelSpec] = {
    'model1': ModelSpec(
        name='model1',
        title=r'$H=J\,ZZ + h\,X$,  jump $\sqrt{\gamma}\,S^-$  (J=1)',
        p1_name='h',     p1_label=r'$h$',       p1_vals=_arange(-2, 2, 0.2),
        p2_name='gamma', p2_label=r'$\gamma$',  p2_vals=_arange(0, 10, 0.2),
        build=_build_model1, lpdo_init='plus'),
    'model2': ModelSpec(
        name='model2',
        title=r'$H=J\,ZZ + h\,X$,  jump $\sqrt{\gamma}\,|{-}\rangle\langle{+}|$'
              r'  (J=1)',
        p1_name='h',     p1_label=r'$h$',       p1_vals=_arange(-2, 2, 0.2),
        p2_name='gamma', p2_label=r'$\gamma$',  p2_vals=_arange(0, 10, 0.2),
        build=_build_model2, lpdo_init='plus'),
    'model3': ModelSpec(
        name='model3',
        title=r"$H=J\,ZZ$,  jumps $\sqrt{\gamma}\,|{-}\rangle\langle{+}|,\ "
              r"\sqrt{\gamma'}\,Z$  (J=1, h=0)",
        p1_name='gamma',   p1_label=r'$\gamma$',   p1_vals=_arange(0, 10, 0.2),
        p2_name='gamma_p', p2_label=r"$\gamma'$",  p2_vals=_arange(0, 10, 0.2),
        build=_build_model3, lpdo_init='plus'),
    'model4': ModelSpec(
        name='model4',
        title=r"$H=J\,ZZ + 1.5\,X$,  jumps $\sqrt{\gamma}\,|{-}\rangle\langle{+}|,\ "
              r"\sqrt{\gamma'}\,Z$  (J=1, h=1.5)",
        p1_name='gamma',   p1_label=r'$\gamma$',   p1_vals=_arange(0, 10, 0.2),
        p2_name='gamma_p', p2_label=r"$\gamma'$",  p2_vals=_arange(0, 10, 0.2),
        build=_build_model4, lpdo_init='plus'),
    # model5 is the old (2.x) model3 Heisenberg model with gamma extended to 20.
    'model5': ModelSpec(
        name='model5',
        title=r'$H=J_x XX + J_y YY + J_z ZZ$,  jump $\sqrt{\gamma}\,S^-$  '
              r'($J_z=1,\ J_x=0.9$)',
        p1_name='J_y',   p1_label=r'$J_y$',     p1_vals=_arange(0, 4, 0.2),
        p2_name='gamma', p2_label=r'$\gamma$',  p2_vals=_arange(0, 20, 0.2),
        build=_build_model5),
    # model6 (added 2026-07, no version bump: existing model data is unchanged):
    # H = J ZZ with the S^- jump rotated around the Bloch ball; the scan axes
    # are the rotation angles rho, sigma in [0, pi/2] step pi/100 (51 x 51).
    'model6': ModelSpec(
        name='model6',
        title=r'$H=J\,ZZ$,  jump $\sqrt{\gamma}\,S^-(\rho,\sigma)$  '
              r'(J=1, $\gamma$=1)',
        p1_name='rho',   p1_label=r'$\rho$',
        p1_vals=_arange(0, np.pi / 2, np.pi * 1e-2),
        p2_name='sigma', p2_label=r'$\sigma$',
        p2_vals=_arange(0, np.pi / 2, np.pi * 1e-2),
        build=_build_model6),
}


# ---------------------------------------------------------------------------
#  Stored quantities and figure layout (shared by collect / refine_collect)
# ---------------------------------------------------------------------------
# (key, label, group letter, is_framability)
QUANTITIES = [
    ('sign_opt',  'Sign problem (min)',         'a', False),
    ('lpdo',      'NESS LPDO bond entropy',      'a', False),
    ('lpdo_max',  'max LPDO bond entropy (path)','a', False),
    ('lind_rate', 'Liouvillian gap',             'a', False),
    ('mag_z',     r'$\langle Z\rangle$',         'b', False),
    ('mag_x',     r'$\langle X\rangle$',        'b', False),
    ('ss_vn',     'NESS VN entropy',            'b', False),
    ('neg',       'NESS negativity',            'b', False),
    ('stab_fra',  'Stabilizer-3 framability',   'c', True),
    ('pauli_fra', 'Pauli framability',          'c', True),
    ('opt_fra_4', 'Opt framability H (d=4)',    'd', True),
    ('opt_fra_6', 'Opt framability H (d=6)',    'd', True),
    ('gamma_ch1', 'max Janek',                  'd', True),
    ('prod_fra_10', r'Product-state fra ($\chi$=10)', 'e', True),
    ('prod_fra_20', r'Product-state fra ($\chi$=20)', 'e', True),
    ('prod_fra_40', r'Product-state fra ($\chi$=40)', 'e', True),
    ('sch_fra_4', 'Opt Schrod fra (d=4)',       'f', True),
    ('sch_fra_6', 'Opt Schrod fra (d=6)',       'f', True),
    ('sch_fra_8', 'Opt Schrod fra (d=8)',       'f', True),
    ('deph_heis_fra_4',  'Deph opt fra H (d=4)', 'g', True),
    ('deph_schro_fra_4', 'Deph opt fra S (d=4)', 'g', True),
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
                        sch_restarts=2, sch_maxfev=200, verbose=True)
    print(f'{"-"*50}')
    for k, _, _, _ in QUANTITIES:
        print(f'  {k:12s} = {res[k]:.6f}')
    print(f'  elapsed = {time.perf_counter() - t0:.1f}s')
