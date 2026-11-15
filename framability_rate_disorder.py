r"""
framability_rate_disorder.py -- framability rate of the atom-only open Dicke
model with INHOMOGENEOUS atomic frequencies, averaged over the pair patches,
with one shared single-qubit frame or one frame per frequency class.

Model (dicke_lindbladian.md, generalised to unequal omega0)
-----------------------------------------------------------
N atoms in M species of equal abundance, species s at

    omega0_s = omega0 (1 + w (2 s + 1 - M) / M),      s = 0 .. M-1,

the midpoints of M equal-probability bins of the uniform distribution on
[omega0 (1 - w), omega0 (1 + w)] (w = 0: the uniform model, M collapses to 1).
The cavity is eliminated exactly as in the note, a_ad = -i (g/2) sum_j m_j,
every atom keeping its own response in the cavity propagator:

    m_j    = c_-^(j) s^-_j + c_+^(j) s^+_j,   c_pm^(j) = 1 / (kappa + i (omega pm omega0_j))
    H      = sum_j omega0_j / 2 Z_j + (lam^2/N) sum_{i,j} Im(X_i m_j),   Im(O) = (O - O^dag)/(2i)
    L      = (lam / sqrt N) sum_j m_j,   kappa D_note[L]     (D_note = 2 D_repo)

(the note's H_eff = (g/2)(J_x a_ad + a_ad^dag J_x) with g^2/4 = lam^2/N; the
i = j terms are the Lamb shift (lam^2/N) Re B_j Z_j / 2, the i != j terms the
pair coupling).  At w = 0 this is trotter_lindbladian_scan._make_model_dicke
term by term; check_uniform() asserts it to 1e-12 (the worker runs it first).

Pair split (as for the uniform model): L_full = (1/(N-1)) sum_{i<j} B_ij,

    B_ij = l_i + l_j + (N-1) l_ij           (one-site terms at full weight)
    l_ij : (lam^2/N) [ (Im A_i + Im A_j)/2 X_i X_j + Im B_j/2 X_i Y_j + Im B_i/2 Y_i X_j ]
           and 2 kappa lam^2/N D[m_i + m_j]
    l_i  : (omega0_i/2 + (lam^2/N) Re B_i / 2) Z_i,  -2 kappa (N-2) lam^2/N D[m_i] (non-CP),
           on-site decay 2 gamma_down D[s^-], dephasing 2 gamma_phi D[Z]

Average rate
------------
A product frame of the whole system with one frame S_i per atom gives

    mu(L_full) <= (1/(N-1)) sum_{i<j} mu(B_ij; S_i (x) S_j) = (N/2) F,
    F = pair average of mu(B_ij; S_i (x) S_j),

and every pair rate is >= 0 (trace preservation: the identity column has
rate 0), so F = 0 iff every pair is at 0.  F is directly comparable with the
uniform panels (w = 0: F = the bond rate mu*).  Pair weights for equal
abundances n_s = N/M: n_s (n_s - 1)/2 (same species), n_s n_t (s < t),
normalised by N (N - 1)/2.

Two frame assignments:
    shared   one S on every atom (the ansatz of the uniform model);
    species  one S_s per species.  Each atom uses its species' frame in ALL its
             N - 1 pairs, so it is still ONE product frame of the whole system;
             the pair frame S_s (x) S_t is A (x) B for s != t.  This is allowed
             here only because the atoms are physically different (user,
             2026-11-15); the uniform pipelines keep S (x) S.

Optimiser
---------
Seeds (stored uniform-model frames of the same grid point, the single-qubit
seed library, the padded smaller-d optimum) are evaluated exactly; then a
trust-region BUNDLE descent on F = sum_p w_p max_j mu_pj: every pair adds the
linearisations of its near-binding columns (dual witness + envelope gradient
through D = S_a (x) S_b) and the epigraph LP

    min sum_p w_p t_p   s.t.  mu_pj + <g_pj, dS> <= t_p,  t_p >= 0,  |dS|_inf <= radius

picks the step, accepted on the TRUE F.  The species search starts from the
shared optimum, so F_species <= F_shared by construction.

Mean-field transition of the mixture (N -> inf)
-----------------------------------------------
Holstein-Primakoff around the all-down state with one collective boson per
species, B_s = n_s^{-1/2} sum_{j in s} b_j (the other modes do not couple):
the linear equations of <B_s>, <B_s^dag> (mixture_mf_matrix); mixture_lam_c
is the smallest lam with an unstable mode.  M = 1 reproduces
scripts/dicke_n_compare_plot.mf_lam_c (the collect script checks it).

Public API
----------
family_params(family), family_rates(family, y)
species_omegas(omega0, width, M), pair_list(N, M)
pair_lindbladian(N, omega, kappa, lam, om_a, om_b, gamma_down, gamma_phi)
check_uniform(family, N)
build_ensemble(family, N, width, M, lam_ratio, y) -> Ensemble
avg_rate(frames, cls, ens, reference=False), pauli_avg_rate(ens), pair_floors(ens)
bundle_polish(frames, cls, ens), optimise_point(ens, m, ...)
mixture_mf_matrix(...), mixture_lam_c(...)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from dissipative_PT import _I2, _SX, _SY, _SZ, _build_lindbladian
from optimize_framability import _project_columns_bloch, _FIXED_COLS, N_FIXED_COLS
from framability_rate import generator_log_norm, generator_log_norm_reference
from framability_rate_global import column_witness, fit_columns, seed_library
from framability_rate_frames import pauli_rate
from trotter_lindbladian_scan import (MODELS, MODEL_DICKE_PARAMS, dicke_coefficients,
                                      build_bond_lindbladian, _S_MINUS_NOTE, _S_PLUS_NOTE)

DISORDER_VERSION = '1.0-species-mixture'
_TOL = 1e-9
_XX = np.kron(_SX, _SX)
_XY = np.kron(_SX, _SY)
_YX = np.kron(_SY, _SX)


# ---------------------------------------------------------------------------
#  Family parameters
# ---------------------------------------------------------------------------
def family_params(family: str) -> dict:
    """omega, omega0, kappa, axis, deph_ratio of a Dicke family
    (MODEL_DICKE_PARAMS; identical for every N of the family)."""
    for p in MODEL_DICKE_PARAMS.values():
        if p['family'] == family:
            return {k: p[k] for k in ('omega', 'omega0', 'kappa', 'axis', 'deph_ratio')}
    raise KeyError(f'unknown Dicke family {family!r}')


def family_rates(family: str, y: float) -> tuple[float, float]:
    """(gamma_down, gamma_phi) at second-axis value y (note convention)."""
    p = family_params(family)
    if p['axis'] == 'gamma_down':
        return float(y), 0.25 * float(p['deph_ratio'] or 0.0) * float(y)
    return 0.0, float(y)


def lam_c0(family: str) -> float:
    """Uniform mean-field lam_c at the MEAN omega0 (the grid's unit of lam)."""
    p = family_params(family)
    return float(dicke_coefficients(p['omega'], p['omega0'], p['kappa'])['lam_c'])


# ---------------------------------------------------------------------------
#  Species and pairs
# ---------------------------------------------------------------------------
def species_omegas(omega0: float, width: float, M: int) -> np.ndarray:
    """Bin midpoints of the uniform distribution on omega0 (1 -+ width);
    width = 0 -> the single value omega0."""
    if width == 0:
        return np.array([float(omega0)])
    if not 0 < width < 1:
        raise ValueError(f'width must be in [0, 1), got {width}')
    s = np.arange(M)
    return float(omega0) * (1.0 + width * (2 * s + 1 - M) / M)


def pair_list(N: int, M: int):
    """Unordered species pairs (s, t), s <= t, and their normalised weights
    for N atoms in M species of equal abundance."""
    if N < 2 or N < M:
        raise ValueError(f'need N >= max(2, M), got N={N}, M={M}')
    n = N / M
    pairs, w = [], []
    for s in range(M):
        for t in range(s, M):
            pairs.append((s, t))
            w.append(n * (n - 1) / 2 if s == t else n * n)
    w = np.array(w, float)
    return pairs, w / w.sum()


def _response(omega: float, kappa: float, om: float):
    c_p = 1.0 / (kappa + 1j * (omega + om))
    c_m = 1.0 / (kappa + 1j * (omega - om))
    return c_m * _S_MINUS_NOTE + c_p * _S_PLUS_NOTE, c_p + c_m, 1j * (c_p - c_m)


def pair_lindbladian(N: int, omega: float, kappa: float, lam: float,
                     om_a: float, om_b: float, gamma_down: float,
                     gamma_phi: float) -> np.ndarray:
    """16x16 real pair patch B_ab (qubit order a, b) in the 2-qubit Pauli
    basis, generator acting on states (A = B_ab.T acts on observables)."""
    lam2N = lam ** 2 / N
    m_a, A_a, B_a = _response(omega, kappa, om_a)
    m_b, A_b, B_b = _response(omega, kappa, om_b)
    h_a = (0.5 * om_a + 0.5 * lam2N * B_a.real) * _SZ
    h_b = (0.5 * om_b + 0.5 * lam2N * B_b.real) * _SZ
    H = (np.kron(h_a, _I2) + np.kron(_I2, h_b)
         + (N - 1) * lam2N * (0.5 * (A_a.imag + A_b.imag) * _XX
                              + 0.5 * B_b.imag * _XY + 0.5 * B_a.imag * _YX))
    mI, Im = np.kron(m_a, _I2), np.kron(_I2, m_b)
    jumps = []

    def add(weight, L):
        if weight != 0:
            Ld = L.conj().T
            jumps.append((float(weight), L, Ld, Ld @ L))

    add(2.0 * kappa * (N - 1) * lam2N, mI + Im)             # collective pair (CP)
    w_neg = -2.0 * kappa * (N - 2) * lam2N                   # counterterm (non-CP)
    add(w_neg, mI)
    add(w_neg, Im)
    if gamma_down > 0:
        add(2.0 * gamma_down, np.kron(_S_MINUS_NOTE, _I2))
        add(2.0 * gamma_down, np.kron(_I2, _S_MINUS_NOTE))
    if gamma_phi > 0:
        add(2.0 * gamma_phi, np.kron(_SZ, _I2))
        add(2.0 * gamma_phi, np.kron(_I2, _SZ))
    return _build_lindbladian(H.astype(complex), jumps, n=2)


def check_uniform(family: str, N: int,
                  pts=((0.0, 0.0), (0.7, 0.04), (1.0, 0.1), (2.0, 0.3)),
                  tol: float = 1e-12) -> float:
    """max |B(w = 0) - registered uniform bond generator| over a few points;
    raises if above tol (the disordered builder must contain the uniform one)."""
    name = f'{family}_N{N}'
    if name not in MODELS:
        raise KeyError(f'{name} is not registered (MODEL_DICKE_PARAMS)')
    spec = MODELS[name]
    p = family_params(family)
    lc = lam_c0(family)
    worst = 0.0
    for x, y in pts:
        ref = np.asarray(build_bond_lindbladian(*spec.build(x, y), spec.dim)).real
        gd, gp = family_rates(family, y)
        L = pair_lindbladian(N, p['omega'], p['kappa'], x * lc, p['omega0'],
                             p['omega0'], gd, gp)
        worst = max(worst, float(np.max(np.abs(L - ref))))
    if worst > tol:
        raise RuntimeError(f'{name}: disordered pair patch at w = 0 differs from the '
                           f'registered model by {worst:.3e} > {tol:g}')
    return worst


@dataclass
class Ensemble:
    family: str
    N: int
    width: float
    M: int                 # requested species count
    lam_ratio: float
    y: float
    gamma_down: float
    gamma_phi: float
    omegas: np.ndarray     # (M_eff,)
    pairs: list            # [(s, t)]
    pw: np.ndarray         # (P,) weights, sum 1
    Ls: list               # pair generators (16 x 16, on states)
    As: list               # transposes (on observables)

    @property
    def M_eff(self) -> int:
        return len(self.omegas)

    @property
    def P(self) -> int:
        return len(self.pairs)


def build_ensemble(family: str, N: int, width: float, M: int,
                   lam_ratio: float, y: float) -> Ensemble:
    p = family_params(family)
    lam = lam_ratio * lam_c0(family)
    om = species_omegas(p['omega0'], width, M)
    pairs, pw = pair_list(N, len(om))
    gd, gp = family_rates(family, y)
    Ls = [pair_lindbladian(N, p['omega'], p['kappa'], lam, om[s], om[t], gd, gp)
          for s, t in pairs]
    return Ensemble(family=family, N=N, width=float(width), M=M,
                    lam_ratio=float(lam_ratio), y=float(y), gamma_down=gd,
                    gamma_phi=gp, omegas=om, pairs=pairs, pw=pw, Ls=Ls,
                    As=[np.ascontiguousarray(L.T) for L in Ls])


# ---------------------------------------------------------------------------
#  Rates
# ---------------------------------------------------------------------------
def class_map(ens: Ensemble, mode: str) -> np.ndarray:
    """species -> frame index: 'shared' all 0, 'species' the identity."""
    if mode == 'shared':
        return np.zeros(ens.M_eff, int)
    if mode == 'species':
        return np.arange(ens.M_eff)
    raise ValueError(mode)


def avg_rate(frames, cls, ens: Ensemble, *, reference: bool = False):
    """(F, per-pair rates) of the frame assignment frames[cls[s]] on species s."""
    f = generator_log_norm_reference if reference else generator_log_norm
    vals = np.empty(ens.P)
    for p, (s, t) in enumerate(ens.pairs):
        vals[p] = f(np.kron(frames[cls[s]], frames[cls[t]]), ens.As[p])
    F = float(ens.pw @ vals) if np.all(np.isfinite(vals)) else np.inf
    return F, vals


def pauli_avg_rate(ens: Ensemble):
    """(F, per-pair rates) of the Pauli frame (closed form)."""
    vals = np.array([pauli_rate(L) for L in ens.Ls])
    return float(ens.pw @ vals), vals


def pair_floors(ens: Ensemble) -> np.ndarray:
    """Spectral abscissa of every pair patch: lower bound on that pair's rate
    in ANY frame (shared, per species or otherwise)."""
    return np.array([float(np.max(np.linalg.eigvals(L).real)) for L in ens.Ls])


# ---------------------------------------------------------------------------
#  Bundle descent on the pair average
# ---------------------------------------------------------------------------
def _pair_column_grad(Sa, Sb, A, j, w, h):
    """Envelope gradient of column j of mu(B; Sa (x) Sb) w.r.t. Sa and Sb:
    d mu_j / dD = (A^T w) e_j^T - w h^T, D[(al be), (i k)] = Sa[al, i] Sb[be, k]."""
    ma, mb = Sa.shape[1], Sb.shape[1]
    gD = np.zeros((16, ma * mb))
    gD[:, j] += A.T @ w
    gD -= np.outer(w, h)
    T = gD.reshape(4, 4, ma, mb)
    return np.einsum('abij,bj->ai', T, Sb), np.einsum('abij,ai->bj', T, Sa)


def _state(frames, cls, ens):
    out = []
    for p, (s, t) in enumerate(ens.pairs):
        a, b = int(cls[s]), int(cls[t])
        D = np.kron(frames[a], frames[b])
        val, cols, H = generator_log_norm(D, ens.As[p], return_cols=True, return_H=True)
        out.append((float(val), cols, H, D, a, b))
    return out


def _F_of(state, ens):
    vals = np.array([s[0] for s in state])
    return float(ens.pw @ vals) if np.all(np.isfinite(vals)) else np.inf


def _project(S_free):
    return np.hstack([_FIXED_COLS, _project_columns_bloch(S_free)])


def bundle_polish(frames, cls, ens: Ensemble, *, n_iter: int = 60,
                  radius: float = 0.1, band: float = 0.05, max_cols: int = 6,
                  shrink: float = 0.5, grow: float = 1.6, radius_max: float = 0.5,
                  tol: float = _TOL, verbose: bool = False):
    """Trust-region bundle descent on F over the frames (all moved jointly).
    Returns (best F, best frames)."""
    frames = [np.asarray(S, float).copy() for S in frames]
    K = len(frames)
    m = frames[0].shape[1]
    nf = m - N_FIXED_COLS
    nvar = K * 4 * nf
    P = ens.P
    state = _state(frames, cls, ens)
    F = _F_of(state, ens)
    if not np.isfinite(F):
        return F, frames
    best_F, best = F, [S.copy() for S in frames]
    c_lp = np.concatenate([np.zeros(nvar), ens.pw])
    bounds_t = [(0.0, None)] * P                 # every pair rate is >= 0
    for it in range(n_iter):
        if F <= tol:
            break
        rows, rhs = [], []
        for p, (val, cols, H, D, a, b) in enumerate(state):
            have = False
            if H is not None:
                thresh = val - band * max(abs(val), 1e-3)
                for j in np.argsort(-cols)[:max_cols]:
                    if cols[j] < thresh:
                        break
                    w = column_witness(D, ens.As[p], int(j))
                    if w is None:
                        continue
                    ga, gb = _pair_column_grad(frames[a], frames[b], ens.As[p],
                                               int(j), w, H[:, j])
                    G = np.zeros((K, 4, m))
                    G[a] += ga
                    G[b] += gb
                    row = np.zeros(nvar + P)
                    row[:nvar] = G[:, :, N_FIXED_COLS:].ravel()
                    row[nvar + p] = -1.0
                    rows.append(row)
                    rhs.append(-float(cols[j]))
                    have = True
            if not have:                            # pair held at its value
                row = np.zeros(nvar + P)
                row[nvar + p] = -1.0
                rows.append(row)
                rhs.append(-val)
        res = linprog(c_lp, A_ub=np.array(rows), b_ub=np.array(rhs),
                      bounds=[(-radius, radius)] * nvar + bounds_t, method='highs')
        if not res.success:
            radius *= shrink
            if radius < 1e-6:
                break
            continue
        delta = res.x[:nvar].reshape(K, 4, nf)
        new = [_project(frames[k][:, N_FIXED_COLS:] + delta[k]) for k in range(K)]
        st_new = _state(new, cls, ens)
        F_new = _F_of(st_new, ens)
        pred = F - float(ens.pw @ res.x[nvar:])
        if np.isfinite(F_new) and F_new < F - tol and (pred <= 0 or (F - F_new) >= 0.05 * pred):
            frames, state, F = new, st_new, F_new
            if F < best_F:
                best_F, best = F, [S.copy() for S in frames]
            radius = min(radius * grow, radius_max)
        else:
            radius *= shrink
            if radius < 1e-6:
                break
        if verbose and (it + 1) % 10 == 0:
            print(f'      bundle {it + 1}: F={F:.6e} radius={radius:.3g} rows={len(rows)}',
                  flush=True)
    return best_F, best


def _kicked(frames, scale, rng):
    return [_project(S[:, N_FIXED_COLS:] + scale * rng.standard_normal(S[:, N_FIXED_COLS:].shape))
            for S in frames]


def _dedup(scored, tol=1e-9):
    out = []
    for F, name, S in sorted(scored, key=lambda z: z[0]):
        if all(abs(F - G) > tol or np.max(np.abs(S - T)) > 1e-8 for G, _, T in out):
            out.append((F, name, S))
    return out


def _improve(frames, cls, ens, F, *, n_iter, kick_iter, n_kicks, kick_scales, rng, tol):
    """bundle polish (n_iter > 0) + kicked restarts (kick_iter each) from the
    incumbent; never returns anything worse than (F, frames)."""
    best_F, best = F, frames
    if best_F > tol and n_iter > 0:
        F1, fr = bundle_polish(best, cls, ens, n_iter=n_iter, tol=tol)
        if F1 < best_F:
            best_F, best = F1, fr
    for scale in kick_scales:
        for _ in range(n_kicks):
            if best_F <= tol:
                break
            F1, fr = bundle_polish(_kicked(best, scale, rng), cls, ens,
                                   n_iter=kick_iter, tol=tol)
            if F1 < best_F:
                best_F, best = F1, fr
    return best_F, best


def optimise_point(ens: Ensemble, m: int, *, stored_frames=(), prev_shared=None,
                   prev_species=None, n_starts: int = 3, bundle_iters: int = 60,
                   n_kicks: int = 1, kick_scales=(0.03,), max_seed_eval: int = 40,
                   certify: bool = True, rng=None, tol: float = _TOL,
                   verbose: bool = False) -> dict:
    """Shared-frame and per-species optimum of F at frame size m.
    Keys (suffix _<m>): F_seed, seed_name, F_shared, S_shared, pairs_shared,
    worst_shared, F_species, S_species (M_eff x 4 x m), pairs_species,
    worst_species, F_shared_ref / F_species_ref (per-column reference LP),
    F_shared_cert / F_species_cert (reference value when certified), t."""
    rng = np.random.default_rng(0) if rng is None else rng
    t0 = time.time()
    cls_sh = class_map(ens, 'shared')

    # ---- seeds for the shared frame ----------------------------------------
    cands = [(f'lib_{k}', S) for k, S in seed_library(m).items()]
    for k, S in enumerate(list(stored_frames)[:max_seed_eval]):
        S = np.asarray(S, float)
        if S.shape[1] <= m:
            cands.append((f'stored{k}', fit_columns(S, m)))
    if prev_shared is not None:
        cands.append(('padded', fit_columns(np.asarray(prev_shared, float), m)))
    scored = []
    for name, S in cands:
        F, _ = avg_rate([S], cls_sh, ens)
        if np.isfinite(F):
            scored.append((F, name, S))
    scored = _dedup(scored)
    if not scored:
        raise RuntimeError('no seed frame with finite average rate')
    F_seed, seed_name, S_seed = scored[0]
    if verbose:
        print(f'    m={m}: {len(scored)} seeds, best {seed_name} F={F_seed:.6g}', flush=True)

    # ---- shared optimum -----------------------------------------------------
    best_F, best = F_seed, [S_seed]
    if best_F > tol:
        for F0, name, S0 in scored[:n_starts]:
            F1, fr = bundle_polish([S0], cls_sh, ens, n_iter=bundle_iters, tol=tol)
            if F1 < best_F:
                best_F, best = F1, fr
            if best_F <= tol:
                break
        best_F, best = _improve(best, cls_sh, ens, best_F, n_iter=0,
                                kick_iter=max(1, bundle_iters // 2), n_kicks=n_kicks,
                                kick_scales=kick_scales, rng=rng, tol=tol)
    S_sh = best[0]
    F_sh, pairs_sh = avg_rate([S_sh], cls_sh, ens)
    if verbose:
        print(f'    m={m}: shared F={F_sh:.6g} ({time.time() - t0:.0f}s)', flush=True)

    # ---- per-species optimum (starts at the shared optimum) -----------------
    if ens.M_eff > 1:
        cls_sp = class_map(ens, 'species')
        starts = [[S_sh.copy() for _ in range(ens.M_eff)]]
        if prev_species is not None and len(prev_species) == ens.M_eff:
            starts.append([fit_columns(np.asarray(S, float), m) for S in prev_species])
        sp_F, sp = np.inf, None
        for fr0 in starts:
            F0, _ = avg_rate(fr0, cls_sp, ens)
            F1, fr1 = _improve(fr0, cls_sp, ens, F0, n_iter=bundle_iters,
                               kick_iter=max(1, bundle_iters // 2), n_kicks=n_kicks,
                               kick_scales=kick_scales, rng=rng, tol=tol)
            if F1 < sp_F:
                sp_F, sp = F1, fr1
        if sp_F > F_sh:                            # cannot happen (start = shared)
            sp_F, sp = F_sh, [S_sh.copy() for _ in range(ens.M_eff)]
        S_sp = np.stack(sp)
        F_sp, pairs_sp = avg_rate(list(S_sp), cls_sp, ens)
    else:
        cls_sp = cls_sh
        S_sp = S_sh[None].copy()
        F_sp, pairs_sp = F_sh, pairs_sh.copy()
    if verbose:
        print(f'    m={m}: species F={F_sp:.6g} ({time.time() - t0:.0f}s)', flush=True)

    out = {f'F_seed_{m}': F_seed, f'seed_name_{m}': seed_name,
           f'F_shared_{m}': F_sh, f'S_shared_{m}': S_sh, f'pairs_shared_{m}': pairs_sh,
           f'worst_shared_{m}': float(np.max(pairs_sh)),
           f'F_species_{m}': F_sp, f'S_species_{m}': S_sp, f'pairs_species_{m}': pairs_sp,
           f'worst_species_{m}': float(np.max(pairs_sp))}
    for key, fr, cls, F in (('shared', [S_sh], cls_sh, F_sh),
                            ('species', list(S_sp), cls_sp, F_sp)):
        if certify:
            F_ref, _ = avg_rate(fr, cls, ens, reference=True)
            out[f'F_{key}_ref_{m}'] = F_ref
            out[f'F_{key}_cert_{m}'] = F_ref if np.isfinite(F_ref) else F
        else:
            out[f'F_{key}_ref_{m}'] = np.nan
            out[f'F_{key}_cert_{m}'] = F
    out[f't_{m}'] = time.time() - t0
    return out


# ---------------------------------------------------------------------------
#  Mean-field transition of the species mixture (N -> inf)
# ---------------------------------------------------------------------------
def mixture_mf_matrix(lam: float, omegas, abund, omega: float, kappa: float,
                      gamma_perp: float) -> np.ndarray:
    """Linear drift of (<B_s>, <B_s^dag>) around the all-down state.

    xi = (B_0 .. B_{M-1}, B_0^dag .. B_{M-1}^dag);  in HP, with x = sqrt(abund),
        H = sum_s omega0_s B_s^dag B_s + lam^2 Im(Xt Mt),
        Xt = sum_s x_s (B_s + B_s^dag),  Mt = sum_s x_s (c_-^s B_s + c_+^s B_s^dag),
        jump (repo D) sqrt(2 kappa) lam Mt,  on-site coherence decay gamma_perp
    (gamma_perp = gamma_down + 4 gamma_phi)."""
    om = np.asarray(omegas, float)
    M = len(om)
    x = np.sqrt(np.asarray(abund, float))
    c_m = 1.0 / (kappa + 1j * (omega - om))
    c_p = 1.0 / (kappa + 1j * (omega + om))
    xv = np.concatenate([x, x]).astype(complex)
    mv = np.concatenate([x * c_m, x * c_p])
    bar = np.concatenate([np.arange(M, 2 * M), np.arange(M)])
    K1 = np.outer(xv, mv)                               # Xt Mt = sum K1_ab xi_a xi_b
    K2 = np.zeros((2 * M, 2 * M), complex)              # (Xt Mt)^dag
    for a in range(2 * M):
        for b in range(2 * M):
            K2[bar[b], bar[a]] += np.conj(mv[b]) * xv[a]
    K = lam ** 2 * (K1 - K2) / 2j
    for s in range(M):
        K[s + M, s] += om[s]
    ell = np.sqrt(2.0 * kappa) * lam * mv               # jump = sum ell_c xi_c
    Mat = np.zeros((2 * M, 2 * M), complex)
    for s in range(M):
        row = -1j * (K[s + M, :] + K[:, s + M])         # -i [B_s, H]
        row = row - 0.5 * np.conj(ell[s]) * ell         # -1/2 conj(ell_s) <L>
        for c in range(2 * M):                          # +1/2 ell_{s+M} <L^dag>
            row[bar[c]] += 0.5 * ell[s + M] * np.conj(ell[c])
        row[s] -= gamma_perp
        Mat[s] = row
    for s in range(M):
        for c in range(2 * M):
            Mat[s + M, bar[c]] = np.conj(Mat[s, c])
    return Mat


def mixture_lam_c(gamma_perp: float, omegas, abund, omega: float, kappa: float,
                  lam_max: float, n_grid: int = 400, tol: float = 1e-12) -> float:
    """Smallest lam <= lam_max with an unstable mode (nan if none)."""
    def unstable(lam):
        ev = np.linalg.eigvals(mixture_mf_matrix(lam, omegas, abund, omega, kappa,
                                                 gamma_perp))
        return float(np.max(ev.real)) > tol
    grid = np.linspace(lam_max / n_grid, lam_max, n_grid)
    prev = 0.0
    for lam in grid:
        if unstable(lam):
            lo, hi = prev, lam
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if unstable(mid):
                    hi = mid
                else:
                    lo = mid
            return float(hi)
        prev = lam
    return float('nan')
