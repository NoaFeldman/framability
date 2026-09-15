r"""
liouvillian_quality.py -- quality factor Q of the relaxation modes of a
Lindbladian, from its FULL spectrum.

A mode lambda_k = -Gamma_k + i omega_k decays at Gamma_k while ringing at
omega_k, so

    Q_k = |Im lambda_k| / |Re lambda_k| = omega_k / Gamma_k

is the angle (radians) it turns per e-folding: Q = 1 loses the oscillation
within about one radian (amplitude x e^{-2 pi} per period), Q >> 1 rings for
many cycles.  The headline quantity is

    Q_max = max_k Q_k      over the DAMPED modes (Re lambda_k < -tol),

the MOST COHERENT relaxation mode -- as opposed to the Liouvillian gap, which is
the SLOWEST one.  Q_max is the spectral quantity the framability rate
constrains:

  * mu*(D) <= 0 turns the rate certificate into a classical sub-Markov rate
    matrix on the 2 m^2 signed frame labels (+1 cemetery state), and every
    eigenvalue of the bond generator is one of its eigenvalues, hence lies in
    the wedge |Im| <= cot(pi/n) |Re|, n = 2 m^2 + 1.  Rigorous but loose:
    cot(pi/33) ~ 10.5 for d_ext_single = 4.
  * Locally it is sharp: a rotation plane carried by an n-gon of frame
    directions allows Q <= cot(pi/n) at mu* = 0, and the correlated planes of a
    product frame (e.g. X(x)I <-> Y(x)Z under ZZ) only carry the square, Q <= 1.
    Exact example: model3 at gamma = 0 (bond convention dim = 2) has
    Q_max = 4J/gamma' and Pauli rate max(0, 2J - gamma'/2) -- mu* = 0 exactly
    where Q_max = 1.

Why the full spectrum (and not nonequilibrium_phase_characterizers.oscillation_rate)
------------------------------------------------------------------------------
oscillation_rate's sparse path maximises over the k rightmost (slowest) modes,
which is a lower bound on the maximum -- but for Q it can be an arbitrarily bad
one: the most coherent mode need not be slow (model3 at gamma = 0: the ZZ beats
X_i (x) P_nn decay at 2 gamma', while pump-limited modes can sit far closer to
the axis).  It also counts near-steady modes with a numerically tiny Re as
enormous ratios (the ~1e18 values on the 8q panel).  Here the spectrum is always
complete (dense, so N <= 6 qubits by default) and every eigenvalue is
classified with one relative tolerance:

    steady      |lambda|  <= tol                         (dropped)
    undamped    |Re|      <= tol < |lambda|              (reported separately:
                                                          omega_undamped, never
                                                          folded into Q_max)
    damped      Re        <  -tol                        (enter Q_max, gap)
    growing     Re        >  +tol                        (warning: not a valid
                                                          Lindbladian spectrum)

tol = tol_rel * max|lambda|.  The default tol_rel = 1e-7 sits above the
~sqrt(eps) ~ 1.5e-8 splitting a size-2 Jordan block (exceptional point) suffers
under floating point, so a defective steady cluster is not misread as a pair of
undamped modes.

Observable quality factor (not a spectral quantity)
---------------------------------------------------
Q_max is the coherence of the dressed eigenmodes, so it is blind to rotations
that the dissipation Zeno-freezes into an overdamped (real) spectrum -- e.g.
model4's field rotating the weakly damped Z axis into the strongly dephased Y.
The observable quality factor measures the bare rotation of product
observables instead: for A = L^T in the two-qubit Pauli basis,

    Q_obs(P) = sum_{P' != P} |A_{P'P}| / (-A_{PP}),     Q_obs = max_P Q_obs(P),

the rate at which the generator moves weight off the Pauli string P (coherent
rotation and drift) in units of P's own decay rate.  Exactly:
  * Q_obs <= 1  <=>  the Pauli-frame rate mu*_Pauli vanishes;
  * Q_obs(P) > 1 <=> the Pauli-l1 weight of the Heisenberg-evolved P grows at t=0+.
observable_quality_opt minimises Q_obs over one local rotation R (the same on
both qubits) and axis lengths w, i.e. over frames S = diag(1, R diag(w)) -- the
zero set of that minimum is that of the d_ext = 4 rate restricted to rotated,
rescaled Pauli frames.

Public API
----------
quality_from_eigenvalues(evals)     classification + Q_max / Q_slowest / gap
quality_factor(L)                   the same from a dense or sparse Liouvillian
observable_quality(L)               Q_obs of a 16x16 Pauli-basis bond generator
observable_quality_opt(L)           Q_obs minimised over the local basis

    python liouvillian_quality.py   # self-test (exact model3 gamma=0 values)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QUALITY_VERSION = '1.0-full-spectrum'

TOL_REL_DEFAULT = 1e-7
DENSE_MAX_DIM_DEFAULT = 4096          # 4^6: N <= 6 qubits, ~270 MB dense


@dataclass
class QualityResult:
    """See quality_from_eigenvalues."""
    Q_max: float              # max Q over damped modes (nan if none)
    lam_Q: complex | None     # eigenvalue attaining Q_max
    Q_slowest: float          # Q of the slowest damped mode(s) = the gap mode
    gap: float                # min Gamma over damped modes (nan if none)
    n_damped: int
    n_undamped: int           # oscillating modes with |Re| <= tol
    n_steady: int
    n_growing: int
    omega_undamped: float     # max |Im| over undamped modes (0 if none)
    tol: float
    dim: int
    warnings: list = field(default_factory=list)

    @property
    def Q_effective(self) -> float:
        """Q_max, or inf when an undamped oscillating mode exists."""
        return float('inf') if self.n_undamped else self.Q_max


def quality_from_eigenvalues(evals, *, tol_rel: float = TOL_REL_DEFAULT
                             ) -> QualityResult:
    """Classify a (complete) Liouvillian spectrum and return its Q statistics."""
    lam = np.asarray(evals, dtype=complex).ravel()
    warnings: list[str] = []
    scale = float(np.max(np.abs(lam))) if lam.size else 0.0
    tol = tol_rel * scale if scale > 0 else tol_rel

    re, im = lam.real, lam.imag
    steady = np.abs(lam) <= tol
    damped = ~steady & (re < -tol)
    undamped = ~steady & (np.abs(re) <= tol)
    growing = ~steady & (re > tol)
    if growing.any():
        warnings.append(f'{int(growing.sum())} eigenvalue(s) with Re > tol='
                        f'{tol:.3g} (max Re {re[growing].max():.3g}); not a '
                        f'contractive Lindbladian spectrum -- excluded.')

    omega_undamped = float(np.max(np.abs(im[undamped]))) if undamped.any() else 0.0

    if not damped.any():
        return QualityResult(np.nan, None, np.nan, np.nan, 0, int(undamped.sum()),
                             int(steady.sum()), int(growing.sum()),
                             omega_undamped, tol, lam.size, warnings)

    lam_d = lam[damped]
    Gamma = -lam_d.real
    omega = np.where(np.abs(lam_d.imag) <= tol, 0.0, np.abs(lam_d.imag))
    Q = omega / Gamma
    k = int(np.argmax(Q))
    gap = float(Gamma.min())
    Q_slowest = float(Q[Gamma <= gap + tol].max())
    return QualityResult(float(Q[k]), complex(lam_d[k]), Q_slowest, gap,
                         int(damped.sum()), int(undamped.sum()),
                         int(steady.sum()), int(growing.sum()),
                         omega_undamped, tol, lam.size, warnings)


def quality_factor(L, *, tol_rel: float = TOL_REL_DEFAULT,
                   dense_max_dim: int = DENSE_MAX_DIM_DEFAULT,
                   return_eigenvalues: bool = False):
    """Q statistics of the Liouvillian L (dense ndarray or scipy sparse).

    Any basis works (Pauli-basis real generators, computational-basis
    column-stacking superoperators): only the spectrum is used.  The full
    spectrum is always computed; L larger than dense_max_dim is refused rather
    than silently approximated (see the module docstring for why a partial
    spectrum is not a usable bound for Q).
    """
    import scipy.sparse as sp

    dim = L.shape[0]
    if dim > dense_max_dim:
        raise ValueError(
            f'quality_factor: dim={dim} exceeds dense_max_dim={dense_max_dim}. '
            f'Q_max needs the complete spectrum; use a smaller system or raise '
            f'dense_max_dim if the memory allows (~{dim * dim * 16 / 2**30:.1f} '
            f'GB dense before LAPACK workspace).')
    Ld = L.toarray() if sp.issparse(L) else np.asarray(L)
    evals = np.linalg.eigvals(Ld)
    res = quality_from_eigenvalues(evals, tol_rel=tol_rel)
    return (res, evals) if return_eigenvalues else res


# ---------------------------------------------------------------------------
#  Observable quality factor (Pauli-basis column dominance)
# ---------------------------------------------------------------------------
OBS_QUALITY_VERSION = '1.0-column-dominance'
OBS_W_MIN = 1e-2          # shortest rotated axis allowed in observable_quality_opt
_OBS_BIG = 1e6            # stand-in for +inf inside the optimiser


@dataclass
class ObservableQualityResult:
    """See observable_quality."""
    Q_obs: float            # max_P Q_obs(P); inf if an undamped string is pushed
    label: str              # two-letter string attaining it ('ZI'; lowercase
                            # letters = rotated axes, named by nearest Pauli axis)
    cols: np.ndarray        # Q_obs(P) for all 16 strings (identity column: 0)
    rotvec: np.ndarray      # local basis rotation (zeros = Pauli basis)
    w: np.ndarray           # lengths of the three rotated axes (ones = Pauli)


def _local_frame(rotvec=None, w=None):
    """4x4 single-qubit frame S: identity column, then the columns of R diag(w)."""
    from scipy.spatial.transform import Rotation
    R = (np.eye(3) if rotvec is None
         else Rotation.from_rotvec(np.asarray(rotvec, float)).as_matrix())
    w = np.ones(3) if w is None else np.asarray(w, float)
    S = np.zeros((4, 4))
    S[0, 0] = 1.0
    S[1:, 1:] = R * w                       # column k scaled by w[k]
    return S, R


def column_dominance_ratios(H, tol):
    """sum_{k != j} |H_kj| / (-H_jj) for every column j of H (H acts on
    coefficient vectors).  Columns with -H_jj <= tol give +inf when they still
    leak (off-diagonal > tol) and 0 when they are conserved (the identity)."""
    H = np.asarray(H, float)
    diag = np.diag(H)
    off = np.abs(H).sum(axis=0) - np.abs(diag)
    damp = -diag
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(damp > tol, off / damp,
                        np.where(off > tol, np.inf, 0.0))


def observable_quality(L, *, rotvec=None, w=None,
                       tol_rel: float = TOL_REL_DEFAULT) -> ObservableQualityResult:
    """Q_obs of a two-qubit Pauli-basis generator L (the 16x16 Schrodinger
    matrix M[a,b] = Tr(P_a L(P_b))/4 of build_bond_lindbladian, index 4a+b <->
    P_a (x) P_b) in the local basis (rotvec, w); default: the Pauli basis.

    In a basis S the generator on frame coefficients is H = D^{-1} A D with
    A = L^T and D = S (x) S (square and invertible, so H is unique), and
    max_j (H_jj + sum_{k!=j}|H_kj|) is exactly that frame's rate mu*.
    """
    A = np.asarray(L, float).T
    S, R = _local_frame(rotvec, w)
    D = np.kron(S, S)
    H = np.linalg.solve(D, A @ D)
    tol = tol_rel * max(float(np.max(np.abs(H))), 1e-300)
    cols = column_dominance_ratios(H, tol)
    k = int(np.argmax(cols))
    if rotvec is None:
        letters = 'IXYZ'
    else:
        letters = 'I' + ''.join('xyz'[int(np.argmax(np.abs(R[:, c])))]
                                for c in range(3))
    label = letters[k // 4] + letters[k % 4]
    return ObservableQualityResult(
        float(cols[k]), label, cols,
        np.zeros(3) if rotvec is None else np.asarray(rotvec, float),
        np.ones(3) if w is None else np.asarray(w, float))


def observable_quality_opt(L, *, n_restarts: int = 8, maxfev: int = 2000,
                           seed: int = 0, w_min: float = OBS_W_MIN,
                           tol_rel: float = TOL_REL_DEFAULT
                           ) -> ObservableQualityResult:
    """min over a local rotation R in SO(3) (same on both qubits) and axis
    lengths w in [w_min, 1]^3 of observable_quality.  Nelder-Mead on
    (rotation vector, logit of w) from the Pauli basis plus random rotations.
    Never worse than the Pauli basis (it is compared at the end)."""
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation

    L = np.asarray(L, float)
    rng = np.random.default_rng(seed)
    t_full = 6.0                                  # logit -> w ~ 1

    def unpack(x):
        return x[:3], w_min + (1.0 - w_min) / (1.0 + np.exp(-x[3:]))

    def objective(x):
        rv, w = unpack(x)
        try:
            q = observable_quality(L, rotvec=rv, w=w, tol_rel=tol_rel).Q_obs
        except np.linalg.LinAlgError:
            return _OBS_BIG
        return min(q, _OBS_BIG)

    starts = [np.r_[np.zeros(3), np.full(3, t_full)]]
    while len(starts) < n_restarts:
        rv = Rotation.random(random_state=int(rng.integers(2**31))).as_rotvec()
        starts.append(np.r_[rv, rng.uniform(-2.0, t_full, 3)])
    steps = np.r_[np.full(3, 0.4), np.full(3, 1.5)]

    best_x, best_f = None, np.inf
    for x0 in starts:
        simplex = np.vstack([x0] + [x0 + steps[i] * np.eye(6)[i] for i in range(6)])
        res = minimize(objective, x0, method='Nelder-Mead',
                       options=dict(maxfev=maxfev, xatol=1e-7, fatol=1e-10,
                                    initial_simplex=simplex))
        if res.fun < best_f:
            best_x, best_f = res.x, float(res.fun)

    rv, w = unpack(best_x)
    out = observable_quality(L, rotvec=rv, w=w, tol_rel=tol_rel)
    pauli = observable_quality(L, rotvec=np.zeros(3), w=np.ones(3), tol_rel=tol_rel)
    return pauli if pauli.Q_obs <= out.Q_obs else out


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    from trotter_lindbladian_scan import MODELS, build_bond_lindbladian
    from n_qubit_lindbladian import build_lindbladian_comp
    from dissipative_PT import bonds_2d

    J = 1.0
    m3 = MODELS['model3']

    # 1. model3 bond generator at gamma = 0 (dim = 2 -> single-site share 1/4):
    #    X (x) P^{+-} modes, lambda = -gamma'/2 +- 2iJ  ->  Q_max = 4J/gamma'.
    for gp in (1.0, 2.0, 4.0, 8.0):
        L = build_bond_lindbladian(*m3.build(0.0, gp), 2).real
        r = quality_factor(L)
        want = 4 * J / gp
        print(f"[bond]    model3 gamma=0 gamma'={gp:<4}  Q_max={r.Q_max:.10f}  "
              f'expected {want:.10f}  lam_Q={r.lam_Q:.4f}')
        assert abs(r.Q_max - want) < 1e-9, 'bond Q_max off the analytic value'

    # 2. model3 on a 2x3 open lattice at gamma = 0: X_i (x) prod_j P_j^{s_j}
    #    rotates at 2J sum_j s_j and decays at 2 gamma'  ->  Q_max = z_max J/gamma'
    #    with z_max = 3 (the two middle sites).  Tests the lattice builder's
    #    conventions against the bond ones.
    lx, ly = 3, 2
    edges = bonds_2d(lx, ly)
    z_max = max(np.bincount(np.array(edges).ravel()))
    for gp in (1.0, 3.0):
        Lc = build_lindbladian_comp(J, 0.0, gp, lx * ly, edges)
        r = quality_factor(Lc)
        want = z_max * J / gp
        print(f"[lattice] model3 gamma=0 gamma'={gp:<4}  Q_max={r.Q_max:.10f}  "
              f'expected {want:.10f}  (n_steady={r.n_steady})')
        assert abs(r.Q_max - want) < 1e-7, 'lattice Q_max off the analytic value'

    # 3. closed bond dynamics (gamma = gamma' = 0): no damped mode at all, the
    #    oscillation is reported as undamped rather than as a divergent ratio.
    L = build_bond_lindbladian(*m3.build(0.0, 0.0), 2).real
    r = quality_factor(L)
    print(f'[closed]  Q_max={r.Q_max}  n_undamped={r.n_undamped}  '
          f'omega_undamped={r.omega_undamped:.4f}  Q_effective={r.Q_effective}')
    assert np.isnan(r.Q_max) and r.n_undamped > 0 and np.isinf(r.Q_effective)

    # 4. observable quality factor.  Pauli basis: Q_obs <= 1 <=> Pauli rate 0
    #    (exactly, any point); model4 at gamma=3, gamma'=8 is bound by the field
    #    rotating Z, Q_obs = (h/2)/(gamma/8) = 4h/gamma = 2 (ties ZI, IZ, ZZ).
    from framability_rate_frames import pauli_rate
    from trotter_lindbladian_scan import MODEL4_H
    m4 = MODELS['model4']
    for g, gp in ((3.0, 8.0), (8.0, 6.0), (6.0, 4.0), (2.0, 2.0), (10.0, 1.0)):
        L = build_bond_lindbladian(*m4.build(g, gp), 2).real
        ro = observable_quality(L)
        mu = pauli_rate(L)
        print(f"[obs]     model4 gamma={g:<4} gamma'={gp:<4}  Q_obs={ro.Q_obs:.6f} "
              f'({ro.label})  Pauli rate={mu:+.6f}')
        assert (ro.Q_obs <= 1 + 1e-9) == (mu <= 1e-9), 'Q_obs <= 1 != Pauli rate 0'
    L = build_bond_lindbladian(*m4.build(3.0, 8.0), 2).real
    ro = observable_quality(L)
    assert abs(ro.Q_obs - 4 * MODEL4_H / 3.0) < 1e-9 and ro.label in ('ZI', 'IZ', 'ZZ')
    ro_opt = observable_quality_opt(L, n_restarts=4, maxfev=800)
    print(f'[obs-opt] model4 gamma=3 gamma\'=8  Q_obs_opt={ro_opt.Q_obs:.6f} '
          f'({ro_opt.label})  w={np.round(ro_opt.w, 3)}')
    assert ro_opt.Q_obs <= ro.Q_obs + 1e-12

    print('liouvillian_quality self-test passed.')


if __name__ == '__main__':
    _self_test()
