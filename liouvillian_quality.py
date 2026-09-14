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

Public API
----------
quality_from_eigenvalues(evals)     classification + Q_max / Q_slowest / gap
quality_factor(L)                   the same from a dense or sparse Liouvillian

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

    print('liouvillian_quality self-test passed.')


if __name__ == '__main__':
    _self_test()
