"""Magic / framability scan of a nearest-neighbour qubit model over a 2D
parameter grid.  Default model: the XXZ chain in a longitudinal field,

    H = J * ( sum_<ij> [ X_i X_j + Y_i Y_j + Delta Z_i Z_j ]  +  h * sum_i Z_i )

with J = 1 and the scan axes

    Delta in [0.2 i for i in range(-14, 15)]      (29 values, -2.8 .. 2.8)
    h     in [0.4 i for i in range(-10, 11)]      (21 values, -4.0 .. 4.0)

The model is a `MagicModel`: it only has to hand back the one-qubit term H1
(2x2) and the nearest-neighbour term H2 (4x4) for a parameter pair, so a
different Hamiltonian is a different `build` function -- everything downstream
is model-agnostic.  Extra fixed couplings live in `consts`.

Quantities computed at every grid point
---------------------------------------
fra_D1, fra_D2
    dt -> 0 limit of the stabilizer-3 framability (framability.
    stabilizer_3_framability) of the two-qubit nearest-neighbour Trotter gate,
    for spatial dimension D = 1 and D = 2.  The gate is
    trotter_lindbladian_scan.bond_trotter_gate, i.e. the 16x16 real Pauli
    transfer matrix of expm(L_bond dt) with

        H_bond = H2 + 1/(2D) ( H1 (x) I + I (x) H1 ),

    so each one-qubit term is divided by 2 in 1D and by 4 in 2D (a site is
    shared by 2D bonds).  The gate is evaluated on the dt ladder

        dt_i = 0.1 i * dt_point,    i = 1 .. 10,     dt_point = choose_dt(...)

    and the limit is taken with trotter_rom_dtbase.extrapolate_to_zero, i.e.
    the dt = 0 value of framability**(1/dt) -- the raw framability tends
    trivially to 1 (identity gate), the per-unit-time rate does not.

nc_2a .. nc_2e
    Non-cliffordness (magic_measures.noncliffordness = alpha=2 stabilizer Renyi
    entropy of the Choi state) of the EXACT propagator of the full 2x2 lattice
    (N = 4 qubits, so the Choi state has 8 qubits), by exact diagonalization:

        2a  U = exp(i H dt_min),  dt_min = min over the whole grid of choose_dt
        2b  U = exp(i H dt_pt),   dt_pt  = choose_dt at this point
        2c  U = exp(i H T),       T = 1e5 * dt_min
        2d  U = exp(i H T),       T = 1e5 * dt_pt
        2e  U = exp(i H T),       T = 100 / gap,  gap = E_1 - E_0 of H

    On the 2x2 lattice H carries the model's H1 on every site and H2 on every
    nearest-neighbour bond at full strength (no 1/2D bond share -- that is a
    Trotter convention, not part of the physical generator).

    The gap is to the first eigenvalue distinct from the ground energy
    (tolerance GAP_TOL); the gap to the literal second eigenvalue is stored
    alongside as `gap_next` so a degenerate-ground-state reading can be
    recovered without recomputing.

    The sign of the exponent is immaterial for these numbers: exp(-iHt) gives
    the complex-conjugate Choi state, whose Pauli expectations differ only by a
    sign on the Y-containing strings, and M_2 sees only their fourth powers.
    EXP_SIGN follows the prompt's exp(+iHt).

Cluster pipeline
----------------
    scripts/xxz_magic_worker.py        per-point array worker (all quantities)
    scripts/xxz_magic.slurm.sh         200-task data array
    scripts/xxz_magic_collect.py       aggregation + the seven colormaps
    scripts/xxz_magic_collect.slurm.sh dependent plotting job
    scripts/submit_xxz_magic.sh        submits both, chained with afterok

Usage (local, one point):
    python xxz_magic_scan.py --self_test
    python xxz_magic_scan.py --model xxz --p1 0.6 --p2 1.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from dissipative_PT import _I2, _SX, _SY, _SZ, _site_op, bonds_2d
from trotter_lindbladian_scan import (
    DT_BASE, LATTICE_LX, LATTICE_LY, choose_dt, bond_trotter_gate,
    _embed_two_site,
)
from trotter_rom_dtbase import extrapolate_to_zero, FIT_N_DEFAULT, DEG_DEFAULT
from framability import stabilizer_3_framability
from magic_measures import noncliffordness

# Version stamp for cached results; bump when any stored quantity changes.
XXZ_MAGIC_VERSION = '1.0'

# --- conventions -------------------------------------------------------------
EXP_SIGN = +1.0          # U = expm(EXP_SIGN * 1j * H * t)  (prompt: exp(+iHt))
DIMS = (1, 2)            # spatial dimensions for the framability maps
DT_FRACS = np.array([0.1 * i for i in range(1, 11)])   # dt ladder / dt_point
T_LONG_FACTOR = 1.0e5    # 2c/2d: T = T_LONG_FACTOR * dt
GAP_FACTOR = 100.0       # 2e:    T = GAP_FACTOR / gap
GAP_TOL = 1e-9           # degeneracy tolerance for "first excited state"

# Non-cliffordness lattice (4 qubits -> 8-qubit Choi state, exact).
NC_LX, NC_LY = LATTICE_LX, LATTICE_LY

# Everything the worker stores per point, in plot order.
TASK_KEYS = ('fra_D1', 'fra_D2', 'nc_2a', 'nc_2b', 'nc_2c', 'nc_2d', 'nc_2e')
TASK_LABELS = {
    'fra_D1': r'3-stabilizer framability, $dt\to0$  (D = 1)',
    'fra_D2': r'3-stabilizer framability, $dt\to0$  (D = 2)',
    'nc_2a': r'non-cliffordness of $e^{iH\,dt_{\min}}$  (2a)',
    'nc_2b': r'non-cliffordness of $e^{iH\,dt(\Delta,h)}$  (2b)',
    'nc_2c': r'non-cliffordness of $e^{iH\,T}$,  $T = 10^5 dt_{\min}$  (2c)',
    'nc_2d': r'non-cliffordness of $e^{iH\,T}$,  $T = 10^5 dt(\Delta,h)$  (2d)',
    'nc_2e': r'non-cliffordness of $e^{iH\,T}$,  $T = 100/\mathrm{gap}$  (2e)',
}
FRA_KEYS = ('fra_D1', 'fra_D2')
NC_KEYS = ('nc_2a', 'nc_2b', 'nc_2c', 'nc_2d', 'nc_2e')


# ---------------------------------------------------------------------------
#  Model specification
# ---------------------------------------------------------------------------
@dataclass
class MagicModel:
    """A nearest-neighbour qubit model scanned over a 2D parameter grid.

    build(p1, p2) -> (H1, H2) with

        H1 : (2, 2) one-qubit term placed on every site, or None
        H2 : (4, 4) nearest-neighbour term placed on every bond, or None

    at FULL coupling strength; the 1/(2D) Trotter bond share is applied
    downstream by trotter_lindbladian_scan.bond_trotter_gate.  This is exactly
    the (H1, H2) convention of trotter_lindbladian_scan, so choose_dt and the
    bond-gate builder take these matrices unchanged.
    """
    name: str
    p1_name: str
    p2_name: str
    p1_vals: np.ndarray
    p2_vals: np.ndarray
    build: Callable[[float, float], tuple[np.ndarray | None, np.ndarray | None]]
    consts: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.p1_vals), len(self.p2_vals)

    @property
    def n_points(self) -> int:
        return len(self.p1_vals) * len(self.p2_vals)

    def point(self, ix: int, iy: int) -> tuple[float, float]:
        return float(self.p1_vals[ix]), float(self.p2_vals[iy])


# --- the XXZ model in a field ------------------------------------------------
J_DEFAULT = 1.0
DELTA_VALS = np.array([i * 0.2 for i in range(-14, 15)], dtype=float)
H_VALS = np.array([i * 0.4 for i in range(-10, 11)], dtype=float)


def build_xxz(delta: float, h: float, J: float = J_DEFAULT):
    """(H1, H2) of  H = J( sum_<ij>[XX + YY + Delta ZZ] + h sum_i Z )."""
    H2 = J * (np.kron(_SX, _SX) + np.kron(_SY, _SY) + delta * np.kron(_SZ, _SZ))
    H1 = J * h * _SZ
    return H1, H2


def _xxz_model(J: float = J_DEFAULT) -> MagicModel:
    return MagicModel(
        name='xxz',
        p1_name='Delta', p2_name='h',
        p1_vals=DELTA_VALS, p2_vals=H_VALS,
        build=lambda d, h, _J=J: build_xxz(d, h, _J),
        consts={'J': J},
    )


MODELS: dict[str, MagicModel] = {'xxz': _xxz_model()}


# ---------------------------------------------------------------------------
#  Trotter step
# ---------------------------------------------------------------------------
def point_dt(model: MagicModel, p1: float, p2: float,
             base: float = DT_BASE) -> float:
    """choose_dt at one grid point (no jump operators: a closed system).

    dt = base / ||H||_1 with the Pauli 1-norm of the raw H1 and H2 terms, i.e.
    trotter_lindbladian_scan.choose_dt's `2b` step.
    """
    H1, H2 = model.build(p1, p2)
    return float(choose_dt(H1, H2, (), (), base=base))


_MIN_DT_CACHE: dict[tuple[str, float], float] = {}


def min_dt_over_grid(model: MagicModel, base: float = DT_BASE) -> float:
    """`2a` step: the minimum of choose_dt over the whole parameter grid.

    Cheap (a Pauli 1-norm per point) and deterministic, so every array task
    recomputes it instead of depending on a shared file.
    """
    key = (model.name, base)
    if key not in _MIN_DT_CACHE:
        vals = [point_dt(model, float(p1), float(p2), base)
                for p1 in model.p1_vals for p2 in model.p2_vals]
        _MIN_DT_CACHE[key] = float(np.min(vals))
    return _MIN_DT_CACHE[key]


def dt_ladder(dt_point: float) -> np.ndarray:
    """The ten steps 0.1 i * dt_point, i = 1..10, used for the dt -> 0 limit."""
    return DT_FRACS * dt_point


# ---------------------------------------------------------------------------
#  (1) stabilizer-3 framability of the bond Trotter gate, extrapolated to dt=0
# ---------------------------------------------------------------------------
def framability_ladder(model: MagicModel, p1: float, p2: float, dim: int, *,
                       dts: np.ndarray | None = None,
                       verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(dt values, stabilizer-3 framability) of the D = `dim` bond gate."""
    H1, H2 = model.build(p1, p2)
    dts = dt_ladder(point_dt(model, p1, p2)) if dts is None else np.asarray(dts, float)
    vals = np.empty(dts.size, dtype=float)
    for i, dt in enumerate(dts):
        gate = bond_trotter_gate(H1, H2, (), (), dim, float(dt))
        vals[i] = stabilizer_3_framability(gate)
        if verbose:
            print(f'    D={dim} dt={dt:.6g}  fra={vals[i]:.9f}', flush=True)
    return dts, vals


def framability_limit(dts: np.ndarray, vals: np.ndarray, *,
                      fit_n: int = FIT_N_DEFAULT, deg: int = DEG_DEFAULT,
                      raw: bool = False) -> float:
    """dt -> 0 limit of framability**(1/dt) (raw=True: of framability itself)."""
    return float(extrapolate_to_zero(dts, vals, fit_n=fit_n, deg=deg, raw=raw))


def framability_rate(dts: np.ndarray, vals: np.ndarray, *,
                     fit_n: int = FIT_N_DEFAULT, deg: int = DEG_DEFAULT) -> float:
    """dt = 0 intercept of ln(framability)/dt, i.e. log(framability_limit).

    extrapolate_to_zero returns exp() of this number, which overflows float64
    once the rate passes ~709 -- reachable here because the ladder sits at
    dt ~ 1e-4..1e-3 (trotter_rom_state guards the same overflow with its
    log10_stab_fra_pow).  The intercept is therefore also fitted directly, with
    exactly the same fit, so the map stays finite and plottable everywhere.
    """
    dts = np.asarray(dts, float)
    vals = np.asarray(vals, float)
    m = np.isfinite(dts) & np.isfinite(vals) & (dts > 0)
    dt, val = dts[m], vals[m]
    if dt.size < 2:
        return float('nan')
    order = np.argsort(dt)                      # ascending dt: nearest dt=0 first
    dt, val = dt[order], val[order]
    k = int(min(fit_n, dt.size))
    d = int(min(deg, k - 1))
    rate = np.log(np.maximum(val[:k], 1.0)) / dt[:k]     # framability is >= 1
    return float(np.polyfit(dt[:k], rate, d)[-1])


# ---------------------------------------------------------------------------
#  (2) non-cliffordness of the exact 2x2-lattice propagator
# ---------------------------------------------------------------------------
def lattice_hamiltonian(model: MagicModel, p1: float, p2: float,
                        Lx: int = NC_LX, Ly: int = NC_LY) -> np.ndarray:
    """(2^N, 2^N) Hamiltonian of the Lx x Ly open-boundary lattice, N = Lx Ly.

    H1 on every site and H2 on every nearest-neighbour bond, at full coupling
    strength -- the Hamiltonian counterpart of
    trotter_lindbladian_scan.build_full_lindbladian_model.
    """
    H1, H2 = model.build(p1, p2)
    N = Lx * Ly
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    if H1 is not None:
        H1 = np.asarray(H1, dtype=complex)
        for s in range(N):
            H = H + _site_op(H1, s, N)
    if H2 is not None:
        H2 = np.asarray(H2, dtype=complex)
        for (i, j) in bonds_2d(Lx, Ly):
            H = H + _embed_two_site(H2, i, j, N)
    return H


def hamiltonian_gaps(evals: np.ndarray, tol: float = GAP_TOL) -> tuple[float, float]:
    """(gap to the first DISTINCT level, gap to the literal second eigenvalue).

    The first is E_1 - E_0 with E_1 the smallest eigenvalue that differs from
    the ground energy by more than `tol`; it is NaN if the spectrum is
    completely degenerate.  The second is evals[1] - evals[0], which is 0 for a
    degenerate ground state.
    """
    e = np.sort(np.asarray(evals, dtype=float))
    gap_next = float(e[1] - e[0]) if e.size > 1 else float('nan')
    above = e[e > e[0] + tol]
    gap = float(above[0] - e[0]) if above.size else float('nan')
    return gap, gap_next


def propagator(evals: np.ndarray, evecs: np.ndarray, t: float) -> np.ndarray:
    """expm(EXP_SIGN i H t) from an eigendecomposition of the Hermitian H."""
    phases = np.exp(EXP_SIGN * 1j * np.asarray(evals, dtype=float) * float(t))
    return (evecs * phases) @ evecs.conj().T


def noncliffordness_at_times(H: np.ndarray, times: dict[str, float], *,
                             log_base: float | None = None,
                             verbose: bool = False) -> tuple[dict[str, float], dict]:
    """Non-cliffordness of exp(i H t) for several t, from one diagonalization.

    Returns ({key: M_2}, diagnostics) with diagnostics carrying the gaps.  A
    non-finite time (e.g. 100/gap with a degenerate ground state) yields NaN.
    """
    evals, evecs = np.linalg.eigh(np.asarray(H, dtype=complex))
    gap, gap_next = hamiltonian_gaps(evals)
    out: dict[str, float] = {}
    for key, t in times.items():
        if not np.isfinite(t):
            out[key] = float('nan')
            continue
        U = propagator(evals, evecs, float(t))
        out[key] = float(noncliffordness(U, log_base=log_base))
        if verbose:
            print(f'    {key}: t={t:.6g}  NC={out[key]:.9f}', flush=True)
    return out, {'gap': gap, 'gap_next': gap_next,
                 'e_min': float(evals.min()), 'e_max': float(evals.max())}


def times_for_point(dt_pt: float, dt_min: float, gap: float) -> dict[str, float]:
    """The five evolution times of tasks 2a .. 2e."""
    t_gap = GAP_FACTOR / gap if (np.isfinite(gap) and gap > 0) else float('nan')
    return {
        'nc_2a': dt_min,
        'nc_2b': dt_pt,
        'nc_2c': T_LONG_FACTOR * dt_min,
        'nc_2d': T_LONG_FACTOR * dt_pt,
        'nc_2e': t_gap,
    }


# ---------------------------------------------------------------------------
#  One grid point: everything
# ---------------------------------------------------------------------------
def compute_point(model: MagicModel, p1: float, p2: float, *,
                  dt_min: float | None = None,
                  groups: tuple[str, ...] = ('fra', 'nc'),
                  dims: tuple[int, ...] = DIMS,
                  fit_n: int = FIT_N_DEFAULT, deg: int = DEG_DEFAULT,
                  verbose: bool = False) -> dict:
    """All scan quantities at one (p1, p2) point.

    The returned dict holds the seven TASK_KEYS, the raw framability ladders
    (`fra_dts`, `fra_raw_D1`, `fra_raw_D2`) so the dt -> 0 fit can be redone at
    collect time without recomputing any LP, the overflow-safe log of the
    framability limits (`fra_rate_D1`, `fra_rate_D2`), the five evolution
    times, the two gaps, and both log conventions for the non-cliffordness
    (`*_bits`).
    """
    dt_pt = point_dt(model, p1, p2)
    dt_min = min_dt_over_grid(model) if dt_min is None else float(dt_min)
    out: dict = {'p1': p1, 'p2': p2, 'dt_pt': dt_pt, 'dt_min': dt_min}

    if 'fra' in groups:
        dts = dt_ladder(dt_pt)
        out['fra_dts'] = dts
        for dim in dims:
            _, vals = framability_ladder(model, p1, p2, dim, dts=dts, verbose=verbose)
            out[f'fra_raw_D{dim}'] = vals
            out[f'fra_D{dim}'] = framability_limit(dts, vals, fit_n=fit_n, deg=deg)
            out[f'fra_rate_D{dim}'] = framability_rate(dts, vals, fit_n=fit_n, deg=deg)
            out[f'fra_raw_lim_D{dim}'] = framability_limit(
                dts, vals, fit_n=fit_n, deg=deg, raw=True)

    if 'nc' in groups:
        H = lattice_hamiltonian(model, p1, p2)
        evals = np.linalg.eigvalsh(H)
        gap, gap_next = hamiltonian_gaps(evals)
        times = times_for_point(dt_pt, dt_min, gap)
        nc, diag = noncliffordness_at_times(H, times, verbose=verbose)
        out.update(nc)
        out.update({f'{k}_bits': v / np.log(2.0) for k, v in nc.items()})
        out.update({f't_{k[-2:]}': v for k, v in times.items()})
        out.update(diag)

    return out


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------
def self_test() -> None:
    model = MODELS['xxz']

    # 1. Grid.
    assert model.shape == (29, 21), model.shape
    assert abs(model.p1_vals[0] + 2.8) < 1e-12 and abs(model.p1_vals[-1] - 2.8) < 1e-12
    assert abs(model.p2_vals[0] + 4.0) < 1e-12 and abs(model.p2_vals[-1] - 4.0) < 1e-12
    assert model.n_points == 609

    # 2. Hermiticity and the Pauli-1-norm step.
    H1, H2 = model.build(0.6, 1.2)
    assert np.allclose(H2, H2.conj().T) and np.allclose(H1, H1.conj().T)
    # ||H||_1 = |J|(XX) + |J|(YY) + |J Delta|(ZZ) + |J h|(Z)
    expect = DT_BASE / (1 + 1 + 0.6 + 1.2)
    assert abs(point_dt(model, 0.6, 1.2) - expect) < 1e-15, point_dt(model, 0.6, 1.2)
    dt_min = min_dt_over_grid(model)
    assert abs(dt_min - DT_BASE / (1 + 1 + 2.8 + 4.0)) < 1e-15, dt_min

    # 3. dt ladder.
    dts = dt_ladder(point_dt(model, 0.6, 1.2))
    assert dts.size == 10 and abs(dts[-1] - expect) < 1e-15
    assert abs(dts[0] - 0.1 * expect) < 1e-15

    # 3b. framability_rate is exactly log(framability_limit) on a clean ladder,
    #     and stays finite where the limit itself overflows float64.
    dts_t = dt_ladder(0.002)
    for rate_true in (0.0, 3.7, 900.0):
        vals_t = np.exp(rate_true * dts_t)             # framability = e^{rate dt}
        assert abs(framability_rate(dts_t, vals_t) - rate_true) < 1e-6 * max(rate_true, 1.0)
    assert abs(framability_limit(dts_t, np.exp(3.7 * dts_t)) - np.exp(3.7)) < 1e-6
    assert not np.isfinite(framability_limit(dts_t, np.exp(900.0 * dts_t)))
    assert np.isfinite(framability_rate(dts_t, np.exp(900.0 * dts_t)))

    # 4. Bond gate: the 1/(2D) share of the field is the only D dependence.
    g1 = bond_trotter_gate(H1, H2, (), (), 1, 0.01)
    g2 = bond_trotter_gate(H1, H2, (), (), 2, 0.01)
    assert g1.shape == (16, 16) and not np.allclose(g1, g2)
    # D = 1 at field h equals D = 2 at field 2h.
    H1b, H2b = model.build(0.6, 2.4)
    assert np.allclose(g1, bond_trotter_gate(H1b, H2b, (), (), 2, 0.01), atol=1e-12)

    # 5. Lattice Hamiltonian: 4 sites, 4 bonds on the 2x2 plaquette.
    H = lattice_hamiltonian(model, 0.6, 1.2)
    assert H.shape == (16, 16) and np.allclose(H, H.conj().T)
    assert len(bonds_2d(NC_LX, NC_LY)) == 4
    # trace = 0 for any traceless-term model; the XXZ terms are all traceless.
    assert abs(np.trace(H)) < 1e-10
    # h -> -h is a spin flip (X^{(x)4} conjugation) of the Delta-preserving part.
    flip = _site_op(_SX, 0, 4) @ _site_op(_SX, 1, 4) @ _site_op(_SX, 2, 4) @ _site_op(_SX, 3, 4)
    assert np.allclose(flip @ H @ flip, lattice_hamiltonian(model, 0.6, -1.2), atol=1e-10)

    # 6. Propagator and gaps.
    evals, evecs = np.linalg.eigh(H)
    U = propagator(evals, evecs, 0.37)
    assert np.allclose(U.conj().T @ U, np.eye(16), atol=1e-12)
    from scipy.linalg import expm
    assert np.allclose(U, expm(EXP_SIGN * 1j * H * 0.37), atol=1e-10)
    gap, gap_next = hamiltonian_gaps(evals)
    assert gap > 0 and gap_next <= gap + 1e-12

    # 7. Times.
    t = times_for_point(0.002, 0.001, 0.5)
    assert t['nc_2a'] == 0.001 and t['nc_2b'] == 0.002
    assert t['nc_2c'] == 100.0 and t['nc_2d'] == 200.0 and t['nc_2e'] == 200.0
    assert np.isnan(times_for_point(0.002, 0.001, float('nan'))['nc_2e'])

    # 8. Non-cliffordness at t = 0 is 0 (identity is Clifford).
    try:
        nc, diag = noncliffordness_at_times(H, {'nc_2a': 0.0})
        assert nc['nc_2a'] < 1e-9, nc
        assert np.isfinite(diag['gap'])
        print(f'non-cliffordness path OK (gap = {diag["gap"]:.6f})')
    except RuntimeError as exc:                          # no RoM-handbook here
        print(f'[skip] non-cliffordness path: {exc}')

    print('xxz_magic_scan self-test passed.')


def main() -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--self_test', action='store_true')
    ap.add_argument('--model', default='xxz', choices=list(MODELS))
    ap.add_argument('--p1', type=float, help='first parameter (Delta)')
    ap.add_argument('--p2', type=float, help='second parameter (h)')
    ap.add_argument('--groups', default='fra,nc',
                    help="comma list of 'fra' and/or 'nc'")
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return
    if a.p1 is None or a.p2 is None:
        ap.error('--p1 and --p2 are required (or pass --self_test)')

    model = MODELS[a.model]
    t0 = time.perf_counter()
    res = compute_point(model, a.p1, a.p2,
                        groups=tuple(g.strip() for g in a.groups.split(',') if g.strip()),
                        verbose=a.verbose)
    print(f'{model.name}  {model.p1_name}={a.p1}  {model.p2_name}={a.p2}')
    for k in TASK_KEYS:
        if k in res:
            print(f'  {k:8s} = {res[k]:.9f}')
    print(f'  dt_pt = {res["dt_pt"]:.6g}   dt_min = {res["dt_min"]:.6g}')
    if 'gap' in res:
        print(f'  gap = {res["gap"]:.6g}   gap_next = {res["gap_next"]:.6g}')
    print(f'  ({time.perf_counter() - t0:.1f}s)')


if __name__ == '__main__':
    main()
