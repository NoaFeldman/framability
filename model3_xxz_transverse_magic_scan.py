"""Magic / framability scan of the XXZ chain in a transverse field, over a 2D
parameter grid of exchange anisotropy and field.

    H = J * sum_<ij> [ X_i X_j + Y_i Y_j + Delta Z_i Z_j ]  +  h * sum_i X_i

with J = 1 and the scan axes

    Delta in linspace(-2, 2, 20)
    h     in linspace(0, 2, 20)

Delta alone (h = 0) gives the standard XXZ phase diagram (gapless XY phase
for |Delta| < 1, gapped Neel/ferromagnetic order for |Delta| > 1); the
transverse field h does not commute with the XXZ chain's U(1) symmetry, so it
breaks integrability and competes with the Delta-driven order.  See
unitary_models_for_magic.tex (Model 3).  Compare with xxz_magic_scan.py's
own model, which instead applies the field longitudinally (h sum_i Z).

This is a thin model definition on top of magic_scan_common, which carries
every quantity downstream of (H1, H2) unchanged from xxz_magic_scan.py: the
Trotter dt -> 0 stabilizer-3 framability (fra_D1, fra_D2) and the exact
Choi-state non-cliffordness (nc_2a .. nc_2e).  See magic_scan_common.py and
xxz_magic_scan.py for the full quantity definitions.

Cluster pipeline (generic, shared with the other 4 models)
------------------------------------------------------------
    scripts/magic_worker.py           per-point array worker (all quantities)
    scripts/magic_scan.slurm.sh       200-task data array
    scripts/magic_collect.py          aggregation + the seven colormaps
    scripts/magic_collect_all.slurm.sh dependent plotting job (all 5 models)
    scripts/submit_unitary_magic.sh   submits all 5 data arrays + the collect job

Usage (local, one point):
    python model3_xxz_transverse_magic_scan.py --self_test
    python model3_xxz_transverse_magic_scan.py --p1 1.3 --p2 0.6
"""

from __future__ import annotations

import numpy as np

from dissipative_PT import _SX, _SY, _SZ, _site_op, bonds_2d
from magic_scan_common import (
    MagicModel, TASK_KEYS,
    point_dt, min_dt_over_grid, dt_ladder, bond_trotter_gate,
    lattice_hamiltonian, hamiltonian_gaps, propagator,
    noncliffordness_at_times, compute_point,
)

# Version stamp for cached results; bump when any stored quantity changes.
MAGIC_VERSION = '1.0'

# --- the XXZ chain in a transverse field --------------------------------------
J_DEFAULT = 1.0
DELTA_VALS = np.linspace(-2.0, 2.0, 20)
H_VALS = np.linspace(0.0, 2.0, 20)


def build_xxz_transverse(delta: float, h: float, J: float = J_DEFAULT):
    """(H1, H2) of  H = J sum_<ij>[XX + YY + Delta ZZ] + h sum_i X."""
    H2 = J * (np.kron(_SX, _SX) + np.kron(_SY, _SY) + delta * np.kron(_SZ, _SZ))
    H1 = h * _SX
    return H1, H2


def _model(J: float = J_DEFAULT) -> MagicModel:
    return MagicModel(
        name='xxz_transverse',
        p1_name='Delta', p2_name='h',
        p1_vals=DELTA_VALS, p2_vals=H_VALS,
        build=lambda d, h, _J=J: build_xxz_transverse(d, h, _J),
        consts={'J': J},
    )


MODELS: dict[str, MagicModel] = {'xxz_transverse': _model()}


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------
def self_test() -> None:
    model = MODELS['xxz_transverse']

    # 1. Grid.
    assert model.shape == (20, 20), model.shape
    assert abs(model.p1_vals[0] + 2.0) < 1e-12 and abs(model.p1_vals[-1] - 2.0) < 1e-12
    assert abs(model.p2_vals[0] - 0.0) < 1e-12 and abs(model.p2_vals[-1] - 2.0) < 1e-12
    assert model.n_points == 400

    # 2. Hermiticity and the Pauli-1-norm step.
    H1, H2 = model.build(1.3, 0.6)
    assert np.allclose(H2, H2.conj().T) and np.allclose(H1, H1.conj().T)
    # ||H||_1 = |J|(XX) + |J|(YY) + |J Delta|(ZZ) + |h|(X)
    from trotter_lindbladian_scan import DT_BASE
    expect = DT_BASE / (1 + 1 + 1.3 + 0.6)
    assert abs(point_dt(model, 1.3, 0.6) - expect) < 1e-15, point_dt(model, 1.3, 0.6)
    dt_min = min_dt_over_grid(model)
    assert abs(dt_min - DT_BASE / (1 + 1 + 2.0 + 2.0)) < 1e-15, dt_min

    # 3. dt ladder.
    dts = dt_ladder(point_dt(model, 1.3, 0.6))
    assert dts.size == 10 and abs(dts[-1] - expect) < 1e-15

    # 4. Bond gate: the 1/(2D) share of the field is the only D dependence.
    g1 = bond_trotter_gate(H1, H2, (), (), 1, 0.01)
    g2 = bond_trotter_gate(H1, H2, (), (), 2, 0.01)
    assert g1.shape == (16, 16) and not np.allclose(g1, g2)

    # 5. Lattice Hamiltonian: 4 sites, 4 bonds on the 2x2 plaquette, Hermitian.
    H = lattice_hamiltonian(model, 1.3, 0.6)
    assert H.shape == (16, 16) and np.allclose(H, H.conj().T)
    assert len(bonds_2d(2, 2)) == 4
    # trace = 0: X, XX, YY, ZZ are all traceless.
    assert abs(np.trace(H)) < 1e-10
    # h -> -h is a spin flip (Z^{(x)4} conjugation) of the Delta-preserving part.
    flip = _site_op(_SZ, 0, 4) @ _site_op(_SZ, 1, 4) @ _site_op(_SZ, 2, 4) @ _site_op(_SZ, 3, 4)
    assert np.allclose(flip @ H @ flip, lattice_hamiltonian(model, 1.3, -0.6), atol=1e-10)

    # 6. Propagator and gaps.
    evals, evecs = np.linalg.eigh(H)
    U = propagator(evals, evecs, 0.37)
    assert np.allclose(U.conj().T @ U, np.eye(16), atol=1e-12)
    from scipy.linalg import expm
    assert np.allclose(U, expm(1j * H * 0.37), atol=1e-10)
    gap, gap_next = hamiltonian_gaps(evals)
    assert gap > 0 and gap_next <= gap + 1e-12

    # 7. Non-cliffordness at t = 0 is 0 (identity is Clifford).
    try:
        nc, diag = noncliffordness_at_times(H, {'nc_2a': 0.0})
        assert nc['nc_2a'] < 1e-9, nc
        assert np.isfinite(diag['gap'])
        print(f'non-cliffordness path OK (gap = {diag["gap"]:.6f})')
    except RuntimeError as exc:                          # no RoM-handbook here
        print(f'[skip] non-cliffordness path: {exc}')

    print('model3_xxz_transverse_magic_scan self-test passed.')


def main() -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--self_test', action='store_true')
    ap.add_argument('--model', default='xxz_transverse', choices=list(MODELS))
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
