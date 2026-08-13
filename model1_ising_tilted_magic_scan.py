"""Magic / framability scan of the Ising chain in a tilted field, over a 2D
parameter grid of the transverse and longitudinal field components.

    H = J * sum_<ij> Z_i Z_j  +  h_x * sum_i X_i  +  h_z * sum_i Z_i

with J = 1 and the scan axes

    h_x in linspace(0, 2, 20)
    h_z in linspace(-1, 1, 20)

h_x drives the paramagnet-ferromagnet quantum critical point at h_x = J; h_z
confines the domain-wall excitations and breaks integrability away from the
h_z = 0 line.  See unitary_models_for_magic.tex (Model 1).

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
    python model1_ising_tilted_magic_scan.py --self_test
    python model1_ising_tilted_magic_scan.py --p1 1.0 --p2 0.3
"""

from __future__ import annotations

import numpy as np

from dissipative_PT import _SX, _SZ, _site_op, bonds_2d
from magic_scan_common import (
    MagicModel, TASK_KEYS,
    point_dt, min_dt_over_grid, dt_ladder, bond_trotter_gate,
    lattice_hamiltonian, hamiltonian_gaps, propagator,
    noncliffordness_at_times, compute_point,
)

# Version stamp for cached results; bump when any stored quantity changes.
MAGIC_VERSION = '1.0'

# --- the Ising-in-a-tilted-field model ---------------------------------------
J_DEFAULT = 1.0
HX_VALS = np.linspace(0.0, 2.0, 20)
HZ_VALS = np.linspace(-1.0, 1.0, 20)


def build_ising_tilted(hx: float, hz: float, J: float = J_DEFAULT):
    """(H1, H2) of  H = J sum_<ij> ZZ  +  hx sum_i X  +  hz sum_i Z."""
    H2 = J * np.kron(_SZ, _SZ)
    H1 = hx * _SX + hz * _SZ
    return H1, H2


def _model(J: float = J_DEFAULT) -> MagicModel:
    return MagicModel(
        name='ising_tilted',
        p1_name='hx', p2_name='hz',
        p1_vals=HX_VALS, p2_vals=HZ_VALS,
        build=lambda hx, hz, _J=J: build_ising_tilted(hx, hz, _J),
        consts={'J': J},
    )


MODELS: dict[str, MagicModel] = {'ising_tilted': _model()}


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------
def self_test() -> None:
    model = MODELS['ising_tilted']

    # 1. Grid.
    assert model.shape == (20, 20), model.shape
    assert abs(model.p1_vals[0] - 0.0) < 1e-12 and abs(model.p1_vals[-1] - 2.0) < 1e-12
    assert abs(model.p2_vals[0] + 1.0) < 1e-12 and abs(model.p2_vals[-1] - 1.0) < 1e-12
    assert model.n_points == 400

    # 2. Hermiticity and the Pauli-1-norm step.
    H1, H2 = model.build(0.7, 0.3)
    assert np.allclose(H2, H2.conj().T) and np.allclose(H1, H1.conj().T)
    # ||H||_1 = |J|(ZZ) + |hx|(X) + |hz|(Z)
    from trotter_lindbladian_scan import DT_BASE
    expect = DT_BASE / (1.0 + 0.7 + 0.3)
    assert abs(point_dt(model, 0.7, 0.3) - expect) < 1e-15, point_dt(model, 0.7, 0.3)
    dt_min = min_dt_over_grid(model)
    assert abs(dt_min - DT_BASE / (1.0 + 2.0 + 1.0)) < 1e-15, dt_min

    # 3. dt ladder.
    dts = dt_ladder(point_dt(model, 0.7, 0.3))
    assert dts.size == 10 and abs(dts[-1] - expect) < 1e-15

    # 4. Bond gate: the 1/(2D) share of the field is the only D dependence.
    g1 = bond_trotter_gate(H1, H2, (), (), 1, 0.01)
    g2 = bond_trotter_gate(H1, H2, (), (), 2, 0.01)
    assert g1.shape == (16, 16) and not np.allclose(g1, g2)

    # 5. Lattice Hamiltonian: 4 sites, 4 bonds on the 2x2 plaquette, Hermitian.
    H = lattice_hamiltonian(model, 0.7, 0.3)
    assert H.shape == (16, 16) and np.allclose(H, H.conj().T)
    assert len(bonds_2d(2, 2)) == 4
    # trace = 0: Z, ZZ, X are all traceless.
    assert abs(np.trace(H)) < 1e-10
    # hz -> -hz, hx -> -hx is a spin flip (X^{(x)4} conjugation).
    flip = _site_op(_SX, 0, 4) @ _site_op(_SX, 1, 4) @ _site_op(_SX, 2, 4) @ _site_op(_SX, 3, 4)
    assert np.allclose(flip @ H @ flip, lattice_hamiltonian(model, 0.7, -0.3), atol=1e-10)

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

    print('model1_ising_tilted_magic_scan self-test passed.')


def main() -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--self_test', action='store_true')
    ap.add_argument('--model', default='ising_tilted', choices=list(MODELS))
    ap.add_argument('--p1', type=float, help='first parameter (hx)')
    ap.add_argument('--p2', type=float, help='second parameter (hz)')
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
