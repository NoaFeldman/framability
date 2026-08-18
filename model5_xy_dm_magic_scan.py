"""Magic / framability scan of the XY chain with a Dzyaloshinskii-Moriya
interaction in a transverse field, over a 2D parameter grid of DM strength
and field.

    H = J * sum_<ij> [ X_i X_j + Y_i Y_j ]
      + D * sum_<ij> [ X_i Y_j - Y_i X_j ]
      + h * sum_i Z_i

with J = 1 and the scan axes

    D in linspace(0, 2, 20)
    h in linspace(0, 2, 20)

The bond term is the isotropic XY exchange plus a DM coupling of strength D
along the bond direction fixed by bonds_ring's nearest-neighbour ordering
(every bond carries the same chirality).  The DM term breaks parity and
competes with the field-driven polarized phase at large h, producing chiral
analogues of the XY model's critical line.  See unitary_models_for_magic.tex
(Model 5).

This is a thin model definition on top of magic_scan_common, which carries
every quantity downstream of (H1, H2) unchanged from xxz_magic_scan.py: the
Trotter dt -> 0 stabilizer-3 framability (fra_D1 only) and the exact
Choi-state non-cliffordness of exp(i H_n t) on a periodic-boundary (ring)
chain of n qubits, n in RING_NS = (4, 5, 6), for t = dt_min * 10**k,
k = 0 .. 5 (nc_n{n}_t{k}, spanning [dt_min, 1e5 dt_min]).  See
magic_scan_common.py and xxz_magic_scan.py for the full quantity definitions.

Cluster pipeline (generic, shared with the other 4 models)
------------------------------------------------------------
    scripts/magic_worker.py           per-point array worker (all quantities)
    scripts/magic_scan.slurm.sh       200-task data array
    scripts/magic_collect.py          aggregation + the colormaps
    scripts/magic_collect_all.slurm.sh dependent plotting job (all 5 models)
    scripts/submit_unitary_magic.sh   submits all 5 data arrays + the collect job

Usage (local, one point):
    python model5_xy_dm_magic_scan.py --self_test
    python model5_xy_dm_magic_scan.py --p1 0.5 --p2 0.9
"""

from __future__ import annotations

import numpy as np

from dissipative_PT import _SX, _SY, _SZ, _site_op
from magic_scan_common import (
    MagicModel, TASK_KEYS, RING_NS, T_LONG_FACTOR,
    point_dt, min_dt_over_grid, dt_ladder, bond_trotter_gate,
    ring_hamiltonian, bonds_ring, hamiltonian_gaps, propagator,
    noncliffordness_at_times, times_for_ring, compute_point,
)

# Version stamp for cached results; bump when any stored quantity changes.
MAGIC_VERSION = '2.0'

# --- the XY chain with a DM interaction in a transverse field ----------------
J_DEFAULT = 1.0
D_VALS = np.linspace(0.0, 2.0, 20)
H_VALS = np.linspace(0.0, 2.0, 20)


def build_xy_dm(D: float, h: float, J: float = J_DEFAULT):
    """(H1, H2) of  H = J sum_<ij>[XX + YY] + D sum_<ij>[XY - YX] + h sum_i Z."""
    H2 = J * (np.kron(_SX, _SX) + np.kron(_SY, _SY)) \
        + D * (np.kron(_SX, _SY) - np.kron(_SY, _SX))
    H1 = h * _SZ
    return H1, H2


def _model(J: float = J_DEFAULT) -> MagicModel:
    return MagicModel(
        name='xy_dm',
        p1_name='D', p2_name='h',
        p1_vals=D_VALS, p2_vals=H_VALS,
        build=lambda D, h, _J=J: build_xy_dm(D, h, _J),
        consts={'J': J},
    )


MODELS: dict[str, MagicModel] = {'xy_dm': _model()}


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------
def self_test() -> None:
    model = MODELS['xy_dm']

    # 1. Grid.
    assert model.shape == (20, 20), model.shape
    assert abs(model.p1_vals[0] - 0.0) < 1e-12 and abs(model.p1_vals[-1] - 2.0) < 1e-12
    assert abs(model.p2_vals[0] - 0.0) < 1e-12 and abs(model.p2_vals[-1] - 2.0) < 1e-12
    assert model.n_points == 400

    # 2. Hermiticity (the DM term X(x)Y - Y(x)X is Hermitian since X(x)Y and
    #    Y(x)X are each Hermitian) and the Pauli-1-norm step.
    H1, H2 = model.build(0.5, 0.9)
    assert np.allclose(H2, H2.conj().T) and np.allclose(H1, H1.conj().T)
    # ||H||_1 = |J|(XX) + |J|(YY) + |D|(XY) + |D|(YX) + |h|(Z) = 2|J| + 2|D| + |h|
    from trotter_lindbladian_scan import DT_BASE
    expect = DT_BASE / (2.0 * 1.0 + 2.0 * 0.5 + 0.9)
    assert abs(point_dt(model, 0.5, 0.9) - expect) < 1e-15, point_dt(model, 0.5, 0.9)
    dt_min = min_dt_over_grid(model)
    assert abs(dt_min - DT_BASE / (2.0 * 1.0 + 2.0 * 2.0 + 2.0)) < 1e-15, dt_min

    # 3. dt ladder.
    dts = dt_ladder(point_dt(model, 0.5, 0.9))
    assert dts.size == 10 and abs(dts[-1] - expect) < 1e-15

    # 4. Bond gate: the 1/(2D) share of the field is the only spatial-D dependence.
    g1 = bond_trotter_gate(H1, H2, (), (), 1, 0.01)
    g2 = bond_trotter_gate(H1, H2, (), (), 2, 0.01)
    assert g1.shape == (16, 16) and not np.allclose(g1, g2)

    # 5. Ring Hamiltonian, n = 4: 4 sites, 4 bonds (the ring C4 is isomorphic to
    #    the old 2x2 open-boundary plaquette, so the shape/trace checks carry
    #    over unchanged).
    H = ring_hamiltonian(model, 0.5, 0.9, 4)
    assert H.shape == (16, 16) and np.allclose(H, H.conj().T)
    assert len(bonds_ring(4)) == 4
    # trace = 0: Z, XX, YY, XY, YX are all traceless.
    assert abs(np.trace(H)) < 1e-10
    # D = 0 recovers the plain XY-in-transverse-field bond term.
    _, H2_xy = model.build(0.0, 0.9)
    assert np.allclose(H2_xy, np.kron(_SX, _SX) + np.kron(_SY, _SY), atol=1e-12)
    # X^{(x)4} conjugation leaves XX and YY invariant, flips the sign of Z (the
    # field) AND of the antisymmetric XY - YX (DM) term, so it maps (D, h) to
    # (-D, -h), not to (D, -h).
    flip = _site_op(_SX, 0, 4) @ _site_op(_SX, 1, 4) @ _site_op(_SX, 2, 4) @ _site_op(_SX, 3, 4)
    assert np.allclose(flip @ H @ flip, ring_hamiltonian(model, -0.5, -0.9, 4), atol=1e-10)
    # Other ring sizes: right shape, Hermitian.
    for n in (5, 6):
        Hn = ring_hamiltonian(model, 0.5, 0.9, n)
        assert Hn.shape == (2 ** n, 2 ** n) and np.allclose(Hn, Hn.conj().T)
        assert len(bonds_ring(n)) == n

    # 6. Propagator and gaps.
    evals, evecs = np.linalg.eigh(H)
    U = propagator(evals, evecs, 0.37)
    assert np.allclose(U.conj().T @ U, np.eye(16), atol=1e-12)
    from scipy.linalg import expm
    assert np.allclose(U, expm(1j * H * 0.37), atol=1e-10)
    gap, gap_next = hamiltonian_gaps(evals)
    assert not np.isfinite(gap) or (gap > 0 and gap_next <= gap + 1e-12)

    # 7. Times: t_k = dt_min * 10**k spans exactly [dt_min, 1e5 dt_min].
    t = times_for_ring(0.001)
    assert len(t) == 6
    assert abs(t['t0'] - 0.001) < 1e-15
    assert abs(t['t5'] - 100.0) < 1e-9
    assert abs(t['t5'] / t['t0'] - T_LONG_FACTOR) < 1e-6

    # 8. Non-cliffordness at t = 0 is 0 (identity is Clifford).
    try:
        nc, diag = noncliffordness_at_times(H, {'nc_n4_t0': 0.0})
        assert nc['nc_n4_t0'] < 1e-9, nc
        assert np.isfinite(diag['gap'])
        print(f'non-cliffordness path OK (gap = {diag["gap"]:.6f})')
    except RuntimeError as exc:                          # no RoM-handbook here
        print(f'[skip] non-cliffordness path: {exc}')

    print('model5_xy_dm_magic_scan self-test passed.')


def main() -> None:
    import argparse
    import time

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--self_test', action='store_true')
    ap.add_argument('--model', default='xy_dm', choices=list(MODELS))
    ap.add_argument('--p1', type=float, help='first parameter (D)')
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
    for n in RING_NS:
        if f'gap_n{n}' in res:
            print(f'  n={n}: gap = {res[f"gap_n{n}"]:.6g}   '
                  f'gap_next = {res[f"gap_next_n{n}"]:.6g}')
    print(f'  ({time.perf_counter() - t0:.1f}s)')


if __name__ == '__main__':
    main()
