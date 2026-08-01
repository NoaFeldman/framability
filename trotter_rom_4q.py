"""
Sub-pipeline of trotter_lindbladian_scan: stabilizer-3 framability of the
two-qubit bond Trotter gate vs the Choi-state Robustness of Magic (RoM) of the
matching 4-qubit (2x2-lattice) Trotter gate.

    NOTE -- two RoM sub-pipelines exist; pick the right one.

    trotter_rom_4q (this module)   RoM of the 4-qubit GATE, via its 8-qubit
                                   Choi state.  Needs column generation, the
                                   compiled C++ enumerator and Gurobi; hours
                                   per point, which is why it only ever ran on
                                   the reduced ROM_GRIDS below.

    trotter_rom_state              RoM of the 4-qubit STATE obtained by
                                   applying the lattice step once to the
                                   lpdo_max start state.  One sparse HiGHS LP
                                   over the full 4-qubit stabilizer matrix:
                                   numpy + scipy only, seconds per point, and
                                   it covers the FULL grids of models 1-6.

    The two quantities are different and neither module imports the other.

Models, bond gates and the per-point adaptive dt are exactly those of
trotter_lindbladian_scan (MODELS / build_bond_lindbladian / choose_dt).  For
one (p1, p2) point of a model this module computes:

  1. the stabilizer-3 framability of the two-qubit bond Trotter gate
     U_2 = expm(L_bond dt)  (framability.stabilizer_3_framability).  For
     models 1-5 this is already stored in results_trotter_v3 and the cluster
     worker reuses it instead of recomputing (pass stab_fra=... here).

  2. the RoM of the 4-qubit gate of the 2x2-qubit lattice.  Sites are
     row-major (0 1 / 2 3), so the four nearest-neighbour bonds are
     (0,1), (2,3), (0,2), (1,3) and the 4-qubit generator is defined as

         L_4 = L_2^(01) + L_2^(23) + L_2^(02) + L_2^(13),
         L_2^(ij) = log U_2 = L_bond dt embedded on qubits (i, j),
         U_4 = expm(L_4)                       (256x256 Pauli transfer matrix)

     i.e. the 4-qubit and two-qubit Lindbladians match in the small-dt limit
     by construction (log U_4 = sum_ij log U_2^(ij) exactly).

     RoM(U_4) := RoM of the Choi state J = (id (x) U_4)(|Omega><Omega|) of the
     channel -- an 8-qubit (generally mixed) state -- computed with the
     rom_of_gate machinery (Hamaguchi-Hamada-Yoshioka, arXiv:2311.01362;
     column generation at n_choi = 8).  For a channel with PTM R in the repo
     convention R[a, b] = Tr(P_a Lambda(P_b)) / 2^n, the Choi Pauli vector is
     closed form:

         b_(A,B) = tr(J (A (x) B)) = s_A R[B, A],   s_A = (-1)^(#Y in A),

     with A on the reference register -- the same qubit ordering and
     normalisation as rom_of_gate.choi_pauli_vector (checked by --self_test).

rom_of_gate (and hence the RoM-handbook clone) is imported lazily, so the
model / gate part of this module also works on machines without the handbook.

Usage:
    python trotter_rom_4q.py --self_test
    python trotter_rom_4q.py --model model6 --p1 0.0 --p2 0.0

Cluster pipeline: scripts/trotter_rom_worker.py (per-point array worker),
scripts/trotter_rom.slurm.sh (200-task array), scripts/trotter_rom_collect.py
(side-by-side stab-3 framability | RoM colormaps).  The pipeline runs on the
reduced ROM_GRIDS parameter ranges (2697 points total), each an exact subset
of the model's full scan grid.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import expm

from trotter_lindbladian_scan import (
    MODELS, DIM_DEFAULT, choose_dt, build_bond_lindbladian, bond_trotter_gate,
)
from framability import stabilizer_3_framability

# Version stamp for cached results; bump when any stored quantity changes.
ROM_VERSION = '1.0'

# 2x2 open-boundary lattice, sites row-major (0 1 / 2 3): the four bonds of
# L_4 = L_2^(01) + L_2^(23) + L_2^(02) + L_2^(13).
LATTICE_BONDS_2X2 = ((0, 1), (2, 3), (0, 2), (1, 3))


# ---------------------------------------------------------------------------
#  Restricted scan grids of the RoM sub-pipeline
# ---------------------------------------------------------------------------
def _restrict(vals: np.ndarray, lo: float, hi: float,
              stride: int = 1) -> np.ndarray:
    """Subset of a model grid: values in [lo, hi], optionally strided."""
    vals = np.asarray(vals)
    return vals[(vals >= lo - 1e-9) & (vals <= hi + 1e-9)][::stride]


# The RoM sub-pipeline runs on reduced parameter ranges (2026-07-17).  Every
# grid is an exact subset of the model's full scan grid in MODELS (same step,
# 0.2, for models 1-5; model6 keeps the full [0, pi/2] range but at half the
# resolution, step 2 pi 1e-2 = every 2nd point of the fine pi 1e-2 grid), so
# each point maps 1:1 onto a main-scan point and its stored stab_fra.
#
#   model1   6 x 31 = 186     model2   6 x 21 = 126
#   model3  21 x 22 = 462     model4  31 x 31 = 961
#   model5  11 x 26 = 286     model6  26 x 26 = 676     (total 2697)
ROM_GRIDS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    'model1': (_restrict(MODELS['model1'].p1_vals, 0.0, 1.0),    # h
               _restrict(MODELS['model1'].p2_vals, 0.0, 6.0)),   # gamma
    'model2': (_restrict(MODELS['model2'].p1_vals, 0.0, 1.0),    # h
               _restrict(MODELS['model2'].p2_vals, 0.0, 4.0)),   # gamma
    'model3': (_restrict(MODELS['model3'].p1_vals, 0.0, 4.0),    # gamma
               _restrict(MODELS['model3'].p2_vals, 0.0, 4.2)),   # gamma'
    'model4': (_restrict(MODELS['model4'].p1_vals, 0.0, 6.0),    # gamma
               _restrict(MODELS['model4'].p2_vals, 0.0, 6.0)),   # gamma'
    'model5': (_restrict(MODELS['model5'].p1_vals, 0.0, 2.0),    # J_y
               _restrict(MODELS['model5'].p2_vals, 0.0, 5.0)),   # gamma
    'model6': (MODELS['model6'].p1_vals[::2],                    # rho
               MODELS['model6'].p2_vals[::2]),                   # sigma
}


# ---------------------------------------------------------------------------
#  4-qubit gate from the bond generator
# ---------------------------------------------------------------------------
def embed_superop(M: np.ndarray, sites: tuple[int, int], n: int = 4) -> np.ndarray:
    """Embed a two-qubit Pauli-basis superoperator (16x16, index 4*a_i + a_j)
    onto qubits `sites` of an n-qubit register (identity on the rest), in the
    big-endian base-4 Pauli ordering of dissipative_PT._pauli_tensor."""
    i, j = sites
    rest = [q for q in range(n) if q not in (i, j)]
    perm = [i, j] + rest                       # qubit order of the kron build
    big = np.kron(np.asarray(M, dtype=float), np.eye(4 ** (n - 2)))
    big = big.reshape([4] * (2 * n))           # out axes (perm), in axes (perm)
    inv = list(np.argsort(perm))               # natural axis -> build position
    big = big.transpose(inv + [n + k for k in inv])
    return big.reshape(4 ** n, 4 ** n)


def four_qubit_gate_ptm(H1, H2, jumps1, jumps2, dim: int, dt: float) -> np.ndarray:
    """256x256 real PTM U_4 = expm(sum_bonds embed(L_bond dt)) on the 2x2
    lattice.  L_bond carries the same 1/(2*dim) one-qubit bond share as the
    two-qubit Trotter gate, so log U_4 = sum_ij log U_2^(ij) exactly."""
    L16 = build_bond_lindbladian(H1, H2, jumps1, jumps2, dim) * dt
    L4 = sum(embed_superop(L16, b, 4) for b in LATTICE_BONDS_2X2)
    return expm(L4).real


# ---------------------------------------------------------------------------
#  Choi Pauli vector of a channel and its RoM
# ---------------------------------------------------------------------------
def _y_parity_signs(n: int) -> np.ndarray:
    """signs[A] = (-1)^(#Y digits of A) over the 4^n big-endian Pauli index."""
    idx = np.arange(4 ** n)
    count = np.zeros_like(idx)
    for _ in range(n):
        count += (idx % 4 == 2)
        idx = idx // 4
    return np.where(count % 2 == 1, -1.0, 1.0)


def choi_pauli_vec_from_ptm(R: np.ndarray) -> np.ndarray:
    """Pauli vector b_(A,B) = tr(J (A (x) B)) of the Choi state
    J = (id (x) Lambda)(|Omega><Omega|), length 16^n, index A * 4^n + B
    (A on the reference register).  With the repo PTM convention
    R[a, b] = Tr(P_a Lambda(P_b)) / 2^n this is b_(A,B) = s_A R[B, A]."""
    R = np.asarray(R, dtype=float)
    n = round(np.log2(R.shape[0]) / 2)
    assert 4 ** n == R.shape[0], f'PTM dimension {R.shape[0]} is not 4^n'
    return (_y_parity_signs(n)[:, None] * R.T).reshape(-1)


def rom_of_channel_ptm(R: np.ndarray, method: str = 'auto', solver: str = 'auto',
                       K: float | None = None, certify: bool = True,
                       verbose: bool = False) -> dict:
    """Choi-state RoM of an n-qubit channel given its PTM (n <= 4, so that the
    Choi state has n_choi = 2n <= 8 qubits).  Same solver stack and result
    keys as rom_of_gate.compute_gate_rom."""
    import rom_of_gate as rog          # lazy: needs the RoM-handbook clone

    R = np.asarray(R, dtype=float)
    n = round(np.log2(R.shape[0]) / 2)
    n_choi = 2 * n
    assert n_choi <= 8, f'{n}-qubit channel -> {n_choi}-qubit Choi: out of reach'

    rho_vec = choi_pauli_vec_from_ptm(R)
    if method == 'auto':
        method = 'naive' if n_choi <= 5 else 'cg'
    solver = rog.pick_solver(solver, n_choi)
    K_used = rog.DEFAULT_K[n_choi] if K is None else K

    t0 = time.perf_counter()
    if method == 'naive':
        rom, coeff, Amat, info = rog.rom_naive(n_choi, rho_vec, solver, verbose)
    elif method == 'cg':
        rom, coeff, Amat, info = rog.rom_column_generation(
            n_choi, rho_vec, K_used, solver, verbose, certify=certify)
    elif method == 'approx':
        rom, coeff, Amat, info = rog.rom_approx(
            n_choi, rho_vec, K_used, solver, verbose)
    else:
        raise ValueError(f'unknown method {method!r}')
    t_lp = time.perf_counter() - t0

    residual = float(np.max(np.abs(Amat @ coeff - rho_vec)))
    assert residual < 1e-6, f'decomposition residual too large: {residual:.2e}'

    return {
        'rom': float(rom),
        'log2_rom': float(np.log2(rom)),
        'rom_method': method,
        'rom_solver': solver,
        'rom_K': float(K_used),
        'rom_certified': bool(info['certified']),
        'rom_cert_value': (float('nan') if info['cert_value'] is None
                           else float(info['cert_value'])),
        'rom_cg_iterations': int(info['iterations']),
        'rom_residual_inf': residual,
        'rom_n_decomp_terms': int(np.sum(np.abs(coeff) > 1e-12)),
        'rom_time_lp_s': t_lp,
    }


# ---------------------------------------------------------------------------
#  Per-point computation
# ---------------------------------------------------------------------------
# '2x2': RoM of the 4-qubit lattice gate (8-qubit Choi, column generation).
# '2x1': RoM of the two-qubit bond gate itself -- a 2x1 lattice has the single
#        bond (0,1), so its gate IS U_2 = expm(L_bond dt), exactly the gate the
#        stabilizer-3 framability refers to (4-qubit Choi, cheap naive LP, no
#        C++ enumeration involved).
SYSTEMS = ('2x2', '2x1')


def compute_rom_point(model, p1: float, p2: float, *, system: str = '2x2',
                      dim: int | None = None, dt: float | None = None,
                      stab_fra: float | None = None, method: str = 'auto',
                      solver: str = 'auto', K: float | None = None,
                      certify: bool = True, verbose: bool = False) -> dict:
    """Stabilizer-3 framability of the two-qubit bond gate and Choi-state RoM
    of the matching lattice gate (see SYSTEMS) at one (p1, p2) point of
    `model`.

    Pass stab_fra to reuse an already-computed framability (models 1-5 in
    results_trotter_v3) instead of recomputing it; it MUST come from the same
    dim / dt (the worker checks the stored dt before reusing).
    """
    assert system in SYSTEMS, f'unknown system {system!r}'
    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    if dim is None:
        dim = model.dim
    if dt is None:
        dt = model.dt if model.dt is not None else choose_dt(H1, H2, jumps1, jumps2)

    out: dict = dict(p1=p1, p2=p2, dim=dim, dt=dt, system=system)

    gate2 = None
    if stab_fra is None:
        gate2 = bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)
        out['stab_fra'] = float(stabilizer_3_framability(gate2))
        out['stab_fra_source'] = 'computed'
    else:
        out['stab_fra'] = float(stab_fra)
        out['stab_fra_source'] = 'scan'
    if verbose:
        print(f'  stab_fra = {out["stab_fra"]:.6f} ({out["stab_fra_source"]})',
              flush=True)

    if system == '2x1':
        gate = gate2 if gate2 is not None else \
            bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)
    else:
        gate = four_qubit_gate_ptm(H1, H2, jumps1, jumps2, dim, dt)
    out.update(rom_of_channel_ptm(gate, method=method, solver=solver, K=K,
                                  certify=certify, verbose=verbose))
    if verbose:
        print(f'  rom = {out["rom"]:.6f}  (log2 = {out["log2_rom"]:.4f}, '
              f'certified = {out["rom_certified"]}, '
              f'{out["rom_time_lp_s"]:.0f}s)', flush=True)
    return out


# ---------------------------------------------------------------------------
#  Self-test: conventions and anchors
# ---------------------------------------------------------------------------
def _ptm_of_unitary(U: np.ndarray) -> np.ndarray:
    """PTM R[a, b] = Tr(P_a U P_b U^dag) / 2^n of a unitary channel."""
    from dissipative_PT import _pauli_tensor
    n = round(np.log2(U.shape[0]))
    P = _pauli_tensor(n)
    d = 2 ** n
    Rmat = np.einsum('aij,jk->aik', P, U)          # P_a U
    Rmat = np.einsum('aik,bkl,li->ab', Rmat, P, U.conj().T)
    return Rmat.real / d


def self_test() -> None:
    import rom_of_gate as rog
    from dissipative_PT import _build_lindbladian, _site_op
    from trotter_lindbladian_scan import _embed_two_site

    # 1. Choi Pauli vector of a channel vs rom_of_gate's unitary Choi vector.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    U, _ = np.linalg.qr(A)                          # Haar-ish 2-qubit unitary
    vec_ptm = choi_pauli_vec_from_ptm(_ptm_of_unitary(U))
    vec_ref = rog.choi_pauli_vector(U)
    err = np.max(np.abs(vec_ptm - vec_ref))
    print(f'[1] Choi vector, random 2q unitary: max err = {err:.2e}')
    assert err < 1e-10

    # 2. embed_superop / L_4: the bond-sum generator must equal the direct
    #    4-qubit Lindbladian with the bond-shared couplings (each site sits on
    #    exactly 2 of the 4 bonds, each bond carries f1 = 1/(2 dim) of every
    #    one-qubit term).
    model = MODELS['model1']
    H1, H2, jumps1, jumps2 = model.build(1.3, 0.7)
    dim = DIM_DEFAULT
    L16 = build_bond_lindbladian(H1, H2, jumps1, jumps2, dim)
    L4_embed = sum(embed_superop(L16, b, 4) for b in LATTICE_BONDS_2X2)

    f1 = 1.0 / (2.0 * dim)
    Hfull = np.zeros((16, 16), dtype=complex)
    jumps = []
    for (i, j) in LATTICE_BONDS_2X2:
        if H2 is not None:
            Hfull += _embed_two_site(H2, i, j, 4)
        for s in (i, j):
            if H1 is not None:
                Hfull += f1 * _site_op(np.asarray(H1, dtype=complex), s, 4)
            for L1 in (jumps1 or []):
                Lf = _site_op(np.asarray(L1, dtype=complex), s, 4)
                Ld = Lf.conj().T
                jumps.append((f1, Lf, Ld, Ld @ Lf))
        for L2 in (jumps2 or []):
            Lf = _embed_two_site(np.asarray(L2, dtype=complex), i, j, 4)
            Ld = Lf.conj().T
            jumps.append((1.0, Lf, Ld, Ld @ Lf))
    L4_direct = _build_lindbladian(Hfull, jumps, n=4)
    err = np.max(np.abs(L4_embed - L4_direct))
    print(f'[2] L_4 bond sum vs direct 4-qubit Lindbladian: max err = {err:.2e}')
    assert err < 1e-10

    # 3. RoM anchors through the channel path (naive LP, n_choi in {2, 4}):
    #    Clifford -> 1, T -> sqrt(2), CX -> 1.
    for spec, want in (('T', np.sqrt(2.0)), ('H', 1.0), ('CX', 1.0)):
        R = _ptm_of_unitary(rog.parse_gate_spec(spec))
        rom = rom_of_channel_ptm(R, method='naive')['rom']
        print(f'[3] RoM({spec}) = {rom:.10f}  (expect {want:.10f})')
        assert abs(rom - want) < 1e-7

    # 4. model6 sanity: at (rho, sigma) = (0, 0) the jump is S^- exactly.
    from trotter_lindbladian_scan import rotated_s_minus, S_MINUS
    err = np.max(np.abs(rotated_s_minus(0.0, 0.0) - S_MINUS))
    print(f'[4] rotated_s_minus(0, 0) vs S^-: max err = {err:.2e}')
    assert err < 1e-12

    print('self-test passed.')


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--self_test', action='store_true')
    p.add_argument('--system', type=str, default='2x2', choices=SYSTEMS)
    p.add_argument('--model',  type=str, choices=list(MODELS))
    p.add_argument('--p1',     type=float)
    p.add_argument('--p2',     type=float)
    p.add_argument('--dim',    type=int, default=None, choices=(1, 2, 3))
    p.add_argument('--dt',     type=float, default=None)
    p.add_argument('--method', type=str, default='auto',
                   choices=['auto', 'naive', 'cg', 'approx'])
    p.add_argument('--solver', type=str, default='auto',
                   choices=['auto', 'gurobi', 'scipy'])
    p.add_argument('--K',      type=float, default=None)
    p.add_argument('--no_certify', action='store_true')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.model is None or args.p1 is None or args.p2 is None:
        p.error('--model, --p1 and --p2 are required (or use --self_test)')

    model = MODELS[args.model]
    print(f'[{model.name}] {model.p1_name}={args.p1} {model.p2_name}={args.p2}')
    res = compute_rom_point(model, args.p1, args.p2, system=args.system,
                            dim=args.dim, dt=args.dt,
                            method=args.method, solver=args.solver, K=args.K,
                            certify=not args.no_certify, verbose=True)
    for k, v in res.items():
        print(f'  {k:20s} = {v}')


if __name__ == '__main__':
    main()
