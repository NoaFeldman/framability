"""
Sub-pipeline of trotter_lindbladian_scan: stabilizer-3 framability of the
two-qubit bond Trotter gate vs the Robustness of Magic (RoM) of the 2x2-lattice
STATE produced by one application of the lattice gate.

For one (p1, p2) point of a model (MODELS['model1'] .. MODELS['model6']) this
module computes three quantities on the model's FULL scan grid:

  1. stab_fra -- the stabilizer-3 framability of the two-qubit bond Trotter gate
     U_2 = expm(L_bond dt)  (framability.stabilizer_3_framability), exactly the
     quantity trotter_lindbladian_scan stores under the same key.  It is NOT
     recomputed when the main scan already holds it: the cluster worker reuses
     results_trotter_v3/<model>/pt_<ix>_<iy>.npz whenever that file is current
     and carries the same dt.  model6 was never scanned, so its framability is
     computed here from the bond gate.

  2. stab_fra**(1/dt) -- the same number as a per-unit-time rate.  dt is O(1e-3),
     so the exponent is O(1e3) and the raw power overflows the moment stab_fra
     exceeds ~1.002; the log of the quantity,

         log10_stab_fra_pow = log10(stab_fra) / dt,

     is therefore the stored (always finite) form and the raw power is stored
     alongside it for convenience, as +inf where it overflows.

  3. rom -- the Robustness of Magic of the four-qubit state

         rho_1 = U_4 rho_0,
         U_4   = expm(L_full dt),
         L_full = build_full_lindbladian_model(H1, H2, jumps1, jumps2)

     i.e. the EXACT 2x2 open-boundary lattice propagator at the bond gate's own
     Trotter step dt (full coupling strength -- the 1/(2 dim) bond share of
     build_bond_lindbladian is a Trotter-gate convention, not part of the
     physical lattice generator), applied once to rho_0, the |0>^4 or |+>^4
     start state of the model's lpdo_max relaxation path
     (trotter_lindbladian_scan.lpdo_init_vector, per-model via
     ModelSpec.lpdo_init).

     rho_0 is a stabilizer state, so rom = 1 at dt = 0 and rom = 1 + O(dt) after
     one step; log2(rom)/dt is stored as the corresponding magic RATE, which
     divides out the strong per-point variation of dt = DT_BASE / scale.

RoM method
----------
rho_1 lives on N = LATTICE_LX * LATTICE_LY = 4 qubits, so its RoM is the exact
LP over the FULL precomputed four-qubit stabilizer matrix (256 x 36720),

    RoM(rho) = min ||x||_1  s.t.  A x = b,   b_P = tr(rho P),

solved as the sparse split-variable program min 1^T(u + v) s.t. [A, -A](u; v) = b
with scipy/HiGHS -- the `rom_of_gate.rom_naive_scipy` method, seconds per point
and certified by construction (the full stabilizer set is used).  This is the
cheap corner of the RoM machinery: no column generation, no compiled C++
enumerator and no Gurobi.

Relation to trotter_rom_4q
--------------------------
trotter_rom_4q computes a DIFFERENT quantity -- the Choi-state RoM of the
4-qubit *gate*, an 8-qubit mixed-state RoM needing column generation, the C++
enumerator and Gurobi (hours per point, which is why it only ever ran on the
reduced ROM_GRIDS).  This module computes the RoM of a 4-qubit *state* and is
cheap enough for the full grids of all six models.  The two modules share only
trotter_lindbladian_scan; neither imports the other.

Usage:
    python trotter_rom_state.py --self_test
    python trotter_rom_state.py --model model6 --p1 0.0 --p2 0.0

Cluster pipeline:
    scripts/trotter_rom_state_worker.py    per-point array worker
    scripts/trotter_rom_state.slurm.sh     200-task array (one model per submit)
    scripts/submit_trotter_rom_state.sh    submits all six models
    scripts/trotter_rom_state_collect.py   per-model colormap figure + summary
"""

from __future__ import annotations

import os
import time

import numpy as np
import scipy.sparse
from scipy.linalg import expm
from scipy.optimize import linprog

from trotter_lindbladian_scan import (
    MODELS, LATTICE_LX, LATTICE_LY, choose_dt, bond_trotter_gate,
    build_full_lindbladian_model, lpdo_init_vector,
)
from framability import stabilizer_3_framability

# Version stamp for cached results; bump when any stored quantity changes.
ROM_STATE_VERSION = '1.0'

# The models this sub-pipeline covers, each on its FULL trotter_lindbladian_scan
# grid (the fast state RoM removes the need for the reduced trotter_rom_4q
# ROM_GRIDS).  Grid sizes:
#   model1  21 x  51 = 1071      model2  21 x  51 = 1071
#   model3  51 x  51 = 2601      model4  51 x  51 = 2601
#   model5  21 x 101 = 2121      model6  51 x  51 = 2601   (total 12066)
STATE_ROM_MODELS = ('model1', 'model2', 'model3', 'model4', 'model5', 'model6')

# Sites of the lattice the state lives on (2x2 = 4 qubits).
N_LATTICE = LATTICE_LX * LATTICE_LY

# Above this the raw stab_fra**(1/dt) overflows a float64; the stored value is
# then +inf and only log10_stab_fra_pow carries the information.
_LOG10_OVERFLOW = 308.0


def grid_of(model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """(p1_vals, p2_vals) of a model: its full trotter_lindbladian_scan grid."""
    m = MODELS[model_name]
    return np.asarray(m.p1_vals, float), np.asarray(m.p2_vals, float)


# ---------------------------------------------------------------------------
#  Four-qubit stabilizer matrix and the HiGHS RoM LP
# ---------------------------------------------------------------------------
def _handbook_dir() -> str:
    """Root of the RoM-handbook clone (only its data/Amat/*.npz is used)."""
    return os.environ.get(
        'ROM_HANDBOOK_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RoM-handbook'))


_AMAT_CACHE: dict[int, scipy.sparse.csr_matrix] = {}
_AEQ_CACHE: dict[int, scipy.sparse.csr_matrix] = {}


def load_amat(n_qubit: int) -> scipy.sparse.csr_matrix:
    """(4^n, #pure stabilizer states) stabilizer matrix in the Pauli basis.

    Identical to exputils.actual_Amat.get_actual_Amat(n_qubit), but the npz is
    loaded directly instead of through the handbook package: every exputils
    module pulls in numba, which this pipeline otherwise does not need at all.
    Only numpy and scipy are required to compute a state RoM here.
    """
    assert 1 <= n_qubit <= 5, 'the precomputed Amat only goes up to 5 qubits'
    if n_qubit not in _AMAT_CACHE:
        path = os.path.join(_handbook_dir(), 'data', 'Amat', f'Amat{n_qubit}.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'stabilizer matrix not found at {path!r}. Clone the handbook '
                '(git clone https://github.com/quantum-programming/RoM-handbook.git) '
                'or set ROM_HANDBOOK_DIR.')
        _AMAT_CACHE[n_qubit] = scipy.sparse.csr_matrix(
            scipy.sparse.load_npz(path).astype(np.float64))
    return _AMAT_CACHE[n_qubit]


def _lp_matrix(n_qubit: int) -> scipy.sparse.csr_matrix:
    """Cached split-variable LP matrix [A, -A] (x = u - v).  Only rho_vec
    changes from point to point, so this is built once per process."""
    if n_qubit not in _AEQ_CACHE:
        A = load_amat(n_qubit)
        _AEQ_CACHE[n_qubit] = scipy.sparse.hstack([A, -A], format='csr')
    return _AEQ_CACHE[n_qubit]


def rom_of_pauli_vec(rho_vec: np.ndarray, n_qubit: int = N_LATTICE,
                     verbose: bool = False) -> dict:
    """Exact RoM of an n-qubit state from its Pauli vector b_P = tr(rho P).

    Sparse split-variable LP over the full stabilizer matrix, solved by
    scipy/HiGHS -- the rom_of_gate.rom_naive_scipy method.  [A, -A] is kept
    SPARSE: at n = 4 it is 256 x ~73440, and a dense hstack would be a ~150 MB
    array that chokes the solver.  Exact and certified by construction, since
    the full stabilizer set is used.
    """
    rho_vec = np.asarray(rho_vec, dtype=float)
    A = load_amat(n_qubit)
    assert rho_vec.shape == (A.shape[0],), \
        f'Pauli vector has length {rho_vec.shape[0]}, expected {A.shape[0]}'
    N = A.shape[1]

    t0 = time.perf_counter()
    res = linprog(np.ones(2 * N), A_eq=_lp_matrix(n_qubit), b_eq=rho_vec,
                  bounds=(0, None), method='highs')
    t_lp = time.perf_counter() - t0
    assert res.success, f'RoM LP failed: {res.message}'

    coeff = res.x[:N] - res.x[N:]
    residual = float(np.max(np.abs(A @ coeff - rho_vec)))
    assert residual < 1e-6, f'decomposition residual too large: {residual:.2e}'
    rom = float(res.fun)
    if verbose:
        print(f'  [RoM] n={n_qubit}, cols={N}: RoM={rom:.10f} ({t_lp:.1f}s)',
              flush=True)
    return {
        'rom': rom,
        'log2_rom': float(np.log2(rom)),
        'rom_n_decomp_terms': int(np.sum(np.abs(coeff) > 1e-12)),
        'rom_residual_inf': residual,
        'rom_time_lp_s': float(t_lp),
    }


# ---------------------------------------------------------------------------
#  The evolved lattice state
# ---------------------------------------------------------------------------
def lattice_step_ptm(H1, H2, jumps1, jumps2, dt: float) -> np.ndarray:
    """(4^N x 4^N) one-step propagator expm(L_full dt) of the LATTICE_LX x
    LATTICE_LY open-boundary lattice, acting on Pauli-coefficient vectors."""
    return expm(build_full_lindbladian_model(H1, H2, jumps1, jumps2) * dt)


def evolved_state_pauli_vec(model, H1, H2, jumps1, jumps2,
                            dt: float) -> np.ndarray:
    """Pauli-COEFFICIENT vector c (rho = sum_P c_P P) of rho_1 = U_4 rho_0, with
    rho_0 the model's lpdo_max path start state (|0>^N or |+>^N)."""
    c0 = lpdo_init_vector(model, N_LATTICE)
    return lattice_step_ptm(H1, H2, jumps1, jumps2, dt) @ c0


def coeffs_to_pauli_vec(c: np.ndarray, n_qubit: int = N_LATTICE) -> np.ndarray:
    """Repo Pauli COEFFICIENTS (rho = sum_P c_P P) -> handbook Pauli VECTOR
    (b_P = tr(rho P)).  Tr(P Q) = 2^n delta_PQ, so b = 2^n c.  Both use the same
    big-endian (I, X, Y, Z)^{(x)n} index order (dissipative_PT._pauli_tensor and
    exputils.state.state_in_pauli_basis agree; checked by trotter_rom_4q)."""
    return (2 ** n_qubit) * np.asarray(c, dtype=float)


# ---------------------------------------------------------------------------
#  Per-point computation
# ---------------------------------------------------------------------------
def compute_state_rom_point(model, p1: float, p2: float, *,
                            dim: int | None = None, dt: float | None = None,
                            stab_fra: float | None = None,
                            verbose: bool = False) -> dict:
    """stab_fra of the bond gate, its 1/dt power, and the RoM of the once-evolved
    2x2 lattice state, at one (p1, p2) point of `model`.

    Pass stab_fra to reuse an already-computed framability (models 1-5 in
    results_trotter_v3) instead of recomputing it; it MUST come from the same
    dim / dt (the worker checks the stored dt before reusing).
    """
    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    if dim is None:
        dim = model.dim
    if dt is None:
        dt = model.dt if model.dt is not None else choose_dt(H1, H2, jumps1, jumps2)

    out: dict = dict(p1=p1, p2=p2, dim=dim, dt=dt, lpdo_init=model.lpdo_init)

    # -- 1/2. stabilizer-3 framability of the two-qubit bond gate, and its rate
    if stab_fra is None:
        gate2 = bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)
        out['stab_fra'] = float(stabilizer_3_framability(gate2))
        out['stab_fra_source'] = 'computed'
    else:
        out['stab_fra'] = float(stab_fra)
        out['stab_fra_source'] = 'scan'
    log10_pow = float(np.log10(out['stab_fra']) / dt)
    out['log10_stab_fra_pow'] = log10_pow
    out['stab_fra_pow'] = (float('inf') if log10_pow > _LOG10_OVERFLOW
                           else float(10.0 ** log10_pow))
    if verbose:
        print(f'  stab_fra = {out["stab_fra"]:.6f} ({out["stab_fra_source"]}), '
              f'^(1/dt) = 1e{log10_pow:.3f}', flush=True)

    # -- 3. RoM of the once-evolved lattice state
    c1 = evolved_state_pauli_vec(model, H1, H2, jumps1, jumps2, dt)
    out.update(rom_of_pauli_vec(coeffs_to_pauli_vec(c1), N_LATTICE, verbose))
    out['rom_rate'] = float(out['log2_rom'] / dt)      # log2(RoM) per unit time
    if verbose:
        print(f'  rom = {out["rom"]:.8f}  (log2 = {out["log2_rom"]:.6f}, '
              f'rate = {out["rom_rate"]:.4f})', flush=True)
    return out


# ---------------------------------------------------------------------------
#  Self-test: conventions and anchors
# ---------------------------------------------------------------------------
def self_test() -> None:
    from dissipative_PT import _pauli_tensor, pauli_to_rho

    # 1. The four-qubit stabilizer matrix is present and has the known shape:
    #    4^4 = 256 Pauli rows, 36720 pure four-qubit stabilizer states.
    A = load_amat(N_LATTICE)
    print(f'[1] Amat{N_LATTICE}: shape = {A.shape}')
    assert A.shape == (4 ** N_LATTICE, 36720), A.shape

    # 2. Pauli-vector convention: b_P = tr(rho P) for the lpdo start states.
    for init, ket in (('zero', '|0>'), ('plus', '|+>')):
        model = MODELS['model5' if init == 'zero' else 'model1']
        assert model.lpdo_init == init
        c0 = lpdo_init_vector(model, N_LATTICE)
        b = coeffs_to_pauli_vec(c0)
        rho = pauli_to_rho(c0, N=N_LATTICE)
        b_ref = np.einsum('aij,ji->a', _pauli_tensor(N_LATTICE), rho).real
        err = float(np.max(np.abs(b - b_ref)))
        print(f'[2] {ket}^4 ({init}): b[0] = {b[0]:.6f}, '
              f'tr(rho^2) = {np.trace(rho @ rho).real:.6f}, max err = {err:.2e}')
        assert abs(b[0] - 1.0) < 1e-12 and err < 1e-12
        assert abs(np.trace(rho @ rho).real - 1.0) < 1e-12       # pure

    # 3. RoM anchors: both lpdo start states are stabilizer states -> RoM = 1,
    #    and one T state tensored with three |0>s -> RoM = sqrt(2).
    for init in ('zero', 'plus'):
        model = MODELS['model5' if init == 'zero' else 'model1']
        rom = rom_of_pauli_vec(coeffs_to_pauli_vec(
            lpdo_init_vector(model, N_LATTICE)))['rom']
        print(f'[3] RoM(lpdo start state, init={init}) = {rom:.10f}')
        assert abs(rom - 1.0) < 1e-7

    t_state = np.array([1.0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])   # T|+>
    zero = np.array([1.0, 0.0, 0.0, 1.0])                            # |0>
    b = t_state
    for _ in range(N_LATTICE - 1):
        b = np.kron(b, zero)
    rom = rom_of_pauli_vec(b)['rom']
    print(f'[3] RoM(T (x) |0>^3) = {rom:.10f}  (expect {np.sqrt(2):.10f})')
    assert abs(rom - np.sqrt(2)) < 1e-7

    # 4. The lattice step is trace preserving (b[0] stays 1) and lands on a
    #    physical state, and dt -> 0 recovers the (stabilizer) start state.
    model = MODELS['model1']
    H1, H2, j1, j2 = model.build(1.0, 2.0)
    dt = choose_dt(H1, H2, j1, j2)
    c1 = evolved_state_pauli_vec(model, H1, H2, j1, j2, dt)
    rho1 = pauli_to_rho(c1, N=N_LATTICE)
    evals = np.linalg.eigvalsh(rho1)
    print(f'[4] one step at dt = {dt:.3e}: b[0] = '
          f'{coeffs_to_pauli_vec(c1)[0]:.10f}, min eig = {evals.min():.3e}, '
          f'tr = {evals.sum():.10f}')
    assert abs(coeffs_to_pauli_vec(c1)[0] - 1.0) < 1e-10
    assert evals.min() > -1e-10

    c_zero = evolved_state_pauli_vec(model, H1, H2, j1, j2, 0.0)
    err = float(np.max(np.abs(c_zero - lpdo_init_vector(model, N_LATTICE))))
    print(f'[4] dt = 0 reproduces the start state: max err = {err:.2e}')
    assert err < 1e-12

    # 5. RoM of the once-evolved state is 1 + O(dt): tiny but > 1 here, and it
    #    must shrink towards 1 as dt does.
    roms = []
    for f in (1.0, 0.1):
        c = evolved_state_pauli_vec(model, H1, H2, j1, j2, f * dt)
        roms.append(rom_of_pauli_vec(coeffs_to_pauli_vec(c))['rom'])
    print(f'[5] RoM at dt = {dt:.3e}: {roms[0]:.10f};  '
          f'at dt/10: {roms[1]:.10f}')
    assert roms[0] >= 1.0 - 1e-7 and roms[1] <= roms[0] + 1e-9

    print('self-test passed.')


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--self_test', action='store_true')
    p.add_argument('--model', type=str, choices=list(STATE_ROM_MODELS))
    p.add_argument('--p1',    type=float)
    p.add_argument('--p2',    type=float)
    p.add_argument('--dim',   type=int, default=None, choices=(1, 2, 3))
    p.add_argument('--dt',    type=float, default=None)
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.model is None or args.p1 is None or args.p2 is None:
        p.error('--model, --p1 and --p2 are required (or use --self_test)')

    model = MODELS[args.model]
    print(f'[{model.name}] {model.p1_name}={args.p1} {model.p2_name}={args.p2}')
    res = compute_state_rom_point(model, args.p1, args.p2, dim=args.dim,
                                  dt=args.dt, verbose=True)
    for k, v in res.items():
        print(f'  {k:22s} = {v}')


if __name__ == '__main__':
    main()
