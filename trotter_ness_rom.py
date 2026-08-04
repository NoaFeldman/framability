"""
Robustness of Magic of the NESS of the 2x2 lattice Lindbladian.

Companion to trotter_rom_state (RoM of the state after ONE Trotter step) and
trotter_rom_dtbase (its DT_BASE sweep).  For one (p1, p2) point of a model this
computes the RoM of the steady state itself:

    L_full  = build_full_lindbladian_model(H1, H2, jumps1, jumps2)   (2x2 lattice)
    rho_ss  = the unique unit-trace null vector of L_full
    ness_rom = RoM(rho_ss)

Unlike everything in trotter_rom_dtbase this quantity does NOT depend on the
Trotter step: the NESS is a property of the generator alone, so there is nothing
to extrapolate and nothing to sweep.  It is kept in its own (cheap) pipeline
rather than folded into the DT_BASE worker precisely for that reason -- one LP
per grid point instead of one per (point, base), and adding it to the sweep
would have invalidated every base line already computed.

The NESS is generally MIXED, and a sufficiently mixed state lies inside the
stabilizer polytope, where the RoM is exactly 1.  Regions of ness_rom == 1 are
therefore physically meaningful (the steady state is a stabilizer mixture), not
a failure of the solver; tr(rho_ss^2) is stored alongside so those regions can
be read against the purity.

Where the null space is degenerate -- e.g. every gamma = 0 edge, which has no
unique steady state -- steady_state_and_decay returns None and this module
reports NaN.  Those cells are expected to be blank on the colormaps.

Usage:
    python trotter_ness_rom.py --self_test
    python trotter_ness_rom.py --model model1 --p1 0.4 --p2 1.0

Cluster pipeline:
    scripts/trotter_ness_rom_worker.py       per-point array worker
    scripts/trotter_ness_rom.slurm.sh        200-task array (one model per submit)
    scripts/submit_trotter_rom_dtbase_all.sh submits this, the DT_BASE sweep, and
                                             the dependent collect job in one go
    scripts/trotter_rom_dtbase_extrap.py     draws the NESS panel on the --raw figure
"""

from __future__ import annotations

import time

import numpy as np

from dissipative_PT import steady_state_and_decay, pauli_to_rho
from trotter_lindbladian_scan import MODELS, build_full_lindbladian_model
from trotter_rom_state import (
    N_LATTICE, STATE_ROM_MODELS, coeffs_to_pauli_vec, rom_of_pauli_vec,
)

# Version stamp for cached results; bump when any stored quantity changes.
NESS_ROM_VERSION = '1.0'


def ness_coeffs(model, p1: float, p2: float) -> tuple[np.ndarray | None, float]:
    """(Pauli-coefficient vector of the NESS, Liouvillian gap) of the 2x2
    lattice at one parameter point.  The vector is None where the steady state
    is not unique."""
    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    L_full = build_full_lindbladian_model(H1, H2, jumps1, jumps2)
    return steady_state_and_decay(L_full, N=N_LATTICE)


def compute_ness_rom_point(model, p1: float, p2: float, *,
                           verbose: bool = False) -> dict:
    """RoM of the NESS at one (p1, p2) point, with its purity and the gap.

    Every field is NaN when the steady state is not unique.
    """
    t0 = time.perf_counter()
    c_ss, decay = ness_coeffs(model, p1, p2)

    out: dict = dict(p1=p1, p2=p2, lind_rate=float(decay))
    if c_ss is None:
        out.update(ness_rom=float('nan'), log2_ness_rom=float('nan'),
                   ness_purity=float('nan'), ness_ok=False,
                   ness_time_s=time.perf_counter() - t0)
        if verbose:
            print('  NESS is not unique at this point -> NaN', flush=True)
        return out

    rho = pauli_to_rho(c_ss, N=N_LATTICE)
    purity = float(np.trace(rho @ rho).real)
    r = rom_of_pauli_vec(coeffs_to_pauli_vec(c_ss), N_LATTICE, verbose)

    out.update(ness_rom=r['rom'], log2_ness_rom=r['log2_rom'],
               ness_purity=purity, ness_ok=True,
               ness_time_s=time.perf_counter() - t0)
    if verbose:
        print(f'  ness_rom = {r["rom"]:.8f}  (purity = {purity:.6f}, '
              f'gap = {decay:.4e}, {out["ness_time_s"]:.1f}s)', flush=True)
    return out


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------
def self_test() -> None:
    # 1. The maximally mixed state is the extreme case of a stabilizer mixture:
    #    b = (1, 0, ..., 0) must give RoM exactly 1.
    b = np.zeros(4 ** N_LATTICE)
    b[0] = 1.0
    rom = rom_of_pauli_vec(b, N_LATTICE)['rom']
    print(f'[1] RoM(maximally mixed 4-qubit state) = {rom:.10f}')
    assert abs(rom - 1.0) < 1e-9

    # 2. A real NESS: unit trace, positive semidefinite, annihilated by L_full.
    model = MODELS['model1']
    p1, p2 = 0.4, 1.0
    H1, H2, j1, j2 = model.build(p1, p2)
    L_full = build_full_lindbladian_model(H1, H2, j1, j2)
    c_ss, decay = steady_state_and_decay(L_full, N=N_LATTICE)
    assert c_ss is not None, 'model1 (h=0.4, gamma=1) should have a unique NESS'
    resid = float(np.max(np.abs(L_full @ c_ss)))
    rho = pauli_to_rho(c_ss, N=N_LATTICE)
    evals = np.linalg.eigvalsh(rho)
    print(f'[2] model1 (h={p1}, gamma={p2}): |L c_ss|_inf = {resid:.2e}, '
          f'tr = {evals.sum():.10f}, min eig = {evals.min():.2e}, '
          f'gap = {decay:.4e}')
    assert resid < 1e-8
    assert abs(evals.sum() - 1.0) < 1e-9 and evals.min() > -1e-9
    assert abs(coeffs_to_pauli_vec(c_ss)[0] - 1.0) < 1e-10

    # 3. The full per-point result at the same point.
    res = compute_ness_rom_point(model, p1, p2, verbose=True)
    print(f'[3] ness_rom = {res["ness_rom"]:.8f}, '
          f'purity = {res["ness_purity"]:.6f}')
    assert res['ness_ok'] and np.isfinite(res['ness_rom'])
    assert res['ness_rom'] >= 1.0 - 1e-7
    assert 0.0 < res['ness_purity'] <= 1.0 + 1e-9

    # 4. gamma = 0 removes the only jump operator, leaving purely Hamiltonian
    #    dynamics with no unique steady state -> NaN, not a crash.
    res0 = compute_ness_rom_point(model, p1, 0.0)
    print(f'[4] gamma=0: ness_ok = {res0["ness_ok"]}, '
          f'ness_rom = {res0["ness_rom"]}')
    assert not res0['ness_ok'] and np.isnan(res0['ness_rom'])

    print('self-test passed.')


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--self_test', action='store_true')
    p.add_argument('--model', type=str, choices=list(STATE_ROM_MODELS))
    p.add_argument('--p1',    type=float)
    p.add_argument('--p2',    type=float)
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.model is None or args.p1 is None or args.p2 is None:
        p.error('--model, --p1 and --p2 are required (or use --self_test)')

    model = MODELS[args.model]
    print(f'[{model.name}] {model.p1_name}={args.p1} {model.p2_name}={args.p2}')
    res = compute_ness_rom_point(model, args.p1, args.p2, verbose=True)
    for k, v in res.items():
        print(f'  {k:16s} = {v}')


if __name__ == '__main__':
    main()
