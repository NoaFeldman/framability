"""
Item 6: "spectral oscillation" measure (nonequilibrium_phase_characterizers.
spectral_oscillation) for a full many-body Lindbladian on a ring, over the
same model3 physics and coarsened (gamma, gamma') grid as
scripts/eight_qubit_gap_worker.py.

N=6, not 8: nonequilibrium_phase_characterizers.spectral_oscillation dense-
diagonalizes the FULL Liouvillian (all eigenvalues *and* eigenvectors,
biorthonormalized) -- its own docstring requires a system "small enough to
diagonalize exactly".  For N=8 the Liouvillian is 65536x65536 (dense
diagonalization needs ~34GB+ and is not tractable); for N=6 it is 4096x4096,
the same scale six_qubit_lindbladian.py already densifies elsewhere in this
repo.  Per the pipeline design discussion, N is reduced to 6 for this measure
specifically (item 4's gap stays at the literal N=8, via sparse eigs only).

Output: <out_dir>/pt_<ig:03d>_<igp:03d>.npz with gap, omega1, Q1, N1 (the
dominant-mode triple) plus the full nonzero-mode Gamma/omega/Q/N arrays.

Usage:
    python scripts/six_qubit_spectral_osc_worker.py --task_id 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from n_qubit_lindbladian import ring_edges, build_dense_H_jumps
from nonequilibrium_phase_characterizers import spectral_oscillation
from eight_qubit_gap_worker import grid_vals   # same coarsened model3 grid

N_QUBITS = 6                 # reduced from the requested 8 -- see module docstring
J = 1.0                      # matches model3 (J=1, h=0)


def run_point(ig: int, igp: int, *, stride: int, out_dir: Path) -> None:
    vals = grid_vals(stride)
    gamma, gamma_p = float(vals[ig]), float(vals[igp])

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'pt_{ig:03d}_{igp:03d}.npz'
    if out.exists():
        print(f'[skip] {out.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'point ({ig},{igp})  gamma={gamma:.3f} '
          f"gamma'={gamma_p:.3f}  N={N_QUBITS}  building dense Liouvillian "
          f'and diagonalizing...', flush=True)

    edges = ring_edges(N_QUBITS)
    H, jumps = build_dense_H_jumps(J, gamma, gamma_p, N_QUBITS, edges)
    res = spectral_oscillation(H=H, c_ops=jumps)

    for w in res.warnings:
        print(f'  WARNING: {w}', flush=True)

    np.savez(out, ig=ig, igp=igp, gamma=gamma, gamma_p=gamma_p, J=J,
             n_qubits=N_QUBITS,
             gap=res.gap, omega1=res.omega1, Q1=res.Q1, N1=res.N1,
             Gamma=res.Gamma, omega=res.omega, Q=res.Q, N_cycles=res.N,
             n_warnings=len(res.warnings))
    print(f'  saved {out.name}  gap={res.gap:.6g} omega1={res.omega1:.6g} '
          f'Q1={res.Q1:.6g} N1={res.N1:.6g}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, required=True,
                   help='flat grid index 0..(n_grid**2 - 1); ig=task_id//n_grid, '
                        'igp=task_id%n_grid')
    p.add_argument('--stride',  type=int, default=5,
                   help='stride on model3 p1_vals/p2_vals (default 5 -> 11x11 grid, '
                        'matching eight_qubit_gap_worker.py)')
    p.add_argument('--out_dir', type=str, default='results_8q/spectral_osc_ring6')
    args = p.parse_args()

    n_grid = len(grid_vals(args.stride))
    n_total = n_grid * n_grid
    if not (0 <= args.task_id < n_total):
        print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
        sys.exit(1)
    ig, igp = divmod(args.task_id, n_grid)

    run_point(ig, igp, stride=args.stride, out_dir=Path(args.out_dir))


if __name__ == '__main__':
    main()
