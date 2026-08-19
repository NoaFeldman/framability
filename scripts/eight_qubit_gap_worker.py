"""
Item 4: Lindbladian gap (gap between the two minimal eigenvalues of the
Lindbladian) for a full system of 8 qubits, on a ring ("circle") and on a
2x4 open-boundary lattice, over the same model3 physics (H = J*sum_<i,j> Z_i
Z_j, jumps sqrt(gamma)|-><+|_i, sqrt(gamma')Z_i per site) as
trotter_lindbladian_scan's (gamma, gamma') grid -- see n_qubit_lindbladian.py.

Grid: model3's own (gamma, gamma') axes, strided down (STRIDE=5 by default ->
11x11=121 points/topology) since each point is a sparse shift-invert
eigendecomposition of a 65536x65536 matrix, ~25x more grid points than that
would be too expensive in aggregate (per the pipeline design discussion: same
model3 physics, coarser grid).

Output: <out_dir>/<topology>/pt_<ig:03d>_<igp:03d>.npz  (gamma, gamma', gap)
one point per array task (121 <= 200, no chunking needed).

Usage:
    python scripts/eight_qubit_gap_worker.py --topology ring --task_id 0
    python scripts/eight_qubit_gap_worker.py --topology lattice --task_id 0 --stride 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS
from dissipative_PT import bonds_2d
from n_qubit_lindbladian import ring_edges, build_lindbladian_comp, lindbladian_gap

N_QUBITS = 8
J = 1.0                    # matches model3 (J=1, h=0)
LATTICE_LX, LATTICE_LY = 4, 2   # 2x4 lattice, 8 sites


def topology_edges(topology: str):
    if topology == 'ring':
        return ring_edges(N_QUBITS)
    if topology == 'lattice':
        return bonds_2d(LATTICE_LX, LATTICE_LY)
    raise ValueError(f'unknown topology: {topology!r}')


def grid_vals(stride: int) -> np.ndarray:
    """Coarsened model3 (gamma, gamma') axis (both axes identical: _arange(0,10,0.2))."""
    return np.asarray(MODELS['model3'].p1_vals[::stride], dtype=float)


def run_point(topology: str, ig: int, igp: int, *, stride: int, out_dir: Path,
             k: int, sigma: float, noise_floor: float) -> None:
    vals = grid_vals(stride)
    gamma, gamma_p = float(vals[ig]), float(vals[igp])

    pt_dir = out_dir / topology
    out = pt_dir / f'pt_{ig:03d}_{igp:03d}.npz'
    if out.exists():
        print(f'[skip] {topology}/{out.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{topology}] point ({ig},{igp})  gamma={gamma:.3f} '
          f"gamma'={gamma_p:.3f}  N={N_QUBITS}  building sparse Liouvillian...",
          flush=True)

    edges = topology_edges(topology)
    L = build_lindbladian_comp(J, gamma, gamma_p, N_QUBITS, edges)
    try:
        gap, evals = lindbladian_gap(L, k=k, sigma=sigma, noise_floor=noise_floor)
    except RuntimeError as e:
        print(f'  WARNING: {e}', flush=True)
        gap, evals = np.nan, np.full(k, np.nan, dtype=complex)

    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, topology=topology, ig=ig, igp=igp, gamma=gamma, gamma_p=gamma_p,
             J=J, N=N_QUBITS, gap=gap, evals=evals, k=k, sigma=sigma)
    print(f'  saved {topology}/{out.name}  gap={gap:.6g}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--topology', type=str, required=True, choices=('ring', 'lattice'))
    p.add_argument('--task_id',  type=int, required=True,
                   help='flat grid index 0..(n_grid**2 - 1); ig=task_id//n_grid, '
                        'igp=task_id%n_grid')
    p.add_argument('--stride',   type=int, default=5,
                   help='stride on model3 p1_vals/p2_vals (default 5 -> 11x11 grid)')
    p.add_argument('--out_dir',  type=str, default='results_8q')
    p.add_argument('--k',        type=int, default=6,
                   help='number of eigenvalues requested near sigma (shift-invert)')
    p.add_argument('--sigma',    type=float, default=-1e-3)
    p.add_argument('--noise_floor', type=float, default=1e-6)
    args = p.parse_args()

    n_grid = len(grid_vals(args.stride))
    n_total = n_grid * n_grid
    if not (0 <= args.task_id < n_total):
        print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
        sys.exit(1)
    ig, igp = divmod(args.task_id, n_grid)

    run_point(args.topology, ig, igp, stride=args.stride, out_dir=Path(args.out_dir),
             k=args.k, sigma=args.sigma, noise_floor=args.noise_floor)


if __name__ == '__main__':
    main()
