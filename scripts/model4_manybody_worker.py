"""
model4 many-body panels (7-8 of the model4 rate figure): oscillation rate and
Lindbladian gap of the FULL Lindbladian of a 2x4 lattice of 8 qubits.

Physics: trotter_lindbladian_scan's model4 on the open-boundary 2x4 lattice --
H = J sum_<ij> Z_i Z_j + h sum_i X_i  (J = 1, h = MODEL4_H = 1.5), jumps
sqrt(gamma)|-><+|_i and sqrt(gamma')Z_i on every site.  This is exactly
n_qubit_lindbladian.build_lindbladian_comp with h_x = MODEL4_H and the edge
list dissipative_PT.bonds_2d(4, 2), i.e. the same sparse builder the model3
item-4 workers use; model4 differs from model3 only by that transverse field.

  7. osc_rate  max_k |Im(lambda_k)/Re(lambda_k)|
               nonequilibrium_phase_characterizers.oscillation_rate
  8. gap       slowest nonzero decay rate min_j { -Re lambda_j > noise_floor }
               n_qubit_lindbladian.lindbladian_gap

Both come off the same 65536 x 65536 sparse Liouvillian, which is built once
per point.  The two spectra are taken by two SEPARATE eigs calls on purpose:
the gap only needs the handful of rightmost modes (k_gap = 12) while the
oscillation rate wants many more (k_osc = 64) to tighten its lower bound, and
each function is used exactly as it is used elsewhere in the repo rather than
being reimplemented on a shared spectrum.  The small k_gap call is the cheap
one, so this costs little over the oscillation rate alone.

Sparse only, never dense: at N = 8 the Liouvillian is 65536 x 65536 (~69 GB
dense), so oscillation_rate is called with method='sparse' and lindbladian_gap
in ARPACK regular mode (sigma=None -- shift-invert needs a sparse LU whose
fill-in does not fit in a job's memory at this size; see lindbladian_gap's
docstring).  The stored osc_rate is therefore a LOWER BOUND on the true
maximum over the full spectrum and `osc_exact` records that.

Grid: model4's own (gamma, gamma') axes, strided down (STRIDE default 5 ->
11 x 11 = 121 points), since each point is a sparse eigendecomposition of a
65536-dimensional non-normal operator.  Use --stride 1 for the full 51 x 51
grid of the framability panels if the budget allows.

Output: <out_dir>/model4_8q/pt_<ix:03d>_<iy:03d>.npz
        (gamma, gamma', osc_rate, gap, evals, ...)

Usage:
    python scripts/model4_manybody_worker.py --task_id 0 --n_chunks 121
    python scripts/model4_manybody_worker.py --task_id 0 --n_chunks 200 --stride 1
"""

from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, MODEL4_H                    # noqa: E402
from dissipative_PT import bonds_2d                                      # noqa: E402
from n_qubit_lindbladian import build_lindbladian_comp, lindbladian_gap  # noqa: E402
from nonequilibrium_phase_characterizers import oscillation_rate         # noqa: E402

MODEL_NAME = 'model4'
TAG = 'model4_8q'

J = 1.0                          # matches model4 (J = 1)
LATTICE_LX, LATTICE_LY = 4, 2    # 2x4 open-boundary lattice
N_QUBITS = LATTICE_LX * LATTICE_LY


def lattice_edges():
    """Bonds of the 2x4 open-boundary lattice (row-major site numbering)."""
    return bonds_2d(LATTICE_LX, LATTICE_LY)


def grid_vals(stride: int):
    """model4's (gamma, gamma') axes, optionally strided."""
    m = MODELS[MODEL_NAME]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def run_point(ix: int, iy: int, args) -> None:
    p1_vals, p2_vals = grid_vals(args.stride)
    gamma, gamma_p = float(p1_vals[ix]), float(p2_vals[iy])

    pt_dir = Path(args.out_dir) / TAG
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists():
        print(f'[skip] {TAG}/{out_f.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{TAG}] point ({ix},{iy})  gamma={gamma:.3f} '
          f"gamma'={gamma_p:.3f}  N={N_QUBITS} lattice "
          f'{LATTICE_LX}x{LATTICE_LY}  h={MODEL4_H}', flush=True)

    L = build_lindbladian_comp(J, gamma, gamma_p, N_QUBITS, lattice_edges(),
                               h_x=MODEL4_H)
    print(f'  Liouvillian built: {L.shape[0]}x{L.shape[1]}, {L.nnz} nnz '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)

    # ---- panel 7: oscillation rate (k_osc rightmost modes) -----------------
    try:
        res = oscillation_rate(L=L, method=args.method, k=args.k_osc,
                               which=args.which, maxiter=args.maxiter)
        osc, lam, osc_exact, n_modes = res.rate, res.lam, res.exact, res.n_modes
        for w in res.warnings:
            print(f'  WARNING (osc): {w}', flush=True)
    except Exception as e:
        print(f'  ERROR (osc): {type(e).__name__}: {e}', flush=True)
        osc, lam, osc_exact, n_modes = np.nan, None, False, 0

    # ---- panel 8: Lindbladian gap (k_gap rightmost modes) ------------------
    try:
        gap, evals = lindbladian_gap(L, k=args.k_gap, sigma=args.sigma,
                                     which=args.which, maxiter=args.maxiter,
                                     noise_floor=args.noise_floor)
    except Exception as e:
        print(f'  ERROR (gap): {type(e).__name__}: {e}', flush=True)
        gap, evals = np.nan, np.full(args.k_gap, np.nan, dtype=complex)

    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_f, model=MODEL_NAME, ix=ix, iy=iy, stride=args.stride,
             gamma=gamma, gamma_p=gamma_p, J=J, h=MODEL4_H, N=N_QUBITS,
             topology='lattice', Lx=LATTICE_LX, Ly=LATTICE_LY,
             osc_rate=osc,
             osc_lam_re=(np.nan if lam is None else lam.real),
             osc_lam_im=(np.nan if lam is None else lam.imag),
             osc_exact=osc_exact, osc_n_modes=n_modes, k_osc=args.k_osc,
             method=args.method,
             gap=gap, evals=evals, k_gap=args.k_gap,
             sigma=(np.nan if args.sigma is None else args.sigma),
             which=args.which, noise_floor=args.noise_floor)
    print(f'  saved {out_f.name}  osc_rate={osc:.6g}  gap={gap:.6g}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks; '
                        'n_chunks<=1 means task_id is a single flat grid index')
    p.add_argument('--out_dir',  type=str, default='results_model4_rate')
    p.add_argument('--stride',   type=int, default=5,
                   help='stride on the model4 grid (5 -> 11x11 = 121 points; '
                        '1 = full 51x51, matching the framability panels)')
    p.add_argument('--method',   type=str, default='sparse',
                   choices=('auto', 'dense', 'sparse'),
                   help="'sparse' is required at N=8 (dense is ~69 GB/point)")
    p.add_argument('--k_osc',    type=int, default=64,
                   help='rightmost modes the oscillation rate maximises over '
                        '(higher = tighter lower bound, slower)')
    p.add_argument('--k_gap',    type=int, default=12,
                   help='rightmost modes the gap is read off (must exceed the '
                        'steady-state degeneracy)')
    p.add_argument('--which',    type=str, default='LR',
                   help="'LR' = largest real part = slowest decaying, which is "
                        'both where |Im/Re| peaks and where the gap lives')
    p.add_argument('--sigma',    type=float, default=None,
                   help='shift-invert target for the gap.  Default None = '
                        'ARPACK regular mode (matvec only), the only tractable '
                        'option at N=8; set it only for small N')
    p.add_argument('--noise_floor', type=float, default=1e-6,
                   help='decay rates at or below this are steady-state modes')
    p.add_argument('--maxiter',  type=int, default=10000)
    args = p.parse_args()

    p1_vals, p2_vals = grid_vals(args.stride)
    nx, ny = len(p1_vals), len(p2_vals)
    n_total = nx * ny

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
            sys.exit(1)
        run_point(args.task_id // ny, args.task_id % ny, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {TAG}: {len(ids)} of '
          f'{n_total} points ({nx}x{ny} grid)', flush=True)
    for pid in ids:
        run_point(pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
