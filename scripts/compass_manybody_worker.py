"""
Compass-chain many-body panels (7-8 of each case's figure): oscillation rate
and Lindbladian gap of the FULL Liouvillian of an 8-qubit chain, over the
case's (gamma, h) grid.

Physics: compass_chain.chain_lindbladian -- H = -Jx sum X_{2i-1}X_{2i}
- Jy sum Y_{2i}Y_{2i+1} - h sum_j (n_x X_j + n_y Y_j) on an open 8-site chain,
jumps sqrt(gamma) Z_j on every site, at full coupling.

  7. osc_rate  max_k |Im(lambda_k)/Re(lambda_k)|
               nonequilibrium_phase_characterizers.oscillation_rate
  8. gap       slowest nonzero decay rate min_j { -Re lambda_j > noise_floor }
               n_qubit_lindbladian.lindbladian_gap

exactly as scripts/model4_manybody_worker.py uses them: one 65536 x 65536 sparse
Liouvillian per point, two separate ARPACK regular-mode eigs calls (k_osc
rightmost modes for the oscillation rate, k_gap for the gap), never dense and
never shift-invert.  The stored osc_rate is therefore a LOWER BOUND on the
maximum over the full spectrum (`osc_exact` records that).

gamma = 0 is recorded without diagonalising: L = -i[H, .] then has a purely
imaginary spectrum, so no mode decays (gap = nan, as
analysis.gap_from_eigenvalues defines it) and |Im/Re| diverges (osc_rate =
inf); ARPACK 'LR' on an all-tied real part would also be its slowest case.

Output: <out_dir>/<case>/manybody/pt_<ig:03d>_<ih:03d>.npz

Usage:
    python scripts/compass_manybody_worker.py --case jx1.0_jy1.0_hx --task_id 0 --n_chunks 121
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
from compass_chain import (CASES, GAMMA_VALS, H_VALS, chain_lindbladian,  # noqa: E402
                           COMPASS_VERSION)
from n_qubit_lindbladian import lindbladian_gap                          # noqa: E402
from nonequilibrium_phase_characterizers import oscillation_rate         # noqa: E402

TAG = 'manybody'
N_QUBITS = 8


def point_path(out_dir, case_name: str, ig: int, ih: int) -> Path:
    return Path(out_dir) / case_name / TAG / f'pt_{ig:03d}_{ih:03d}.npz'


def _save_atomic(path: Path, **arrays) -> None:
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def run_point(case, ig: int, ih: int, args) -> None:
    gamma, h = float(GAMMA_VALS[ig]), float(H_VALS[ih])
    out_f = point_path(args.out_dir, case.name, ig, ih)
    if out_f.exists():
        print(f'[skip] {case.name}/{TAG}/{out_f.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{case.name}] point ({ig},{ih})  gamma={gamma:.3f}  h={h:.3f}  '
          f'N={args.n_qubits} {args.topology}', flush=True)

    osc, lam, osc_exact, n_modes = np.inf, None, False, 0
    gap, evals = np.nan, np.full(args.k_gap, np.nan, dtype=complex)
    if gamma > 0.0:
        L = chain_lindbladian(case, gamma, h, n_qubits=args.n_qubits,
                              topology=args.topology)
        print(f'  Liouvillian built: {L.shape[0]}x{L.shape[1]}, {L.nnz} nnz '
              f'({time.perf_counter() - t0:.0f}s)', flush=True)

        # ---- panel 7: oscillation rate (k_osc rightmost modes) -------------
        try:
            res = oscillation_rate(L=L, method=args.method, k=args.k_osc,
                                   which=args.which, maxiter=args.maxiter)
            osc, lam, osc_exact, n_modes = (res.rate, res.lam, res.exact,
                                            res.n_modes)
            for w in res.warnings:
                print(f'  WARNING (osc): {w}', flush=True)
        except Exception as e:
            print(f'  ERROR (osc): {type(e).__name__}: {e}', flush=True)
            osc = np.nan

        # ---- panel 8: Lindbladian gap (k_gap rightmost modes) --------------
        try:
            gap, evals = lindbladian_gap(L, k=args.k_gap, sigma=args.sigma,
                                         which=args.which, maxiter=args.maxiter,
                                         noise_floor=args.noise_floor)
        except Exception as e:
            print(f'  ERROR (gap): {type(e).__name__}: {e}', flush=True)
    else:
        print('  gamma = 0: purely imaginary spectrum, stored gap=nan, '
              'osc_rate=inf without diagonalising', flush=True)

    out_f.parent.mkdir(parents=True, exist_ok=True)
    _save_atomic(out_f, case=case.name, Jx=case.Jx, Jy=case.Jy, field=case.field,
                 ig=ig, ih=ih, gamma=gamma, h=h, N=args.n_qubits,
                 topology=args.topology,
                 osc_rate=osc,
                 osc_lam_re=(np.nan if lam is None else lam.real),
                 osc_lam_im=(np.nan if lam is None else lam.imag),
                 osc_exact=osc_exact, osc_n_modes=n_modes, k_osc=args.k_osc,
                 method=args.method,
                 gap=gap, evals=evals, k_gap=args.k_gap,
                 sigma=(np.nan if args.sigma is None else args.sigma),
                 which=args.which, noise_floor=args.noise_floor,
                 compass_version=COMPASS_VERSION)
    print(f'  saved {out_f.name}  osc_rate={osc:.6g}  gap={gap:.6g}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--case',     type=str, required=True, choices=list(CASES))
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks; '
                        'n_chunks<=1 means task_id is a single flat grid index')
    p.add_argument('--out_dir',  type=str, default='results_compass')
    p.add_argument('--n_qubits', type=int, default=N_QUBITS)
    p.add_argument('--topology', type=str, default='chain',
                   choices=('chain', 'ring'))
    p.add_argument('--method',   type=str, default='sparse',
                   choices=('auto', 'dense', 'sparse'),
                   help="'sparse' is required at N=8 (dense is ~69 GB/point)")
    p.add_argument('--k_osc',    type=int, default=64,
                   help='rightmost modes the oscillation rate maximises over '
                        '(higher = tighter lower bound, slower)')
    p.add_argument('--k_gap',    type=int, default=12,
                   help='rightmost modes the gap is read off (must exceed the '
                        'steady-state degeneracy: 2 at h = 0)')
    p.add_argument('--which',    type=str, default='LR')
    p.add_argument('--sigma',    type=float, default=None,
                   help='shift-invert target for the gap; default None = ARPACK '
                        'regular mode, the only tractable option at N=8')
    p.add_argument('--noise_floor', type=float, default=1e-6)
    p.add_argument('--maxiter',  type=int, default=10000)
    args = p.parse_args()

    case = CASES[args.case]
    ng, nh = len(GAMMA_VALS), len(H_VALS)
    n_total = ng * nh

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
            sys.exit(1)
        run_point(case, args.task_id // nh, args.task_id % nh, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {case.name}/{TAG}: {len(ids)} '
          f'of {n_total} points ({ng}x{nh} grid)', flush=True)
    for pid in ids:
        run_point(case, pid // nh, pid % nh, args)


if __name__ == '__main__':
    main()
