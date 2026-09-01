"""
Oscillation rate of the full 8-qubit ring Lindbladian, over model3's
(gamma, gamma') grid -- the extra panel on the dtbase-line framability figure.

For each grid point the full N-qubit Liouvillian (model3 physics:
H = J sum_<ij> Z_i Z_j, jumps sqrt(gamma)|-><+|_i and sqrt(gamma')Z_i on every
site, on a ring) is built and its spectrum examined for

    oscillation rate = max_k |Im(lambda_k) / Re(lambda_k)|

i.e. radians of ringing per e-folding of decay, maximized over modes.

N=8 and the sparse spectrum
---------------------------
The requested system is 8 qubits, whose Liouvillian is 65536x65536.  A dense
diagonalization of that is ~69 GB and ~O(d^3) work PER GRID POINT, so it is not
merely slow but impossible across a 2601-point grid; nonequilibrium_phase_
characterizers.oscillation_rate refuses it outright above dense_max_dim.

The measure itself makes the sparse route natural: |Im/Re| diverges as
Re(lambda) -> 0, so it is dominated by the rightmost (slowest-decaying) modes,
which is exactly what scipy's eigs(which='LR') targets.  The stored value is
therefore the maximum over the K rightmost modes -- a LOWER BOUND on the true
maximum over the full spectrum, and the npz records `exact=False` to say so.
Raise --k to tighten the bound; use --n_qubits 6 --method dense for an exact
(but smaller-system) value.

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz
        (gamma, gamma', osc_rate, lam_re, lam_im, exact, n_modes)

Usage:
    python scripts/osc_rate_worker.py --model model3 --task_id 0 --n_chunks 200
    python scripts/osc_rate_worker.py --model model3 --task_id 0 --n_qubits 6 --method dense
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
from trotter_lindbladian_scan import MODELS
from n_qubit_lindbladian import ring_edges, build_lindbladian_comp
from nonequilibrium_phase_characterizers import oscillation_rate

J = 1.0                      # matches model3 (J=1, h=0)
N_QUBITS_DEFAULT = 8


def grid_vals(model: str, stride: int):
    """model3's (gamma, gamma') axes, optionally strided."""
    m = MODELS[model]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def run_point(model: str, ix: int, iy: int, args) -> None:
    p1_vals, p2_vals = grid_vals(model, args.stride)
    gamma, gamma_p = float(p1_vals[ix]), float(p2_vals[iy])

    pt_dir = Path(args.out_dir) / model
    out = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out.exists():
        print(f'[skip] {model}/{out.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{model}] point ({ix},{iy})  gamma={gamma:.3f} '
          f"gamma'={gamma_p:.3f}  N={args.n_qubits} ring  method={args.method}",
          flush=True)

    L = build_lindbladian_comp(J, gamma, gamma_p, args.n_qubits,
                               ring_edges(args.n_qubits))
    try:
        res = oscillation_rate(L=L, method=args.method, k=args.k,
                               which=args.which, maxiter=args.maxiter)
        rate, lam, exact, n_modes = res.rate, res.lam, res.exact, res.n_modes
        for w in res.warnings:
            print(f'  WARNING: {w}', flush=True)
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        rate, lam, exact, n_modes = np.nan, None, False, 0

    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out, model=model, ix=ix, iy=iy, gamma=gamma, gamma_p=gamma_p,
             J=J, n_qubits=args.n_qubits, topology='ring',
             osc_rate=rate,
             lam_re=(np.nan if lam is None else lam.real),
             lam_im=(np.nan if lam is None else lam.imag),
             exact=exact, n_modes=n_modes, k=args.k, method=args.method)
    print(f'  saved {model}/{out.name}  osc_rate={rate:.6g}  '
          f'lam={lam}  ({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, default='model3', choices=list(MODELS))
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--out_dir',  type=str, default='results_osc_rate')
    p.add_argument('--stride',   type=int, default=1,
                   help='stride on the model grid (1 = full 51x51, matching the '
                        'framability panels)')
    p.add_argument('--n_qubits', type=int, default=N_QUBITS_DEFAULT)
    p.add_argument('--method',   type=str, default='sparse',
                   choices=('auto', 'dense', 'sparse'),
                   help="'sparse' (default) is required at n_qubits=8; 'dense' "
                        'gives the exact maximum but only fits n_qubits<=6')
    p.add_argument('--k',        type=int, default=64,
                   help='sparse: number of rightmost modes maximized over '
                        '(higher = tighter lower bound, slower)')
    p.add_argument('--which',    type=str, default='LR',
                   help="sparse: eigenvalue selection ('LR' = largest real "
                        'part = slowest decaying = where |Im/Re| peaks)')
    p.add_argument('--maxiter',  type=int, default=10000)
    args = p.parse_args()

    p1_vals, p2_vals = grid_vals(args.model, args.stride)
    nx, ny = len(p1_vals), len(p2_vals)
    n_total = nx * ny

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
            sys.exit(1)
        run_point(args.model, args.task_id // ny, args.task_id % ny, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {args.model}: {len(ids)} points',
          flush=True)
    for pid in ids:
        run_point(args.model, pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
