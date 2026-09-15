"""
model4 many-body panels (7-8 of the model4 rate figure): oscillation rate and
Lindbladian gap of the FULL Lindbladian of a 2x4 lattice of 8 qubits.

Physics: trotter_lindbladian_scan's model4 on the open-boundary 2x4 lattice --
H = J sum_<ij> Z_i Z_j + h sum_i X_i  (J = 1, h = MODEL4_H = 1.5), jumps
sqrt(gamma)|-><+|_i and sqrt(gamma')Z_i on every site.  This is exactly
n_qubit_lindbladian.build_lindbladian_comp with h_x = MODEL4_H and the edge
list dissipative_PT.bonds_2d(4, 2), i.e. the same sparse builder the model3
item-4 workers use; model4 differs from model3 only by that transverse field.
--model model8 runs the same two panels for model8 (longitudinal field h Z,
gamma fixed) on its (h, gamma') grid, writing <out_dir>/model8_8q/; the scan
point -> builder mapping is n_qubit_lindbladian.model_lattice_params.
--model model10 (the Shibata-Katsura dissipative quantum Ising chain,
https://arxiv.org/abs/1904.12505) is a 1D model: its two panels use an 8-site
PERIODIC ring -- the paper's boundary condition, and every site sits on two
bonds as in the model's dim = 1 bond gate -- built from the model's own scan
terms by n_qubit_lindbladian.build_lindbladian_from_terms (see mb_geometry),
writing <out_dir>/model10_8q/.

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
from trotter_lindbladian_scan import MODELS                              # noqa: E402
from dissipative_PT import bonds_2d                                      # noqa: E402
from n_qubit_lindbladian import (build_lindbladian_comp, lindbladian_gap,  # noqa: E402
                                 model_lattice_params, model_terms_lindbladian,
                                 ring_edges, RING_MODELS)
from nonequilibrium_phase_characterizers import oscillation_rate         # noqa: E402

MODEL_NAME = 'model4'                  # default --model
TAG = 'model4_8q'                      # = mb_tag(MODEL_NAME), kept for importers
SUPPORTED_MODELS = ('model4', 'model8', 'model10')


def mb_tag(model: str) -> str:
    """Per-point output subdirectory of `model`'s many-body panels."""
    return f'{model}_8q'

J = 1.0                          # matches model4 (J = 1)
LATTICE_LX, LATTICE_LY = 4, 2    # 2x4 open-boundary lattice
N_QUBITS = LATTICE_LX * LATTICE_LY


def lattice_edges():
    """Bonds of the 2x4 open-boundary lattice (row-major site numbering)."""
    return bonds_2d(LATTICE_LX, LATTICE_LY)


def mb_geometry(model: str) -> dict:
    """The N_QUBITS sites and bonds of `model`'s many-body panels: the 2x4
    open-boundary lattice for the 2D models, a periodic ring for the 1D
    RING_MODELS (model10).  topology / Lx / Ly are stored per point; `label`
    goes into the figure title."""
    if model in RING_MODELS:
        return dict(edges=ring_edges(N_QUBITS), topology='ring', Lx=N_QUBITS,
                    Ly=1, label='periodic ring')
    return dict(edges=lattice_edges(), topology='lattice', Lx=LATTICE_LX,
                Ly=LATTICE_LY, label=f'{LATTICE_LY}x{LATTICE_LX} lattice')


def grid_vals(stride: int, model: str = MODEL_NAME):
    """`model`'s scan axes (p1, p2), optionally strided."""
    m = MODELS[model]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def run_point(ix: int, iy: int, args) -> None:
    model = getattr(args, 'model', MODEL_NAME)
    tag = mb_tag(model)
    m = MODELS[model]
    p1_vals, p2_vals = grid_vals(args.stride, model)
    p1, p2 = float(p1_vals[ix]), float(p2_vals[iy])
    # model4/model8: build_lindbladian_comp arguments; the RING_MODELS carry
    # their physics in MODELS[model].build instead
    params = ({} if model in RING_MODELS
              else model_lattice_params(model, p1, p2))
    geo = mb_geometry(model)

    pt_dir = Path(args.out_dir) / tag
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists():
        print(f'[skip] {tag}/{out_f.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{tag}] point ({ix},{iy})  {m.p1_name}={p1:.3f} '
          f"{m.p2_name}={p2:.3f}  N={N_QUBITS} {geo['label']}  {params}",
          flush=True)

    if model in RING_MODELS:
        L = model_terms_lindbladian(model, p1, p2, N_QUBITS, geo['edges'])
    else:
        L = build_lindbladian_comp(J, params['gamma'], params['gamma_p'],
                                   N_QUBITS, geo['edges'], h_x=params['h_x'],
                                   h_z=params['h_z'])
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
    np.savez(out_f, model=model, ix=ix, iy=iy, stride=args.stride,
             p1=p1, p2=p2, p1_name=m.p1_name, p2_name=m.p2_name,
             J=J, N=N_QUBITS, **params,
             topology=geo['topology'], Lx=geo['Lx'], Ly=geo['Ly'],
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
    p.add_argument('--model',    type=str, default=MODEL_NAME,
                   choices=SUPPORTED_MODELS)
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks; '
                        'n_chunks<=1 means task_id is a single flat grid index')
    p.add_argument('--out_dir',  type=str, default=None,
                   help='default results_<model>_rate')
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
    if args.out_dir is None:
        args.out_dir = f'results_{args.model}_rate'

    p1_vals, p2_vals = grid_vals(args.stride, args.model)
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
    print(f'[chunk {args.task_id}/{args.n_chunks}] {mb_tag(args.model)}: {len(ids)} of '
          f'{n_total} points ({nx}x{ny} grid)', flush=True)
    for pid in ids:
        run_point(pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
