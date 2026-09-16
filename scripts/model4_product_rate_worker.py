"""
model4 product-state framability RATE panels (chi = 10, 40).

For every (gamma, gamma') point of the model4 rate grid (the grid of
scripts/model4_rate_panels_worker.py: MODELS['model4'].p1_vals x p2_vals,
51 x 51 = 2601 points at stride 1) this computes the dt-free framability rate

    mu*(D) = lim_{dt -> 0} (framability(expm(dt L), D) - 1) / dt

of the two-qubit bond generator L = build_bond_lindbladian(...) for the random
product-state frames D = kron(D_1, D_1), D_1 = chi random pure single-qubit
states, chi in PROD_RATE_CHIS = (10, 40).

Nothing is reimplemented here: the rate is framability_rate_frames.
product_state_rate (the dt-free companion of framability.
product_state_framability), and the frames are the very frames of the scan's
prod_fra_{chi}: trotter_lindbladian_scan reseeds the global RNG with
PROD_FRAME_SEED and then draws the chi in PROD_CHIS = (10, 20, 40) IN ORDER,
so the chi=40 frame is the one drawn after the chi=10 and chi=20 frames.
build_prod_frames replays that exact sequence once per task.

Output: <out_dir>/<model>_product/pt_<ix:03d>_<iy:03d>.npz  with keys
rate_prod_10, rate_prod_40 (one file per grid point; existing files are
skipped, so a partial array can simply be resubmitted).  The directory is
separate from <out_dir>/<model>/ so these files never collide with the six-rate
pipeline's files or its refine rounds.  scripts/model4_rate_panels_collect.py
reads it and adds the two panels to <model>_rate_panels.png.

Usage:
    python scripts/model4_product_rate_worker.py --task_id 0 --n_chunks 200
    python scripts/model4_product_rate_worker.py --task_id 7 --n_chunks 200 --stride 5
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,   # noqa: E402
                                      PROD_CHIS, PROD_FRAME_SEED)
from framability import make_product_state_D                             # noqa: E402
from framability_rate import spectral_abscissa                           # noqa: E402
from framability_rate_frames import (product_state_rate,                 # noqa: E402
                                     RATE_FRAMES_VERSION)
from model4_rate_panels_worker import (MODEL_NAME, SUPPORTED_MODELS,     # noqa: E402
                                       grid_vals)

PROD_RATE_CHIS = (10, 40)

# (npz key, panel label) in figure order.
PROD_RATE_KEYS = [(f'rate_prod_{chi}', rf'Product-state rate ($\chi={chi}$)')
                  for chi in PROD_RATE_CHIS]


def prod_tag(model: str) -> str:
    """Per-point subdirectory of this pipeline under the rate out_dir."""
    return f'{model}_product'


def build_prod_frames() -> dict:
    """{chi: D} for PROD_RATE_CHIS, replaying the scan's draw sequence
    (reseed, then every chi in PROD_CHIS in order) so each D is the frame of
    the scan's prod_fra_{chi}."""
    np.random.seed(PROD_FRAME_SEED)
    frames = {chi: make_product_state_D(chi) for chi in PROD_CHIS}
    return {chi: frames[chi] for chi in PROD_RATE_CHIS}


def run_point(ix: int, iy: int, frames: dict, args) -> None:
    model = args.model
    m = MODELS[model]
    p1_vals, p2_vals = grid_vals(args.stride, model)
    p1, p2 = float(p1_vals[ix]), float(p2_vals[iy])

    pt_dir = Path(args.out_dir) / prod_tag(model)
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists():
        print(f'[skip] {prod_tag(model)}/{out_f.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{model}] point ({ix},{iy})  {m.p1_name}={p1:.3f} '
          f'{m.p2_name}={p2:.3f}', flush=True)

    try:
        H1, H2, jumps1, jumps2 = m.build(p1, p2)
        L = build_bond_lindbladian(H1, H2, jumps1, jumps2, args.dim).real
        res = dict(p1=p1, p2=p2, **{m.p1_name: p1, m.p2_name: p2},
                   dim=args.dim, floor=spectral_abscissa(L.T))
        for chi, D in frames.items():
            t1 = time.perf_counter()
            res[f'rate_prod_{chi}'] = product_state_rate(chi, L, D=D)
            res[f't_prod_{chi}'] = time.perf_counter() - t1
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        return

    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_f, model=model, ix=ix, iy=iy, stride=args.stride,
             prod_frame_seed=PROD_FRAME_SEED, prod_chis_drawn=np.array(PROD_CHIS),
             rate_frames_version=RATE_FRAMES_VERSION, **res)
    vals = '  '.join(f'{k}={res[k]:+.5f}' for k, _ in PROD_RATE_KEYS)
    print(f'  saved {out_f.name}  {vals}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks '
                        '(200 = the job cap; n_chunks<=1 means task_id is a '
                        'single flat grid index)')
    p.add_argument('--model',    type=str, default=MODEL_NAME,
                   choices=SUPPORTED_MODELS)
    p.add_argument('--out_dir',  type=str, default=None,
                   help='default results_<model>_rate')
    p.add_argument('--stride',   type=int, default=1,
                   help='stride on the model grid (1 = full 51x51 = 2601 pts)')
    p.add_argument('--dim',      type=int, default=None,
                   help="lattice dimension of the bond Trotter convention; "
                        "default the model's ModelSpec.dim (must match the "
                        'six-rate pipeline)')
    args = p.parse_args()
    if args.out_dir is None:
        args.out_dir = f'results_{args.model}_rate'
    if args.dim is None:
        args.dim = MODELS[args.model].dim

    p1_vals, p2_vals = grid_vals(args.stride, args.model)
    nx, ny = len(p1_vals), len(p2_vals)
    n_total = nx * ny
    frames = build_prod_frames()

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
            sys.exit(1)
        run_point(args.task_id // ny, args.task_id % ny, frames, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {args.model} product rates '
          f'chi={PROD_RATE_CHIS}: {len(ids)} of {n_total} points '
          f'({nx}x{ny} grid)', flush=True)
    for pid in ids:
        run_point(pid // ny, pid % ny, frames, args)


if __name__ == '__main__':
    main()
