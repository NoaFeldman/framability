"""
model4 framability-RATE panels (1-6 of the model4 rate figure).

For every (gamma, gamma') point of trotter_lindbladian_scan's model4 grid
(MODELS['model4'].p1_vals x p2_vals, 51 x 51 = 2601 points) this computes the
dt-free framability rate

    mu*(D) = lim_{dt -> 0} (framability(expm(dt L), D) - 1) / dt

of the two-qubit bond generator L = build_bond_lindbladian(...) for six frames:

  1. rate_stab3   stabilizer-3 frame        framability_rate_frames.stabilizer_3_rate
  2. rate_pauli   Pauli frame               framability_rate_frames.pauli_rate
  3. rate_heis_4  optimised Heisenberg, d_ext_single=4   framability_rate.minimize_rate
  4. rate_heis_6  optimised Heisenberg, d_ext_single=6   framability_rate.minimize_rate
  5. rate_schro_4 optimised Schrodinger, d_ext_single=4  framability_rate_state.minimize_state_rate
  6. rate_schro_6 optimised Schrodinger, d_ext_single=6  framability_rate_state.minimize_state_rate

Nothing about the measures themselves is (re)implemented here: 1-2 are the
fixed-frame evaluators of framability_rate_frames, 3-4 the observable-frame
generator optimiser of framability_rate, 5-6 its state-frame counterpart.  This
file only maps grid points onto array tasks and stores results.

The optimised frames (S) are stored alongside the values so a later neighbour
refinement (framability_rate.neighbor_refine_rates) can cross-evaluate them --
the cross-evaluation that the model3 gp2 row established as mandatory for the
optimised rates.

Output: <out_dir>/model4/pt_<ix:03d>_<iy:03d>.npz   (one file per grid point;
existing files are skipped, so a partial array can simply be resubmitted).

Usage:
    python scripts/model4_rate_panels_worker.py --task_id 0 --n_chunks 200
    python scripts/model4_rate_panels_worker.py --task_id 7 --n_chunks 200 --stride 5
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
from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,   # noqa: E402
                                      DIM_DEFAULT)
from framability_rate import (minimize_rate, spectral_abscissa,          # noqa: E402
                              RATE_VERSION)
from framability_rate_frames import (pauli_rate, stabilizer_3_rate,      # noqa: E402
                                     RATE_FRAMES_VERSION)
from framability_rate_state import (minimize_state_rate,                 # noqa: E402
                                    RATE_STATE_VERSION)

MODEL_NAME = 'model4'

# The six rate panels, in figure order: (npz key, human label).
RATE_KEYS = [
    ('rate_stab3',   'Stabilizer-3 framability rate'),
    ('rate_pauli',   'Pauli framability rate'),
    ('rate_heis_4',  r'Opt Heisenberg rate ($d_{\rm ext}=4$)'),
    ('rate_heis_6',  r'Opt Heisenberg rate ($d_{\rm ext}=6$)'),
    ('rate_schro_4', r'Opt Schrodinger rate ($d_{\rm ext}=4$)'),
    ('rate_schro_6', r'Opt Schrodinger rate ($d_{\rm ext}=6$)'),
]


def grid_vals(stride: int):
    """model4's (gamma, gamma') axes, optionally strided."""
    m = MODELS[MODEL_NAME]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def compute_rates(gamma: float, gamma_p: float, args) -> dict:
    """All six framability rates of the model4 bond generator at one point."""
    m = MODELS[MODEL_NAME]
    H1, H2, jumps1, jumps2 = m.build(gamma, gamma_p)
    L = build_bond_lindbladian(H1, H2, jumps1, jumps2, args.dim).real
    A = L.T                                   # Heisenberg picture generator

    out: dict = dict(gamma=gamma, gamma_p=gamma_p, dim=args.dim,
                     floor=spectral_abscissa(A))

    # ---- 1-2: fixed frames (no optimisation; stab3 is 1080 per-column LPs) --
    t0 = time.perf_counter()
    out['rate_pauli'] = pauli_rate(L)
    out['rate_stab3'] = stabilizer_3_rate(L)
    out['t_fixed'] = time.perf_counter() - t0

    # ---- 3-4: optimised observable frames (alternating certificate scheme) --
    for de in (4, 6):
        t0 = time.perf_counter()
        S, mu, info = minimize_rate(
            A, de, n_restarts=args.heis_restarts, maxfev=args.heis_maxfev,
            seed=args.seed + de, verbose=False, polish_iters=args.polish,
            check_swap=not args.no_swap_check)
        out[f'rate_heis_{de}'] = mu
        out[f'S_heis_{de}'] = S
        out[f'mu_search_heis_{de}'] = info['mu_search']
        out[f'mu_polish_heis_{de}'] = info['mu_polish']
        out[f't_heis_{de}'] = time.perf_counter() - t0

    # ---- 5-6: optimised state frames (Nelder-Mead on the same rate LP) ------
    for de in (4, 6):
        t0 = time.perf_counter()
        S, mu, x = minimize_state_rate(
            L, de, n_restarts=args.schro_restarts, maxfev=args.schro_maxfev,
            seed=args.seed + de, verbose=False, return_x=True)
        out[f'rate_schro_{de}'] = mu
        out[f'S_schro_{de}'] = S
        out[f'x_schro_{de}'] = x
        out[f't_schro_{de}'] = time.perf_counter() - t0

    return out


def run_point(ix: int, iy: int, args) -> None:
    p1_vals, p2_vals = grid_vals(args.stride)
    gamma, gamma_p = float(p1_vals[ix]), float(p2_vals[iy])

    pt_dir = Path(args.out_dir) / MODEL_NAME
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists():
        print(f'[skip] {MODEL_NAME}/{out_f.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{MODEL_NAME}] point ({ix},{iy})  gamma={gamma:.3f} '
          f"gamma'={gamma_p:.3f}", flush=True)

    try:
        res = compute_rates(gamma, gamma_p, args)
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        return

    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_f, model=MODEL_NAME, ix=ix, iy=iy, stride=args.stride,
             rate_version=RATE_VERSION, rate_frames_version=RATE_FRAMES_VERSION,
             rate_state_version=RATE_STATE_VERSION, **res)
    vals = '  '.join(f'{k}={res[k]:+.5f}' for k, _ in RATE_KEYS)
    print(f'  saved {out_f.name}  {vals}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks '
                        '(200 = the job cap; n_chunks<=1 means task_id is a '
                        'single flat grid index)')
    p.add_argument('--out_dir',  type=str, default='results_model4_rate')
    p.add_argument('--stride',   type=int, default=1,
                   help='stride on the model4 grid (1 = full 51x51 = 2601 pts)')
    p.add_argument('--dim',      type=int, default=DIM_DEFAULT,
                   help='lattice dimension of the bond Trotter convention '
                        '(each qubit sits on 2*dim bonds); must match the scan')
    p.add_argument('--heis_restarts', type=int, default=8,
                   help='minimize_rate restarts for the observable frames')
    p.add_argument('--heis_maxfev',   type=int, default=3000)
    p.add_argument('--polish',        type=int, default=300,
                   help='Polyak subgradient polish iterations (0 disables)')
    p.add_argument('--schro_restarts', type=int, default=5,
                   help='minimize_state_rate restarts for the state frames')
    p.add_argument('--schro_maxfev',   type=int, default=800)
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--no_swap_check', action='store_true',
                   help='skip minimize_rate\'s swap-symmetry assertion on the '
                        'bond generator (model4 is symmetric; this is an escape '
                        'hatch for numerical edge cases)')
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
    print(f'[chunk {args.task_id}/{args.n_chunks}] {MODEL_NAME}: '
          f'{len(ids)} of {n_total} points ({nx}x{ny} grid)', flush=True)
    for pid in ids:
        run_point(pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
