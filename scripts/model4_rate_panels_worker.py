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

Other models: --model model8 runs the identical six measures on model8's
(h, gamma') grid, --model model10 on the (Delta1, Delta2) grid of the
Shibata-Katsura dissipative quantum Ising chain
(https://arxiv.org/abs/1904.12505); default out_dir results_<model>_rate.
The bond generator takes the model's own lattice dimension (ModelSpec.dim:
2 for model4 / model8, 1 for the model10 chain) unless --dim is given.

Frame seeds (--frame_seeds)
---------------------------
'default' = the optimisers' own seed sets (minimize_rate: extended-Pauli
octagon in the X-Z plane at d=6, cycling identity, random; minimize_state_rate:
octahedron, SIC tetrahedron, random).  'ring' ADDS seeds whose columns form a
ring in the plane rotated by the model's single-site field (RING_AXIS; model8:
the X-Y plane about Z):
  * observable frames: identity, the axis Pauli and d-2 ring directions at
    angles pi j/(d-2) -- a 2(d-2)-gon (square at d=4, octagon at d=6) -- once
    per ring length in RING_LENGTHS;
  * state frames: d pure states equally spaced on the ring, tilted alternately
    toward +-axis (STATE_RING_TILTS; keeps the Pauli-support penalty at zero).
'auto' = 'ring' for models in RING_AXIS, 'default' otherwise.  Without the ring
seeds the d=6 optimisers start from X-Z-plane / octahedral frames and may stall
at the square's value on model8.  The seeds are appended to (not substituted
for) the default restarts, so each optimisation runs 3 (Heisenberg) / 2 (state)
more restarts.

Output: <out_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz   (one file per grid point;
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

MODEL_NAME = 'model4'                   # default --model
SUPPORTED_MODELS = ('model4', 'model8', 'model10')

# Ring frame seeds (see module docstring): model -> Pauli axis (1=X, 2=Y, 3=Z)
# its single-site field rotates about; the ring lies in the other two axes.
RING_AXIS = {'model8': 3}
RING_LENGTHS = (1.0, 0.5, 0.2)          # observable-frame ring column lengths
STATE_RING_TILTS = (0.15, 0.4)          # state ring: sine of the tilt to the axis


def _ring_plane(axis: int):
    """Pauli row indices (u, v) of the plane rotated about `axis`."""
    return {1: (2, 3), 2: (3, 1), 3: (1, 2)}[axis]


def ring_heis_seeds(d_ext_single: int, axis: int) -> list:
    """Observable-frame seeds S (4 x d) for minimize_rate(extra_init_S=...):
    identity, the axis Pauli, and d-2 ring directions pi j/(d-2), one S per
    ring length.  Columns obey |c_I| + |b| <= 1; column 0 is the identity."""
    k = d_ext_single - 2
    u, v = _ring_plane(axis)
    th = np.pi * np.arange(k) / k
    seeds = []
    for a in RING_LENGTHS:
        S = np.zeros((4, d_ext_single))
        S[0, 0] = 1.0
        S[axis, 1] = 1.0
        S[u, 2:] = a * np.cos(th)
        S[v, 2:] = a * np.sin(th)
        seeds.append(S)
    return seeds


def ring_state_seeds(d_ext_single: int, axis: int) -> list:
    """State-frame seeds for minimize_state_rate(extra_init_xs=...): flat
    3 x d Bloch blocks (the encoding of optimize_framability._state_params_to_S)
    of d pure states equally spaced on the ring, tilted alternately to +-axis."""
    u, v = _ring_plane(axis)
    th = 2 * np.pi * np.arange(d_ext_single) / d_ext_single
    sign = np.where(np.arange(d_ext_single) % 2 == 0, 1.0, -1.0)
    seeds = []
    for s in STATE_RING_TILTS:
        B = np.zeros((4, d_ext_single))       # rows I, X, Y, Z; row I unused
        c = np.sqrt(1.0 - s * s)
        B[u] = 0.5 * c * np.cos(th)
        B[v] = 0.5 * c * np.sin(th)
        B[axis] = 0.5 * s * sign
        seeds.append(B[1:].ravel())
    return seeds


def frame_seed_mode(model: str, requested: str) -> str:
    """Resolve --frame_seeds ('auto' | 'default' | 'ring') for `model`."""
    if requested == 'auto':
        return 'ring' if model in RING_AXIS else 'default'
    if requested == 'ring' and model not in RING_AXIS:
        raise ValueError(f'--frame_seeds ring: no RING_AXIS entry for {model}')
    return requested

# The six rate panels, in figure order: (npz key, human label).
RATE_KEYS = [
    ('rate_stab3',   'Stabilizer-3 framability rate'),
    ('rate_pauli',   'Pauli framability rate'),
    ('rate_heis_4',  r'Opt Heisenberg rate ($d_{\rm ext}=4$)'),
    ('rate_heis_6',  r'Opt Heisenberg rate ($d_{\rm ext}=6$)'),
    ('rate_schro_4', r'Opt Schrodinger rate ($d_{\rm ext}=4$)'),
    ('rate_schro_6', r'Opt Schrodinger rate ($d_{\rm ext}=6$)'),
]


def grid_vals(stride: int, model: str = MODEL_NAME):
    """`model`'s scan axes (p1, p2), optionally strided."""
    m = MODELS[model]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def compute_rates(p1: float, p2: float, args) -> dict:
    """All six framability rates of the model's bond generator at one point."""
    model = getattr(args, 'model', MODEL_NAME)
    m = MODELS[model]
    H1, H2, jumps1, jumps2 = m.build(p1, p2)
    L = build_bond_lindbladian(H1, H2, jumps1, jumps2, args.dim).real
    A = L.T                                   # Heisenberg picture generator

    seeds = frame_seed_mode(model, getattr(args, 'frame_seeds', 'auto'))
    axis = RING_AXIS.get(model)
    # {p1_name: p1, p2_name: p2} keeps model4's gamma / gamma_p keys unchanged
    out: dict = dict(p1=p1, p2=p2, **{m.p1_name: p1, m.p2_name: p2},
                     dim=args.dim, floor=spectral_abscissa(A),
                     frame_seeds=seeds)

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
            check_swap=not args.no_swap_check,
            extra_init_S=(ring_heis_seeds(de, axis) if seeds == 'ring'
                          else None))
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
            seed=args.seed + de, verbose=False, return_x=True,
            extra_init_xs=(ring_state_seeds(de, axis) if seeds == 'ring'
                           else None))
        out[f'rate_schro_{de}'] = mu
        out[f'S_schro_{de}'] = S
        out[f'x_schro_{de}'] = x
        out[f't_schro_{de}'] = time.perf_counter() - t0

    return out


def run_point(ix: int, iy: int, args) -> None:
    model = getattr(args, 'model', MODEL_NAME)
    m = MODELS[model]
    p1_vals, p2_vals = grid_vals(args.stride, model)
    p1, p2 = float(p1_vals[ix]), float(p2_vals[iy])

    pt_dir = Path(args.out_dir) / model
    out_f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists():
        print(f'[skip] {model}/{out_f.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{model}] point ({ix},{iy})  {m.p1_name}={p1:.3f} '
          f'{m.p2_name}={p2:.3f}', flush=True)

    try:
        res = compute_rates(p1, p2, args)
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        return

    pt_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_f, model=model, ix=ix, iy=iy, stride=args.stride,
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
    p.add_argument('--model',    type=str, default=MODEL_NAME,
                   choices=SUPPORTED_MODELS)
    p.add_argument('--frame_seeds', type=str, default='auto',
                   choices=('auto', 'default', 'ring'),
                   help="optimiser frame seeds: 'ring' adds rings in the "
                        "model's field plane (auto: ring for models in "
                        'RING_AXIS, i.e. model8)')
    p.add_argument('--out_dir',  type=str, default=None,
                   help='default results_<model>_rate')
    p.add_argument('--stride',   type=int, default=1,
                   help='stride on the model grid (1 = full 51x51 = 2601 pts)')
    p.add_argument('--dim',      type=int, default=None,
                   help='lattice dimension of the bond Trotter convention '
                        '(each qubit sits on 2*dim bonds); must match the scan. '
                        "Default: the model's ModelSpec.dim (DIM_DEFAULT = "
                        f'{DIM_DEFAULT} for model4 / model8, 1 for model10)')
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
    if args.out_dir is None:
        args.out_dir = f'results_{args.model}_rate'
    if args.dim is None:
        args.dim = MODELS[args.model].dim

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
    print(f'[chunk {args.task_id}/{args.n_chunks}] {args.model}: '
          f'{len(ids)} of {n_total} points ({nx}x{ny} grid)', flush=True)
    for pid in ids:
        run_point(pid // ny, pid % ny, args)


if __name__ == '__main__':
    main()
