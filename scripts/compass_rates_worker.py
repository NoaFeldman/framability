"""
Compass-chain framability-RATE panels (1-6 of each case's figure).

For one case of compass_chain.CASES and every (gamma, h) point of its grid
(GAMMA_VALS x H_VALS, 11 x 11) this computes six dt-free framability rates

  1. rate_pauli    Pauli frame               framability_rate_frames.pauli_rate
  2. rate_stab3    stabilizer-3 frame        framability_rate_frames.stabilizer_3_rate
  3. rate_heis_4   optimised Heisenberg, d_ext_single=4   framability_rate.minimize_rate
  4. rate_heis_6   optimised Heisenberg, d_ext_single=6   framability_rate.minimize_rate
  5. rate_schro_4  optimised Schrodinger, d_ext_single=4  framability_rate_state.minimize_state_rate
  6. rate_schro_6  optimised Schrodinger, d_ext_single=6  framability_rate_state.minimize_state_rate

of the chain's two bond gates.  The chain has two inequivalent bonds (Jx XX and
Jy YY, one of each on every qubit) and the site dephasing may be split between
them in any ratio: the XX gate carries alpha * gamma, the YY gate
(1 - alpha) * gamma.  For EACH measure separately, dephasing_split.balance_split
searches alpha until the two gates' rates agree to --rel_tol (default 0.1), and
the stored panel value is

    rate = max(mu_XX(alpha), mu_YY(1 - alpha))

at that split, the cost of the more expensive gate.  The field is always shared
evenly (build_bond_lindbladian's 1/2 bond share in 1D).  At gamma = 0 there is
nothing to split and a single evaluation at alpha = 1/2 is stored.

Nothing about the measures is (re)implemented here: this file maps units of
work onto array tasks, hands the existing rate functions to the split search,
and stores results.

Unit of work = (grid point, measure): 11 * 11 * 6 = 726 units per case, strided
across the array so every task mixes cheap and expensive measures.

Output: <out_dir>/<case>/rates/pt_<ig:03d>_<ih:03d>_<measure>.npz
        (one file per unit, written atomically; existing files are skipped, so a
        partial array can simply be resubmitted)

Usage:
    python scripts/compass_rates_worker.py --case jx1.0_jy1.0_hx --task_id 0 --n_chunks 200
    python scripts/compass_rates_worker.py --case jx0.5_jy1.0_hxy --task_id 5 --measures rate_pauli
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
from compass_chain import (CASES, GAMMA_VALS, H_VALS, bond_lindbladian,  # noqa: E402
                           COMPASS_VERSION)
from dephasing_split import balance_split, SPLIT_VERSION                 # noqa: E402
from framability_rate import minimize_rate, RATE_VERSION                 # noqa: E402
from framability_rate_frames import (pauli_rate, stabilizer_3_rate,      # noqa: E402
                                     RATE_FRAMES_VERSION)
from framability_rate_state import (minimize_state_rate,                 # noqa: E402
                                    RATE_STATE_VERSION)

# The six rate panels, in figure order: (npz measure key, human label).
RATE_KEYS = [
    ('rate_pauli',   'Pauli framability rate'),
    ('rate_stab3',   'Stabilizer-3 framability rate'),
    ('rate_heis_4',  r'Opt Heisenberg rate ($d_{\rm ext}=4$)'),
    ('rate_heis_6',  r'Opt Heisenberg rate ($d_{\rm ext}=6$)'),
    ('rate_schro_4', r'Opt Schrodinger rate ($d_{\rm ext}=4$)'),
    ('rate_schro_6', r'Opt Schrodinger rate ($d_{\rm ext}=6$)'),
]
MEASURES = [k for k, _ in RATE_KEYS]


def unit_path(out_dir, case_name: str, ig: int, ih: int, key: str) -> Path:
    return Path(out_dir) / case_name / 'rates' / f'pt_{ig:03d}_{ih:03d}_{key}.npz'


def make_rate_fn(key: str, case, bond: str, gamma: float, h: float, args):
    """frac -> (mu, info): measure `key` of the `bond` gate when it carries the
    fraction frac of the site dephasing.  The optimised frames warm-start from
    the frame found at the previous split tried for the same gate, so the split
    search moves one frame along instead of re-optimising from scratch."""
    warm: dict = {}

    def fn(frac: float):
        L = bond_lindbladian(case, bond, gamma, h, dephasing_frac=frac)
        if key == 'rate_pauli':
            return pauli_rate(L), None
        if key == 'rate_stab3':
            return stabilizer_3_rate(L), None
        de = int(key.rsplit('_', 1)[1])
        if key.startswith('rate_heis_'):
            S, mu, info = minimize_rate(
                L.T, de, n_restarts=args.heis_restarts, maxfev=args.heis_maxfev,
                seed=args.seed + de, verbose=False, polish_iters=args.polish,
                check_swap=not args.no_swap_check,
                extra_init_S=[warm['S']] if 'S' in warm else None)
            warm['S'] = S
            return mu, dict(S=S, mu_search=info['mu_search'],
                            mu_polish=info['mu_polish'])
        if key.startswith('rate_schro_'):
            S, mu, x = minimize_state_rate(
                L, de, n_restarts=args.schro_restarts, maxfev=args.schro_maxfev,
                seed=args.seed + de, verbose=False, return_x=True,
                extra_init_xs=[warm['x']] if 'x' in warm else None)
            warm['x'] = x
            return mu, dict(S=S, x=x)
        raise ValueError(f'unknown measure {key!r}')

    return fn


def _save_atomic(path: Path, **arrays) -> None:
    """np.savez to a temporary name, then rename: a job killed mid-write never
    leaves a truncated file that a resubmission would skip as done."""
    tmp = path.with_name(path.stem + '.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def run_unit(case, ig: int, ih: int, key: str, args) -> None:
    gamma, h = float(GAMMA_VALS[ig]), float(H_VALS[ih])
    out_f = unit_path(args.out_dir, case.name, ig, ih, key)
    if out_f.exists():
        print(f'[skip] {case.name}/rates/{out_f.name} already exists', flush=True)
        return

    t0 = time.perf_counter()
    print(f'[{case.name}] point ({ig},{ih})  gamma={gamma:.3f}  h={h:.3f}  '
          f'{key}', flush=True)

    # Without dephasing the split is void: one evaluation at the even share.
    max_evals = 1 if gamma == 0.0 else args.max_evals
    try:
        res = balance_split(make_rate_fn(key, case, 'xx', gamma, h, args),
                            make_rate_fn(key, case, 'yy', gamma, h, args),
                            rel_tol=args.rel_tol, abs_tol=args.abs_tol,
                            max_evals=max_evals)
    except Exception as e:
        print(f'  ERROR: {type(e).__name__}: {e}', flush=True)
        return

    extra = {}
    for tag, info in (('xx', res.info_a), ('yy', res.info_b)):
        for k, v in (info or {}).items():
            extra[f'{k}_{tag}'] = v

    out_f.parent.mkdir(parents=True, exist_ok=True)
    _save_atomic(out_f, case=case.name, Jx=case.Jx, Jy=case.Jy, field=case.field,
                 ig=ig, ih=ih, gamma=gamma, h=h, measure=key,
                 rate=res.rate, alpha=res.alpha,
                 mu_xx=res.mu_a, mu_yy=res.mu_b, mismatch=res.mismatch,
                 balanced=res.balanced, n_evals=res.n_evals,
                 alphas=res.alphas, mus_xx=res.mus_a, mus_yy=res.mus_b,
                 rel_tol=args.rel_tol, abs_tol=args.abs_tol,
                 compass_version=COMPASS_VERSION, split_version=SPLIT_VERSION,
                 rate_version=RATE_VERSION,
                 rate_frames_version=RATE_FRAMES_VERSION,
                 rate_state_version=RATE_STATE_VERSION,
                 t_total=time.perf_counter() - t0, **extra)
    print(f'  saved {out_f.name}  rate={res.rate:+.5f}  alpha={res.alpha:.4f}  '
          f'mu_xx={res.mu_a:+.5f}  mu_yy={res.mu_b:+.5f}  '
          f'balanced={res.balanced}  evals={res.n_evals}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--case',     type=str, required=True, choices=list(CASES))
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the units into this many strided array tasks '
                        '(200 = the job cap); n_chunks<=1 means task_id is a '
                        'single flat unit index')
    p.add_argument('--out_dir',  type=str, default='results_compass')
    p.add_argument('--measures', nargs='+', default=MEASURES, choices=MEASURES)
    p.add_argument('--rel_tol',  type=float, default=0.1,
                   help='relative error at which the two bond gates count as even')
    p.add_argument('--abs_tol',  type=float, default=1e-6,
                   help='absolute agreement that also counts as even (both '
                        'rates at the floor)')
    p.add_argument('--max_evals', type=int, default=12,
                   help='split evaluations per unit (each one evaluates the '
                        'measure on both bond gates)')
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
                   help="skip minimize_rate's swap-symmetry assertion (both "
                        'bond gates are swap symmetric; escape hatch for '
                        'numerical edge cases)')
    args = p.parse_args()

    case = CASES[args.case]
    ng, nh = len(GAMMA_VALS), len(H_VALS)
    units = [(ig, ih, key) for ig in range(ng) for ih in range(nh)
             for key in args.measures]
    n_total = len(units)

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
            sys.exit(1)
        run_unit(case, *units[args.task_id], args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    ids = list(range(args.task_id, n_total, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {case.name}: {len(ids)} of '
          f'{n_total} units ({ng}x{nh} grid x {len(args.measures)} measures)',
          flush=True)
    for uid in ids:
        run_unit(case, *units[uid], args)


if __name__ == '__main__':
    main()
