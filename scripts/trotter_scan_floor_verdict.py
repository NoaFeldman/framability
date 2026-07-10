"""
High-precision floor verdict for near-floor Trotter-scan framabilities.

The production LP evaluators run HiGHS at its default feasibility/optimality
tolerances (~1e-7), so stored values with 0 < opt_fra - 1 <~ 1e-7 are
numerically indistinguishable from the floor, and the two evaluators (batched
fast LP vs per-column reference LP) disagree at the 1e-8 level.  This script
settles those points WITHOUT any optimisation: it re-evaluates each point's
best stored frame on its own gate with HiGHS tolerances tightened to
--lp_tol (default 1e-10, the tightest HiGHS accepts), pushing the resolution
down two to three orders of magnitude.

For every point whose best-known value sits in the ambiguous band
1 + band_min < opt_fra <= 1 + band_max, the verdict is:

  CERTIFIED   tight value <= 1 + snap_tol  ->  the point is framable beyond
              the resolution of float64 + LP; exactly 1.0 is written so
              contours at 1.0 and island counts come out clean.
  IMPROVED    tight value below the stored value (but above snap_tol) ->
              the tighter value is written.
  GAP         tight value holds a plateau above the stored value ->
              nothing is written for the key (the merge takes minima anyway);
              the point genuinely sits above the floor at this resolution.

The raw tight value is always recorded as <key>_tight for the record.
Output files pt_verdict_r<NN>_<ix>_<iy>.npz (only for points with a
CERTIFIED / IMPROVED key) are picked up by the standard refine collect.

Usage (cheap - one whole model per invocation, LP solves only):
    python scripts/trotter_scan_floor_verdict.py --model model1 --round 1 \
        [--out_dir results_trotter_v3] [--band_min 1e-12] [--band_max 1e-4] \
        [--snap_tol 1e-9] [--lp_tol 1e-10]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS
from dissipative_PT import _kron_power
from trotter_refine_common import KEYS, base_path, best_known, build_gate

TOL = 1e-12


def framability_lp_tight(D: np.ndarray, gate: np.ndarray,
                         lp_tol: float) -> float:
    """Framability of `gate` in the fixed frame D at tightened LP tolerances.

    Same quantity as dissipative_PT._framability_lp (max over target columns
    of the atomic norm min ||u||_1 s.t. D u = gate^T d_j), but solved through
    scipy.optimize.linprog(method='highs') with primal/dual feasibility
    tolerances set to lp_tol instead of the ~1e-7 HiGHS defaults.  The LPs
    are tiny (2*d_ext variables, 16 equalities), so this stays cheap.
    """
    D = np.asarray(D, dtype=float)
    nrows, d_ext = D.shape
    Y = gate.real.T @ D
    c = np.ones(2 * d_ext)                      # u = u+ - u-;  ||u||_1 = sum
    A_eq = np.hstack([D, -D])
    bounds = [(0.0, None)] * (2 * d_ext)
    opts = {'presolve': False,
            'primal_feasibility_tolerance': lp_tol,
            'dual_feasibility_tolerance': lp_tol}
    best = 0.0
    for j in range(d_ext):
        r = linprog(c, A_eq=A_eq, b_eq=Y[:, j], bounds=bounds,
                    method='highs', options=opts)
        if not r.success:
            return np.inf
        best = max(best, float(r.fun))
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, required=True, choices=list(MODELS))
    p.add_argument('--round',    type=int, required=True,
                   help='label of the pt_verdict_r<NN> output files')
    p.add_argument('--out_dir',  type=str, default='results_trotter_v3')
    p.add_argument('--band_min', type=float, default=1e-12,
                   help='only points with opt_fra - 1 > band_min are checked')
    p.add_argument('--band_max', type=float, default=1e-4,
                   help='only points with opt_fra - 1 <= band_max are checked')
    p.add_argument('--snap_tol', type=float, default=1e-9,
                   help='tight value <= 1 + snap_tol certifies the point as '
                        'framable (exactly 1.0 is stored)')
    p.add_argument('--lp_tol',   type=float, default=1e-10,
                   help='HiGHS primal/dual feasibility tolerance (1e-10 is '
                        'the tightest HiGHS accepts)')
    args = p.parse_args()

    model = MODELS[args.model]
    out_dir = Path(args.out_dir)
    mdir = out_dir / model.name
    t0 = time.perf_counter()

    n_checked = 0
    n_certified = {k: 0 for k in KEYS}
    n_improved = {k: 0 for k in KEYS}
    gaps: dict = {k: [] for k in KEYS}          # surviving genuine gaps
    n_written = 0

    print(f'[{model.name}] floor verdict: band (1+{args.band_min:g}, '
          f'1+{args.band_max:g}], snap {args.snap_tol:g}, '
          f'lp_tol {args.lp_tol:g}', flush=True)

    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            base = base_path(out_dir, model, ix, iy)
            if not base.exists():
                continue

            todo = {}
            for key, (s_key, _) in KEYS.items():
                v, S = best_known(out_dir, model, ix, iy, key, s_key)
                if (S is not None and np.isfinite(v)
                        and args.band_min < v - 1.0 <= args.band_max):
                    todo[key] = (v, S)
            if not todo:
                continue

            gate = build_gate(model, ix, iy, np.load(base))
            n_checked += 1
            payload = {'round': np.array(args.round),
                       'ix': np.array(ix), 'iy': np.array(iy)}
            write = False

            for key, (v, S) in todo.items():
                s_key = KEYS[key][0]
                f_tight = framability_lp_tight(_kron_power(S, 2), gate,
                                               args.lp_tol)
                payload[f'{key}_tight'] = np.array(f_tight)
                if f_tight <= 1.0 + args.snap_tol:
                    payload[key] = np.array(1.0)
                    payload[s_key] = np.asarray(S)
                    n_certified[key] += 1
                    write = True
                    verdict = 'CERTIFIED -> 1.0'
                elif f_tight < v - TOL:
                    payload[key] = np.array(f_tight)
                    payload[s_key] = np.asarray(S)
                    n_improved[key] += 1
                    write = True
                    verdict = 'IMPROVED'
                else:
                    gaps[key].append((ix, iy, v, f_tight))
                    verdict = 'GAP'
                print(f'  ({ix:3d},{iy:3d}) {key}: stored 1+{v-1.0:.3e}  '
                      f'tight 1+{f_tight-1.0:.3e}  [{verdict}]', flush=True)

            if write:
                mdir.mkdir(parents=True, exist_ok=True)
                np.savez(mdir /
                         f'pt_verdict_r{args.round:02d}_{ix:03d}_{iy:03d}.npz',
                         **payload)
                n_written += 1

    print(f'\n[{model.name}] verdict summary '
          f'({n_checked} point(s) in band, {n_written} file(s) written, '
          f'{time.perf_counter()-t0:.0f}s):', flush=True)
    for key in KEYS:
        print(f'  {key}: certified {n_certified[key]}, '
              f'improved {n_improved[key]}, '
              f'genuine gaps {len(gaps[key])}', flush=True)
        for ix, iy, v, f_tight in gaps[key]:
            print(f'    gap at ({ix:3d},{iy:3d}) '
                  f'{model.p1_name}={model.p1_vals[ix]:.3f} '
                  f'{model.p2_name}={model.p2_vals[iy]:.3f}: '
                  f'stored 1+{v-1.0:.3e}, tight 1+{f_tight-1.0:.3e}',
                  flush=True)


if __name__ == '__main__':
    main()
