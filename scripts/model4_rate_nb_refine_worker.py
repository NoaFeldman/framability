"""
FULL neighbour refining of the optimised HEISENBERG framability RATES
(rate_heis_4 / rate_heis_6) of a rate-panel scan -- one round, one point.

This is the full twin of scripts/model4_rate_quick_refine_worker.py, whose
helpers it reuses (best_known, neighbor_frames, propagate, _embed_S).  The
quick worker only touches points on the boundary of the mu* = 0 region; this
one refines EVERY grid point, so it also lowers the optimised rates inside the
mu* > 0 region, where independent per-point restarts leave jagged values and
islands.  It is the rate-picture analogue of scripts/trotter_scan_refine_worker.py.

Per point, for d_ext_single = 4 and 6:
  1. propagation (no optimisation): every 4-connected neighbour's best-known
     frame is scored on THIS point's generator with the batched LP
     generator_log_norm; an improvement is confirmed by the independent
     per-column LP generator_log_norm_reference before it is accepted.  Every
     finite value is a certified upper bound, so min(own, neighbours) is
     always sound.
  2. re-optimisation: framability_rate.minimize_rate seeded with the incumbent
     and every neighbour frame (reduced restarts / maxfev); kept only if it
     strictly beats the incumbent.  Skipped for points already at the floor
     mu* = 0, which no frame can go below.
Then the cross-d_ext step: a d=4 frame better than the best d=6 value is
embedded into d=6 and re-optimised (six columns contain four, so
rate_heis_6 <= rate_heis_4 must hold).

Rounds are sequential: round r reads the base scan plus every earlier round of
both flavours (quick _qrefine_r*, full _nrefine_r*), so a good frame spreads at
least one grid ring per round.  Unlike the quick worker, every processed point
writes its round file -- also when nothing improved -- because that file is
the resume marker: a resubmitted round skips finished points.  Files are
written atomically (hidden temporary name, then rename), so a killed task
never leaves a truncated marker behind.

Reads:  results_<model>_rate/<model>/pt_<ix>_<iy>.npz                (base scan)
        results_<model>_rate/<model>/pt_<ix>_<iy>_*refine_r*.npz      (earlier rounds)
Writes: results_<model>_rate/<model>/pt_<ix>_<iy>_nrefine_r<NN>.npz   (every point)

scripts/model4_rate_panels_collect.py takes the minimum over the base scan and
every refine round automatically.  Run 10 rounds + the collect via
scripts/submit_model4_rate_nb_refine.sh.  model10 is the Shibata-Katsura
dissipative quantum Ising chain (https://arxiv.org/abs/1904.12505).

Usage:
    python scripts/model4_rate_nb_refine_worker.py --model model10 --round 1 --task_id 0 --n_chunks 200
    python scripts/model4_rate_nb_refine_worker.py --model model10 --round 1 --task_id 17   # single point
    python scripts/model4_rate_nb_refine_worker.py --model model10 --round 1 --report       # round summary
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
                                      DIM_DEFAULT)
from optimize_framability import _kron_power                            # noqa: E402
from framability_rate import (minimize_rate,                            # noqa: E402
                              generator_log_norm_reference, RATE_VERSION)
from model4_rate_panels_worker import MODEL_NAME, SUPPORTED_MODELS      # noqa: E402
from model4_rate_quick_refine_worker import (KEYS, TOL, RATE_FLOOR,     # noqa: E402
                                             best_known, neighbor_frames,
                                             propagate, _embed_S)

ROUND_TAG = 'nrefine'


def round_path(pt_dir: Path, ix: int, iy: int, round_: int) -> Path:
    """This point's full-refine file of round `round_`."""
    return pt_dir / f'pt_{ix:03d}_{iy:03d}_{ROUND_TAG}_r{round_:02d}.npz'


def _save_atomic(path: Path, **arrays) -> None:
    """np.savez to a hidden temporary name, then rename.  A killed task never
    leaves a truncated file that a resubmission would take for a finished
    point, and the temporary name matches none of the readers' pt_* globs."""
    tmp = path.with_name('.tmp_' + path.name)
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _reoptimise(A, d_ext: int, val: float, S, seeds, *, maxfev: int, seed: int,
                args):
    """minimize_rate seeded with `seeds`; returns the better of the incumbent
    (val, S) and the new result.  Skipped at the floor, which no frame beats."""
    if val <= RATE_FLOOR + args.rate_tol:
        return val, S
    S_new, mu_new, _ = minimize_rate(
        A, d_ext, n_restarts=args.n_restarts, maxfev=maxfev, seed=seed,
        verbose=False, polish_iters=args.polish, extra_init_S=seeds or None)
    if np.isfinite(mu_new) and mu_new < val - TOL:
        return mu_new, S_new
    return val, S


def run_point(model, point_id: int, args) -> None:
    pt_dir = Path(args.out_dir) / model.name
    ix, iy = point_id // model.N_Y, point_id % model.N_Y
    p1, p2 = float(model.p1_vals[ix]), float(model.p2_vals[iy])

    out = round_path(pt_dir, ix, iy, args.round)
    if out.exists():
        print(f'[skip] {out.name} already exists', flush=True)
        return
    if not (pt_dir / f'pt_{ix:03d}_{iy:03d}.npz').exists():
        return                      # point not scanned yet -- nothing to refine

    t0 = time.perf_counter()
    print(f'[point {point_id}/{model.N_TOTAL}] {model.name} full round '
          f'{args.round}  {model.p1_name}={p1:.3f} {model.p2_name}={p2:.3f}',
          flush=True)
    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    A = build_bond_lindbladian(H1, H2, jumps1, jumps2, args.dim).real.T
    seed = args.seed + point_id + 100000 * args.round
    maxfev = {'rate_heis_4': args.maxfev_4, 'rate_heis_6': args.maxfev_6}

    prev, results = {}, {}
    for off, (key, (s_key, d_ext)) in enumerate(KEYS.items()):
        self_val, self_S = best_known(pt_dir, ix, iy, key, s_key)
        nb_list = neighbor_frames(pt_dir, model, ix, iy, key, s_key)
        prev[key] = self_val

        # stage 1: propagation (a handful of LPs)
        val, S, j = propagate(A, self_val, self_S, nb_list)
        # stage 2: re-optimisation seeded with the incumbent + every neighbour
        seeds = ([S] if S is not None else []) + [f for _, f in nb_list]
        val, S = _reoptimise(A, d_ext, val, S, seeds, maxfev=maxfev[key],
                             seed=seed + off, args=args)
        results[key] = (val, S)
        print(f'  d{d_ext}: {self_val:.6e} -> {val:.6e}'
              + (f'  (propagated from neighbour {j})' if j >= 0 else '')
              + f'  [{time.perf_counter() - t0:.0f}s]', flush=True)

    # ── cross-d_ext step: the d=6 frame family contains every d=4 frame ─────
    (v4, S4), (v6, S6) = results['rate_heis_4'], results['rate_heis_6']
    if S4 is not None and v4 < v6 - TOL:
        seed6 = _embed_S(S4, 4, 6)
        v_emb = generator_log_norm_reference(_kron_power(seed6, 2), A)
        if np.isfinite(v_emb) and v_emb < v6 - TOL:
            v6, S6 = v_emb, seed6
        v6, S6 = _reoptimise(A, 6, v6, S6,
                             [seed6] + ([S6] if S6 is not None else []),
                             maxfev=args.maxfev_6, seed=seed + 7, args=args)
        print(f'  cross d4->d6: {results["rate_heis_6"][0]:.6e} -> {v6:.6e}',
              flush=True)
        results['rate_heis_6'] = (v6, S6)

    improved = {k: bool(results[k][0] < prev[k] - TOL) for k in KEYS}
    # {p1_name: p1, p2_name: p2} keeps model4's gamma / gamma_p keys unchanged
    payload = dict(model=model.name, ix=ix, iy=iy,
                   **{model.p1_name: p1, model.p2_name: p2},
                   dim=args.dim, round=args.round, refine_mode='full',
                   rate_version=RATE_VERSION, rate_floor=RATE_FLOOR,
                   rate_tol=args.rate_tol, t_refine=time.perf_counter() - t0)
    for key, (s_key, _) in KEYS.items():
        val, S = results[key]
        payload[f'{key}_prev'] = prev[key]
        payload[f'{key}_improved'] = improved[key]
        if np.isfinite(val) and S is not None:
            payload[key] = val
            payload[s_key] = S
    pt_dir.mkdir(parents=True, exist_ok=True)
    _save_atomic(out, **payload)
    print(f'  saved {out.name}  d4={results["rate_heis_4"][0]:.6e} '
          f'd6={results["rate_heis_6"][0]:.6e}  improved={improved}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


def report_round(pt_dir: Path, round_: int) -> None:
    """One-line summary of a round on disk: points processed / improved."""
    n_base = len(list(pt_dir.glob('pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz')))
    files = sorted(pt_dir.glob(f'pt_*_*_{ROUND_TAG}_r{round_:02d}.npz'))
    n_imp = dict.fromkeys(KEYS, 0)
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            continue
        for k in KEYS:
            if f'{k}_improved' in d.files and bool(d[f'{k}_improved']):
                n_imp[k] += 1
    print(f'[{pt_dir.name} full refine round {round_}] {len(files)}/{n_base} '
          'scanned point(s) processed; improved: '
          + ', '.join(f'{k} {n}' for k, n in n_imp.items()), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, default=MODEL_NAME,
                   choices=SUPPORTED_MODELS)
    p.add_argument('--round',    type=int, required=True,
                   help='full neighbour-refine round (1..10); round r reads the '
                        'base scan and every earlier round')
    p.add_argument('--task_id',  type=int, default=None,
                   help='chunk id (with --n_chunks > 1) or flat grid index')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--report',   action='store_true',
                   help='only summarise the round already on disk')
    p.add_argument('--out_dir',  type=str, default=None,
                   help='default results_<model>_rate')
    p.add_argument('--dim',      type=int, default=None,
                   help="bond Trotter convention; default: the model's "
                        f'ModelSpec.dim (DIM_DEFAULT = {DIM_DEFAULT})')
    p.add_argument('--rate_tol', type=float, default=1e-6,
                   help='a rate within this of 0 sits on the floor and is not '
                        're-optimised (matches the collect script\'s CONTOUR_TOL)')
    p.add_argument('--n_restarts', type=int, default=3,
                   help='minimize_rate restarts on top of the neighbour seeds '
                        '(the base scan used 8)')
    p.add_argument('--maxfev_4', type=int, default=1000)
    p.add_argument('--maxfev_6', type=int, default=500)
    p.add_argument('--polish',   type=int, default=150)
    p.add_argument('--seed',     type=int, default=0)
    args = p.parse_args()

    model = MODELS[args.model]
    if args.out_dir is None:
        args.out_dir = f'results_{args.model}_rate'
    if args.dim is None:
        args.dim = model.dim

    if args.report:
        report_round(Path(args.out_dir) / model.name, args.round)
        return
    if args.task_id is None:
        p.error('--task_id is required unless --report is given')

    n_total = model.N_TOTAL
    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            print(f'ERROR: task_id must be in [0, {n_total})', file=sys.stderr)
            sys.exit(1)
        ids = [args.task_id]
    else:
        if not (0 <= args.task_id < args.n_chunks):
            print(f'ERROR: chunk id must be in [0, {args.n_chunks})',
                  file=sys.stderr)
            sys.exit(1)
        ids = list(range(args.task_id, n_total, args.n_chunks))
        print(f'[chunk {args.task_id}/{args.n_chunks}] {model.name} full round '
              f'{args.round}: {len(ids)} of {n_total} points', flush=True)

    for pid in ids:
        try:
            run_point(model, pid, args)
        except Exception as e:      # no marker written -> retried on resubmit
            print(f'  ERROR point {pid}: {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    main()
