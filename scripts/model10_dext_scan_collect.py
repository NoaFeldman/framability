"""
Collect of the model10 d_ext scan (scripts/model10_dext_scan_worker.py).

Per scanned (Delta, d_ext): the best certified rate over every search
restart and hop round, and how far it can be trusted:
  hits        search restarts within tol of the best,
              tol = max(--tol_abs, --tol_rel * best);  cold_hits: those
              among the cold restarts (never saw a pipeline frame -- the
              independent reproduction; seeded restarts share their seed
              pool, so their agreement alone is weaker evidence)
  last_gain   best before the last hop round - best after it
  converged   hits >= --min_hits, cold_hits >= --min_cold_hits and
              last_gain <= tol (no hop round: the hit test alone)
The pipelines' values at the same points fill the other d_ext: best_known
over model10_seeded (d_ext 4, 8) and model10_seeded_d12 (12), and the
original scan's stored rate_heis_4 / 6 / 8 files.  Padding a frame never
raises its rate, so the value at d_ext m is the minimum over every value at
d_ext m' <= m (scan and pipelines); entries lowered that way are reported.

Writes <out_dir>/model10_dext_scan/summary.npz:
  deltas (P,), d_ext (M,), rate (P, M) [nested], rate_scan, rate_pipe,
  source (P, M) 'scan' | 'pipeline' | 'padded d<k>' | '',
  scanned, converged (P, M) bool, hits, cold_hits, n_search, n_cold,
  n_hop_rounds (P, M) int, last_gain (P, M), label (P, M) best frame origin,
  search_vals (P, M, R) certified search results (nan-padded),
  search_cold (P, M, R) bool, S_<ip>_<m> best scanned frames.
The figure is drawn by scripts/model10_opt_rate_vs_dext_plot.py from it.

Usage:
    python scripts/model10_dext_scan_collect.py [--deltas 0.5 0.6 0.7 --d_exts 6 10]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from model10_seeded_worker import MODEL, grid_vals, STORED_DIRS_DEFAULT   # noqa: E402
from model10_seeded_qrefine_worker import best_known                      # noqa: E402
from model10_d12_worker import d12_dir, d8_dir                            # noqa: E402
from model10_dext_scan_worker import (SCAN_DIR, PIPE_D_EXTS,              # noqa: E402
                                      STORED_D_EXTS, combo_dir, diag_points)
from rate_gopt_worker import stored_frames                                # noqa: E402


def load_combo(out_dir: str, m: int, ix: int, iy: int) -> dict:
    """Every result file of one (point, d_ext): search values / kinds,
    hop values per round, best frame."""
    d = combo_dir(out_dir, m, ix, iy)
    search, cold, hops = [], [], {}
    best = (np.inf, None, '')
    for f in sorted(d.glob('search_r[0-9][0-9][0-9].npz')) + \
            sorted(d.glob('hop[0-9][0-9]_r[0-9][0-9][0-9].npz')):
        try:
            z = np.load(f, allow_pickle=True)
            v = float(z['rate'])
        except Exception:                                   # noqa: BLE001
            continue
        if f.name.startswith('search'):
            search.append(v)
            cold.append(str(z['kind']) == 'cold')
        else:
            hops.setdefault(int(f.name[3:5]), []).append(v)
        if np.isfinite(v) and v < best[0]:
            best = (v, np.asarray(z['S'], float), f'{f.stem}: {z["label"]}')
    return dict(search=np.array(search, float), cold=np.array(cold, bool),
                hops=hops, best=best)


def pipeline_values(args, ix: int, iy: int) -> dict:
    """{d_ext: lowest stored rate} of the earlier pipelines at the point."""
    vals: dict = {}

    def put(k, v):
        if np.isfinite(v):
            vals[k] = min(vals.get(k, np.inf), float(v))

    for d in args.stored_dirs:
        pt = Path(d) / args.model
        if pt.is_dir():
            for k, lst in stored_frames(pt, ix, iy, STORED_D_EXTS).items():
                for v, _ in lst:
                    put(k, v)
    pt_seed, pt_d12 = d8_dir(args.out_dir, 1, args.model), d12_dir(args.out_dir, 1, args.model)
    for k in PIPE_D_EXTS:
        v, S, _ = best_known(pt_d12 if k == 12 else pt_seed, ix, iy, f'rate_{k}', f'S_{k}')
        if S is not None:
            put(k, v)
    return vals


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--deltas', type=float, nargs='+', default=[0.5, 0.6, 0.7])
    p.add_argument('--d_exts', type=int, nargs='+', default=[6, 10],
                   help='the scanned d_ext (as passed to the worker)')
    p.add_argument('--model', type=str, default=MODEL, choices=sorted(MODELS))
    p.add_argument('--out_dir', type=str, default='results_model10_rate')
    p.add_argument('--stored_dirs', type=str, nargs='*',
                   default=list(STORED_DIRS_DEFAULT))
    p.add_argument('--tol_abs', type=float, default=1e-5)
    p.add_argument('--tol_rel', type=float, default=1e-4)
    p.add_argument('--min_hits', type=int, default=3)
    p.add_argument('--min_cold_hits', type=int, default=1,
                   help='hits needed among the cold restarts (an independent '
                        'reproduction; seeded restarts share their seed pool)')
    args = p.parse_args()

    pts = diag_points(args.deltas, args.model)
    scan_m = sorted(set(args.d_exts))
    combos = {(ip, m): load_combo(args.out_dir, m, ix, iy)
              for ip, (ix, iy) in enumerate(pts) for m in scan_m}
    pipe = [pipeline_values(args, ix, iy) for ix, iy in pts]
    d_ext = sorted(set(scan_m) | {k for pv in pipe for k in pv})
    P, M = len(pts), len(d_ext)
    R = max([len(c['search']) for c in combos.values()] + [1])

    rate_scan = np.full((P, M), np.nan)
    rate_pipe = np.full((P, M), np.nan)
    hits = np.zeros((P, M), int)
    cold_hits = np.zeros((P, M), int)
    n_search = np.zeros((P, M), int)
    n_cold = np.zeros((P, M), int)
    n_rounds = np.zeros((P, M), int)
    last_gain = np.full((P, M), np.nan)
    converged = np.zeros((P, M), bool)
    scanned = np.zeros((P, M), bool)
    label = np.full((P, M), '', dtype=object)
    search_vals = np.full((P, M, R), np.nan)
    search_cold = np.zeros((P, M, R), bool)
    frames = {}
    for ip in range(P):
        for im, m in enumerate(d_ext):
            rate_pipe[ip, im] = pipe[ip].get(m, np.nan)
            if m not in scan_m:
                continue
            c = combos[(ip, m)]
            v_best = c['best'][0]
            if not np.isfinite(v_best):
                continue
            scanned[ip, im] = True
            rate_scan[ip, im] = v_best
            label[ip, im] = c['best'][2]
            frames[f'S_{ip}_{m}'] = c['best'][1]
            s, cl = c['search'], c['cold']
            search_vals[ip, im, :len(s)] = s
            search_cold[ip, im, :len(s)] = cl
            tol = max(args.tol_abs, args.tol_rel * abs(v_best))
            hit = np.isfinite(s) & (s <= v_best + tol)
            hits[ip, im], cold_hits[ip, im] = int(hit.sum()), int((hit & cl).sum())
            n_search[ip, im], n_cold[ip, im] = len(s), int(cl.sum())
            n_rounds[ip, im] = len(c['hops'])
            if c['hops']:
                last = max(c['hops'])
                before = [s] + [np.array(v) for r, v in c['hops'].items() if r < last]
                v_before = float(np.nanmin(np.concatenate(before))) if any(
                    len(b) for b in before) else np.inf
                last_gain[ip, im] = max(v_before - v_best, 0.0)
            gain_ok = n_rounds[ip, im] == 0 or last_gain[ip, im] <= tol
            converged[ip, im] = (hits[ip, im] >= args.min_hits
                                 and cold_hits[ip, im] >= args.min_cold_hits
                                 and gain_ok)

    # merge with the pipelines, then nest in d_ext
    rate = np.fmin(rate_scan, rate_pipe)
    source = np.full((P, M), '', dtype=object)
    source[np.isfinite(rate)] = 'pipeline'
    source[np.isfinite(rate_scan) & (rate_scan <= np.nan_to_num(rate_pipe, nan=np.inf))] = 'scan'
    for ip in range(P):
        for im in range(1, M):
            if rate[ip, im - 1] < np.nan_to_num(rate[ip, im], nan=np.inf):
                rate[ip, im] = rate[ip, im - 1]
                src = str(source[ip, im - 1])
                source[ip, im] = src if src.startswith('padded') else f'padded d{d_ext[im - 1]}'

    # report
    print(f'[{args.model} dext scan] {P} points x d_ext {d_ext} (scanned {scan_m}); '
          f'tol = max({args.tol_abs:g}, {args.tol_rel:g} x rate), converged = '
          f'>= {args.min_hits} hits' + (f', >= {args.min_cold_hits} cold'
                                         if args.min_cold_hits else '')
          + ', last hop round gain <= tol')
    print(f'{"Delta":>6s} {"d_ext":>5s} {"rate":>12s} {"source":>10s} '
          f'{"scan":>12s} {"pipeline":>12s} {"hits":>7s} {"cold":>6s} '
          f'{"rounds":>6s} {"last gain":>10s}  status')
    for ip, (ix, iy) in enumerate(pts):
        for im, m in enumerate(d_ext):
            if scanned[ip, im]:
                st = 'converged' if converged[ip, im] else 'NOT CONVERGED'
                hs = f'{hits[ip, im]}/{n_search[ip, im]}'
                cs = f'{cold_hits[ip, im]}/{n_cold[ip, im]}'
                lg = f'{last_gain[ip, im]:10.2e}' if n_rounds[ip, im] else f'{"-":>10s}'
            else:
                st, hs, cs, lg = 'pipeline only', '-', '-', f'{"-":>10s}'
            print(f'{args.deltas[ip]:6.2f} {m:5d} {rate[ip, im]:12.8f} '
                  f'{str(source[ip, im]):>10s} {rate_scan[ip, im]:12.8f} '
                  f'{rate_pipe[ip, im]:12.8f} {hs:>7s} {cs:>6s} '
                  f'{n_rounds[ip, im]:6d} {lg}  {st}')
    n_bad = int((scanned & ~converged).sum())
    if n_bad:
        print(f'  {n_bad} scanned (Delta, d_ext) not converged: run more hop rounds '
              '(SKIP_SEARCH=1 bash scripts/submit_model10_dext_scan.sh) or more '
              'search replicas')
    for ip in range(P):
        for im, m in enumerate(d_ext):
            if str(source[ip, im]).startswith('padded'):
                print(f'  Delta={args.deltas[ip]:.2f} d_ext={m}: value from '
                      f'{source[ip, im]} ({rate[ip, im]:.8f}) below its own best '
                      f'({np.fmin(rate_scan[ip, im], rate_pipe[ip, im]):.8f})')

    out = Path(args.out_dir) / SCAN_DIR / 'summary.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.npz')
    np.savez(tmp, model=args.model, deltas=np.array(args.deltas, float),
             ix=np.array([ix for ix, _ in pts]), iy=np.array([iy for _, iy in pts]),
             d_ext=np.array(d_ext), scan_d_ext=np.array(scan_m), rate=rate,
             rate_scan=rate_scan, rate_pipe=rate_pipe, source=source.astype(str),
             scanned=scanned, converged=converged, hits=hits, cold_hits=cold_hits,
             n_search=n_search, n_cold=n_cold, n_hop_rounds=n_rounds,
             last_gain=last_gain, label=label.astype(str), search_vals=search_vals,
             search_cold=search_cold, tol_abs=args.tol_abs, tol_rel=args.tol_rel,
             min_hits=args.min_hits, min_cold_hits=args.min_cold_hits,
             title=MODELS[args.model].title, **frames)
    os.replace(tmp, out)
    print(f'[{args.model} dext scan] wrote {out}', flush=True)


if __name__ == '__main__':
    main()
