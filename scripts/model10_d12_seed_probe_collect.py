"""
Summary of the d_ext = 12 seed probe (scripts/model10_d12_seed_probe_worker.py):
which of the new structured families beat the best known d_ext = 12 rate, by
how much, and where.  Read-only unless --adopt.

Prints
  * targets / processed / improved / visible (> --visible) / new rate-0 points,
    summed, mean and largest drop;
  * the control: how often the re-run yz6xy4 (already in the pipeline) alone
    beats the reference, and the incumbent-polish control if it was run --
    improvements that the control reproduces are the budget, not the seeds;
  * per family: points where its optimum alone beats the reference, where it
    was the best new candidate, where it is the origin of the final certified
    improvement, and the median gap (family - reference);
  * the --n_show largest improvements with their coordinates.
Writes <probe_dir>/probe_summary.npz (drop grid, flags, per-family tables).

--adopt promotes every improved point into the pipeline directory as a
randomised-refine round file  pt_<ix>_<iy>_rrefine_r<NN>.npz  (NN = last
rrefine round on disk + 1, keys rate_12 / S_12 / label_12 / rate_12_prev), so
best_known, the panels and refine_status pick it up and the next margin /
refine rounds spread it.  Run it only while no d_ext = 12 round is queued.

Usage:
    python scripts/model10_d12_seed_probe_collect.py
    python scripts/model10_d12_seed_probe_collect.py --out_dir results_model10_rate_dim2
    python scripts/model10_d12_seed_probe_collect.py --adopt
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from model10_seeded_worker import MODEL, grid_vals                        # noqa: E402
from model10_seeded_qrefine_worker import RREFINE_TAG                     # noqa: E402
from model10_d12_worker import d12_dir                                    # noqa: E402
from model10_d12_seed_probe_worker import (probe_dir, CONTROL_NAMES,      # noqa: E402
                                           KEY, S_KEY, LAB_KEY, M)


def load_probe(pd: Path) -> list:
    rows = []
    for f in sorted(pd.glob('pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz')):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception as e:                              # noqa: BLE001
            print(f'  warning: {f.name}: {e}', flush=True)
            continue
        rows.append(dict(
            file=f, ix=int(d['ix']), iy=int(d['iy']),
            v_ref=float(d[f'{KEY}_prev']), v_new=float(d[KEY]),
            label=str(d[LAB_KEY]), improved=bool(d['improved']),
            new_zero=bool(d['new_zero']),
            names=[str(s) for s in d['fam_names']],
            vals=np.asarray(d['fam_vals'], float),
            pol_names=[str(s) for s in d['polished_names']],
            pol_vals=np.asarray(d['polished_vals'], float),
            ctrl_fam=float(d['control_fam_polished']),
            ctrl_inc=float(d['control_incumbent_polished']),
            t=float(d['t'])))
    return rows


def last_round(pt: Path, tag: str) -> int:
    rs = [int(m.group(1)) for f in pt.glob(f'pt_*{tag}[0-9][0-9].npz')
          if (m := re.search(rf'{tag}(\d\d)\.npz$', f.name))]
    return max(rs) if rs else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default=MODEL, choices=sorted(MODELS))
    ap.add_argument('--out_dir', type=str, default='results_model10_rate')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--gain', type=float, default=1e-6,
                    help='a drop below this is not counted as an improvement')
    ap.add_argument('--visible', type=float, default=1e-3,
                    help='a drop above this shows on the colour scale')
    ap.add_argument('--n_show', type=int, default=15)
    ap.add_argument('--adopt', action='store_true',
                    help='write improved points into the d12 pipeline dir as '
                         'an rrefine round (see module docstring)')
    args = ap.parse_args()
    spec = MODELS[args.model]
    pd = probe_dir(args.out_dir, args.stride, args.model)
    p1, p2 = grid_vals(args.stride, args.model)
    nx, ny = len(p1), len(p2)
    tf = pd / 'targets.npz'
    n_targets = len(np.load(tf)['ix']) if tf.exists() else -1
    rows = load_probe(pd)
    print(f'[d12 seed probe] {args.model} {pd}: {len(rows)} processed of '
          f'{n_targets if n_targets >= 0 else "?"} targets', flush=True)
    if not rows:
        return

    drops = np.array([r['v_ref'] - r['v_new'] for r in rows])
    imp = drops > args.gain
    vis = drops > args.visible
    nz = np.array([r['new_zero'] for r in rows])
    print(f'   improved {int(imp.sum())}  visible (> {args.visible:g}) {int(vis.sum())}  '
          f'new rate-0 {int(nz.sum())}  | drop sum {drops[imp].sum():.4g}  '
          f'mean {drops[imp].mean() if imp.any() else 0:.3g}  '
          f'max {drops.max():.3g}  | median time {np.median([r["t"] for r in rows]):.0f}s',
          flush=True)

    # control: the pipeline's own family at the probe budget, incumbent polish
    cf = np.array([[r['ctrl_fam'], r['v_ref']] for r in rows])
    ok = np.isfinite(cf[:, 0])
    if ok.any():
        n_c = int(((cf[ok, 1] - cf[ok, 0]) > args.gain).sum())
        print(f'   control yz6xy4 (re-run + polish): beats the reference at {n_c}/{int(ok.sum())}'
              f' points -- improvements it reproduces are the budget, not the seeds',
              flush=True)
    ci = np.array([[r['ctrl_inc'], r['v_ref']] for r in rows])
    ok = np.isfinite(ci[:, 0])
    if ok.any():
        n_c = int(((ci[ok, 1] - ci[ok, 0]) > args.gain).sum())
        print(f'   control incumbent polish: beats the reference at {n_c}/{int(ok.sum())}',
              flush=True)

    # per family
    fams = sorted({n for r in rows for n in r['names']})
    alone, best_new, origin, gaps = Counter(), Counter(), Counter(), {f: [] for f in fams}
    for r in rows:
        v_new_best = np.inf
        for n, v in zip(r['names'], r['vals']):
            if not np.isfinite(v):
                continue
            gaps[n].append(v - r['v_ref'])
            if r['v_ref'] - v > args.gain:
                alone[n] += 1
            if n not in CONTROL_NAMES and v < v_new_best:
                v_new_best, nb = v, n
        if np.isfinite(v_new_best):
            best_new[nb] += 1
        if r['improved']:
            m = re.match(r'probe (\S+) \+polish', r['label'])
            if m:
                origin[m.group(1)] += 1
    print('\n   family       alone<ref  best-new  origin-of-gain  median(fam-ref)', flush=True)
    for f in fams:
        g = np.median(gaps[f]) if gaps[f] else np.nan
        tag = '  (control)' if f in CONTROL_NAMES else ''
        print(f'   {f:<12s} {alone[f]:9d} {best_new[f]:9d} {origin[f]:15d}  {g:+.3e}{tag}',
              flush=True)

    # largest improvements
    order = np.argsort(-drops)
    print(f'\n   largest drops ({spec.p1_name}, {spec.p2_name}): ref -> new [origin]', flush=True)
    for i in order[:args.n_show]:
        if drops[i] <= args.gain:
            break
        r = rows[i]
        print(f'     ({p1[r["ix"]]:.2f}, {p2[r["iy"]]:.2f})  {r["v_ref"]:.4e} -> '
              f'{r["v_new"]:.4e}  (-{drops[i]:.3e})  [{r["label"]}]'
              f'{"  NEW RATE-0" if r["new_zero"] else ""}', flush=True)

    # grids for plotting later
    drop_grid = np.full((nx, ny), np.nan)
    for r, dr in zip(rows, drops):
        drop_grid[r['ix'], r['iy']] = dr
    np.savez(pd / 'probe_summary.npz', drop=drop_grid, p1=p1, p2=p2,
             ix=[r['ix'] for r in rows], iy=[r['iy'] for r in rows],
             v_ref=[r['v_ref'] for r in rows], v_new=[r['v_new'] for r in rows],
             label=[r['label'] for r in rows], fam_names=fams,
             alone=[alone[f] for f in fams], best_new=[best_new[f] for f in fams],
             origin=[origin[f] for f in fams], model=args.model)
    print(f'\n   summary -> {pd / "probe_summary.npz"}', flush=True)

    if not args.adopt:
        return
    pt12 = d12_dir(args.out_dir, args.stride, args.model)
    rnd = last_round(pt12, RREFINE_TAG) + 1
    if rnd > 99:
        sys.exit('no free 2-digit rrefine round left')
    n_w = 0
    for r, dr in zip(rows, drops):
        if dr <= args.gain:
            continue
        d = np.load(r['file'], allow_pickle=True)
        out = pt12 / f'pt_{r["ix"]:03d}_{r["iy"]:03d}{RREFINE_TAG}{rnd:02d}.npz'
        if out.exists():
            continue
        tmp = out.with_suffix('.tmp.npz')
        np.savez(tmp, model=args.model, ix=r['ix'], iy=r['iy'],
                 **{spec.p1_name: float(d[spec.p1_name]), spec.p2_name: float(d[spec.p2_name])},
                 dim=int(d['dim']), d_ext=M, round=rnd, target='seed probe', mode='probe',
                 rate_version=str(d['rate_version']),
                 rate_margin_version=str(d['rate_margin_version']),
                 **{KEY: r['v_new'], S_KEY: np.asarray(d[S_KEY], float),
                    LAB_KEY: r['label'], f'{KEY}_prev': r['v_ref']})
        os.replace(tmp, out)
        n_w += 1
    print(f'   adopted {n_w} point(s) as {RREFINE_TAG}{rnd:02d} files in {pt12}', flush=True)


if __name__ == '__main__':
    main()
