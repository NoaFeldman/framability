"""
Is another refine / margin round worth it?  Round history and residual
indicators of the seeded model10 rate pipeline (dim = 2 run by default; the
dim = 1 run with --out_dir results_model10_rate).  Read-only.

1. Per round of every round loop (from its files on disk):
     d4/d8 quick refine     model10_seeded/pt_*_qrefine_rNN.npz      rate_4, rate_8
     d12 refine             model10_seeded_d12/pt_*_qrefine_rNN.npz  rate_12
     d8 / d12 margin        .../pt_*_margin_rNN.npz                  rate_<m>, mode
     randomised refine      .../pt_*_rrefine_rNN.npz (if any)        rate_<m>
   the points whose rate dropped by more than --gain against the best value
   known when the round started (rate_<m>_prev), the summed and largest drop,
   and the NEW rate-0 points (the white contour moves).  Margin rounds also
   count their modes: 'margin' / 'gossip' only add slack at rate-0 points,
   'transfer' makes a new rate-0 point, 'push' lowers a positive rate.
   A round that improved nothing writes no file, so it does not show up here;
   the driver log says "... wrote nothing -- stopping" for it.
2. The current panels (min over every file; d8 and d12 also take the smaller
   d_ext's value, as in the figure):
     boundary     positive points with a rate-0 4-neighbour: the only points
                  quick refine re-optimises
     near-zero    rate in (--tol, 1e-3): rate-0 frames without slack, the
                  margin rounds' target
     non-monotone rate rising by > 1e-4 one grid step towards MORE noise
                  (a likely missed minimum), split by whether the high point
                  lies within MARGIN_RADIUS of the rate-0 region (reachable
                  by margin push / refine) or not (only a re-optimisation of
                  the point itself reaches it: model10_rrefine_worker.py
                  --targets nonmono)
3. A verdict per loop from its last round on disk.

--round_gain DIR TAG ROUND prints only "<largest drop> <new rate-0 points>
<files>" of one round (every rate_<m> in its files); the stop rule of
scripts/submit_model10_dim2.sh.

Usage (on the cluster, from the repo root; a minute or so of file reading):
    python scripts/model10_refine_status.py
    python scripts/model10_refine_status.py --out_dir results_model10_rate
    python scripts/model10_refine_status.py --round_gain \\
        results_model10_rate_dim2/model10_seeded _rrefine_r 7
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model10_seeded_worker import grid_vals, pt_dir_name                  # noqa: E402
from model10_seeded_qrefine_worker import (ROUND_TAG, MARGIN_TAG,         # noqa: E402
                                           RREFINE_TAG)
from model10_d12_worker import d12_dir                                    # noqa: E402
from model10_seeded_panels_collect import load_best                       # noqa: E402

MARGIN_RADIUS = 2          # model10_margin_worker.py --radius default
STEP_TOL = 1e-4            # non-monotone threshold of the figure logs


def history(pt: Path, tag: str, d_exts, tol: float, gain: float) -> dict:
    """{round: stats} of one round loop, from its files in pt."""
    out = {}
    if not pt.is_dir():
        return out
    for f in pt.glob(f'pt_*{tag}[0-9][0-9].npz'):
        r = int(re.search(rf'{tag}(\d\d)\.npz$', f.name).group(1))
        st = out.setdefault(r, dict(files=0, modes=Counter(),
                                    drop={m: dict(n=0, sum=0.0, max=0.0, zeros=0)
                                          for m in d_exts}))
        try:
            d = np.load(f, allow_pickle=True)
        except Exception as e:                              # noqa: BLE001
            print(f'  warning: {f.name}: {e}', flush=True)
            continue
        st['files'] += 1
        if 'mode' in d.files:
            st['modes'][str(d['mode'])] += 1
        for m in d_exts:
            k = f'rate_{m}'
            if k not in d.files or f'{k}_prev' not in d.files:
                continue
            new, prev = float(d[k]), float(d[f'{k}_prev'])
            if not (np.isfinite(new) and np.isfinite(prev)):
                continue
            s = st['drop'][m]
            if prev - new > gain:
                s['n'] += 1
                s['sum'] += prev - new
                s['max'] = max(s['max'], prev - new)
            if prev > tol >= new:
                s['zeros'] += 1
    return dict(sorted(out.items()))


def round_gain(pt: Path, tag: str, r: int, tol: float):
    """(largest drop, new rate-0 points, files) of round r, over every
    rate_<m> with a stored rate_<m>_prev."""
    mx, zeros, n = 0.0, 0, 0
    for f in pt.glob(f'pt_*{tag}{r:02d}.npz'):
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:                                   # noqa: BLE001
            continue
        n += 1
        for k in d.files:
            if not re.fullmatch(r'rate_\d+_prev', k) or k[:-5] not in d.files:
                continue
            new, prev = float(d[k[:-5]]), float(d[k])
            if np.isfinite(new) and np.isfinite(prev):
                mx = max(mx, prev - new)
                zeros += prev > tol >= new
    return mx, zeros, n


def report_loop(label: str, hist: dict, visible: float) -> str:
    """Print the round table of one loop; return its verdict line."""
    print(f'\n== {label}: {len(hist)} round(s) on disk', flush=True)
    if not hist:
        return f'{label}: no rounds on disk'
    for r, st in hist.items():
        parts = [f'd{m}: {s["n"]:4d} down (sum {s["sum"]:.3g}, max {s["max"]:.2e}), '
                 f'{s["zeros"]:3d} new rate-0' for m, s in st['drop'].items()]
        modes = ', '.join(f'{k} {v}' for k, v in sorted(st['modes'].items()))
        print(f'   r{r:02d} {st["files"]:5d} files | ' + ' | '.join(parts)
              + (f' | modes: {modes}' if modes else ''), flush=True)
    r = max(hist)
    last = hist[r]
    zeros = sum(s['zeros'] for s in last['drop'].values())
    mx = max(s['max'] for s in last['drop'].values())
    slack = last['modes'].get('margin', 0) + last['modes'].get('gossip', 0)
    if zeros or mx > visible:
        return (f'{label}: STILL MOVING -- r{r:02d} made {zeros} new rate-0 '
                f'point(s), largest drop {mx:.2e}; more rounds change the figure')
    if slack:
        return (f'{label}: no visible change in r{r:02d}, but {slack} rate-0 '
                'frame(s) gained slack; one more round shows whether that turns '
                'into transfers')
    return (f'{label}: EXHAUSTED for the figure -- r{r:02d} only made drops '
            f'<= {visible:g} and no new rate-0 point')


def indicators(R, m: int, p1, p2, tol: float, n_show: int) -> None:
    z = R <= tol
    pos = np.isfinite(R) & ~z
    k4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)
    boundary = pos & binary_dilation(z, structure=k4)
    reach = binary_dilation(z, structure=np.ones((2 * MARGIN_RADIUS + 1,) * 2, bool))
    near = (R > tol) & (R < 1e-3)
    steps = []                          # (jump, (ix, iy) low-noise, high-noise)
    for ax in (0, 1):
        dR = np.diff(R, axis=ax)
        for i, j in zip(*np.nonzero(dR > STEP_TOL)):
            hi = (i + 1, j) if ax == 0 else (i, j + 1)
            steps.append((float(dR[i, j]), (i, j), hi))
    steps.sort(reverse=True)
    n_reach = sum(1 for s in steps if reach[s[2]])
    print(f'   d{m}: {int(z.sum())} rate-0 | {int(boundary.sum())} boundary | '
          f'{int(near.sum())} near-zero | {len(steps)} non-monotone steps'
          + (f' (largest +{steps[0][0]:.3g}): {n_reach} within reach of '
             f'refine/margin, {len(steps) - n_reach} beyond' if steps else ''),
          flush=True)
    for jump, lo, hi in steps[:n_show]:
        print(f'        +{jump:.3e}  ({p1[lo[0]]:.2f}, {p2[lo[1]]:.2f}) {R[lo]:.4f}'
              f' -> ({p1[hi[0]]:.2f}, {p2[hi[1]]:.2f}) {R[hi]:.4f}'
              f'{"" if reach[hi] else "  [beyond]"}', flush=True)
    if near.any():
        pts = ', '.join(f'({p1[i]:.2f}, {p2[j]:.2f}) {R[i, j]:.1e}'
                        for i, j in list(zip(*np.nonzero(near)))[:n_show])
        print(f'        near-zero: {pts}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default='results_model10_rate_dim2')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--tol', type=float, default=1e-6,
                    help='rate counted as 0 (the figure contour tolerance)')
    ap.add_argument('--gain', type=float, default=1e-6,
                    help='a drop below this is not counted')
    ap.add_argument('--visible', type=float, default=1e-3,
                    help='a drop below this does not show on the colour scale')
    ap.add_argument('--n_show', type=int, default=5,
                    help='non-monotone steps / near-zero points listed per d_ext')
    ap.add_argument('--round_gain', nargs=3, metavar=('DIR', 'TAG', 'ROUND'),
                    help='only print "<largest drop> <new rate-0> <files>" of '
                         'one round and exit')
    args = ap.parse_args()
    if args.round_gain:
        d, tag, r = args.round_gain
        mx, zeros, n = round_gain(Path(d), tag, int(r), args.tol)
        print(f'{mx:.6e} {zeros} {n}')
        return

    pt8 = Path(args.out_dir) / pt_dir_name(args.stride)
    pt12 = d12_dir(args.out_dir, args.stride)
    loops = [('d4/d8 quick refine', pt8, ROUND_TAG, (4, 8)),
             ('d12 refine', pt12, ROUND_TAG, (12,)),
             ('d8 margin', pt8, MARGIN_TAG, (8,)),
             ('d12 margin', pt12, MARGIN_TAG, (12,)),
             ('d8 randomised refine', pt8, RREFINE_TAG, (8,)),
             ('d12 randomised refine', pt12, RREFINE_TAG, (12,))]
    verdicts = []
    for label, pt, tag, d_exts in loops:
        hist = history(pt, tag, d_exts, args.tol, args.gain)
        if hist or 'randomised' not in label:
            verdicts.append(report_loop(label, hist, args.visible))

    p1, p2 = grid_vals(args.stride)
    nx, ny = len(p1), len(p2)
    print('\n== current panels', flush=True)
    s = {4: load_best(pt8, nx, ny, 4), 8: load_best(pt8, nx, ny, 8),
         12: load_best(pt12, nx, ny, 12)}
    panel = {4: s[4]['grid']}
    panel[8] = np.fmin(s[8]['grid'], panel[4])
    panel[12] = np.fmin(s[12]['grid'], panel[8])
    for m in (4, 8, 12):
        indicators(panel[m], m, p1, p2, args.tol, args.n_show)
    for m, k in ((8, 4), (12, 8)):
        gain = panel[k] - panel[m]
        print(f'   d{m} below d{k} at {int((gain > args.gain).sum())} points '
              f'(largest gain {np.nanmax(gain):.3g})', flush=True)

    print('\n== verdict', flush=True)
    for v in verdicts:
        print(f'   {v}', flush=True)


if __name__ == '__main__':
    main()
