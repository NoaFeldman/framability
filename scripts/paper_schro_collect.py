"""
Collect the optimised SCHRODINGER (state-frame) rate panel of one model for
the paper figures.

Reads the per-point files of the Schrodinger rerun
(scripts/model4_rate_panels_worker.py --measures schro --tag <tag>):

    <out_dir>/<model>_<tag>/pt_<ix:03d>_<iy:03d>.npz   keys rate_schro_<d>

and, optionally, an earlier panels npz with rate_schro_4 / rate_schro_6 on the
same grid (--ref_npz; e.g. results_model4_rate/model4_rate_panels.npz).  Every
value is an upper bound on mu*_Schrodinger (a certified rate of a concrete
frame), so the panel is the point-wise minimum over all of them.

Writes <out_dir>/<model>_schro_panels.npz (or --out):
    p1_vals, p2_vals            the model's grid (stride 1)
    rate_schro_best (nx, ny)    point-wise minimum (nan where nothing exists)
    rate_schro_<d>  (nx, ny)    the rerun, per state-frame size d
    ref_rate_schro_<d>          the --ref_npz arrays that were used
    best_src (nx, ny) str       which array gave the minimum
    n_points                    rerun points found

Usage:
    python scripts/paper_schro_collect.py --model model4 --out_dir results_model4_rate \\
        --ref_npz results_model4_rate/model4_rate_panels.npz
    python scripts/paper_schro_collect.py --model dicke_dec_N10 --out_dir results_dicke_dec_N10_seeded
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS                              # noqa: E402

PT_RE = re.compile(r'pt_(\d{3})_(\d{3})\.npz$')


def load_rerun(pt_dir: Path, nx: int, ny: int) -> tuple[dict, int]:
    """{d: (nx, ny) array} of the rerun's rate_schro_<d>, and #points read."""
    out: dict = {}
    n = 0
    for f in sorted(pt_dir.glob('pt_*.npz')):
        mt = PT_RE.search(f.name)
        if not mt:
            continue
        ix, iy = int(mt.group(1)), int(mt.group(2))
        if not (0 <= ix < nx and 0 <= iy < ny):
            print(f'  skip {f.name}: outside the {nx}x{ny} grid', flush=True)
            continue
        try:
            with np.load(f, allow_pickle=True) as z:
                for k in z.files:
                    mk = re.fullmatch(r'rate_schro_(\d+)', k)
                    if mk and np.ndim(z[k]) == 0:
                        d = int(mk.group(1))
                        out.setdefault(d, np.full((nx, ny), np.nan))[ix, iy] = float(z[k])
        except Exception as e:                    # truncated file of a killed task
            print(f'  skip {f.name}: {type(e).__name__}: {e}', flush=True)
            continue
        n += 1
    return out, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=sorted(MODELS))
    ap.add_argument('--out_dir', required=True,
                    help='root of the rerun (the worker\'s --out_dir)')
    ap.add_argument('--tag', default='schro', help='the worker\'s --tag')
    ap.add_argument('--ref_npz', default=None,
                    help='earlier panels npz with rate_schro_4 / rate_schro_6 '
                         '(same model, same dim, stride 1)')
    ap.add_argument('--out', default=None,
                    help='default <out_dir>/<model>_schro_panels.npz')
    args = ap.parse_args()

    spec = MODELS[args.model]
    p1, p2 = np.asarray(spec.p1_vals, float), np.asarray(spec.p2_vals, float)
    nx, ny = len(p1), len(p2)
    pt_dir = Path(args.out_dir) / f'{args.model}_{args.tag}'
    rerun, n = load_rerun(pt_dir, nx, ny) if pt_dir.is_dir() else ({}, 0)
    print(f'[{args.model}] rerun {pt_dir}: {n} of {nx * ny} points, '
          f'd = {sorted(rerun)}', flush=True)

    cands = {f'rate_schro_{d}': a for d, a in sorted(rerun.items())}
    refs = {}
    if args.ref_npz:
        with np.load(args.ref_npz, allow_pickle=True) as z:
            for k in z.files:
                if re.fullmatch(r'rate_schro_\d+', k) and z[k].shape == (nx, ny):
                    refs[f'ref_{k}'] = np.asarray(z[k], float)
        print(f'[{args.model}] reference {args.ref_npz}: {sorted(refs)}', flush=True)
    cands.update(refs)
    if not cands:
        print(f'[{args.model}] ERROR: no Schrodinger data at all', flush=True)
        sys.exit(1)

    names = list(cands)
    stack = np.stack([cands[k] for k in names])            # (K, nx, ny)
    finite = np.isfinite(stack)
    best = np.where(finite.any(0), np.nanmin(np.where(finite, stack, np.inf), 0), np.nan)
    arg = np.argmin(np.where(finite, stack, np.inf), 0)
    src = np.where(finite.any(0), np.array(names, dtype=object)[arg], '').astype(str)

    for k in names:
        a = cands[k]
        ok = np.isfinite(a)
        if ok.any():
            print(f'  {k:>20s}: {ok.sum():5d} pts, min {np.nanmin(a):.4f}, '
                  f'best at {(src == k).sum()} pts', flush=True)
    for k in [k for k in names if k.startswith('ref_')]:
        both = np.isfinite(cands[k]) & np.isfinite(best)
        if both.any():
            gain = cands[k][both] - best[both]
            print(f'  gain over {k}: max {gain.max():.4f}, mean {gain.mean():.4f}, '
                  f'> 1e-3 at {(gain > 1e-3).sum()} pts', flush=True)

    out = Path(args.out or Path(args.out_dir) / f'{args.model}_schro_panels.npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, model=args.model, p1_vals=p1, p2_vals=p2,
             p1_name=spec.p1_name, p2_name=spec.p2_name,
             rate_schro_best=best, best_src=src, n_points=n, **cands)
    print(f'[{args.model}] wrote {out}  ({np.isfinite(best).sum()} of {nx * ny} '
          f'points; min {np.nanmin(best):.4f})', flush=True)


if __name__ == '__main__':
    main()
