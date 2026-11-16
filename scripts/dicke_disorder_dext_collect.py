"""
Summary of the disordered-Dicke d_ext scan (scripts/dicke_disorder_rate_worker.py
with --points / --init_root, run by scripts/submit_paper_dext_scan.sh), in the
layout scripts/paper_dext_plot.py reads:

    <root>/<family>_N<N>_M<M>_w<w:.2f>_dext_summary.npz
        model ('<family>_N<N>'), width, M, frames, d_ext, p1, p2, p1_name, p2_name,
        rate (n_points, n_d)   best certified pair-averaged rate with <= d_ext
                               columns (running min over d_ext: a frame of
                               fewer columns pads to more),
        raw  (n_points, n_d)   the value found at exactly that d_ext,
        converged (n_points, n_d) all False: this scan has no restart /
                               basin-hopping convergence test (open markers)

--frames species (default; per-species frames, F_species_cert_<d>) or shared.

Usage:
    python scripts/dicke_disorder_dext_collect.py --p1 0.5 1.0 1.5 --p2 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from trotter_lindbladian_scan import MODELS                           # noqa: E402


def grid_index(vals, v) -> int:
    i = int(np.argmin(np.abs(np.asarray(vals, float) - v)))
    if abs(vals[i] - v) > 1e-9:
        raise ValueError(f'{v} not on the grid')
    return i


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', nargs='+', default=['dicke_dec', 'dicke_dec_k025'])
    ap.add_argument('--ns', nargs='+', type=int, default=[100])
    ap.add_argument('--widths', nargs='+', type=float, default=[0.5, 0.8])
    ap.add_argument('--M', type=int, default=4)
    ap.add_argument('--p1', nargs='+', type=float, required=True)
    ap.add_argument('--p2', nargs='+', type=float, required=True)
    ap.add_argument('--d_exts', nargs='+', type=int, default=[4, 6, 8, 10, 12, 14, 16])
    ap.add_argument('--frames', choices=('species', 'shared'), default='species')
    ap.add_argument('--root', default='results_dicke_disorder_dext')
    args = ap.parse_args()

    root = REPO / args.root
    d_ext = np.asarray(sorted(args.d_exts), int)
    for fam in args.families:
        for N in args.ns:
            spec = MODELS[f'{fam}_N{N}']
            pts = [(a, b) for a in args.p1 for b in args.p2]
            for w in args.widths:
                folder = root / f'{fam}_N{N}_M{args.M}_w{w:.2f}'
                raw = np.full((len(pts), len(d_ext)), np.nan)
                for ip, (a, b) in enumerate(pts):
                    ix, iy = grid_index(spec.p1_vals, a), grid_index(spec.p2_vals, b)
                    f = folder / f'pt_{ix:03d}_{iy:03d}.npz'
                    if not f.exists():
                        continue
                    with np.load(f, allow_pickle=True) as z:
                        for j, m in enumerate(d_ext):
                            k = f'F_{args.frames}_cert_{m}'
                            if k in z.files:
                                raw[ip, j] = float(z[k])
                rate = np.fmin.accumulate(raw, axis=1)
                print(f'[{fam}_N{N} M={args.M} w={w:.2f}] {args.frames} frames', flush=True)
                for ip, (a, b) in enumerate(pts):
                    print(f'  {spec.p1_name}={a:g} {spec.p2_name}={b:g}: '
                          + '  '.join(f'{m}:{v:.4f}' for m, v in zip(d_ext, rate[ip])),
                          flush=True)
                path = root / f'{fam}_N{N}_M{args.M}_w{w:.2f}_dext_summary.npz'
                root.mkdir(parents=True, exist_ok=True)
                np.savez(path, model=f'{fam}_N{N}', width=w, M=args.M, frames=args.frames,
                         d_ext=d_ext, p1=np.array([p[0] for p in pts]),
                         p2=np.array([p[1] for p in pts]), p1_name=spec.p1_name,
                         p2_name=spec.p2_name, rate=rate, raw=raw,
                         converged=np.zeros_like(rate, bool))
                print(f'  wrote {path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
