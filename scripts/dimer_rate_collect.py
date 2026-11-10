"""
Collect scripts/dimer_rate_worker.py over a model's grid and draw the
dimer-frame rate panels against the stored product-frame rates.

Reads   <in_dir>/<model>_dimer_<tiling>_<mode>[_s<stride>]/pt_<ix>_<iy>.npz
Writes  <in_dir>/<model>_dimer_<tiling>_<mode>_panels.npz
        <in_dir>/<model>_dimer_<tiling>_<mode>_panels.png
            row 1: dimer F_split | dimer F_joint | mu_in (bond inside a dimer)
                   | mu_cut (bond across dimers; weighted mean over cut types)
            row 2: best stored product rate (any single-qubit d_ext)
                   | product d4 (best S1 seed) | gain over product d4
                   | gain over the best stored product (coolwarm, > 0 = dimer wins)
        white contours: rate at the floor mu* = 0.
All rates are per-bond averages, so every panel shares one scale convention.

Usage:
    python scripts/dimer_rate_collect.py --model model4
    python scripts/dimer_rate_collect.py --model model10 --dim 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                              # noqa: E402
from rate_gopt_collect import _panel, CONTOUR_TOL                        # noqa: E402
from dimer_rate_worker import grid_vals, pt_dir_name                    # noqa: E402
from framability_rate_dimer import TILINGS                              # noqa: E402

SCALARS = ('rate_dimer', 'rate_split', 'rate_joint', 'mu_in', 'mu_seed',
           'mu_weights', 'mu_rw', 'mu_polish', 'mu_reference', 't')


def load_grids(model: str, pt_dir: Path, stride: int, m: int, cuts) -> dict:
    p1, p2 = grid_vals(model, stride)
    nx, ny = len(p1), len(p2)
    nan = lambda: np.full((nx, ny), np.nan)                 # noqa: E731
    g = {f'{k}_{m}': nan() for k in SCALARS}
    g.update({f'mu_cut_{c}_{m}': nan() for c in cuts})
    g.update({f'mu_joint_{c}_{m}': nan() for c in cuts})
    g.update(rate_pauli=nan(), prod_rate_4=nan(), prod_best=nan(),
             prod_best_k=nan())
    found = 0
    for ix in range(nx):
        for iy in range(ny):
            f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
            except Exception as e:                          # noqa: BLE001
                print(f'  warning: {f.name}: {e}', flush=True)
                continue
            found += 1
            for k in list(g):
                if k in d.files and k not in ('prod_best', 'prod_best_k'):
                    try:
                        g[k][ix, iy] = float(d[k])
                    except (TypeError, ValueError):
                        pass
            best, best_k = np.inf, np.nan
            for k in d.files:
                if k.startswith('prod_best_'):
                    v = float(d[k])
                    if np.isfinite(v) and v < best:
                        best, best_k = v, float(k.rsplit('_', 1)[1])
            if np.isfinite(best):
                g['prod_best'][ix, iy] = best
                g['prod_best_k'][ix, iy] = best_k
    g.update(p1_vals=p1, p2_vals=p2, found=found, n_total=nx * ny)
    return g


def plot(model, g, m, cuts, cut_w, title, png: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = MODELS[model]
    xv, yv = g['p1_vals'], g['p2_vals']
    wsum = sum(cut_w.values())
    cut_mean = sum(cut_w[c] * g[f'mu_cut_{c}_{m}'] for c in cuts) / wsum
    F = g[f'rate_split_{m}']
    gain4 = g['prod_rate_4'] - F
    gainb = g['prod_best'] - F
    fig, ax = plt.subplots(2, 4, figsize=(22.4, 9.6), constrained_layout=True)
    fig.suptitle(f'{spec.title}\n{title}  (per-bond rates; white: $\\mu^*=0$; '
                 f'{g["found"]}/{g["n_total"]} points)', fontsize=12)
    n0 = int((np.abs(F) < CONTOUR_TOL).sum())
    _panel(fig, ax[0, 0], spec, xv, yv, F,
           f'dimer frame, split  ({n0} pts at floor)')
    _panel(fig, ax[0, 1], spec, xv, yv, g[f'rate_joint_{m}'],
           'dimer frame, joint patch')
    _panel(fig, ax[0, 2], spec, xv, yv, g[f'mu_in_{m}'],
           r'$\mu_{\rm in}$: bond inside a dimer ($S$)')
    _panel(fig, ax[0, 3], spec, xv, yv, cut_mean,
           r'$\mu_{\rm cut}$: bond across dimers ($S\otimes S$)')
    kb = g['prod_best_k']
    ks = sorted({int(k) for k in kb[np.isfinite(kb)]})
    _panel(fig, ax[1, 0], spec, xv, yv, g['prod_best'],
           f'best stored product rate (d_ext {ks})')
    _panel(fig, ax[1, 1], spec, xv, yv, g['prod_rate_4'],
           r'product $S_1\otimes S_1$, $d_{\rm ext}=4$')
    _panel(fig, ax[1, 2], spec, xv, yv, gain4,
           'gain over product d4 (product $-$ dimer)', floor=None,
           cmap='coolwarm', symmetric=True)
    _panel(fig, ax[1, 3], spec, xv, yv, gainb,
           'gain over best stored product', floor=None, cmap='coolwarm',
           symmetric=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[{model} dimer] wrote {png}', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=list(MODELS))
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--tiling', default=None)
    p.add_argument('--mode', default='split', choices=['split', 'joint'])
    p.add_argument('--in_dir', default=None)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--m', type=int, default=16)
    p.add_argument('--no_plot', action='store_true')
    args = p.parse_args()
    spec = MODELS[args.model]
    dim = spec.dim if args.dim is None else args.dim
    tiling = args.tiling or ('chain' if dim == 1 else 'brick')
    in_dir = Path(args.in_dir or (f'results_{args.model}_dimer' if dim == spec.dim
                                  else f'results_{args.model}_dim{dim}_dimer'))
    name = pt_dir_name(args.model, tiling, args.mode, args.stride)
    cut_w = TILINGS[dim][tiling]
    cuts = list(cut_w)
    g = load_grids(args.model, in_dir / name, args.stride, args.m, cuts)
    print(f'[{args.model} dimer] {g["found"]}/{g["n_total"]} points from '
          f'{in_dir / name}', flush=True)
    if g['found'] == 0:
        sys.exit('no points found')

    F = g[f'rate_split_{args.m}']
    for lab, ref in (('product d4', g['prod_rate_4']),
                     ('best stored product', g['prod_best'])):
        ok = np.isfinite(F) & np.isfinite(ref)
        if ok.any():
            d = ref[ok] - F[ok]
            print(f'  vs {lab}: dimer lower at {int((d > 1e-6).sum())}/{ok.sum()} '
                  f'points, max gain {d.max():.4f}, median {np.median(d):.4f}, '
                  f'max loss {max(0.0, -d.min()):.4f}', flush=True)

    out_npz = in_dir / f'{name}_panels.npz'
    np.savez(out_npz, model=args.model, dim=dim, tiling=tiling, mode=args.mode,
             stride=args.stride, m=args.m,
             **{k: v for k, v in g.items() if isinstance(v, np.ndarray)})
    print(f'[{args.model} dimer] wrote {out_npz}', flush=True)
    if not args.no_plot:
        title = (f'dimer-frame Heisenberg rate, dim={dim}, {tiling} tiling, '
                 f'optimised on {args.mode}, $m={args.m}$')
        plot(args.model, g, args.m, cuts, cut_w, title,
             in_dir / f'{name}_panels.png')


if __name__ == '__main__':
    main()
