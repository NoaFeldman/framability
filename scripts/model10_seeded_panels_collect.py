"""
Draw the model10 rate figure:

    row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext=4 | d_ext=8 | d_ext=12
    row 2 | product-state chi=10 | chi=40 | 8q osc rate | 8q gap

Per point, each optimised Heisenberg panel shows the minimum over every
certified upper bound on that d_ext's optimum:
    d_ext = 4 :  seeded worker, collect cross-evaluation and quick-refine
                 rounds (<out_dir>/model10_seeded/), and the earlier
                 optimiser's rate_heis_4
    d_ext = 8 :  the same seeded files, plus the d_ext = 4 panel and the
                 earlier rate_heis_6 (a 4- or 6-column frame padded by
                 repeated columns is an 8-column frame with the same rate)
    d_ext = 12:  <out_dir>/model10_seeded_d12/ (seed stage + refine rounds),
                 plus the d_ext = 8 panel (padding again)
The fallbacks only fill points a run has not reached; the log counts them.
The product-state rates (chi = 10, 40) come from
scripts/model4_product_rate_worker.py (<prod_dir>/model10_product/).

Base panels (stabilizer-3, Pauli, earlier opt d_ext = 4 / 6, many-body):
the first of --base_in_dirs holding a model10/ subdirectory, else the stored
figure data --base_npz (scripts/model10_panels_common.load_base).

Outputs:
    --out_png   (default results_model4_rate/model10_rate_panels.png)
    <out_dir>/model10_seeded_panels.npz

Usage:
    python scripts/model10_seeded_panels_collect.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from model10_seeded_worker import MODEL, grid_vals, pt_dir_name           # noqa: E402
from model10_seeded_qrefine_worker import (ROUND_TAG, MARGIN_TAG,         # noqa: E402
                                           best_known)
from model10_d12_worker import d12_dir                                    # noqa: E402
import model10_panels_common as common                                    # noqa: E402

D_EXTS = (4, 8, 12)


def rounds_on_disk(pt: Path, tag: str = ROUND_TAG) -> list:
    return sorted({int(m.group(1)) for f in pt.glob(f'pt_*{tag}*.npz')
                   if (m := re.search(rf'{tag}(\d\d)\.npz$', f.name))})


def load_best(pt: Path, nx: int, ny: int, m: int) -> dict:
    """rate_<m> grid: min over the worker, xeval and qrefine files of pt."""
    g = np.full((nx, ny), np.nan)
    found = 0
    for ix in range(nx):
        for iy in range(ny):
            if not (pt / f'pt_{ix:03d}_{iy:03d}.npz').exists():
                continue
            found += 1
            v, _, _ = best_known(pt, ix, iy, f'rate_{m}', f'S_{m}')
            if np.isfinite(v):
                g[ix, iy] = v
    rounds = rounds_on_disk(pt) if pt.is_dir() else []
    margin = rounds_on_disk(pt, MARGIN_TAG) if pt.is_dir() else []
    print(f'[{MODEL} panels] d_ext={m}: {found}/{nx * ny} points in {pt}; '
          f'refine rounds on disk: {rounds or "none"}; margin rounds: '
          f'{margin or "none"}', flush=True)
    return dict(grid=g, n_points=found, rounds=rounds, margin=margin)


def on_base_grid(Z, stride: int, base_stride: int, shape) -> np.ndarray:
    """A grid of the seeded pipeline placed on the base grid (NaN elsewhere)."""
    out = np.full(shape, np.nan)
    if stride % base_stride:
        return out
    r = stride // base_stride
    for ix in range(Z.shape[0]):
        for iy in range(Z.shape[1]):
            jx, jy = ix * r, iy * r
            if jx < shape[0] and jy < shape[1]:
                out[jx, jy] = Z[ix, iy]
    return out


def _rounds_txt(s: dict, what: str) -> str:
    n, k = len(s['rounds']), len(s['margin'])
    txt = f'{what} + {n} refine round{"s" if n != 1 else ""}'
    return txt + (f' + {k} margin round{"s" if k != 1 else ""}' if k else '')


def plot(panels: list, png: Path) -> None:
    """panels: (kind, x, y, Z, title) with kind 'rate' (viridis + floor
    contour) or 'mb' (magma); drawn row by row on a 2 x 5 grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    m = MODELS[MODEL]
    lab = dict(xlabel=m.p1_label, ylabel=m.p2_label)
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(27.5, 5 * nrow),
                             constrained_layout=True)
    fig.suptitle(
        f'{MODEL}:  {m.title}'
        '\n'
        r'framability rates $\mu^*=\lim_{dt\to0}({\rm fra}-1)/dt$ of the bond '
        r'generator (white: $\mu^*=0$)  |  opt Heisenberg: optimised observable '
        r'frames  |  product-state: random product frames  |  '
        rf"{common.N_QUBITS}q panels: full {common.mb_geometry(MODEL)['label']} "
        r'Lindbladian',
        fontsize=13)
    flat = list(np.atleast_1d(axes).flat)
    for ax, (kind, xv, yv, Z, title) in zip(flat, panels):
        if kind == 'rate':
            common.draw_panel(fig, ax, xv, yv, Z, title, common.FRA_CMAP,
                              floor_contour=0.0, **lab)
        else:
            common.draw_panel(fig, ax, xv, yv, Z, title, common.MB_CMAP, **lab)
    for ax in flat[len(panels):]:
        ax.axis('off')
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[{MODEL} panels] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, default='results_model10_rate',
                    help='seeded pipeline root (model10_seeded/, model10_seeded_d12/)')
    ap.add_argument('--stride', type=int, default=1,
                    help='stride of the seeded / d_ext = 12 runs')
    ap.add_argument('--out_png', type=str,
                    default='results_model4_rate/model10_rate_panels.png')
    ap.add_argument('--base_in_dirs', type=str, nargs='*',
                    default=['results_model10_rate', 'results_model4_rate'])
    ap.add_argument('--base_npz', type=str, nargs='*',
                    default=['results_model4_rate/model10_rate_panels.npz',
                             'results_model10_rate/model10_rate_panels.npz'])
    ap.add_argument('--base_stride', type=int, default=1)
    ap.add_argument('--mb_stride', type=int, default=5)
    ap.add_argument('--prod_dirs', type=str, nargs='*',
                    default=['results_model10_rate', 'results_model4_rate'],
                    help='roots searched for model10_product/ (first with data)')
    ap.add_argument('--prod_stride', type=int, default=1)
    args = ap.parse_args()

    bd = common.load_base(args.base_in_dirs, args.base_npz,
                          base_stride=args.base_stride, mb_stride=args.mb_stride)
    if bd is None:
        sys.exit('no base figure data found (neither per-point dirs in '
                 f'{args.base_in_dirs} nor {args.base_npz})')
    rates, mb = bd['rates'], bd['mb']
    xv, yv = rates['p1_vals'], rates['p2_vals']
    old4 = np.asarray(rates['rate_heis_4'], float)
    old6 = np.asarray(rates['rate_heis_6'], float)
    shape = old4.shape

    p1, p2 = grid_vals(args.stride)
    nx, ny = len(p1), len(p2)
    s4 = load_best(Path(args.out_dir) / pt_dir_name(args.stride), nx, ny, 4)
    s8 = load_best(Path(args.out_dir) / pt_dir_name(args.stride), nx, ny, 8)
    s12 = load_best(d12_dir(args.out_dir, args.stride), nx, ny, 12)
    new = {mm: on_base_grid(s['grid'], args.stride, bd['stride'], shape)
           for mm, s in ((4, s4), (8, s8), (12, s12))}
    panel = {4: np.fmin(new[4], old4)}
    panel[8] = np.fmin.reduce([new[8], panel[4], old6])
    panel[12] = np.fmin(new[12], panel[8])
    for mm, ref in ((4, old4), (8, np.fmin(old4, old6)), (12, panel[8])):
        ok = np.isfinite(new[mm])
        fb = np.isfinite(panel[mm]) & ~ok
        both = ok & np.isfinite(ref)
        print(f'  d{mm}: own data at {int(ok.sum())} pts, fallback at '
              f'{int(fb.sum())}; rate 0 at {int((panel[mm] <= common.CONTOUR_TOL).sum())}; '
              f'below the {"previous" if mm < 12 else "d_ext=8"} value at '
              f'{int(((ref - panel[mm])[both] > 1e-6).sum())}', flush=True)

    prod = common.load_prod(args.prod_dirs, args.prod_stride)
    if prod is None:
        print(f'[{MODEL} panels] no product-state data under '
              f'{[str(Path(d) / "model10_product") for d in args.prod_dirs]}; '
              'those panels are left empty', flush=True)

    t4 = _rounds_txt(s4, 'seeded')
    t8 = _rounds_txt(s8, 'seeded')
    t12 = _rounds_txt(s12, 'seeded from $d=8$')
    dl = r'$d_{\rm ext}'
    panels = [
        ('rate', xv, yv, rates['rate_stab3'], 'Stabilizer-3 framability rate'),
        ('rate', xv, yv, rates['rate_pauli'], 'Pauli framability rate'),
        ('rate', xv, yv, panel[4], rf'Opt Heisenberg rate ({dl}=4$)' f'\n{t4}'),
        ('rate', xv, yv, panel[8], rf'Opt Heisenberg rate ({dl}=8$)' f'\n{t8}'),
        ('rate', xv, yv, panel[12], rf'Opt Heisenberg rate ({dl}=12$)' f'\n{t12}'),
    ]
    for key, label in common.PROD_RATE_KEYS:
        if prod is None:
            panels.append(('rate', xv, yv, np.full(shape, np.nan), label))
        else:
            panels.append(('rate', prod['p1_vals'], prod['p2_vals'], prod[key], label))
    for key, label in common.MB_KEYS:
        panels.append(('mb', mb['p1_vals'], mb['p2_vals'], mb[key], label))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y = MODELS[MODEL].p1_name, MODELS[MODEL].p2_name
    npz = out_dir / f'{MODEL}_seeded_panels.npz'
    np.savez(npz, model=MODEL, stride=args.stride, base_stride=bd['stride'],
             qrefine_rounds=np.array(s4['rounds'], int),
             d12_refine_rounds=np.array(s12['rounds'], int),
             **{f'{x}_vals': xv, f'{y}_vals': yv},
             panel_rate_heis_4=panel[4], panel_rate_heis_8=panel[8],
             panel_rate_heis_12=panel[12],
             seeded_rate_4=new[4], seeded_rate_8=new[8], d12_rate_12=new[12],
             prev_rate_heis_4=old4, prev_rate_heis_6=old6,
             **({} if prod is None else
                {f'prod_{x}_vals': prod['p1_vals'], f'prod_{y}_vals': prod['p2_vals'],
                 **{k: prod[k] for k, _ in common.PROD_RATE_KEYS}}))
    print(f'[{MODEL} panels] wrote {npz}', flush=True)
    plot(panels, Path(args.out_png))


if __name__ == '__main__':
    main()
