"""
model10 framability-rate figure on the 2D square lattice (bond generator with
dim = 2: each qubit sits on 4 bonds, so the one-site terms h X and Delta1 X
enter a bond with weight 1/4 instead of the chain's 1/2), and the neighbour
cross-evaluation stage of that pipeline.

The framability panels of results_model4_rate/model10_rate_panels.png, same
positions and style, without the 8q many-body panels:

    row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext=4 | d_ext=8 | d_ext=12
    row 2 | product-state chi=10 | chi=40

Data (all under --out_dir, default results_model10_rate_dim2; written by
scripts/model10_dim2.slurm.sh, driven by scripts/submit_model10_dim2.sh):
    model10_seeded/      stabilizer-3 / Pauli (worker --fixed_frames) and the
                         d_ext = 4 / 8 rates: worker (seeded from the dim = 1
                         frames), _xeval, quick-refine, margin files
    model10_seeded_d12/  d_ext = 12: seed stage, refine rounds, margin files
    model10_product/     product-state rates chi = 10, 40
Per point each optimised panel shows the minimum over every certified upper
bound on that d_ext's optimum (model10_seeded_panels_collect.load_best), and
the d_ext = 8 / 12 panels also take the smaller d_ext's value (a frame padded
by repeated columns keeps its rate).

--xeval   neighbour cross-evaluation of the d_ext = 4 / 8 frames
          (model10_seeded_collect.xeval: Jacobi sweeps, per-column-LP
          certified, writes pt_<ix>_<iy>_xeval.npz) on the dim-d generators;
          runs between the seed stage and the quick-refine rounds
--no_plot skip the figure

Outputs:
    --out_png  (default results_model4_rate/model10_dim2_rate_panels.png)
    <out_dir>/model10_dim<d>_rate_panels.npz

Usage:
    python scripts/model10_dim2_collect.py --xeval --no_plot --n_proc 8
    python scripts/model10_dim2_collect.py
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from model10_seeded_worker import MODEL, grid_vals, pt_dir_name           # noqa: E402
from model10_d12_worker import d12_dir                                    # noqa: E402
from model10_seeded_collect import load_seeded, xeval                     # noqa: E402
from model10_seeded_panels_collect import load_best                       # noqa: E402
import model10_panels_common as common                                    # noqa: E402

DIM_DEFAULT = 2
FIXED_KEYS = [('rate_stab3', 'Stabilizer-3 framability rate'),
              ('rate_pauli', 'Pauli framability rate')]
LATTICE = {1: '1D chain', 2: '2D square lattice', 3: '3D cubic lattice'}


def load_fixed(pt: Path, nx: int, ny: int) -> dict:
    """rate_stab3 / rate_pauli grids from the worker files of pt."""
    g = {k: np.full((nx, ny), np.nan) for k, _ in FIXED_KEYS}
    for ix in range(nx):
        for iy in range(ny):
            f = pt / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
            except Exception as e:                          # noqa: BLE001
                print(f'  warning: {f.name}: {e}', flush=True)
                continue
            for k, _ in FIXED_KEYS:
                if k in d.files:
                    g[k][ix, iy] = float(d[k])
    print(f'[{MODEL} fixed frames] stab3 at {int(np.isfinite(g["rate_stab3"]).sum())}, '
          f'Pauli at {int(np.isfinite(g["rate_pauli"]).sum())} of {nx * ny} points',
          flush=True)
    return g


def summary(R, ref, m: int) -> None:
    """Log line per optimised panel (same counters as the dim = 1 figure)."""
    up = int((R[1:, :] - R[:-1, :] > 1e-4).sum() + (R[:, 1:] - R[:, :-1] > 1e-4).sum())
    near = int(((R > common.CONTOUR_TOL) & (R < 1e-3)).sum())
    txt = ''
    if ref is not None:
        both = np.isfinite(R) & np.isfinite(ref)
        txt = (f'; below d_ext={m - 4 if m == 8 else 8} at '
               f'{int(((ref - R)[both] > 1e-6).sum())}')
    print(f'  d{m}: data at {int(np.isfinite(R).sum())} pts; rate 0 at '
          f'{int((R <= common.CONTOUR_TOL).sum())}{txt}; non-monotone steps {up}; '
          f'near-zero (1e-6..1e-3) {near}', flush=True)


def rounds_txt(s: dict, what: str) -> str:
    """Two short title lines: frame source, then the rounds behind it."""
    parts = [f'{len(s["rounds"])} refine']
    parts += [f'{len(s["margin"])} margin'] if s['margin'] else []
    parts += [f'{len(s["rref"])} randomised'] if s['rref'] else []
    return f'{what}\n+ ' + ' + '.join(parts) + ' rounds'


def plot(panels: list, png: Path, dim: int) -> None:
    """panels: (x, y, Z, title), drawn row by row on a 2 x 5 grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    m = MODELS[MODEL]
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(27.5, 5 * nrow),
                             constrained_layout=True)
    title = m.title.replace(LATTICE[1], LATTICE.get(dim, f'{dim}D lattice'))
    fig.suptitle(
        f'{MODEL}:  {title}'
        '\n'
        r'framability rates $\mu^*=\lim_{dt\to0}({\rm fra}-1)/dt$ of the bond '
        rf'generator, one-site terms $\times 1/{2 * dim}$ per bond '
        r'(white: $\mu^*=0$)  |  opt Heisenberg: optimised observable frames, '
        r'seeded from the 1D-chain optima  |  product-state: random product frames',
        fontsize=13)
    flat = list(np.atleast_1d(axes).flat)
    for ax, (xv, yv, Z, t) in zip(flat, panels):
        common.draw_panel(fig, ax, xv, yv, Z, t, common.FRA_CMAP,
                          floor_contour=0.0, xlabel=m.p1_label, ylabel=m.p2_label)
    for ax in flat[len(panels):]:
        ax.axis('off')
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[{MODEL} dim={dim}] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=DIM_DEFAULT)
    ap.add_argument('--out_dir', type=str, default=None,
                    help='pipeline root (default results_model10_rate_dim<dim>)')
    ap.add_argument('--stride', type=int, default=1,
                    help='stride of the seeded / d_ext = 12 arrays')
    ap.add_argument('--prod_stride', type=int, default=None,
                    help='stride of the product-state run (default --stride)')
    ap.add_argument('--out_png', type=str, default=None,
                    help='default results_model4_rate/model10_dim<dim>_rate_panels.png')
    ap.add_argument('--xeval', action='store_true',
                    help='run the d_ext = 4 / 8 neighbour cross-evaluation first')
    ap.add_argument('--radius4', type=int, default=2)
    ap.add_argument('--radius8', type=int, default=1)
    ap.add_argument('--max_sweeps', type=int, default=6)
    ap.add_argument('--n_proc', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.out_dir or f'results_model10_rate_dim{args.dim}')
    png = Path(args.out_png or
               f'results_model4_rate/{MODEL}_dim{args.dim}_rate_panels.png')
    prod_stride = args.prod_stride or args.stride

    p1, p2 = grid_vals(args.stride)
    nx, ny = len(p1), len(p2)
    pt = out_dir / pt_dir_name(args.stride)

    if args.xeval:
        g = load_seeded(pt, nx, ny)
        if g['n_points'] == 0:
            sys.exit(f'no seeded worker output under {pt}')
        n_imp = xeval(g, p1, p2, pt, dim=args.dim,
                      radius={4: args.radius4, 8: args.radius8},
                      n_proc=args.n_proc, max_sweeps=args.max_sweeps)
        print(f'[{MODEL} dim={args.dim}] neighbour cross-evaluation improved '
              f'{n_imp} point(s)', flush=True)
    if args.no_plot:
        return

    fixed = load_fixed(pt, nx, ny)
    s4 = load_best(pt, nx, ny, 4)
    s8 = load_best(pt, nx, ny, 8)
    s12 = load_best(d12_dir(str(out_dir), args.stride), nx, ny, 12)
    panel = {4: s4['grid']}
    panel[8] = np.fmin(s8['grid'], panel[4])
    panel[12] = np.fmin(s12['grid'], panel[8])
    summary(panel[4], None, 4)
    summary(panel[8], panel[4], 8)
    summary(panel[12], panel[8], 12)

    prod = common.load_prod([str(out_dir)], prod_stride)
    if prod is None:
        print(f'[{MODEL} dim={args.dim}] no product-state data under '
              f'{out_dir / "model10_product"}; those panels are left empty',
              flush=True)

    dl = r'$d_{\rm ext}'
    panels = [(p1, p2, fixed[k], label) for k, label in FIXED_KEYS]
    panels += [
        (p1, p2, panel[4], rf'Opt Heisenberg rate ({dl}=4$)'
                           f'\n{rounds_txt(s4, "seeded from 1D")}'),
        (p1, p2, panel[8], rf'Opt Heisenberg rate ({dl}=8$)'
                           f'\n{rounds_txt(s8, "seeded from 1D")}'),
        (p1, p2, panel[12], rf'Opt Heisenberg rate ({dl}=12$)'
                            f'\n{rounds_txt(s12, "seeded from 1D + $d=8$")}'),
    ]
    for key, label in common.PROD_RATE_KEYS:
        if prod is None:
            panels.append((p1, p2, np.full((nx, ny), np.nan), label))
        else:
            panels.append((prod['p1_vals'], prod['p2_vals'], prod[key], label))

    out_dir.mkdir(parents=True, exist_ok=True)
    x, y = MODELS[MODEL].p1_name, MODELS[MODEL].p2_name
    npz = out_dir / f'{MODEL}_dim{args.dim}_rate_panels.npz'
    np.savez(npz, model=MODEL, dim=args.dim, stride=args.stride,
             **{f'{x}_vals': p1, f'{y}_vals': p2},
             **{k: fixed[k] for k, _ in FIXED_KEYS},
             panel_rate_heis_4=panel[4], panel_rate_heis_8=panel[8],
             panel_rate_heis_12=panel[12],
             seeded_rate_4=s4['grid'], seeded_rate_8=s8['grid'],
             d12_rate_12=s12['grid'],
             qrefine_rounds=np.array(s4['rounds'], int),
             d12_refine_rounds=np.array(s12['rounds'], int),
             **({} if prod is None else
                {f'prod_{x}_vals': prod['p1_vals'], f'prod_{y}_vals': prod['p2_vals'],
                 **{k: prod[k] for k, _ in common.PROD_RATE_KEYS}}))
    print(f'[{MODEL} dim={args.dim}] wrote {npz}', flush=True)
    plot(panels, png, args.dim)


if __name__ == '__main__':
    main()
