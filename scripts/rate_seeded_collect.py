"""
Framability-rate figure of the seeded rate pipeline for any model of
trotter_lindbladian_scan.MODELS (scripts/submit_rate_seeded.sh), and the
pipeline's neighbour cross-evaluation stage.

Panels (same positions and style as the model10 figures):

    row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext=4 | d_ext=8 | d_ext=12
    row 2 | product-state chi=10 | chi=40

Data, all under --out_dir (default results_<model>_seeded):
    <model>_seeded/      stabilizer-3 / Pauli (worker --fixed_frames) and the
                         d_ext = 4 / 8 rates: worker, _xeval, quick-refine,
                         margin and randomised-refine files
    <model>_seeded_d12/  d_ext = 12: seed stage and every round file
    <model>_product/     product-state rates chi = 10, 40
Each optimised panel shows, per point, the minimum over every certified
upper bound of that d_ext (model10_seeded_panels_collect.load_best); the
d_ext = 8 / 12 panels also take the smaller d_ext's value (padding).

The log reports, per d_ext, the share of the Pauli-positive points brought to
mu* = 0, and -- with --ref_npz -- the comparison with an earlier figure's
optimised panels: rates (rate_heis_4 / rate_heis_6, e.g.
results_model4_rate/model4_rate_panels.npz) or dt -> 0 extrapolated
framabilities fra^(1/dt) ~ exp(mu*) (opt_fra_4 / opt_fra_6, e.g.
results_dtbase_line/model3_dtbase_extrap.npz; 'framable' there is
fra^(1/dt) <= 1.001, i.e. mu* <= log 1.001).

--xeval   neighbour cross-evaluation of the d_ext = 4 / 8 frames
          (model10_seeded_collect.xeval; run after the seed stage)
--no_plot skip the figure

Outputs:
    --out_png  (default results_model4_rate/<model>_seeded_rate_panels.png)
    <out_dir>/<model>_seeded_rate_panels.npz

Usage:
    python scripts/rate_seeded_collect.py --model model4 --xeval --no_plot --n_proc 8
    python scripts/rate_seeded_collect.py --model model4 \\
        --ref_npz results_model4_rate/model4_rate_panels.npz
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
from model10_seeded_worker import grid_vals, pt_dir_name                  # noqa: E402
from model10_d12_worker import d12_dir                                    # noqa: E402
from model10_seeded_collect import load_seeded, xeval                     # noqa: E402
from model10_seeded_panels_collect import load_best                       # noqa: E402
from model4_product_rate_worker import PROD_RATE_KEYS, prod_tag           # noqa: E402
import model10_panels_common as common                                    # noqa: E402

FIXED_KEYS = [('rate_stab3', 'Stabilizer-3 framability rate'),
              ('rate_pauli', 'Pauli framability rate')]
TOL = common.CONTOUR_TOL                  # mu* within this of 0 = rate 0
FRA_TOL = 1.001                           # 'framable' of the finite-dt figures


def load_fixed(pt: Path, nx: int, ny: int, model: str) -> dict:
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
    print(f'[{model} fixed frames] stab3 at {int(np.isfinite(g["rate_stab3"]).sum())}, '
          f'Pauli at {int(np.isfinite(g["rate_pauli"]).sum())} of {nx * ny} points',
          flush=True)
    return g


def rounds_txt(s: dict, what: str, dext_only_refine: bool = False) -> str:
    """Two short title lines: frame source, then the rounds behind it."""
    parts = [f'{len(s["rounds"])} refine']
    if not dext_only_refine:
        parts += [f'{len(s["margin"])} margin'] if s['margin'] else []
        parts += [f'{len(s["rref"])} randomised'] if s['rref'] else []
    return f'{what}\n+ ' + ' + '.join(parts) + ' rounds'


def summary(R, ref, m: int, ref_m: int | None) -> None:
    """Log line per optimised panel (same counters as the model10 figures)."""
    up = int((np.diff(R, axis=0) > 1e-4).sum() + (np.diff(R, axis=1) > 1e-4).sum())
    near = int(((R > TOL) & (R < 1e-3)).sum())
    txt = ''
    if ref is not None:
        both = np.isfinite(R) & np.isfinite(ref)
        txt = f'; below d_ext={ref_m} at {int(((ref - R)[both] > 1e-6).sum())}'
    print(f'  d{m}: data at {int(np.isfinite(R).sum())} pts; rate 0 at '
          f'{int((R <= TOL).sum())}{txt}; non-monotone steps {up}; '
          f'near-zero (1e-6..1e-3) {near}', flush=True)


def gap_report(panel: dict, pauli) -> None:
    """Share of the Pauli-positive points each optimised panel brings to 0."""
    pz = np.isfinite(pauli) & (pauli <= TOL)
    pos = np.isfinite(pauli) & ~pz
    n = int(pos.sum())
    if n == 0:
        return
    parts = []
    for m, R in panel.items():
        z = np.isfinite(R) & (R <= TOL)
        parts.append(f'd{m} {int((z & pos).sum())} ({100 * (z & pos).sum() / n:.1f}%)')
    print(f'  Pauli: rate 0 at {int(pz.sum())}; of its {n} positive points the '
          f'optimised frames bring to rate 0: ' + ' | '.join(parts), flush=True)


def ref_report(ref_npz: Path, panel: dict) -> None:
    """Rate-0 / framable counts of an earlier figure's optimised panels next
    to the new ones (same grid assumed; shapes are checked)."""
    if not ref_npz.exists():
        print(f'  [ref] {ref_npz} not found; comparison skipped', flush=True)
        return
    d = np.load(ref_npz, allow_pickle=True)
    if 'rate_heis_4' in d.files:
        old = {4: np.asarray(d['rate_heis_4'], float)}
        if 'rate_heis_6' in d.files:
            old[6] = np.fmin(old[4], np.asarray(d['rate_heis_6'], float))
        tol, what = TOL, 'rate 0'
    elif 'opt_fra_4' in d.files:
        old = {4: np.log(np.asarray(d['opt_fra_4'], float))}
        if 'opt_fra_6' in d.files:
            old[6] = np.fmin(old[4], np.log(np.asarray(d['opt_fra_6'], float)))
        tol, what = np.log(FRA_TOL), f'framable (fra^(1/dt) <= {FRA_TOL})'
    else:
        print(f'  [ref] {ref_npz}: no rate_heis_4 / opt_fra_4; skipped', flush=True)
        return
    shape = next(iter(panel.values())).shape
    old = {k: v for k, v in old.items() if v.shape == shape}
    if not old:
        print(f'  [ref] {ref_npz}: grid shape differs; skipped', flush=True)
        return
    old_txt = ', '.join(f'old d{k}: {int((v <= tol).sum())}' for k, v in old.items())
    new_txt = ', '.join(f'new d{m}: {int((R <= tol).sum())}' for m, R in panel.items())
    print(f'  [ref] {what} counts with the same tolerance ({ref_npz.name}): '
          f'{old_txt} | {new_txt}', flush=True)
    if what == 'rate 0':
        k = max(old)
        both = np.isfinite(old[k]) & np.isfinite(panel[8])
        gain = np.where(both, old[k] - panel[8], np.nan)
        print(f'  [ref] new d8 below old d{k} by > 1e-6 at '
              f'{int((gain > 1e-6).sum())} points (largest gain {np.nanmax(gain):.3g}); '
              f'above it at {int((gain < -1e-6).sum())}', flush=True)


def plot(panels: list, png: Path, model: str, dim: int) -> None:
    """panels: (x, y, Z, title), drawn row by row on a 2 x 5 grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    m = MODELS[model]
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(27.5, 5 * nrow),
                             constrained_layout=True)
    fig.suptitle(
        f'{model}:  {m.title}'
        '\n'
        r'framability rates $\mu^*=\lim_{dt\to0}({\rm fra}-1)/dt$ of the bond '
        rf'generator (dim $={dim}$; white: $\mu^*=0$)  |  opt Heisenberg: '
        r'optimised observable frames, seeded pipeline  |  product-state: '
        r'random product frames',
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
    print(f'[{model}] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True, choices=sorted(MODELS))
    ap.add_argument('--dim', type=int, default=None,
                    help="bond convention dimension (default: the model's)")
    ap.add_argument('--out_dir', type=str, default=None,
                    help='pipeline root (default results_<model>_seeded)')
    ap.add_argument('--stride', type=int, default=1,
                    help='stride of the seeded / d_ext = 12 arrays')
    ap.add_argument('--prod_stride', type=int, default=None,
                    help='stride of the product-state run (default --stride)')
    ap.add_argument('--out_png', type=str, default=None,
                    help='default results_model4_rate/<model>_seeded_rate_panels.png')
    ap.add_argument('--ref_npz', type=str, default=None,
                    help='earlier figure data to compare the optimised panels with')
    ap.add_argument('--xeval', action='store_true',
                    help='run the d_ext = 4 / 8 neighbour cross-evaluation first')
    ap.add_argument('--radius4', type=int, default=2)
    ap.add_argument('--radius8', type=int, default=1)
    ap.add_argument('--max_sweeps', type=int, default=6)
    ap.add_argument('--n_proc', type=int,
                    default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()
    model = args.model
    dim = args.dim if args.dim is not None else MODELS[model].dim
    out_dir = Path(args.out_dir or f'results_{model}_seeded')
    png = Path(args.out_png or f'results_model4_rate/{model}_seeded_rate_panels.png')
    prod_stride = args.prod_stride or args.stride

    p1, p2 = grid_vals(args.stride, model)
    nx, ny = len(p1), len(p2)
    pt = out_dir / pt_dir_name(args.stride, model)

    if args.xeval:
        g = load_seeded(pt, nx, ny)
        if g['n_points'] == 0:
            sys.exit(f'no seeded worker output under {pt}')
        n_imp = xeval(g, p1, p2, pt, dim=dim,
                      radius={4: args.radius4, 8: args.radius8},
                      n_proc=args.n_proc, max_sweeps=args.max_sweeps, model=model)
        print(f'[{model}] neighbour cross-evaluation improved {n_imp} point(s)',
              flush=True)
    if args.no_plot:
        return

    fixed = load_fixed(pt, nx, ny, model)
    s4 = load_best(pt, nx, ny, 4, model)
    s8 = load_best(pt, nx, ny, 8, model)
    s12 = load_best(d12_dir(str(out_dir), args.stride, model), nx, ny, 12, model)
    panel = {4: s4['grid']}
    panel[8] = np.fmin(s8['grid'], panel[4])
    panel[12] = np.fmin(s12['grid'], panel[8])
    summary(panel[4], None, 4, None)
    summary(panel[8], panel[4], 8, 4)
    summary(panel[12], panel[8], 12, 8)
    gap_report(panel, fixed['rate_pauli'])
    if args.ref_npz:
        ref_report(Path(args.ref_npz), panel)

    prod_pt = out_dir / prod_tag(model)
    prod = (common.load_group(prod_pt, [k for k, _ in PROD_RATE_KEYS], prod_stride,
                              f'{model}-product', model=model)
            if prod_pt.is_dir() else None)
    if prod is None or prod['n_points'] == 0:
        print(f'[{model}] no product-state data under {prod_pt}; those panels '
              'are left empty', flush=True)
        prod = None

    dl = r'$d_{\rm ext}'
    panels = [(p1, p2, fixed[k], label) for k, label in FIXED_KEYS]
    panels += [
        # margin / randomised rounds in that folder run at d_ext = 8 only
        (p1, p2, panel[4], rf'Opt Heisenberg rate ({dl}=4$)'
                           f'\n{rounds_txt(s4, "seeded", dext_only_refine=True)}'),
        (p1, p2, panel[8], rf'Opt Heisenberg rate ({dl}=8$)'
                           f'\n{rounds_txt(s8, "seeded")}'),
        (p1, p2, panel[12], rf'Opt Heisenberg rate ({dl}=12$)'
                            f'\n{rounds_txt(s12, "seeded from $d=8$")}'),
    ]
    for key, label in PROD_RATE_KEYS:
        if prod is None:
            panels.append((p1, p2, np.full((nx, ny), np.nan), label))
        else:
            panels.append((prod['p1_vals'], prod['p2_vals'], prod[key], label))

    out_dir.mkdir(parents=True, exist_ok=True)
    x, y = MODELS[model].p1_name, MODELS[model].p2_name
    npz = out_dir / f'{model}_seeded_rate_panels.npz'
    np.savez(npz, model=model, dim=dim, stride=args.stride,
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
                 **{k: prod[k] for k, _ in PROD_RATE_KEYS}}))
    print(f'[{model}] wrote {npz}', flush=True)
    plot(panels, png, model, dim)


if __name__ == '__main__':
    main()
