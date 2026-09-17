"""
Stage 3 of the product-frame growth pipeline: aggregate and plot.

Reads  <in_dir>/<tag>/frames.npz                     (stage 1 ladders)
       <in_dir>/<tag>/fra_r<r>_<gp>_<field>.npz      (stage 2 framabilities)
Writes <out_dir>/product_frame_grow.npz              (every curve, one array per key)
       <out_dir>/<tag>_grow.png                      (per-case, 2 panels)
       <out_dir>/product_frame_grow_rates.png        (all seven cases, rate panels)

Per case the figure is framability against d_ext (left) and the framability
RATE (f - 1)/dt against d_ext on a log axis (right), with one curve per
(gamma' variant, target variant):

    gamma' = J        (the continuous_simulation.tex threshold)   colour 0
    gamma' = 0.99 J   (detuned)                                   colour 1
    'plain' target    (the bare Euler gate: the default)          dashed
    'free'  target    (optional dt -> 0 reference)                solid

The rate panel is the readable one: at dt = 1e-2 every framability here sits
within ~1e-3 of 1, and the interesting quantities -- the 2|lambda_-| ~ 2e-2
detuning signal and the non-CP floor of the Euler step (see
product_frame_grow's docstring) -- are only visible as rates.  The floor the
worker MEASURED (negativity_floor, a certified lower bound on f for any frame
at this dt) is drawn dotted per gamma' variant, with the analytic corner
2 dt J^2 dash-dotted for reference.  A curve sitting on its floor means the
time step is the limitation, not the frame.

Usage:
    python scripts/product_frame_grow_collect.py
    python scripts/product_frame_grow_collect.py --in_dir results_product_frame_grow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from product_frame_grow import CASES, FIELDS, GP_FACTOR_DEFAULT              # noqa: E402
from product_frame_grow_frames_worker import OUT_DIR_DEFAULT, load_frames    # noqa: E402
from product_frame_grow_worker import GP_VARIANTS, unit_path                 # noqa: E402

GP_COLOUR = {'at': '#1f77b4', 'lo': '#d62728'}
FIELD_STYLE = {'free': '-', 'plain': '--'}
FIELD_MARKER = {'free': 'o', 'plain': 's'}


def gp_label(gp: str, gp_factor: float) -> str:
    return r"$\gamma' = J$" if gp == 'at' else rf"$\gamma' = {gp_factor:g}\,J$"


def load_case(in_dir, tag: str) -> dict | None:
    """{(gp, field): (d_ext, framability, rate)} plus the ladder metadata."""
    lad = load_frames(in_dir, tag)
    if lad is None:
        print(f'  warning: no ladder for {tag}', flush=True)
        return None
    out = dict(tag=tag, d_exts=np.asarray(lad['d_exts'], dtype=int),
               dt=float(np.asarray(lad['dt']).ravel()[0]),
               gate_kind=str(np.asarray(lad['gate_kind']).ravel()[0]),
               J=float(np.asarray(lad['J']).ravel()[0]),
               gamma=float(np.asarray(lad['gamma']).ravel()[0]),
               h=float(np.asarray(lad['h']).ravel()[0]),
               model=str(np.asarray(lad['model']).ravel()[0]),
               rounds=lad['rounds'], curves={})
    gp_factor = float('nan')
    for gp in GP_VARIANTS:
        for field in FIELDS:
            d_ext, fra, floor = [], [], []
            for r, d in enumerate(out['d_exts']):
                path = unit_path(in_dir, tag, r, gp, field)
                if not path.exists():
                    continue
                try:
                    z = np.load(path, allow_pickle=True)
                except Exception as e:
                    print(f'  warning: {path.name}: {e}', flush=True)
                    continue
                d_ext.append(int(np.asarray(z['d_ext']).ravel()[0]))
                fra.append(float(np.asarray(z['framability']).ravel()[0]))
                floor.append(float(np.asarray(z['floor']).ravel()[0])
                             if 'floor' in z.files else np.nan)
                if gp == 'lo':
                    gp_factor = float(np.asarray(z['gp_factor']).ravel()[0])
            order = np.argsort(d_ext)
            d_ext = np.asarray(d_ext, dtype=int)[order]
            fra = np.asarray(fra, dtype=float)[order]
            floor = np.asarray(floor, dtype=float)[order]
            out['curves'][(gp, field)] = (d_ext, fra, (fra - 1.0) / out['dt'])
            out.setdefault('floors', {})[(gp, field)] = (d_ext, floor,
                                                         (floor - 1.0) / out['dt'])
    out['gp_factor'] = gp_factor if np.isfinite(gp_factor) else GP_FACTOR_DEFAULT
    n_have = sum(len(v[0]) for v in out['curves'].values())
    print(f'[{tag}] ladder d_exts={list(out["d_exts"])}, '
          f'{n_have}/{4 * len(out["d_exts"])} framabilities present', flush=True)
    return out


def case_title(rec: dict) -> str:
    return (rf"{rec['model']}:  $J={rec['J']:g}$, $\gamma={rec['gamma']:g}$, "
            rf"$h={rec['h']:g}$,  $\mathrm{{d}}t={rec['dt']:g}$")


def floor_curve(rec: dict, gp: str, field: str = 'plain'):
    """(d_ext, floor rate): the MEASURED certified lower bound of the worker.

    negativity_floor over the very target columns the LP was given, so it is a
    rigorous lower bound on f for any frame at this dt.  Falls back to the
    analytic 2 dt J^2 (the |q| = 1 corner of the tex matrix) for data written
    before the floor was recorded.
    """
    d_ext, _fl, rate = rec.get('floors', {}).get((gp, field), ([], [], []))
    if len(d_ext) and np.all(np.isfinite(rate)):
        return np.asarray(d_ext), np.asarray(rate)
    return np.asarray([]), np.asarray([])


def floor_rate_estimate(rec: dict) -> float:
    """2 dt J^2 |q|^2_max = 2 dt J^2: the analytic corner of the Euler floor."""
    return 2.0 * rec['dt'] * rec['J'] ** 2


def plot_case(rec: dict, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for (gp, field), (d_ext, fra, rate) in rec['curves'].items():
        if len(d_ext) == 0:
            continue
        lab = f"{gp_label(gp, rec['gp_factor'])}, {field}"
        kw = dict(color=GP_COLOUR[gp], ls=FIELD_STYLE[field],
                  marker=FIELD_MARKER[field], ms=4, lw=1.6, label=lab)
        axes[0].plot(d_ext, fra, **kw)
        axes[1].plot(d_ext, np.maximum(rate, 1e-16), **kw)
    for ax, ylab in zip(axes, ('framability $f$',
                               r'rate $(f-1)/\mathrm{d}t$')):
        ax.set_xlabel(r'$d_{\mathrm{ext}}$ (single-qubit frame size)')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[1].set_yscale('log')
    for gp in GP_VARIANTS:
        fd, fr = floor_curve(rec, gp)
        if len(fd):
            axes[1].plot(fd, np.maximum(fr, 1e-16), color=GP_COLOUR[gp], lw=1.0,
                         ls=':', label=f'{gp_label(gp, rec["gp_factor"])}, '
                                       'certified floor')
    axes[1].axhline(floor_rate_estimate(rec), color='0.4', lw=0.8, ls='-.',
                    label=r'$2\,\mathrm{d}t\,J^2$ (analytic corner)')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(case_title(rec))
    fig.tight_layout()
    png = out_dir / f'{rec["tag"]}_grow.png'
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def plot_all(recs: list, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ncol = 4
    nrow = int(np.ceil(len(recs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow),
                             squeeze=False)
    flat = axes.ravel()
    for ax, rec in zip(flat, recs):
        for (gp, field), (d_ext, _fra, rate) in rec['curves'].items():
            if len(d_ext) == 0:
                continue
            ax.plot(d_ext, np.maximum(rate, 1e-16), color=GP_COLOUR[gp],
                    ls=FIELD_STYLE[field], marker=FIELD_MARKER[field], ms=3,
                    lw=1.4, label=f"{gp_label(gp, rec['gp_factor'])}, {field}")
        for gp in GP_VARIANTS:
            fd, fr = floor_curve(rec, gp)
            if len(fd):
                ax.plot(fd, np.maximum(fr, 1e-16), color=GP_COLOUR[gp], lw=1.0,
                        ls=':')
        ax.axhline(floor_rate_estimate(rec), color='0.4', lw=0.8, ls='-.')
        ax.set_yscale('log')
        ax.set_title(case_title(rec), fontsize=9)
        ax.set_xlabel(r'$d_{\mathrm{ext}}$')
        ax.set_ylabel(r'$(f-1)/\mathrm{d}t$')
        ax.grid(alpha=0.3)
    for ax in flat[len(recs):]:
        ax.axis('off')
    handles, labels = flat[0].get_legend_handles_labels()
    if len(recs) < len(flat):
        flat[len(recs)].legend(handles, labels, loc='center', fontsize=9,
                               title='dotted: Euler non-CP floor')
    else:
        fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9)
    fig.suptitle("Product-frame growth: framability rate vs frame size "
                 "(growth at $\\gamma' = J$, octahedron start)")
    fig.tight_layout()
    png = out_dir / 'product_frame_grow_rates.png'
    fig.savefig(png, dpi=150)
    plt.close(fig)
    return png


def save_npz(recs: list, out_dir: Path) -> Path:
    data = {}
    for rec in recs:
        tag = rec['tag']
        data[f'{tag}_d_exts'] = rec['d_exts']
        data[f'{tag}_dt'] = np.array(rec['dt'])
        data[f'{tag}_gamma'] = np.array(rec['gamma'])
        data[f'{tag}_h'] = np.array(rec['h'])
        data[f'{tag}_J'] = np.array(rec['J'])
        for (gp, field), (d_ext, fra, rate) in rec['curves'].items():
            data[f'{tag}_{gp}_{field}_d_ext'] = d_ext
            data[f'{tag}_{gp}_{field}_fra'] = fra
            data[f'{tag}_{gp}_{field}_rate'] = rate
            fl = rec.get('floors', {}).get((gp, field))
            if fl is not None:
                data[f'{tag}_{gp}_{field}_floor'] = fl[1]
                data[f'{tag}_{gp}_{field}_floor_rate'] = fl[2]
    npz = out_dir / 'product_frame_grow.npz'
    np.savez(npz, **data)
    return npz


def report(recs: list) -> None:
    print('\n' + '=' * 92)
    print('framability rate (f-1)/dt at the smallest and largest d_ext reached')
    print('=' * 92)
    head = f'{"case":16s} {"d_ext":>11s}'
    for gp in GP_VARIANTS:
        for field in FIELDS:
            head += f'  {gp + "/" + field:>17s}'
    print(head)
    for rec in recs:
        for which, idx in (('first', 0), ('last', -1)):
            row = f'{rec["tag"]:16s} {which:>11s}'
            for gp in GP_VARIANTS:
                for field in FIELDS:
                    d_ext, _fra, rate = rec['curves'][(gp, field)]
                    row += ('  ' + (f'{rate[idx]:.6e} ({d_ext[idx]:3d})'
                                    if len(d_ext) else f'{"-":>17s}'))
            print(row)
    print('=' * 92)
    print('certified floor (negativity_floor: a lower bound for ANY frame) at '
          'the largest d_ext')
    for rec in recs:
        row = f'{rec["tag"]:16s} {"floor":>11s}'
        for gp in GP_VARIANTS:
            for field in FIELDS:
                fl = rec.get('floors', {}).get((gp, field))
                row += ('  ' + (f'{fl[2][-1]:.6e}      ' if fl is not None
                                and len(fl[0]) and np.isfinite(fl[2][-1])
                                else f'{"-":>17s}'))
        print(row)
    print('=' * 92)
    print("A gamma' = J curve that keeps falling while the gamma' < J one "
          "flattens at ~2|lambda_-| = 2e-2 is the tex threshold showing up.  "
          'If a curve sits ON its certified floor, the time step is the '
          'limitation, not the frame: rerun stages 1-2 with a smaller DT '
          '(dt = 1e-3 clears all seven cases; dt = 1e-2 only clears gamma = 0 '
          'and 2, since the floor grows like dt*gamma^2).')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', type=str, default=OUT_DIR_DEFAULT)
    p.add_argument('--out_dir', type=str, default=OUT_DIR_DEFAULT)
    p.add_argument('--tags', type=str, nargs='*', default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for case in CASES:
        if args.tags and case['tag'] not in args.tags:
            continue
        rec = load_case(args.in_dir, case['tag'])
        if rec is not None:
            recs.append(rec)
    if not recs:
        print('nothing to collect: run stages 1 and 2 first', flush=True)
        return

    for rec in recs:
        print(f'  wrote {plot_case(rec, out_dir)}', flush=True)
    print(f'  wrote {plot_all(recs, out_dir)}', flush=True)
    print(f'  wrote {save_npz(recs, out_dir)}', flush=True)
    report(recs)


if __name__ == '__main__':
    main()
