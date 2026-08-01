"""
dt -> 0 extrapolation of the DT_BASE sweep (trotter_rom_dtbase) into a summary
npz and a two-panel colormap figure per model:

    1. stabilizer-3 framability**(1/dt)  at dt = 0
    2. RoM**(1/dt)                       at dt = 0

Both raw quantities tend trivially to 1 as dt -> 0 (the bond gate tends to the
identity; the evolved state tends back to the stabilizer start state), so the
plotted limit is the per-unit-time power exp(rate0), rate0 = lim ln(value)/dt --
the same construction and the same defaults (fit_n = 15, deg = 1) as
scripts/trotter_dtbase_line_extrap.py, so the framability panel is directly
comparable with results_dtbase_line.

--raw instead extrapolates the raw values, which must come out at 1 everywhere;
that panel is a validation of the sweep rather than a physical result.

Panels reuse the styling of trotter_rom_state_collect (viridis, log colour axis
labelled in powers of ten, white contour at the reference value, colour range
never forced to include it).

Usage:
    python scripts/trotter_rom_dtbase_extrap.py --all --save_npz
    python scripts/trotter_rom_dtbase_extrap.py --model model3 --fit_n 12 --deg 2
    python scripts/trotter_rom_dtbase_extrap.py --all --raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS
from trotter_rom_state import STATE_ROM_MODELS, grid_of
from trotter_rom_dtbase import (
    FIT_N_DEFAULT, DEG_DEFAULT, extrapolate_to_zero,
)
# scripts/ is on sys.path[0]: reuse the panel styling of the state pipeline so
# both figure families look identical.
from trotter_rom_state_collect import _panel

SUMMARY_NAME = 'rom_dtbase_extrap.npz'

# (key, panel title without the dt->0 suffix)
QUANTITIES = [
    ('stab_fra', 'Stabilizer-3 framability'),
    ('rom',      'RoM of the once-evolved 2x2 state'),
]


def load_and_extrapolate(in_dir: Path, model, *, fit_n: int, deg: int,
                         raw: bool) -> dict:
    """(N_X, N_Y) dt=0 limit of each quantity; NaN where a point is missing."""
    p1_vals, p2_vals = grid_of(model.name)
    n1, n2 = len(p1_vals), len(p2_vals)
    out = {k: np.full((n1, n2), np.nan) for k, _ in QUANTITIES}
    out['n_base'] = np.full((n1, n2), np.nan)
    mdir = in_dir / model.name

    n_loaded = 0
    for ix in range(n1):
        for iy in range(n2):
            f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                dt = np.asarray(d['dt'], float)
                out['n_base'][ix, iy] = int(np.sum(np.isfinite(dt)))
                for key, _ in QUANTITIES:
                    out[key][ix, iy] = extrapolate_to_zero(
                        dt, np.asarray(d[key], float),
                        fit_n=fit_n, deg=deg, raw=raw)
                n_loaded += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)

    nb = out['n_base'][np.isfinite(out['n_base'])]
    print(f'Loaded {n_loaded}/{n1 * n2} points for {model.name}'
          + (f' ({int(nb.min())}-{int(nb.max())} bases each).' if nb.size
             else '.'), flush=True)
    out['p1'], out['p2'] = p1_vals, p2_vals
    return out


def plot_model(res: dict, model, out_png: Path, *, raw: bool,
               fit_n: int, deg: int) -> None:
    x_vals, y_vals = res['p1'], res['p2']

    fig, axes = plt.subplots(1, len(QUANTITIES), figsize=(5.5 * len(QUANTITIES), 4.2),
                             constrained_layout=True)
    for ax, (key, label) in zip(np.atleast_1d(axes), QUANTITIES):
        if raw:
            # the raw limit sits at 1 everywhere; plot it linearly
            _panel(ax, fig, x_vals, y_vals, res[key],
                   f'{label}\nraw value at $dt\\to0$', 1.0, model)
        else:
            # exp(rate0) spans decades across a grid; plot log10 with 10^x ticks
            with np.errstate(divide='ignore', invalid='ignore'):
                log10_lim = np.log10(res[key])
            _panel(ax, fig, x_vals, y_vals, log10_lim,
                   f'{label}$^{{1/dt}}$\nat $dt\\to0$', 0.0, model,
                   cbar_pow10=True)

    kind = 'raw value' if raw else r'value$^{1/dt}$'
    fig.suptitle(f'{model.name}:  {model.title}\n'
                 rf'$dt\to0$ extrapolation of {kind}  '
                 rf'($dt=\mathrm{{DT\_BASE}}/\max(\|H\|_1,\{{\gamma_k\}})$, '
                 rf'fit_n={fit_n}, deg={deg})', fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_png}', flush=True)
    plt.close(fig)


def run_model(name: str, in_dir: Path, out_png: Path | None, *,
              save_npz: bool, fit_n: int, deg: int, raw: bool) -> None:
    model = MODELS[name]
    res = load_and_extrapolate(in_dir, model, fit_n=fit_n, deg=deg, raw=raw)
    suffix = '_raw' if raw else ''
    if save_npz:
        npz = in_dir / model.name / SUMMARY_NAME.replace('.npz', f'{suffix}.npz')
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, fit_n=fit_n, deg=deg, raw=raw,
                 **{k: res[k] for k in
                    [q for q, _ in QUANTITIES] + ['n_base', 'p1', 'p2']})
        print(f'Saved {npz}', flush=True)
    png = out_png or (Path('results_plots') /
                      f'trotter_rom_dtbase_extrap{suffix}_{name}.png')
    plot_model(res, model, png, raw=raw, fit_n=fit_n, deg=deg)


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--model', type=str, choices=list(STATE_ROM_MODELS))
    g.add_argument('--all', action='store_true', help='every model in turn')
    p.add_argument('--in_dir',  type=str, default='results_trotter_rom_dtbase')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_rom_dtbase_extrap_<model>.png '
                        '(ignored with --all)')
    p.add_argument('--save_npz', action='store_true',
                   help=f'also save <in_dir>/<model>/{SUMMARY_NAME}')
    p.add_argument('--fit_n', type=int, default=FIT_N_DEFAULT,
                   help='number of points nearest dt=0 used in the fit')
    p.add_argument('--deg',   type=int, default=DEG_DEFAULT,
                   help='polynomial degree of the dt-extrapolation fit')
    p.add_argument('--raw', action='store_true',
                   help='extrapolate the raw values instead of value**(1/dt) '
                        '(a validation panel: the limit must be 1 everywhere)')
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    names = list(STATE_ROM_MODELS) if args.all else [args.model]
    for name in names:
        run_model(name, in_dir,
                  None if args.all or args.out_png is None else Path(args.out_png),
                  save_npz=args.save_npz, fit_n=args.fit_n, deg=args.deg,
                  raw=args.raw)


if __name__ == '__main__':
    main()
