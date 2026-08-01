"""
Collect one model's state-RoM worker results (trotter_rom_state) into a summary
npz and a colormap figure over the model's full scan grid:

    1. stabilizer-3 framability of the two-qubit bond Trotter gate
    2. the same as a per-unit-time rate, stab_fra**(1/dt), on a log colour axis
       (dt is O(1e-3), so the raw power overflows float64 as soon as stab_fra
       exceeds ~1.002 -- the panel therefore plots log10(stab_fra)/dt and labels
       the colourbar with the corresponding powers of ten)
    3. RoM of the 2x2-lattice state after one application of the lattice
       propagator to the lpdo_max start state
    4. the same as a per-unit-time rate, RoM**(1/dt), on the same log colour
       axis as panel 2.  One step is short, so RoM = 1 + O(dt) and panel 3
       largely tracks the per-point variation of dt; this panel divides it out.

--rate appends a fifth panel holding log2(RoM)/dt, which is exactly
log2(panel 4) -- the same information on a linear axis.

Panels 2 and 4 are derived here from the stored rom / log2_rom / rom_rate / dt,
so neither needs a worker re-run.

All panels use the perceptually uniform sequential viridis map (every quantity
is a one-sided magnitude) and draw a thin white contour at the reference value
-- framability = 1: framable; RoM = 1: the state stays stabilizer -- with the
title annotating min-reference where the value never reaches it on the grid.

Usage:
    python scripts/trotter_rom_state_collect.py --model model3
    python scripts/trotter_rom_state_collect.py --all --save_npz
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

# A panel counts as reaching its reference level if its minimum does so to
# within this tolerance.
ONE_TOL = 1e-6

# Keys pulled out of every per-point npz.
LOAD_KEYS = ('stab_fra', 'log10_stab_fra_pow', 'rom', 'log2_rom', 'rom_rate',
             'dt')


def load_results(in_dir: Path, model) -> dict[str, np.ndarray]:
    """Per-quantity (N_X, N_Y) arrays over the model's full grid; NaN where a
    point is missing."""
    p1_vals, p2_vals = grid_of(model.name)
    n1, n2 = len(p1_vals), len(p2_vals)
    out = {k: np.full((n1, n2), np.nan) for k in LOAD_KEYS}
    mdir = in_dir / model.name
    n_loaded = 0
    for ix in range(n1):
        for iy in range(n2):
            f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                for k in LOAD_KEYS:
                    if k in d:
                        out[k][ix, iy] = float(d[k])
                n_loaded += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    print(f'Loaded {n_loaded}/{n1 * n2} points for {model.name}.', flush=True)
    out['p1'], out['p2'] = p1_vals, p2_vals
    return out


def _edges(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 1:
        return np.array([vals[0] - 0.05, vals[0] + 0.05])
    step = np.diff(vals).mean()
    return np.concatenate([[vals[0] - step / 2],
                           (vals[:-1] + vals[1:]) / 2,
                           [vals[-1] + step / 2]])


def _panel(ax, fig, x_vals, y_vals, data, title, level, model,
           cbar_pow10: bool = False) -> None:
    """One pcolormesh panel; `data` is (N_X, N_Y) and is transposed for drawing
    so that x = p1 and y = p2.  `level` is the reference contour value."""
    grid = np.asarray(data).T                      # (N_Y, N_X)
    finite = grid[np.isfinite(grid)]
    if finite.size:
        vmin = min(float(finite.min()), level)
        vmax = max(float(finite.max()), level)
        if vmin == vmax:
            vmin, vmax = vmin - 0.05, vmax + 0.05
    else:
        vmin, vmax = level - 0.05, level + 1.0

    im = ax.pcolormesh(_edges(x_vals), _edges(y_vals), grid, cmap='viridis',
                       vmin=vmin, vmax=vmax, shading='flat')
    ax.set_xlabel(model.p1_label)
    ax.set_ylabel(model.p2_label)
    if finite.size and float(finite.min()) > level + ONE_TOL:
        title += f'\nmin$-{level:g}$ = {float(finite.min()) - level:.3g}'
    ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    ticks = sorted(set(np.linspace(vmin, vmax, 5).tolist() + [level]))
    cbar.set_ticks(ticks)
    if cbar_pow10:
        # the panel holds log10(value); label the bar with the value itself
        cbar.set_ticklabels([f'$10^{{{t:.3g}}}$' for t in ticks])
    try:
        ax.contour(x_vals, y_vals, grid, levels=[level], colors='white',
                   linewidths=1.0)
    except Exception:
        pass


def plot_model(res: dict, model, out_png: Path, rate: bool = False) -> None:
    x_vals, y_vals = res['p1'], res['p2']

    # RoM**(1/dt) gets the same log-axis treatment as stab_fra**(1/dt): the raw
    # power overflows float64 the moment RoM exceeds ~1.002.  log10(RoM)/dt is
    # taken from the stored rate rather than from RoM itself, so it is exactly
    # consistent with it and needs no worker re-run.
    log10_rom_pow = res['rom_rate'] * np.log10(2.0)

    panels = [
        ('Stabilizer-3 framability (2-qubit bond gate)', res['stab_fra'],
         1.0, False),
        (r'Stabilizer-3 framability$^{1/dt}$', res['log10_stab_fra_pow'],
         0.0, True),
        ('RoM of the once-evolved 2x2 state', res['rom'], 1.0, False),
        (r'RoM$^{1/dt}$', log10_rom_pow, 0.0, True),
    ]
    if rate:
        panels.append((r'Magic rate  $\log_2(\mathrm{RoM})/dt$',
                       res['rom_rate'], 0.0, False))

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.2),
                             constrained_layout=True)
    for ax, (title, data, level, pow10) in zip(np.atleast_1d(axes), panels):
        _panel(ax, fig, x_vals, y_vals, data, title, level, model,
               cbar_pow10=pow10)

    fig.suptitle(f'{model.name}:  {model.title}', fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_png}', flush=True)
    plt.close(fig)


def run_model(name: str, in_dir: Path, out_png: Path | None,
              save_npz: bool, rate: bool) -> None:
    model = MODELS[name]
    res = load_results(in_dir, model)
    if save_npz:
        npz = in_dir / model.name / 'rom_state_summary.npz'
        np.savez(npz, **{k: res[k] for k in list(LOAD_KEYS) + ['p1', 'p2']})
        print(f'Saved {npz}', flush=True)
    png = out_png or Path('results_plots') / f'trotter_rom_state_{name}.png'
    plot_model(res, model, png, rate=rate)


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--model', type=str, choices=list(STATE_ROM_MODELS))
    g.add_argument('--all', action='store_true', help='every model in turn')
    p.add_argument('--in_dir',  type=str, default='results_trotter_rom_state')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_rom_state_<model>.png '
                        '(ignored with --all)')
    p.add_argument('--save_npz', action='store_true',
                   help='also save <in_dir>/<model>/rom_state_summary.npz')
    p.add_argument('--rate', action='store_true',
                   help='append a fifth panel with log2(RoM)/dt on a linear '
                        'axis (= log2 of the RoM^(1/dt) panel)')
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    names = list(STATE_ROM_MODELS) if args.all else [args.model]
    for name in names:
        run_model(name, in_dir,
                  None if args.all or args.out_png is None else Path(args.out_png),
                  args.save_npz, args.rate)


if __name__ == '__main__':
    main()
