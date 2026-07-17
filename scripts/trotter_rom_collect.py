"""
Collect one model's 4-qubit Trotter RoM worker results into a summary npz and
a two-panel colormap figure:

    left  : stabilizer-3 framability of the two-qubit bond Trotter gate
    right : Choi-state RoM of the matching 4-qubit (2x2-lattice) gate
            (or log2(RoM) with --log2)

Both panels draw a thin white contour at 1.0 (framability = 1: framable;
RoM = 1: stabilizer-preserving) and annotate the title with min-1 when the
value never reaches 1 on the grid.

Usage:
    python scripts/trotter_rom_collect.py --model model6 \
        --in_dir results_trotter_rom --out_png results_plots/trotter_rom_model6.png
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

# A panel counts as reaching 1 if its minimum does so to within this tolerance.
ONE_TOL = 1e-6


def load_results(in_dir: Path, model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(stab_fra, rom, certified) arrays of shape (N_X, N_Y); NaN if missing."""
    stab = np.full((model.N_X, model.N_Y), np.nan)
    rom = np.full((model.N_X, model.N_Y), np.nan)
    cert = np.full((model.N_X, model.N_Y), np.nan)
    n_loaded = 0
    mdir = in_dir / model.name
    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                stab[ix, iy] = float(d['stab_fra'])
                rom[ix, iy] = float(d['rom'])
                cert[ix, iy] = float(bool(d['rom_certified']))
                n_loaded += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    n_cert = int(np.nansum(cert))
    print(f'Loaded {n_loaded}/{model.N_TOTAL} points for {model.name} '
          f'({n_cert} certified).', flush=True)
    return stab, rom, cert


def _edges(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 1:
        return np.array([vals[0] - 0.05, vals[0] + 0.05])
    step = np.diff(vals).mean()
    return np.concatenate([[vals[0] - step / 2],
                           (vals[:-1] + vals[1:]) / 2,
                           [vals[-1] + step / 2]])


def plot_side_by_side(stab: np.ndarray, rom: np.ndarray, model,
                      out_png: Path, log2: bool = False) -> None:
    x_vals, y_vals = np.array(model.p1_vals), np.array(model.p2_vals)
    x_edges, y_edges = _edges(x_vals), _edges(y_vals)

    rom_plot = np.log2(rom) if log2 else rom
    rom_level = 0.0 if log2 else 1.0
    rom_label = 'log2 RoM of 4-qubit gate' if log2 else 'RoM of 4-qubit gate'

    panels = [
        ('Stabilizer-3 framability (2-qubit gate)', stab, 1.0),
        (rom_label, rom_plot, rom_level),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, (label, arr, level) in zip(axes, panels):
        data = arr.T                       # (N_Y, N_X): y = p2, x = p1
        finite = data[np.isfinite(data)]
        if finite.size:
            vmin = min(float(finite.min()), level)
            vmax = max(float(finite.max()), level)
            if vmin == vmax:
                vmin, vmax = vmin - 0.05, vmax + 0.05
        else:
            vmin, vmax = level - 0.05, level + 1.0
        im = ax.pcolormesh(x_edges, y_edges, data, cmap='viridis',
                           vmin=vmin, vmax=vmax, shading='flat')
        ax.set_xlabel(model.p1_label)
        ax.set_ylabel(model.p2_label)
        title = label
        if finite.size and float(finite.min()) > level + ONE_TOL:
            title += f'\nmin$-{level:g}$ = {float(finite.min()) - level:.3g}'
        ax.set_title(title, fontsize=10)
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        ticks = sorted(set(np.linspace(vmin, vmax, 5).tolist() + [level]))
        cbar.set_ticks(ticks)
        try:
            ax.contour(x_vals, y_vals, data, levels=[level],
                       colors='white', linewidths=1.0)
        except Exception:
            pass

    fig.suptitle(f'{model.name}:  {model.title}', fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_png}', flush=True)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',   type=str, required=True, choices=list(MODELS))
    p.add_argument('--in_dir',  type=str, default='results_trotter_rom')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_rom_<model>.png')
    p.add_argument('--log2', action='store_true',
                   help='plot log2(RoM) instead of RoM')
    p.add_argument('--save_npz', action='store_true',
                   help='also save <in_dir>/<model>/rom_summary.npz')
    args = p.parse_args()

    model = MODELS[args.model]
    in_dir = Path(args.in_dir)
    out_png = Path(args.out_png) if args.out_png else \
        Path('results_plots') / f'trotter_rom_{model.name}.png'

    stab, rom, cert = load_results(in_dir, model)
    if args.save_npz:
        npz = in_dir / model.name / 'rom_summary.npz'
        np.savez(npz, stab_fra=stab, rom=rom, certified=cert,
                 p1=model.p1_vals, p2=model.p2_vals)
        print(f'Saved {npz}', flush=True)

    plot_side_by_side(stab, rom, model, out_png, log2=args.log2)


if __name__ == '__main__':
    main()
