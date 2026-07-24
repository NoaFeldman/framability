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
from trotter_rom_4q import ROM_GRIDS

# A panel counts as reaching 1 if its minimum does so to within this tolerance.
ONE_TOL = 1e-6


def _main_index(vals: np.ndarray, v: float) -> int:
    """Index of reduced-grid value `v` in the model's full grid `vals`."""
    idx = np.where(np.isclose(np.asarray(vals, float), v, rtol=0, atol=1e-9))[0]
    assert idx.size == 1, f'value {v} not on the full model grid'
    return int(idx[0])


def load_results(in_dir: Path, model, p1_vals: np.ndarray,
                 p2_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(stab_fra, rom, certified) arrays over the reduced grid (n1, n2); NaN
    where data is missing.  Files are named by full-grid indices."""
    n1, n2 = len(p1_vals), len(p2_vals)
    stab = np.full((n1, n2), np.nan)
    rom = np.full((n1, n2), np.nan)
    cert = np.full((n1, n2), np.nan)
    n_loaded = 0
    mdir = in_dir / model.name
    for irx, p1 in enumerate(p1_vals):
        ix = _main_index(model.p1_vals, float(p1))
        for iry, p2 in enumerate(p2_vals):
            iy = _main_index(model.p2_vals, float(p2))
            f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                stab[irx, iry] = float(d['stab_fra'])
                rom[irx, iry] = float(d['rom'])
                cert[irx, iry] = float(bool(d['rom_certified']))
                n_loaded += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    n_cert = int(np.nansum(cert))
    print(f'Loaded {n_loaded}/{n1 * n2} points for {model.name} '
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
                      p1_vals: np.ndarray, p2_vals: np.ndarray,
                      out_png: Path, log2: bool = False,
                      system: str = '2x2') -> None:
    x_vals, y_vals = np.asarray(p1_vals, float), np.asarray(p2_vals, float)
    x_edges, y_edges = _edges(x_vals), _edges(y_vals)

    gate_name = {'2x2': '4-qubit gate', '2x1': '2-qubit bond gate'}[system]
    rom_plot = np.log2(rom) if log2 else rom
    rom_level = 0.0 if log2 else 1.0
    rom_label = f'log2 RoM of {gate_name}' if log2 else f'RoM of {gate_name}'

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
    p.add_argument('--model',   type=str, required=True,
                   choices=sorted(ROM_GRIDS))
    p.add_argument('--system',  type=str, default='2x2',
                   choices=('2x2', '2x1'))
    p.add_argument('--in_dir',  type=str, default=None,
                   help='default: results_trotter_rom (2x2) / '
                        'results_trotter_rom2 (2x1)')
    p.add_argument('--out_png', type=str, default=None,
                   help='default results_plots/trotter_rom[2]_<model>.png')
    p.add_argument('--log2', action='store_true',
                   help='plot log2(RoM) instead of RoM')
    p.add_argument('--save_npz', action='store_true',
                   help='also save <in_dir>/<model>/rom_summary.npz')
    args = p.parse_args()

    model = MODELS[args.model]
    tag = 'rom' if args.system == '2x2' else 'rom2'
    in_dir = Path(args.in_dir) if args.in_dir else \
        Path('results_trotter_rom' if args.system == '2x2'
             else 'results_trotter_rom2')
    out_png = Path(args.out_png) if args.out_png else \
        Path('results_plots') / f'trotter_{tag}_{model.name}.png'

    p1_vals, p2_vals = ROM_GRIDS[args.model]
    stab, rom, cert = load_results(in_dir, model, p1_vals, p2_vals)
    if args.save_npz:
        npz = in_dir / model.name / 'rom_summary.npz'
        np.savez(npz, stab_fra=stab, rom=rom, certified=cert,
                 p1=p1_vals, p2=p2_vals)
        print(f'Saved {npz}', flush=True)

    plot_side_by_side(stab, rom, model, p1_vals, p2_vals, out_png,
                      log2=args.log2, system=args.system)


if __name__ == '__main__':
    main()
