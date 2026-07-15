"""
Line-plot of the Trotter-scan framability measures along a single grid line.

Fix one scan parameter and sweep the other: given a model and a value for each
of its two scan axes -- with exactly one of the two set to the string 'all' --
this plots every framability quantity (the is_framability entries of
trotter_lindbladian_scan.QUANTITIES) as a function of the 'all' axis, holding the
other axis at the requested (nearest-grid) value.

The data are the per-point worker files written by trotter_scan_worker.py:

    <in_dir>/<model>/pt_<ix:03d>_<iy:03d>.npz        (ix -> p1/x-axis, iy -> p2/y-axis)

Example
-------
model3 has p1 = gamma (x-axis) and p2 = gamma' (y-axis).  To plot the
framabilities as a function of gamma at fixed gamma' = 3:

    plot_framability_line(model='model3', x_val='all', y_val=3)

or from the shell:

    python scripts/trotter_scan_line_plot.py --model model3 --x all --y 3

Usage
-----
    python scripts/trotter_scan_line_plot.py --model <model> --x <val|all> --y <val|all>
        [--in_dir results_trotter_v3] [--out_png <path>]
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
from trotter_lindbladian_scan import MODELS, QUANTITIES

# The framability quantities (key, label) in QUANTITIES order.
FRA_QUANTITIES = [(k, lbl) for k, lbl, _, is_fra in QUANTITIES if is_fra]


def _nearest_index(vals: np.ndarray, target: float) -> int:
    """Index of the grid value closest to `target`; warns on a poor match."""
    vals = np.asarray(vals, dtype=float)
    i = int(np.argmin(np.abs(vals - target)))
    if abs(vals[i] - target) > 1e-6:
        print(f'  warning: no exact grid value {target}; using nearest '
              f'{vals[i]:.4g} (index {i})', flush=True)
    return i


def _parse_axis(v):
    """'all' (any case) -> the sentinel 'all'; otherwise a float."""
    if isinstance(v, str) and v.strip().lower() == 'all':
        return 'all'
    return float(v)


def load_framability_line(model, in_dir: Path, ix_fixed=None, iy_fixed=None):
    """Framability quantities along the varying grid axis.

    Exactly one of ix_fixed / iy_fixed is an int (the held index); the other is
    None (the swept axis).  Returns (sweep_vals, data) where data has shape
    (n_sweep, n_fra); missing points are NaN.
    """
    mdir = in_dir / model.name
    if iy_fixed is None:                 # sweep p2 (iy), hold p1 (ix)
        sweep_vals = np.asarray(model.p2_vals, dtype=float)
        idx_pairs = [(ix_fixed, iy) for iy in range(model.N_Y)]
    else:                                # sweep p1 (ix), hold p2 (iy)
        sweep_vals = np.asarray(model.p1_vals, dtype=float)
        idx_pairs = [(ix, iy_fixed) for ix in range(model.N_X)]

    data = np.full((len(idx_pairs), len(FRA_QUANTITIES)), np.nan)
    n_loaded = 0
    for row, (ix, iy) in enumerate(idx_pairs):
        f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
        if not f.exists():
            continue
        try:
            d = np.load(f)
            for qi, (key, _) in enumerate(FRA_QUANTITIES):
                if key in d:
                    data[row, qi] = float(d[key])
            n_loaded += 1
        except Exception as e:
            print(f'  warning: {f.name}: {e}', flush=True)
    print(f'Loaded {n_loaded}/{len(idx_pairs)} points for {model.name}.',
          flush=True)
    return sweep_vals, data


def plot_framability_line(model: str, x_val, y_val, *,
                          in_dir: str = 'results_trotter_v3',
                          out_png: str | None = None):
    """Plot the framability measures along the 'all' axis at fixed other axis.

    model         : model name (key of trotter_lindbladian_scan.MODELS).
    x_val, y_val  : value on the x-axis (p1) and y-axis (p2); exactly one must be
                    the string 'all' (the swept axis), the other a number (held,
                    snapped to the nearest grid value).
    in_dir        : directory holding <model>/pt_*.npz worker files.
    out_png       : output path; defaults to
                    results_plots/trotter_<model>_line_<axis>_<fixed>.png.
    """
    if model not in MODELS:
        raise ValueError(f'unknown model {model!r}; choose from {list(MODELS)}')
    m = MODELS[model]
    x_val, y_val = _parse_axis(x_val), _parse_axis(y_val)

    if (x_val == 'all') == (y_val == 'all'):
        raise ValueError("exactly one of x_val / y_val must be 'all' "
                         "(the swept axis); the other must be a number")

    in_dir = Path(in_dir)
    if x_val == 'all':                    # sweep p1 (x-axis), hold p2 (y-axis)
        iy = _nearest_index(m.p2_vals, y_val)
        sweep_vals, data = load_framability_line(m, in_dir, iy_fixed=iy)
        sweep_label = m.p1_label
        held_txt = f'{m.p2_name} = {float(m.p2_vals[iy]):g}'
        tag = f'x_all_{m.p2_name}{float(m.p2_vals[iy]):g}'
    else:                                 # sweep p2 (y-axis), hold p1 (x-axis)
        ix = _nearest_index(m.p1_vals, x_val)
        sweep_vals, data = load_framability_line(m, in_dir, ix_fixed=ix)
        sweep_label = m.p2_label
        held_txt = f'{m.p1_name} = {float(m.p1_vals[ix]):g}'
        tag = f'y_all_{m.p1_name}{float(m.p1_vals[ix]):g}'

    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    cmap = plt.get_cmap('tab20')
    for qi, (key, label) in enumerate(FRA_QUANTITIES):
        y = data[:, qi]
        if not np.any(np.isfinite(y)):
            continue
        ax.plot(sweep_vals, y, marker='o', ms=3, lw=1.3,
                color=cmap(qi % 20), label=label)

    # framable threshold: framability = 1
    ax.axhline(1.0, color='0.4', ls='--', lw=1.0, zorder=0)

    ax.set_xlabel(sweep_label)
    ax.set_ylabel('framability')
    ax.set_title(f'{m.name}:  {m.title}\nframability vs {sweep_label}  '
                 f'(held {held_txt})', fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)

    out = Path(out_png) if out_png else \
        Path('results_plots') / f'trotter_{m.name}_line_{tag}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'Saved {out}', flush=True)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model',   type=str, required=True, choices=list(MODELS))
    p.add_argument('--x',       type=str, required=True,
                   help="x-axis (p1) value, or 'all' to sweep it")
    p.add_argument('--y',       type=str, required=True,
                   help="y-axis (p2) value, or 'all' to sweep it")
    p.add_argument('--in_dir',  type=str, default='results_trotter_v3')
    p.add_argument('--out_png', type=str, default=None)
    args = p.parse_args()
    plot_framability_line(args.model, args.x, args.y,
                          in_dir=args.in_dir, out_png=args.out_png)


if __name__ == '__main__':
    main()
