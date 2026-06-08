"""
Collect dissipative-PT worker results into a summary npz and a colormap figure.

Usage:
    python scripts/dissipative_PT_collect.py \\
        --in_dir results_dpt --out_png results_plots/dissipative_PT.png
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
from dissipative_PT import H_LIST, GAMMA_LIST, N_H, N_G

QUANTITIES = [
    ('pauli_fra',   'Pauli framability'),
    ('opt_fra_4',   'Opt framability (d=4)'),
    ('opt_fra_6',   'Opt framability (d=6)'),
    ('sign_init',   'Sign problem (raw)'),
    ('sign_opt',    'Sign problem (opt)'),
    ('chan_stab',   'Channel stab. purity'),
    ('decay_rate',  'Decay rate'),
    ('ss_vn',       'NESS VN entropy'),
    ('mean_mag',    'Mean magnetisation ⟨Z⟩'),
    ('neg_half',    'Half-system negativity'),
    ('max_lpdo',    'Max LPDO bond entropy'),
]
N_Q = len(QUANTITIES)


def load_results(in_dir: Path) -> np.ndarray:
    """Return (N_H, N_G, N_Q) array; NaN where data is missing."""
    arr = np.full((N_H, N_G, N_Q), np.nan)
    n_loaded = 0
    for ih in range(N_H):
        for ig in range(N_G):
            f = in_dir / f'dpt_{ih:02d}_{ig:02d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f)
                for qi, (key, _) in enumerate(QUANTITIES):
                    if key in d:
                        arr[ih, ig, qi] = float(d[key])
                n_loaded += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    print(f'Loaded {n_loaded}/{N_H * N_G} points.', flush=True)
    return arr


def plot_colormaps(arr: np.ndarray, out_png: Path) -> None:
    h_vals     = np.array(H_LIST)
    gamma_vals = np.array(GAMMA_LIST)

    n_cols = 4
    n_rows = (N_Q + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 3.5 * n_rows),
                             constrained_layout=True)
    axes_flat = axes.flat

    for qi, (key, label) in enumerate(QUANTITIES):
        ax = axes_flat[qi]
        data = arr[:, :, qi].T   # (N_G, N_H) — gamma on y, h on x

        # Use pcolormesh with the actual axis values
        h_edges = _edges(h_vals)
        g_edges = _edges(gamma_vals)
        vmin, vmax = _robust_limits(data)
        im = ax.pcolormesh(h_edges, g_edges, data,
                           cmap='viridis', vmin=vmin, vmax=vmax, shading='flat')
        fig.colorbar(im, ax=ax, pad=0.02)

        ax.set_xlabel('h')
        ax.set_ylabel('γ')
        ax.set_title(label, fontsize=10)

        # Add contour at value=1 for framability panels
        if 'fra' in key:
            try:
                ax.contour(h_vals, gamma_vals, data,
                           levels=[1.0], colors='white', linewidths=1.0)
            except Exception:
                pass

    # hide unused panels
    for qi in range(N_Q, n_rows * n_cols):
        axes_flat[qi].set_visible(False)

    fig.suptitle('Dissipative Phase Transition scan  (J=1, 2×2 lattice)', fontsize=13)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f'Saved {out_png}', flush=True)
    plt.close(fig)


def _edges(vals: np.ndarray) -> np.ndarray:
    """Bin edges for pcolormesh from centre values."""
    if len(vals) == 1:
        return np.array([vals[0] - 0.05, vals[0] + 0.05])
    step = np.diff(vals).mean()
    return np.concatenate([[vals[0] - step / 2],
                           (vals[:-1] + vals[1:]) / 2,
                           [vals[-1] + step / 2]])


def _robust_limits(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.nanpercentile(finite, 2)), float(np.nanpercentile(finite, 98))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',  type=str, default='results_dpt')
    p.add_argument('--out_png', type=str, default='results_plots/dissipative_PT.png')
    p.add_argument('--save_npz', action='store_true',
                   help='Also save a summary dpt_scan.npz in --in_dir')
    args = p.parse_args()

    in_dir  = Path(args.in_dir)
    out_png = Path(args.out_png)

    arr = load_results(in_dir)
    if args.save_npz:
        npz_path = in_dir / 'dpt_scan.npz'
        np.savez(npz_path, arr=arr, h=H_LIST, gamma=GAMMA_LIST,
                 quantities=[k for k, _ in QUANTITIES])
        print(f'Saved {npz_path}', flush=True)

    plot_colormaps(arr, out_png)


if __name__ == '__main__':
    main()
