"""
Collect per-task minimax-frame results and produce a summary npz + plot.

Reads <in_dir>/minimax_<d>_<pi:02d>.npz for d in D_EXT_SINGLES and pi in 0..N_P-1.

Output:
    <out_dir>/minimax_frame.npz
    <out_dir>/minimax_frame.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from minimax_frame_worker import D_EXT_SINGLES, P_VALUES, N_D, N_P, N_GATES
from sweep_depol_gates_worker import GATES


D_COLORS = ['tab:green', 'tab:red', 'tab:purple']
D_MARKERS = ['^', 'D', 'v']
D_STYLES = ['-.', ':', (0, (3, 1, 1, 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_minimax_frame')
    parser.add_argument('--out_dir', type=str, default='results_minimax_frame')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    worst = np.full((N_D, N_P), np.nan)
    per_gate = np.full((N_D, N_P, N_GATES), np.nan)
    missing = []

    for di, d_ext_single in enumerate(D_EXT_SINGLES):
        for pi in range(N_P):
            f = in_dir / f'minimax_{d_ext_single}_{pi:02d}.npz'
            if not f.exists():
                missing.append(str(f))
                continue
            data = np.load(f, allow_pickle=True)
            worst[di, pi] = data['worst']
            per_gate[di, pi, :] = data['framability']

    if missing:
        print(f'[warn] {len(missing)} missing files (first few):')
        for m in missing[:5]:
            print('   ', m)

    # Monotonicity over d_ext_single: a smaller frame embeds into a larger
    # one (pad with zero columns), so the minimax worst-case should be
    # non-increasing in d_ext_single.  Enforce after-the-fact to clean up
    # local-optimiser noise, ignoring NaN entries.
    worst_mono = worst.copy()
    running = np.full(N_P, np.inf)
    for di in range(N_D):
        mask = ~np.isnan(worst_mono[di])
        running[mask] = np.minimum(running[mask], worst_mono[di, mask])
        worst_mono[di, mask] = running[mask]

    p_values = np.array(P_VALUES)
    npz_path = out_dir / 'minimax_frame.npz'
    np.savez(npz_path,
             p_values=p_values,
             d_ext_singles=np.array(D_EXT_SINGLES),
             gates=np.array(GATES),
             worst=worst, worst_mono=worst_mono,
             framability_per_gate=per_gate)
    print(f'[saved] {npz_path}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    ax = axes[0]
    for di, d_ext_single in enumerate(D_EXT_SINGLES):
        ax.plot(p_values, worst_mono[di], linestyle=D_STYLES[di],
                marker=D_MARKERS[di], color=D_COLORS[di],
                label=fr'$d_{{\rm ext,single}}={d_ext_single}$',
                linewidth=1.6, markersize=6)
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_ylabel(r'$\min_S\,\max_g$ framability')
    ax.set_xlabel(r'depolarisation $p$')
    ax.set_title('Worst-case framability over {H, T, CNOT}')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    for di, d_ext_single in enumerate(D_EXT_SINGLES):
        ax.plot(p_values, worst_mono[di] ** 2, linestyle=D_STYLES[di],
                marker=D_MARKERS[di], color=D_COLORS[di],
                label=fr'$d_{{\rm ext,single}}={d_ext_single}$',
                linewidth=1.6, markersize=6)
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_ylabel(r'$\min_S\,\max_g$ framability$^2$')
    ax.set_xlabel(r'depolarisation $p$')
    ax.set_title('Worst-case framability$^2$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle('Minimax frame optimisation across gates {H, T, CNOT} '
                 'under 2-qubit depolarisation')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png_path = out_dir / 'minimax_frame.png'
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    print(f'[saved] {png_path}')


if __name__ == '__main__':
    main()
