"""
Collect per-task `.npz` files written by _sweep_depol_gates_worker.py and
produce the composite figure and a single aggregate `.npz`.

Output:
    <out_dir>/depol_sweep.npz
    <out_dir>/depol_sweep.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from _sweep_depol_gates_worker import (
    GATES, P_VALUES, N_FRAMES, N_P,
    build_channel, ext_pauli_framability,
)

FRAME_LABELS = [
    'Pauli frame',
    'Extended-Pauli frame',
    'Optimized (d_ext_single=4)',
    'Optimized (d_ext_single=6)',
    'Optimized (d_ext_single=8)',
]
FRAME_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
# Distinct linestyles so curves that overlap (typically the Pauli frame is
# hidden underneath identical/coincident curves) remain visually separable.
FRAME_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
FRAME_MARKERS = ['o', 's', '^', 'D', 'v']
FRAME_LW      = [2.6, 1.6, 1.6, 1.6, 1.6]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_depol_sweep')
    parser.add_argument('--out_dir', type=str, default='results_depol_sweep')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_g = len(GATES)
    fra = np.full((n_g, N_P, N_FRAMES), np.nan)
    otoc = np.full((n_g, N_P), np.nan)
    stab = np.full((n_g, N_P), np.nan)
    obe = np.full((n_g, N_P), np.nan)

    missing = []
    for g in range(n_g):
        for pi in range(N_P):
            f = in_dir / f'sweep_{g}_{pi:02d}.npz'
            if not f.exists():
                missing.append(str(f))
                continue
            d = np.load(f, allow_pickle=True)
            fra[g, pi, :] = d['framability']
            otoc[g, pi] = d['otoc']
            stab[g, pi] = d['stab']
            obe[g, pi] = d['obe']
    if missing:
        print(f'[warn] {len(missing)} missing files (first few):')
        for m in missing[:5]:
            print('   ', m)

    # ------------------------------------------------------------------
    # Recompute the Extended-Pauli column with the current worker's
    # extended_pauli_D_xy frame (extension in the X-Y plane), in case the
    # per-task files were written with the older X-Z extension.  The
    # other quantities (Pauli, Optimized, OTOC, stab, obe) are reused.
    # ------------------------------------------------------------------
    for ig, gate in enumerate(GATES):
        for pi, p in enumerate(P_VALUES):
            channel = build_channel(gate, float(p))
            fra[ig, pi, 1] = ext_pauli_framability(channel)

    # ------------------------------------------------------------------
    # Enforce monotonicity over d_ext_single: theoretically
    #   Opt(d_ext_single=k) >= Opt(d_ext_single=k')  for k <= k',
    # because any frame at d=k embeds into d=k' by padding with zero
    # columns (a valid frame).  The local optimiser (COBYQA/Powell with
    # 5 restarts) can fail to find that embedding, so we replace each
    # Opt(d=k') by min(Opt(d=k'') for k'' <= k') across k''=4,6,8.
    # The Pauli and Extended-Pauli columns are left untouched.
    # ------------------------------------------------------------------
    fra[:, :, 2:] = np.minimum.accumulate(fra[:, :, 2:], axis=2)

    p_values = np.array(P_VALUES)
    npz_path = out_dir / 'depol_sweep.npz'
    np.savez(npz_path,
             p_values=p_values, gates=np.array(GATES),
             frame_labels=np.array(FRAME_LABELS),
             framability=fra, otoc=otoc,
             channel_stabilizer_purity=stab,
             operator_bond_entropy=obe)
    print(f'[saved] {npz_path}')

    fig, axes = plt.subplots(n_g, 4, figsize=(20, 4.0 * n_g), sharex=True)
    if n_g == 1:
        axes = axes[np.newaxis, :]
    for ig, gate in enumerate(GATES):
        ax = axes[ig, 0]
        for jf, lbl in enumerate(FRAME_LABELS):
            ax.plot(p_values, fra[ig, :, jf],
                    linestyle=FRAME_STYLES[jf],
                    marker=FRAME_MARKERS[jf],
                    linewidth=FRAME_LW[jf],
                    markersize=6,
                    color=FRAME_COLORS[jf], label=lbl)
        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
        ax.set_ylabel('Framability')
        ax.set_title(f'{gate}: framability')
        ax.grid(alpha=0.3)
        if ig == 0:
            ax.legend(fontsize=8)

        axes[ig, 1].plot(p_values, otoc[ig], 'o-', color='tab:cyan')
        axes[ig, 1].set_ylabel('OTOC')
        axes[ig, 1].set_title(f'{gate}: OTOC')
        axes[ig, 1].grid(alpha=0.3)

        axes[ig, 2].plot(p_values, stab[ig], 'o-', color='tab:olive')
        axes[ig, 2].set_ylabel('Channel stabilizer purity')
        axes[ig, 2].set_title(f'{gate}: stabilizer purity')
        axes[ig, 2].grid(alpha=0.3)

        axes[ig, 3].plot(p_values, obe[ig], 'o-', color='tab:brown')
        axes[ig, 3].set_ylabel('Operator bond entropy')
        axes[ig, 3].set_title(f'{gate}: op. bond entropy')
        axes[ig, 3].grid(alpha=0.3)

        for ax_ in axes[ig]:
            if ig == n_g - 1:
                ax_.set_xlabel(r'depolarisation $p$')

    fig.suptitle('Depolarised gates: framability and channel quantities vs. p '
                 '(H, T lifted to 2 qubits as G$\\otimes$I)')
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png_path = out_dir / 'depol_sweep.png'
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    print(f'[saved] {png_path}')


if __name__ == '__main__':
    main()
