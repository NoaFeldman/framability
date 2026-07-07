"""
Collect per-task `.npz` files written by _sweep_depol_gates_worker.py and
produce the composite figure and a single aggregate `.npz`.

Output:
    <out_dir>/depol_sweep.npz
    results_plots/depol_sweep.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- For optimized framability overlay ---
import sys
sys.path.append(str(Path(__file__).parent))
import unified_framability


# --- Patch: Add sqrtT to GATES and update P_VALUES/N_P if needed ---
GATES = ['CNOT', 'H', 'T', 'sqrtT']
P_VALUES = [0.01 * i for i in range(8)]
N_P = len(P_VALUES)

from sweep_depol_gates_worker import (
    build_channel, ext_pauli_framability, ext_pauli_framability_scaled,
    pauli_framability,
)

# Only keep Pauli, Extended-Pauli (a=1), Extended-Pauli scaled.
FRAME_LABELS = [
    'Pauli frame',
    'Extended-Pauli frame (a=1)',
    r'Extended-Pauli frame (a=$\sqrt{1/2}\cdot 0.84$)',
]
FRAME_COLORS = ['tab:blue', 'tab:orange', 'tab:olive']
FRAME_STYLES = ['-', '--', (0, (5, 2))]
FRAME_MARKERS = ['o', 's', 'P']
FRAME_LW      = [2.6, 1.6, 1.6]
N_FRAMES_PLOT = len(FRAME_LABELS)

# --- Optimized framability overlay settings ---
OPT_FRAME_LABELS = [r'Optimized $d_\mathrm{ext}=4$', r'Optimized $d_\mathrm{ext}=6$', r'Optimized $d_\mathrm{ext}=8$']
OPT_FRAME_COLORS = ['tab:red', 'tab:purple', 'tab:green']
OPT_FRAME_MARKERS = ['^', 'v', 'D']
OPT_FRAME_LW = [2.2, 2.2, 2.2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_depol_sweep')
    parser.add_argument('--out_dir', type=str, default='results_depol_sweep')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_g = len(GATES)
    fra = np.full((n_g, N_P, N_FRAMES_PLOT), np.nan)
    obe = np.full((n_g, N_P), np.nan)

    for ig, gate in enumerate(GATES):
        for pi, p in enumerate(P_VALUES):
            channel = build_channel(gate, float(p))
            fra[ig, pi, 0] = pauli_framability(channel)
            fra[ig, pi, 1] = ext_pauli_framability(channel)
            fra[ig, pi, 2] = ext_pauli_framability_scaled(channel)
            # Operator bond entropy from existing per-task files if present.
            f = in_dir / f'sweep_{ig}_{pi:02d}.npz'
            if f.exists():
                d = np.load(f, allow_pickle=True)
                obe[ig, pi] = d['obe']

    p_values = np.array(P_VALUES)
    npz_path = out_dir / 'depol_sweep.npz'
    np.savez(npz_path,
             p_values=p_values, gates=np.array(GATES),
             frame_labels=np.array(FRAME_LABELS),
             framability=fra,
             operator_bond_entropy=obe)
    print(f'[saved] {npz_path}')

    fig, axes = plt.subplots(n_g, 2, figsize=(11, 4.0 * n_g), sharex=True)
    if n_g == 1:
        axes = axes[np.newaxis, :]

    # --- Load optimized framability unified dataset ---
    try:
        opt_fra_arr, gamma_step = unified_framability.load()
    except Exception as e:
        print(f"[warn] Could not load optimized framability: {e}")
        opt_fra_arr = None

    # Map gate index to optimized framability index (if possible)
    # Assume first n_g gates in opt_fra_arr correspond to GATES order

    for ig, gate in enumerate(GATES):
        ax = axes[ig, 0]
        for jf, lbl in enumerate(FRAME_LABELS):
            ax.plot(p_values, fra[ig, :, jf] ** 2,
                    linestyle=FRAME_STYLES[jf],
                    marker=FRAME_MARKERS[jf],
                    linewidth=FRAME_LW[jf],
                    markersize=6,
                    color=FRAME_COLORS[jf], label=lbl)

        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
        ax.set_ylabel(r'Framability$^2$')
        ax.set_title(f'{gate}: framability$^2$')
        ax.grid(alpha=0.3)
        if ig == 0:
            ax.legend(fontsize=8)

        axes[ig, 1].plot(p_values, obe[ig], 'o-', color='tab:brown')
        axes[ig, 1].set_ylabel('Operator bond entropy')
        axes[ig, 1].set_title(f'{gate}: op. bond entropy')
        axes[ig, 1].grid(alpha=0.3)

        for ax_ in axes[ig]:
            if ig == n_g - 1:
                ax_.set_xlabel(r'depolarisation $p$')

    # --- Overlay optimized framability for gate sets on all subplots ---
    try:
        opt_H_CNOT_T = np.load(str(out_dir / 'depol_opt_fra_H_CNOT_T.npy'))
    except Exception as e:
        opt_H_CNOT_T = None
        print(f"[warn] Could not plot optimized [H,CNOT,T]: {e}")
    try:
        opt_H_CNOT_sqrtT = np.load(str(out_dir / 'depol_opt_fra_H_CNOT_sqrtT.npy'))
    except Exception as e:
        opt_H_CNOT_sqrtT = None
        print(f"[warn] Could not plot optimized [H,CNOT,sqrtT]: {e}")

    for ig, gate in enumerate(GATES):
        ax = axes[ig, 0]
        if opt_H_CNOT_T is not None:
            ax.plot(p_values, opt_H_CNOT_T ** 2, '-o', color='tab:red', label='Optimized [H,CNOT,T]', linewidth=2.5, markersize=7)
        if opt_H_CNOT_sqrtT is not None:
            ax.plot(p_values, opt_H_CNOT_sqrtT ** 2, '-s', color='tab:purple', label='Optimized [H,CNOT,$\\sqrt{T}$]', linewidth=2.5, markersize=7)
        if ig == 0:
            ax.legend(fontsize=8)

    fig.suptitle('Depolarised gates: framability and channel quantities vs. p '
                 '(H, T lifted to 2 qubits as G$\\otimes$I)')
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png_path = Path('results_plots') / 'depol_sweep.png'
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    print(f'[saved] {png_path}')


if __name__ == '__main__':
    main()
