"""
Collect results from depol_fra_worker.py and plot framability vs p.

Reads files  <out_dir>/depol_fra_<gate_idx>_<p_idx:02d>.npy
and produces <out_dir>/depol_fra.png.

Usage
-----
    python depol_fra_collect.py [--out_dir results_depol] [--fig_path PATH]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

P_VALUES = [0.05, 0.07, 0.09, 0.11, 0.13]
GATE_LABELS = [
    r'$\Lambda_p \circ H$',
    r'$\Lambda_p \circ T$',
    r'$\Lambda_p^{\otimes 2} \circ \mathrm{CNOT}$',
]
MARKERS = ['o', 's', '^']


def main():
    pa = argparse.ArgumentParser(
        description='Collect depol framability results and produce figure.'
    )
    pa.add_argument('--out_dir',  type=str, default='results_depol',
                    help='Directory containing depol_fra_*.npy files.')
    pa.add_argument('--fig_path', type=str, default=None,
                    help='Output figure path (default: <out_dir>/depol_fra.png).')
    args = pa.parse_args()

    n_gates = len(GATE_LABELS)
    n_p     = len(P_VALUES)

    fra     = np.full((n_gates, n_p), np.nan)
    missing = []

    for g in range(n_gates):
        for pi in range(n_p):
            fpath = os.path.join(args.out_dir, f'depol_fra_{g}_{pi:02d}.npy')
            if os.path.exists(fpath):
                fra[g, pi] = float(np.load(fpath)[0])
            else:
                missing.append(fpath)

    if missing:
        print('WARNING: missing result files:', file=sys.stderr)
        for f in missing:
            print(f'  {f}', file=sys.stderr)

    # ── print table ──────────────────────────────────────────────────────────
    header = f"{'p':>6s}" + ''.join(f'  {lbl:>30s}' for lbl in GATE_LABELS)
    print(header)
    print('-' * len(header))
    for pi, p in enumerate(P_VALUES):
        row = f'{p:6.3f}'
        for g in range(n_gates):
            val = fra[g, pi]
            row += f'  {val:>30.6f}' if not np.isnan(val) else f'  {"NaN":>30s}'
        print(row)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for g in range(n_gates):
        mask = ~np.isnan(fra[g])
        ax.plot(
            np.array(P_VALUES)[mask],
            fra[g][mask],
            marker=MARKERS[g],
            linewidth=1.8,
            markersize=7,
            label=GATE_LABELS[g],
        )

    ax.set_xlabel(r'Depolarising probability $p$', fontsize=12)
    ax.set_ylabel(r'Product-state framability  ($\chi = 30$)', fontsize=12)
    ax.set_title(r'Framability of $\Lambda_p \circ U$ vs noise strength', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_path = args.fig_path or os.path.join(args.out_dir, 'depol_fra.png')
    os.makedirs(os.path.dirname(os.path.abspath(fig_path)), exist_ok=True)
    plt.savefig(fig_path, dpi=150)
    print(f'\nSaved figure to {fig_path}')


if __name__ == '__main__':
    main()
