"""
Plot the minimax framability across (CNOT, H, T) gates as a function of the
depolarisation rate p, one line per d_ext_single in {4, 6, 8}.

Reads results_minimax_H_CNOT_T/minimax_<d_ext>_<idx>.npz where p = 0.01*idx.

Outputs results_plots/minimax_H_CNOT_T.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


D_EXT_LIST = [4, 6, 8]


def _load(in_dir: Path):
    by_dext = {}
    for d_ext in D_EXT_LIST:
        ps, worst = [], []
        for f in sorted(in_dir.glob(f'minimax_{d_ext}_*.npz')):
            d = np.load(f)
            ps.append(float(d['p']))
            worst.append(float(d['worst']))
        order = np.argsort(ps)
        by_dext[d_ext] = (np.asarray(ps)[order], np.asarray(worst)[order])
    return by_dext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='results_minimax_H_CNOT_T')
    parser.add_argument('--out',    default='results_plots/minimax_H_CNOT_T.png')
    args = parser.parse_args()

    data = _load(Path(args.in_dir))

    fig, ax = plt.subplots(figsize=(7, 5))
    for d_ext, (ps, worst) in data.items():
        if len(ps) == 0:
            continue
        ax.plot(ps, worst, marker='o',
                label=fr'$d_\mathrm{{ext}}={d_ext}$  ({len(ps)} pts)')

    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.0,
               label='framability = 1')
    ax.set_xlabel(r'depolarisation rate $p$')
    ax.set_ylabel(r'minimax framability  $\max_g \mathrm{fra}(g)$')
    ax.set_title(r'Minimax framability over $\{\mathrm{CNOT}, H, T\}$ '
                 r'vs depolarisation rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f'Saved {out_path}')

    for d_ext, (ps, worst) in data.items():
        if len(ps):
            print(f'  d_ext={d_ext}: n={len(ps)}, p in [{ps.min():.2f}, {ps.max():.2f}], '
                  f'worst in [{worst.min():.4f}, {worst.max():.4f}]')


if __name__ == '__main__':
    main()
