"""
Item 4: collect scripts/eight_qubit_gap_worker.py's per-point npz files into a
(gamma, gamma') grid and plot the Lindbladian gap for each topology (ring,
2x4 lattice) as its own colormap.

Usage:
    python scripts/eight_qubit_gap_collect.py
    python scripts/eight_qubit_gap_collect.py --topologies ring lattice --stride 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eight_qubit_gap_worker import grid_vals, N_QUBITS


def load(topology: str, in_dir: Path, stride: int) -> dict:
    vals = grid_vals(stride)
    n = len(vals)
    gap = np.full((n, n), np.nan)
    found = 0
    pt_dir = in_dir / topology
    for ig in range(n):
        for igp in range(n):
            f = pt_dir / f'pt_{ig:03d}_{igp:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                gap[ig, igp] = float(d['gap'])
                found += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    print(f'[8q-gap] {topology}: {found}/{n * n} grid points loaded', flush=True)
    return dict(gamma_vals=vals, gamma_p_vals=vals, gap=gap)


def plot(data_by_topology: dict, png: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    topologies = list(data_by_topology)
    fig, axes = plt.subplots(1, len(topologies), figsize=(7 * len(topologies), 6),
                             constrained_layout=True, squeeze=False)
    fig.suptitle(fr'{N_QUBITS}-qubit Lindbladian gap  '
                 r'($H=J\sum_{\langle ij\rangle}Z_iZ_j$, '
                 r"jumps $\sqrt{\gamma}|-\rangle\langle+|_i,\ \sqrt{\gamma'}Z_i$)",
                 fontsize=13)

    def edges(v):
        v = np.asarray(v, float)
        mid = (v[:-1] + v[1:]) / 2
        return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])

    for ax, topo in zip(axes.flat, topologies):
        d = data_by_topology[topo]
        gx = edges(d['gamma_vals'])
        gy = edges(d['gamma_p_vals'])
        Z = d['gap'].T
        vmax = float(np.nanmax(Z)) if np.isfinite(Z).any() else 1e-6
        pcm = ax.pcolormesh(gx, gy, Z, cmap='viridis', vmin=0.0, vmax=max(vmax, 1e-9),
                            shading='flat')
        ax.set_title(f'{topo}', fontsize=12)
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r"$\gamma'$")
        fig.colorbar(pcm, ax=ax, label='Lindbladian gap')

    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[8q-gap] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--topologies', nargs='+', default=['ring', 'lattice'],
                    choices=['ring', 'lattice'])
    ap.add_argument('--in_dir', type=str, default='results_8q')
    ap.add_argument('--out_dir', type=str, default='results_8q')
    ap.add_argument('--stride', type=int, default=5)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {}
    for topo in args.topologies:
        d = load(topo, in_dir, args.stride)
        np.savez(out_dir / f'eight_qubit_gap_{topo}.npz', **d)
        data[topo] = d

    plot(data, out_dir / 'eight_qubit_gap.png')


if __name__ == '__main__':
    main()
