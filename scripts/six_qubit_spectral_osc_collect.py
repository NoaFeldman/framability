"""
Item 6: collect scripts/six_qubit_spectral_osc_worker.py's per-point npz files
into a (gamma, gamma') grid and plot the four spectral-oscillation summaries
(gap, omega1, Q1, N1) of the 6-qubit ring Lindbladian as colormaps.

Usage:
    python scripts/six_qubit_spectral_osc_collect.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eight_qubit_gap_worker import grid_vals
from six_qubit_spectral_osc_worker import N_QUBITS

MEASURES = [
    ('gap',    'Liouvillian gap $\\Delta$'),
    ('omega1', r'dominant-mode $\omega_1$'),
    ('Q1',     r'dominant-mode $Q_1$'),
    ('N1',     r'dominant-mode $N_1$'),
]


def load(in_dir: Path, stride: int) -> dict:
    vals = grid_vals(stride)
    n = len(vals)
    arrs = {k: np.full((n, n), np.nan) for k, _ in MEASURES}
    found = 0
    for ig in range(n):
        for igp in range(n):
            f = in_dir / f'pt_{ig:03d}_{igp:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                for k, _ in MEASURES:
                    arrs[k][ig, igp] = float(d[k])
                found += 1
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
    print(f'[6q-specosc] {found}/{n * n} grid points loaded', flush=True)
    return dict(gamma_vals=vals, gamma_p_vals=vals, **arrs)


def plot(data: dict, png: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True,
                             squeeze=False)
    fig.suptitle(fr'{N_QUBITS}-qubit ring spectral-oscillation measures  '
                 r'($H=J\sum_{\langle ij\rangle}Z_iZ_j$, '
                 r"jumps $\sqrt\gamma|-\rangle\langle+|_i,\ \sqrt{\gamma'}Z_i$)",
                 fontsize=13)

    def edges(v):
        v = np.asarray(v, float)
        mid = (v[:-1] + v[1:]) / 2
        return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])
    gx = edges(data['gamma_vals'])
    gy = edges(data['gamma_p_vals'])

    for ax, (key, label) in zip(axes.flat, MEASURES):
        Z = data[key].T
        finite = np.isfinite(Z)
        vmax = float(np.nanmax(Z)) if finite.any() else 1.0
        vmin = float(np.nanmin(Z)) if finite.any() else 0.0
        pcm = ax.pcolormesh(gx, gy, Z, cmap='viridis', vmin=vmin, vmax=vmax,
                            shading='flat')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r"$\gamma'$")
        fig.colorbar(pcm, ax=ax)

    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[6q-specosc] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', type=str, default='results_8q/spectral_osc_ring6')
    ap.add_argument('--out_dir', type=str, default='results_8q')
    ap.add_argument('--stride', type=int, default=5)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load(in_dir, args.stride)
    np.savez(out_dir / 'six_qubit_spectral_osc_ring.npz', **data)
    plot(data, out_dir / 'six_qubit_spectral_osc_ring.png')


if __name__ == '__main__':
    main()
