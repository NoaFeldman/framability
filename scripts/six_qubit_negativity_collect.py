"""
Collect six-qubit negativity results and generate plot.

Input per-point files:
    <out_dir>/six_neg_<ig:04d>_<igp:04d>.npy   scalar float

Output:
    <out_dir>/six_negativity_<n_g>x<n_gp>.npy
    results_plots/six_negativity_<n_g>x<n_gp>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description="Collect 6-qubit negativity and plot.")
    p.add_argument("--n_pts_g",    type=int, default=101)
    p.add_argument("--n_pts_gp",   type=int, default=7)
    p.add_argument("--J",          type=float, default=1.0)
    p.add_argument("--gamma_step", type=float, default=0.2)
    p.add_argument("--out_dir",    type=str, default="results_six_neg")
    args = p.parse_args()

    n_g, n_gp = args.n_pts_g, args.n_pts_gp
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-point files
    grid = np.full((n_g, n_gp), np.nan, dtype=float)
    missing = 0
    for ig in range(n_g):
        for igp in range(n_gp):
            f = out_dir / f"six_neg_{ig:04d}_{igp:04d}.npy"
            if f.exists():
                try:
                    grid[ig, igp] = float(np.load(f))
                except Exception as e:
                    print(f"  [skip] {f}: {e}", file=sys.stderr)
                    missing += 1
            else:
                missing += 1

    if missing:
        print(f"WARNING: {missing}/{n_g*n_gp} files missing or corrupted", file=sys.stderr)

    # Save aggregated grid
    out_npy = out_dir / f"six_negativity_{n_g}x{n_gp}.npy"
    np.save(out_npy, grid)
    print(f"[saved] {out_npy}  shape={grid.shape}  coverage={np.sum(np.isfinite(grid))}/{n_g*n_gp}")

    # Plot
    gammas_g = args.gamma_step * np.arange(n_g)
    gammas_gp = args.gamma_step * np.arange(n_gp)
    half = args.gamma_step / 2.0
    extent = [gammas_gp[0] - half, gammas_gp[-1] + half,
              gammas_g[0] - half, gammas_g[-1] + half]

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    im = ax.imshow(grid, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    ax.set_title(f"6-qubit (2x3) Entanglement Negativity\n(3|3 row bipartition, J={args.J})")
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    fig.colorbar(im, ax=ax, label="Negativity")
    fig.tight_layout()

    out_png = Path("results_plots") / f"six_negativity_{n_g}x{n_gp}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
