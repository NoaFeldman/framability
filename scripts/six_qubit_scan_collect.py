"""
Aggregate per-point results from `six_qubit_scan_worker.py` and produce
`<out_dir>/six_qubit_scan_bond_vs_fra.png` (the 2x3-lattice analogue of
`results/two_qubit_scan_full_bond_vs_fra.png`).

Per-point file layout:
    <out_dir>/six_point_<ig:04d>_<igp:04d>.npy   shape (5,)
        [0] max_negativity     (3|3 row bipartition, evolution)
        [1] max_bond_dim       (LPDO 3|3 row bipartition, evolution)
        [2] max_bond_entropy   (LPDO 3|3 row bipartition, evolution)
        [3] pauli_fra          (2-qubit gate)
        [4] min_fra            (2-qubit gate, optimised)

Output figure shows three panels:
    max negativity  |  max LPDO bond dim  |  optimised framability
all for the bipartition between the two 3-qubit rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm


def main() -> None:
    p = argparse.ArgumentParser(
        description="Collect 6-qubit scan results and plot bond vs framability."
    )
    p.add_argument("--n_pts",      type=int,   default=41,
                   help="Square grid size (used if --n_pts_g/--n_pts_gp not given).")
    p.add_argument("--n_pts_g",    type=int,   default=None)
    p.add_argument("--n_pts_gp",   type=int,   default=None)
    p.add_argument("--J",          type=float, default=1.0)
    p.add_argument("--gamma_step", type=float, default=0.2)
    p.add_argument("--out_dir",    type=str,   default="results_six")
    p.add_argument("--out_name",   type=str,
                   default="six_qubit_scan_bond_vs_fra.png")
    args = p.parse_args()

    n_g     = args.n_pts_g  if args.n_pts_g  is not None else args.n_pts
    n_gp    = args.n_pts_gp if args.n_pts_gp is not None else args.n_pts
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load per-point files ───────────────────────────────────────────
    grid = np.full((n_g, n_gp, 5), np.nan, dtype=float)
    missing = []
    for ig in range(n_g):
        for igp in range(n_gp):
            f = out_dir / f"six_point_{ig:04d}_{igp:04d}.npy"
            if not f.exists():
                missing.append(f)
                continue
            grid[ig, igp] = np.load(f)

    if missing:
        print(f"WARNING: {len(missing)} per-point files missing", file=sys.stderr)
        for f in missing[:10]:
            print(f"  {f}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... ({len(missing)-10} more)", file=sys.stderr)

    np.save(out_dir / "six_qubit_scan.npy", grid)
    print(f"[saved] {out_dir/'six_qubit_scan.npy'}  shape={grid.shape}")

    max_neg   = grid[:, :, 0]
    max_chi   = grid[:, :, 1]
    pauli_fra = grid[:, :, 3]
    opt_fra   = grid[:, :, 4]
    min_fra   = np.minimum(opt_fra, pauli_fra)

    gammas_g  = args.gamma_step * np.arange(n_g)
    gammas_gp = args.gamma_step * np.arange(n_gp)
    half      = args.gamma_step / 2.0
    extent    = [gammas_gp[0] - half, gammas_gp[-1] + half,
                 gammas_g[0]  - half, gammas_g[-1]  + half]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Negativity ───────────────────────────────────────────────────────
    ax = axes[0]
    im0 = ax.imshow(max_neg, origin="lower", extent=extent, aspect="auto",
                    cmap="viridis")
    ax.set_title("Max entanglement negativity\n"
                 "(3|3 row bipartition, 2x3 lattice)")
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    fig.colorbar(im0, ax=ax)

    # ── LPDO bond dimension (discrete colormap) ──────────────────────────
    ax = axes[1]
    chi_vals = max_chi[np.isfinite(max_chi)]
    if chi_vals.size:
        chi_max = int(np.nanmax(max_chi))
        chi_min = int(np.nanmin(max_chi))
    else:
        chi_max, chi_min = 1, 0
    n_levels   = max(chi_max - chi_min + 1, 1)
    boundaries = np.arange(chi_min - 0.5, chi_max + 1.5, 1.0)
    cmap       = cm.get_cmap("viridis", n_levels)
    norm       = BoundaryNorm(boundaries, cmap.N)
    im1 = ax.imshow(max_chi, origin="lower", extent=extent, aspect="auto",
                    cmap=cmap, norm=norm)
    ax.set_title("Max LPDO bond dimension\n"
                 "(3|3 row bipartition, 2x3 lattice)")
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    cb1 = fig.colorbar(im1, ax=ax,
                       ticks=np.arange(chi_min, chi_max + 1, 1))
    cb1.set_label(r"$\chi$")

    # ── Optimised framability (2-qubit operator) ────────────────────────
    ax = axes[2]
    im2 = ax.imshow(min_fra, origin="lower", extent=extent, aspect="auto",
                    cmap="viridis")
    ax.set_title("Optimised framability\n"
                 "(2-qubit operator, min of optimised and Pauli)")
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    fig.colorbar(im2, ax=ax)
    if np.nanmin(min_fra) < 1.0 < np.nanmax(min_fra):
        try:
            ax.contour(min_fra, levels=[1.0], colors="white", linewidths=0.8,
                       extent=extent, origin="lower")
        except Exception:
            pass

    fig.suptitle(
        f"6-qubit (2x3) Lindbladian:  3|3 row bipartition diagnostics  "
        f"vs  2-qubit-operator optimised framability\n"
        f"(J={args.J}, step={args.gamma_step}, grid {n_g}x{n_gp})"
    )
    fig.tight_layout()

    out_png = out_dir / args.out_name
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
