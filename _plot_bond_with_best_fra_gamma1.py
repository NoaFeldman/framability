"""
1-D slice at gamma=1 of the three quantities shown in
results_six/bond_entropy_with_best_framability.png:

  - Two-qubit max LPDO bond entropy
  - Six-qubit (2x3) max LPDO bond entropy
  - Best optimized framability (min of 2q & 6q on the union grid)

All three are plotted as a function of gamma'.

Output: results_six/bond_entropy_with_best_framability_gamma1.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import unified_framability


N_QUANTITIES_6Q = 13
PATTERN_6Q = re.compile(r"^six_full_(\d{4})_(\d{4})\.npy$")


def _load_two_qubit_bond(path: Path) -> np.ndarray:
    return np.load(path)[:, :, 11]


def _autodetect_grid(in_dir: Path):
    max_g = max_gp = -1
    for f in in_dir.iterdir():
        m = PATTERN_6Q.match(f.name)
        if m is None:
            continue
        max_g = max(max_g, int(m.group(1)))
        max_gp = max(max_gp, int(m.group(2)))
    if max_g < 0:
        return None
    return max_g + 1, max_gp + 1


def _assemble_six(in_dir: Path, n_g: int, n_gp: int) -> np.ndarray:
    arr = np.full((n_g, n_gp, N_QUANTITIES_6Q), np.nan, dtype=float)
    for ig in range(n_g):
        for igp in range(n_gp):
            f = in_dir / f"six_full_{ig:04d}_{igp:04d}.npy"
            if not f.exists():
                continue
            try:
                v = np.load(f)
                if v.shape == (N_QUANTITIES_6Q,):
                    arr[ig, igp] = v
            except Exception:
                pass
    return arr


def _load_six_qubit_bond(in_dir: Path) -> np.ndarray:
    collected = in_dir / "six_full_scan.npy"
    if collected.exists():
        arr = np.load(collected)
    else:
        det = _autodetect_grid(in_dir)
        if det is None:
            print(f"ERROR: no six-qubit data in {in_dir}", file=sys.stderr)
            sys.exit(1)
        arr = _assemble_six(in_dir, *det)
    return arr[:, :, 2]


def _row_at_gamma(arr: np.ndarray, gamma: float, step: float):
    ig = int(round(gamma / step))
    if ig < 0 or ig >= arr.shape[0]:
        return None, None
    n_gp = arr.shape[1]
    gammas_p = step * np.arange(n_gp)
    return gammas_p, arr[ig]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--two_qubit_file", type=str,
                   default="results/scan_full.npy")
    p.add_argument("--six_qubit_dir",  type=str, default="results_six")
    p.add_argument("--out_path",       type=str,
        default="results_plots/bond_entropy_with_best_framability_gamma1.png")
    p.add_argument("--gamma",          type=float, default=1.0)
    p.add_argument("--gamma_step",     type=float, default=0.2)
    args = p.parse_args()

    two_path = Path(args.two_qubit_file)
    six_dir = Path(args.six_qubit_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bond_2q = _load_two_qubit_bond(two_path)
    bond_6q = _load_six_qubit_bond(six_dir)
    fra_best, _ = unified_framability.load()

    g2_x, g2_y = _row_at_gamma(bond_2q, args.gamma, args.gamma_step)
    g6_x, g6_y = _row_at_gamma(bond_6q, args.gamma, args.gamma_step)
    gu_x, gu_y = _row_at_gamma(fra_best, args.gamma, args.gamma_step)

    if g2_y is None and g6_y is None and gu_y is None:
        print(f"ERROR: gamma={args.gamma} out of range in all arrays",
              file=sys.stderr)
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    ax = axes[0]
    if g2_y is not None:
        ax.plot(g2_x, g2_y, "o-", label="Two-qubit", color="tab:blue")
    if g6_y is not None:
        ax.plot(g6_x, g6_y, "s-", label="Six-qubit (2x3)", color="tab:orange")
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel("Max LPDO bond entropy")
    ax.set_title("Max LPDO bond entropy")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    if gu_y is not None:
        ax.plot(gu_x, gu_y, "d-", color="tab:green",
                label="Best optimized framability")
        ax.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel("Framability")
    ax.set_title("Best optimized framability (min of 2q & 6q)")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle(rf"Slice at $\gamma = {args.gamma:g}$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
