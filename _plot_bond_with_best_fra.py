"""
Three-panel figure:
    panel 0 = two-qubit  max LPDO bond entropy
    panel 1 = six-qubit  max LPDO bond entropy
    panel 2 = "best" optimized framability across the two cases
              (elementwise minimum on the union grid; NaN outside coverage)

The two- and six-qubit scans share the same gate / Hamiltonian and gamma_step,
but cover different (gamma, gamma') ranges:
    2q:  gamma in [0, (n-1)*step],  gamma' in [0, (n-1)*step]
    6q:  gamma in [0, (n_g-1)*step], gamma' in [0, (n_gp-1)*step]
The combined framability panel uses the union grid; per cell we take the
minimum of whichever values are available (smaller = closer to 1 = "better").

Output: results_six/bond_entropy_with_best_framability.png
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


def _load_two_qubit_bond(path: Path):
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


def _assemble_six(in_dir: Path, n_g: int, n_gp: int):
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


def _load_six_qubit_bond(in_dir: Path):
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


def _extent(n_g, n_gp, step):
    half = step / 2.0
    return [-half, (n_gp - 1) * step + half,
            -half, (n_g - 1) * step + half]


def _combine_min(a2: np.ndarray, a6: np.ndarray):
    """Place 2q and 6q on union grid and take elementwise nanmin."""
    ng = max(a2.shape[0], a6.shape[0])
    ngp = max(a2.shape[1], a6.shape[1])
    pad2 = np.full((ng, ngp), np.nan)
    pad6 = np.full((ng, ngp), np.nan)
    pad2[:a2.shape[0], :a2.shape[1]] = a2
    pad6[:a6.shape[0], :a6.shape[1]] = a6
    stacked = np.stack([pad2, pad6], axis=0)
    with np.errstate(invalid="ignore"):
        out = np.nanmin(stacked, axis=0)
    out[~np.isfinite(out)] = np.nan
    return out, ng, ngp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--two_qubit_file", type=str,
                   default="results/scan_full.npy")
    p.add_argument("--six_qubit_dir",  type=str, default="results_six")
    p.add_argument("--out_path",       type=str,
                   default="results_plots/bond_entropy_with_best_framability.png")
    p.add_argument("--gamma_step",     type=float, default=0.2)
    args = p.parse_args()

    two_path = Path(args.two_qubit_file)
    six_dir = Path(args.six_qubit_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not two_path.is_file():
        print(f"ERROR: {two_path} not found", file=sys.stderr)
        sys.exit(1)
    if not six_dir.is_dir():
        print(f"ERROR: {six_dir} not found", file=sys.stderr)
        sys.exit(1)

    bond_2q = _load_two_qubit_bond(two_path)
    bond_6q = _load_six_qubit_bond(six_dir)

    fra_best, _ = unified_framability.load()
    ng_u, ngp_u = fra_best.shape

    def _safe_lims(*arrs):
        finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrs])
        if finite.size == 0:
            return None, None
        return float(finite.min()), float(finite.max())

    bond_vmin, bond_vmax = _safe_lims(bond_2q, bond_6q)

    ext_2q = _extent(*bond_2q.shape, args.gamma_step)
    ext_6q = _extent(*bond_6q.shape, args.gamma_step)
    ext_u = _extent(ng_u, ngp_u, args.gamma_step)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    panels = [
        (axes[0], bond_2q, ext_2q,
         "Two-qubit: max LPDO bond entropy", bond_vmin, bond_vmax, False),
        (axes[1], bond_6q, ext_6q,
         "Six-qubit (2x3): max LPDO bond entropy", bond_vmin, bond_vmax, False),
        (axes[2], fra_best, ext_u,
         "Best optimized framability (min of 2q & 6q)", None, None, True),
    ]

    for ax, data, extent, title, vmin, vmax, is_fra in panels:
        kw = dict(origin="lower", extent=extent, aspect="auto", cmap="viridis")
        if vmin is not None:
            kw.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(data, **kw)
        ax.set_title(title)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r"$\gamma$")
        finite = data[np.isfinite(data)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            try:
                if is_fra and lo < 1.0 < hi:
                    ax.contour(data, levels=[1.0], colors="white",
                               linewidths=0.8, extent=extent, origin="lower")
                if (not is_fra) and (lo < 0.0 < hi or hi > 1e-10):
                    ax.contour(data, levels=[1e-10], colors="white",
                               linewidths=0.8, extent=extent, origin="lower")
            except Exception:
                pass
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
