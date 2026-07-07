"""
Plot the 6-qubit (2x3) comprehensive scan in a 3x4 layout mirroring
results/two_qubit_scan_full.png.

Reads `<in_dir>/six_full_scan.npy` of shape (n_g, n_gp, 13) produced by
six_qubit_full_collect.py.  If this file is absent, falls back to
per-point `<in_dir>/six_full_<ig>_<igp>.npy` files.

Quantity layout (matching `plot_full_scan` in build_two_qubit_scan_full.py):

    Row 1  Von Neumann entropy, Negativity, Max LPDO bond entropy,
           Operator bond entropy
    Row 2  X magnetization, Decay rate, Small-t OTOC, Large-t OTOC,
           Channel stabilizer purity
    Row 3  Pauli framability, Optimized framability,
           Dyadic stabilizer framability, Product-state framability (chi=30)

Framability panels share a colour scale and gain a white contour at fra=1.
Entropy/entanglement panels gain a white contour near 0.

Output: <out_dir>/six_qubit_scan_full.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import unified_framability


N_QUANTITIES = 13
PATTERN = re.compile(r"^six_full_(\d{4})_(\d{4})\.npy$")


def _autodetect_grid(in_dir: Path):
    max_g = -1
    max_gp = -1
    for f in in_dir.iterdir():
        m = PATTERN.match(f.name)
        if m is None:
            continue
        ig = int(m.group(1))
        igp = int(m.group(2))
        if ig > max_g:
            max_g = ig
        if igp > max_gp:
            max_gp = igp
    if max_g < 0:
        return None
    return max_g + 1, max_gp + 1


def _assemble_from_points(in_dir: Path, n_g: int, n_gp: int):
    arr = np.full((n_g, n_gp, N_QUANTITIES), np.nan, dtype=float)
    n_loaded = 0
    for ig in range(n_g):
        for igp in range(n_gp):
            f = in_dir / f"six_full_{ig:04d}_{igp:04d}.npy"
            if not f.exists():
                continue
            try:
                v = np.load(f)
                if v.shape != (N_QUANTITIES,):
                    continue
                arr[ig, igp] = v
                n_loaded += 1
            except Exception:
                continue
    return arr, n_loaded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir",     type=str, default="results_six")
    p.add_argument("--out_dir",    type=str, default="results_plots")
    p.add_argument("--out_name",   type=str, default="six_qubit_scan_full.png")
    p.add_argument("--n_pts_g",    type=int, default=None)
    p.add_argument("--n_pts_gp",   type=int, default=None)
    p.add_argument("--gamma_step", type=float, default=0.2)
    p.add_argument("--J",          type=float, default=1.0)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.is_dir():
        print(f"ERROR: {in_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    collected = in_dir / "six_full_scan.npy"
    if collected.exists():
        arr = np.load(collected)
        if arr.ndim != 3 or arr.shape[2] < N_QUANTITIES:
            print(f"ERROR: {collected} has shape {arr.shape}; expected last axis "
                  f">= {N_QUANTITIES}", file=sys.stderr)
            sys.exit(1)
        print(f"[load] {collected}  shape={arr.shape}")
    else:
        n_g = args.n_pts_g
        n_gp = args.n_pts_gp
        if n_g is None or n_gp is None:
            det = _autodetect_grid(in_dir)
            if det is None:
                print(f"ERROR: no six_full_*.npy or six_full_scan.npy in "
                      f"{in_dir}", file=sys.stderr)
                sys.exit(1)
            d_g, d_gp = det
            if n_g is None:
                n_g = d_g
            if n_gp is None:
                n_gp = d_gp
        arr, n_loaded = _assemble_from_points(in_dir, n_g, n_gp)
        print(f"[load] assembled from per-point files: "
              f"{n_loaded}/{n_g*n_gp} present")

    n_g, n_gp = arr.shape[0], arr.shape[1]

    ss_vn         = arr[:, :, 0]
    ss_neg        = arr[:, :, 1]
    # Prefer the larger standalone negativity scan when available
    # (six_qubit_negativity_collect.py output).  Falls back to col 1 of
    # six_full_scan otherwise.
    ss_neg_extent = None  # use global extent unless overridden
    for cand in [Path("results/six_negativity_101x7.npy"),
                 Path("results_six/six_negativity_101x7.npy")]:
        if cand.is_file():
            try:
                ss_neg_alt = np.load(cand)
                if ss_neg_alt.ndim == 2:
                    ss_neg = ss_neg_alt
                    ng_n, ngp_n = ss_neg.shape
                    half_n = args.gamma_step / 2.0
                    ss_neg_extent = [
                        -half_n, (ngp_n - 1) * args.gamma_step + half_n,
                        -half_n, (ng_n - 1) * args.gamma_step + half_n,
                    ]
                    print(f"[negativity] using {cand}  shape={ss_neg.shape}")
                    break
            except Exception as e:
                print(f"[warn] could not load {cand}: {e}")
    max_bond_S    = arr[:, :, 2]
    operator_bond = arr[:, :, 3]
    mag_x         = arr[:, :, 4]
    decay         = arr[:, :, 5]
    otoc_small    = arr[:, :, 6]
    otoc_large    = arr[:, :, 7]
    chan_stab     = arr[:, :, 8]
    pauli_fra     = arr[:, :, 9]
    # Optimized framability sourced from unified dataset (single source of
    # truth across all scan sizes).  Falls back to the per-scan column 10 if
    # the unified file has not been built yet.
    try:
        fra_unified, _ = unified_framability.load()
        min_fra = unified_framability.crop(fra_unified, n_g, n_gp)
        print(f"[unified] optimized framability sourced from "
              f"{unified_framability.DEFAULT_PATH}")
    except (FileNotFoundError, ValueError) as e:
        print(f"[warn] {e}\n[fallback] using six_full_scan column 10")
        min_fra = arr[:, :, 10]
    dyadic_stab   = arr[:, :, 11]
    chi30         = arr[:, :, 12]

    # l1-coherence panel (optional). Use the standalone scan if available;
    # otherwise no coherence panel is added.
    coherence = None
    coherence_extent = None
    coh_candidates = [
        Path("results_six") / f"six_coherence_{n_g}x{n_gp}.npy",
        Path("results_six_coh") / f"six_coherence_{n_g}x{n_gp}.npy",
        Path("results") / f"six_coherence_{n_g}x{n_gp}.npy",
    ]
    # Also accept any size and crop / use its own extent.
    extra = [p for p in Path("results_six_coh").glob("six_coherence_*x*.npy")] \
        if Path("results_six_coh").is_dir() else []
    extra += [p for p in Path("results_six").glob("six_coherence_*x*.npy")] \
        if Path("results_six").is_dir() else []
    for cand in coh_candidates + extra:
        if cand.is_file():
            try:
                ca = np.load(cand)
                if ca.ndim != 2:
                    continue
                coherence = ca
                ng_c, ngp_c = ca.shape
                half_c = args.gamma_step / 2.0
                coherence_extent = [
                    -half_c, (ngp_c - 1) * args.gamma_step + half_c,
                    -half_c, (ng_c - 1) * args.gamma_step + half_c,
                ]
                print(f"[coherence] using {cand}  shape={ca.shape}")
                break
            except Exception as e:
                print(f"[warn] could not load {cand}: {e}")

    fra_arrays = [pauli_fra, min_fra, dyadic_stab, chi30]
    finite_vals = [a[np.isfinite(a)] for a in fra_arrays]
    finite_vals = [v for v in finite_vals if v.size > 0]
    if finite_vals:
        fra_vmin = float(min(v.min() for v in finite_vals))
        fra_vmax = float(max(v.max() for v in finite_vals))
    else:
        fra_vmin, fra_vmax = 0.0, 1.0

    entropy_panel_ids = {id(ss_vn), id(ss_neg), id(max_bond_S),
                         id(operator_bond)}
    if coherence is not None:
        entropy_panel_ids.add(id(coherence))

    top_row = [
        (ss_vn,         "Von Neumann entropy", None),
        (ss_neg,        "Negativity (3|3 row)", ss_neg_extent),
        (max_bond_S,    "Max LPDO bond entropy (trajectory)", None),
        (operator_bond, "Operator bond entropy", None),
    ]
    if coherence is not None:
        top_row.append(
            (coherence,
             r"$\ell_1$-coherence $\sum_{i\neq j}|\rho_{ij}|$",
             coherence_extent),
        )

    rows = [
        top_row,
        [
            (mag_x,      r"$X$ magnetization", None),
            (decay,      "Decay rate (2q)", None),
            (otoc_small, r"Small $t$ OTOC: $t=0.1\,\min(\gamma,\gamma')$", None),
            (otoc_large, r"Large $t$ OTOC: $t=10\,\max(\gamma,\gamma')$", None),
            (chan_stab,  r"Channel stabilizer purity $M(e^{L\,dt})$", None),
        ],
        [
            (pauli_fra,   "Pauli state framability", None),
            (min_fra,     "Optimized framability", None),
            (dyadic_stab, "Dyadic stabilizer framability", None),
            (chi30,       r"Product-state framability ($\chi=30$)", None),
        ],
    ]

    ncols = max(len(r) for r in rows)
    panel_h = 14 / 3
    fig, axes = plt.subplots(3, ncols, figsize=(panel_h * ncols, 14))

    gammas_y = args.gamma_step * np.arange(n_g)
    gammas_x = args.gamma_step * np.arange(n_gp)
    half = args.gamma_step / 2.0
    extent = [gammas_x[0] - half, gammas_x[-1] + half,
              gammas_y[0] - half, gammas_y[-1] + half]

    def _safe_minmax(a):
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return None, None
        return float(finite.min()), float(finite.max())

    for r, row in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(row):
                ax.set_visible(False)
                continue

            data, title, ext_override = row[c]
            ext_use = ext_override if ext_override is not None else extent
            is_fra = any(data is a for a in fra_arrays)

            if is_fra:
                im = ax.imshow(data, origin="lower", extent=ext_use,
                               aspect="auto", cmap="viridis",
                               vmin=fra_vmin, vmax=fra_vmax)
                lo, hi = _safe_minmax(data)
                if lo is not None and lo < 1.0 < hi:
                    try:
                        ax.contour(data, levels=[1.0], colors="white",
                                   linewidths=0.8, extent=ext_use,
                                   origin="lower")
                    except Exception:
                        pass
            else:
                im = ax.imshow(data, origin="lower", extent=ext_use,
                               aspect="auto", cmap="viridis")
                if id(data) in entropy_panel_ids:
                    lo, hi = _safe_minmax(data)
                    if lo is not None and (lo < 0.0 < hi or hi > 1e-10):
                        try:
                            ax.contour(data, levels=[1e-10], colors="white",
                                       linewidths=0.8, extent=ext_use,
                                       origin="lower")
                        except Exception:
                            pass

            ax.set_title(title)
            ax.set_xlabel(r"$\gamma'$")
            ax.set_ylabel(r"$\gamma$")
            fig.colorbar(im, ax=ax)

    fig.suptitle(f"Six-qubit (2x3) scan summary  "
                 f"(J={args.J}, n_g={n_g}, n_gp={n_gp}, "
                 f"step={args.gamma_step})")
    fig.tight_layout()

    out_png = out_dir / args.out_name
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
