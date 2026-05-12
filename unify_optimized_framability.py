"""
Build a unified "optimized framability" dataset by combining all available
sources into a single grid, taking the elementwise minimum across sources
(the optimized framability is bounded below by 1 in the framable region;
smaller values mean a tighter / better certificate, so min == best).

Sources currently supported:

  * Two-qubit scan : <results_dir>/scan_full.npy
        Column 3  -- "min framability" (optimization output).
        Column 2  -- Pauli framability (used as upper bound on opt fra).
        We use np.minimum(col 3, col 2), matching the convention in
        build_two_qubit_scan_full.plot_full_scan / plot_bond_entropy_vs_framability.

  * Six-qubit (2x3) scan : <results_six_dir>/six_full_scan.npy
        Column 10 -- optimized framability.
        Falls back to assembling per-point `six_full_<ig>_<igp>.npy` files
        when the collected file is absent.

  * (Extensible) other scan_full-like .npy files via --extra
        --extra path,col,col,...
        e.g. --extra results_other/scan_full.npy,3,2

Outputs (default in results/):
  * optimized_framability_unified.npy       -- (n_g, n_gp) float, NaN for no coverage
  * optimized_framability_unified_meta.npz  -- gamma_step, n_g, n_gp, sources

Other modules read this dataset via `unified_framability.load()`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


N_QUANTITIES_6Q = 13
PATTERN_6Q = re.compile(r"^six_full_(\d{4})_(\d{4})\.npy$")


def _autodetect_six_grid(in_dir: Path):
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


def _load_two_qubit(path: Path):
    if not path.is_file():
        return None
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[2] < 4:
        print(f"  [skip] {path}: shape {arr.shape}")
        return None
    fra = np.minimum(arr[:, :, 3], arr[:, :, 2])
    return fra


def _load_six_qubit(in_dir: Path):
    if not in_dir.is_dir():
        return None
    collected = in_dir / "six_full_scan.npy"
    if collected.is_file():
        arr = np.load(collected)
    else:
        det = _autodetect_six_grid(in_dir)
        if det is None:
            return None
        arr = _assemble_six(in_dir, *det)
    if arr.ndim != 3 or arr.shape[2] < 11:
        print(f"  [skip] {in_dir}: assembled shape {arr.shape}")
        return None
    return arr[:, :, 10]


def _load_extra(spec: str):
    """Spec format: 'path,col[,col,...]' -- elementwise min over listed cols."""
    parts = spec.split(",")
    if len(parts) < 2:
        raise ValueError(f"--extra spec '{spec}' needs path,col[,col,...]")
    path = Path(parts[0])
    cols = [int(c) for c in parts[1:]]
    if not path.is_file():
        print(f"  [skip] extra {path} (missing)")
        return None
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[2] <= max(cols):
        print(f"  [skip] extra {path}: shape {arr.shape}, want col(s) {cols}")
        return None
    sub = arr[:, :, cols]
    return np.nanmin(sub, axis=2)


def _combine(sources):
    """sources: list of (name, 2D-array). Returns (combined, n_g, n_gp)."""
    if not sources:
        return None, 0, 0
    n_g = max(a.shape[0] for _, a in sources)
    n_gp = max(a.shape[1] for _, a in sources)
    stack = []
    for name, a in sources:
        pad = np.full((n_g, n_gp), np.nan)
        pad[:a.shape[0], :a.shape[1]] = a
        stack.append(pad)
        print(f"  [src] {name:20s} shape={a.shape} -> padded to ({n_g},{n_gp})")
    stacked = np.stack(stack, axis=0)
    with np.errstate(invalid="ignore"):
        out = np.nanmin(stacked, axis=0)
    out[~np.isfinite(out)] = np.nan
    return out, n_g, n_gp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--two_qubit_file", type=str,
                   default="results/scan_full.npy")
    p.add_argument("--six_qubit_dir",  type=str, default="results_six")
    p.add_argument("--extra", action="append", default=[],
                   help="Additional scan_full-like file: path,col[,col,...]. "
                        "Repeatable.")
    p.add_argument("--out_path", type=str,
                   default="results/optimized_framability_unified.npy")
    p.add_argument("--gamma_step", type=float, default=0.2)
    args = p.parse_args()

    sources = []

    fra2 = _load_two_qubit(Path(args.two_qubit_file))
    if fra2 is not None:
        sources.append(("two_qubit", fra2))

    fra6 = _load_six_qubit(Path(args.six_qubit_dir))
    if fra6 is not None:
        sources.append(("six_qubit_2x3", fra6))

    for spec in args.extra:
        e = _load_extra(spec)
        if e is not None:
            sources.append((f"extra:{spec}", e))

    if not sources:
        print("ERROR: no optimized framability sources found", file=sys.stderr)
        sys.exit(1)

    combined, n_g, n_gp = _combine(sources)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, combined)
    meta_path = out_path.with_name(out_path.stem + "_meta.npz")
    np.savez(meta_path,
             gamma_step=np.float64(args.gamma_step),
             n_g=np.int64(n_g),
             n_gp=np.int64(n_gp),
             sources=np.array([n for n, _ in sources], dtype=object))

    finite = combined[np.isfinite(combined)]
    print(f"[saved] {out_path}  shape=({n_g},{n_gp})  "
          f"coverage={finite.size}/{n_g*n_gp}  "
          f"min={finite.min() if finite.size else float('nan'):.4f}  "
          f"max={finite.max() if finite.size else float('nan'):.4f}")
    print(f"[saved] {meta_path}")


if __name__ == "__main__":
    main()
