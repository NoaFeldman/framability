"""
Collect per-point outputs from six_qubit_full_worker.py into a single grid.

Aggregates `<in_dir>/six_full_<ig:04d>_<igp:04d>.npy` (each shape (13,))
into a (n_g, n_gp, 13) array; missing points are filled with NaN.

Output:
    <out_dir>/six_full_scan.npy   shape (n_g, n_gp, 13)

Usage
-----
    python six_qubit_full_collect.py --n_pts_g 51 --n_pts_gp 21 \
                                     --in_dir results_six --out_dir results_six
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


N_QUANTITIES = 13
PATTERN = re.compile(r"^six_full_(\d{4})_(\d{4})\.npy$")


def _autodetect_grid(in_dir: Path):
    max_g = -1
    max_gp = -1
    found = False
    for f in in_dir.iterdir():
        m = PATTERN.match(f.name)
        if m is None:
            continue
        found = True
        ig  = int(m.group(1))
        igp = int(m.group(2))
        if ig  > max_g:  max_g  = ig
        if igp > max_gp: max_gp = igp
    if not found:
        return None
    return max_g + 1, max_gp + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_pts_g",  type=int, default=None)
    p.add_argument("--n_pts_gp", type=int, default=None)
    p.add_argument("--in_dir",   type=str, default="results_six")
    p.add_argument("--out_dir",  type=str, default="results_six")
    args = p.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_pts_g is None or args.n_pts_gp is None:
        grid = _autodetect_grid(in_dir)
        if grid is None:
            raise SystemExit(f"No six_full_*.npy files found in {in_dir}.")
        n_g, n_gp = grid
        print(f"[auto] detected grid n_g={n_g}  n_gp={n_gp}")
    else:
        n_g, n_gp = args.n_pts_g, args.n_pts_gp

    out = np.full((n_g, n_gp, N_QUANTITIES), np.nan, dtype=float)
    n_present = 0
    n_missing = 0
    for ig in range(n_g):
        for igp in range(n_gp):
            f = in_dir / f"six_full_{ig:04d}_{igp:04d}.npy"
            if not f.exists():
                n_missing += 1
                continue
            v = np.load(f)
            if v.shape[0] != N_QUANTITIES:
                print(f"[warn] {f} has shape {v.shape}, expected ({N_QUANTITIES},); skipping")
                n_missing += 1
                continue
            out[ig, igp] = v
            n_present += 1

    out_path = out_dir / "six_full_scan.npy"
    np.save(out_path, out)
    print(f"[saved] {out_path}  shape={out.shape}  present={n_present}  missing={n_missing}")


if __name__ == "__main__":
    main()
