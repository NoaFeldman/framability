"""Aggregate per-gate JSON results of rom_of_gate.py into one CSV + table.

Usage:
    python collect_rom.py [--dir results_rom] [--csv results_rom/rom_summary.csv]
"""

import argparse
import csv
import glob
import json
import os

COLUMNS = [
    "task_id",
    "label",
    "spec",
    "n_gate",
    "n_choi",
    "rom",
    "log2_rom",
    "method",
    "solver",
    "K",
    "certified",
    "cert_value",
    "cg_iterations",
    "n_decomp_terms",
    "residual_inf",
    "time_lp_s",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_rom")
    ap.add_argument("--csv", default=None, help="output CSV path "
                    "(default: <dir>/rom_summary.csv)")
    args = ap.parse_args()
    out_csv = args.csv or os.path.join(args.dir, "rom_summary.csv")

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "rom_*.json"))):
        with open(path) as f:
            r = json.load(f)
        rows.append({c: r.get(c) for c in COLUMNS})
    if not rows:
        raise SystemExit(f"no rom_*.json files found in {args.dir!r}")

    rows.sort(key=lambda r: (r["task_id"] is None, r["task_id"], r["label"]))

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # aligned console table
    hdr = ["task", "spec", "n", "RoM", "log2", "method", "cert", "iters", "t_LP[s]"]
    print("  ".join(f"{h:>10}" for h in hdr))
    for r in rows:
        print("  ".join([
            f"{(r['task_id'] if r['task_id'] is not None else '-'):>10}",
            f"{r['spec'][:24]:>24}"[:24].rjust(10),
            f"{r['n_gate']:>10}",
            f"{r['rom']:>10.6f}",
            f"{r['log2_rom']:>10.6f}",
            f"{r['method']:>10}",
            f"{str(r['certified']):>10}",
            f"{(r['cg_iterations'] if r['cg_iterations'] is not None else '-'):>10}",
            f"{r['time_lp_s']:>10.2f}",
        ]))
    print(f"\nwrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
