"""
Build the requested two-qubit scan figure with reuse-first logic.

Requested panels
----------------
Row 1:
  a) von Neumann entropy
  b) negativity
  c) max LPDO bond entropy
  d) operator bond entropy
Row 2:
  e) X magnetization
  f) decay rate
  g) small-t OTOC
  h) large-t OTOC
  i) channel stabilizer purity
Row 3:
  j) Pauli state framability
  k) optimized framability
  l) dyadic stabilizer framability
  m) product-state framability (chi=30)

Also supports submitting TWO rounds of neighbor-seeded framability refinement
on SLURM (cluster) before plotting.

Usage
-----
python build_two_qubit_scan_full.py --n_pts 41 --J 1.0 --gamma_step 0.2 --out_dir results

Optional cluster refinement submission:
python build_two_qubit_scan_full.py --submit_neighbor_twice
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian


def run_cmd(cmd, cwd=None):
    print("[run]", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def have_file(path):
    return Path(path).exists()


def all_point_extra_exist(out_dir, n_pts):
    for ig in range(n_pts):
        for igp in range(n_pts):
            f = Path(out_dir) / f"point_extra_{ig:04d}_{igp:04d}.npy"
            if not f.exists():
                return False
    return True


def all_row_files_exist(out_dir, n_pts):
    for ig in range(n_pts):
        f = Path(out_dir) / f"row_{ig:04d}.npy"
        if not f.exists():
            return False
    return True


def ensure_scan_full(args):
    """Ensure out_dir/scan_full.npy exists by reusing or generating prerequisites."""
    scan_full = Path(args.out_dir) / "scan_full.npy"
    if scan_full.exists():
        print(f"[reuse] {scan_full}")
        return

    if not all_row_files_exist(args.out_dir, args.n_pts):
        # Generate missing base rows locally (one row per task_id).
        print("[gen] Missing row_XXXX.npy files -> running scan_worker.py locally.")
        for ig in range(args.n_pts):
            row_file = Path(args.out_dir) / f"row_{ig:04d}.npy"
            if row_file.exists():
                continue
            run_cmd([
                sys.executable,
                "scan_worker.py",
                "--task_id", str(ig),
                "--n_pts", str(args.n_pts),
                "--J", str(args.J),
                "--gamma_step", str(args.gamma_step),
                "--out_dir", args.out_dir,
            ])

    if not all_point_extra_exist(args.out_dir, args.n_pts):
        # Generate missing per-point extras locally.
        print("[gen] Missing point_extra_XXXX_XXXX.npy -> running scan_worker_extra.py locally.")
        total = args.n_pts * args.n_pts
        for tid in range(total):
            ig = tid // args.n_pts
            igp = tid % args.n_pts
            pt = Path(args.out_dir) / f"point_extra_{ig:04d}_{igp:04d}.npy"
            if pt.exists():
                continue
            run_cmd([
                sys.executable,
                "scan_worker_extra.py",
                "--task_id", str(tid),
                "--n_pts", str(args.n_pts),
                "--J", str(args.J),
                "--gamma_step", str(args.gamma_step),
                "--out_dir", args.out_dir,
            ])

    # Merge row + extra (+ optional external columns).
    run_cmd([
        sys.executable,
        "scan_collect.py",
        "--n_pts", str(args.n_pts),
        "--J", str(args.J),
        "--gamma_step", str(args.gamma_step),
        "--out_dir", args.out_dir,
    ])


def submit_neighbor_round(round_index, args, after_job=None):
    """Submit one round of neighbor refinement (array + collect) via sbatch."""
    env = os.environ.copy()
    env.update({
        "N_PTS": str(args.n_pts),
        "J": str(args.J),
        "GAMMA_STEP": str(args.gamma_step),
        "OUT_DIR": str(args.out_dir),
        "N_RESTARTS": str(args.n_restarts),
        "MAXFEV": str(args.maxfev),
    })

    array_end = args.n_pts * args.n_pts - 1

    array_cmd = [
        "sbatch",
        "--parsable",
        f"--array=0-{array_end}%{args.max_concurrent}",
    ]
    if after_job:
        array_cmd.append(f"--dependency=afterok:{after_job}")
    array_cmd.append("neighbor_refine_array.sh")

    print(f"[cluster] round {round_index}: submit array", flush=True)
    array_job = subprocess.check_output(array_cmd, env=env, text=True).strip()
    print(f"[cluster] round {round_index}: array job id = {array_job}")

    collect_cmd = [
        "sbatch",
        "--parsable",
        f"--dependency=afterok:{array_job}",
        "neighbor_refine_collect.sh",
    ]
    print(f"[cluster] round {round_index}: submit collect", flush=True)
    collect_job = subprocess.check_output(collect_cmd, env=env, text=True).strip()
    print(f"[cluster] round {round_index}: collect job id = {collect_job}")

    return collect_job


def maybe_submit_neighbor_twice(args):
    if not args.submit_neighbor_twice:
        return

    if shutil.which("sbatch") is None:
        print("[skip] --submit_neighbor_twice requested but sbatch is not available on this machine.")
        print("       Run this script on the cluster login node to submit the jobs.")
        return

    # Needs scan_full for neighbor_refine_worker.py outlier detection.
    ensure_scan_full(args)

    first_collect = submit_neighbor_round(1, args, after_job=args.after_job)
    second_collect = submit_neighbor_round(2, args, after_job=first_collect)

    print("[cluster] Submitted two refinement rounds.")
    print(f"          round1 collect job: {first_collect}")
    print(f"          round2 collect job: {second_collect}")
    print("          Use: squeue -u $USER")


def ensure_operator_bond_entropy(args):
    f = Path(args.out_dir) / "operator_bond_entropy.npy"
    if f.exists():
        print(f"[reuse] {f}")
        return np.load(f)

    print("[gen] operator_bond_entropy.npy via _plot_operator_bond_entropy.py")
    run_cmd([
        sys.executable,
        "_plot_operator_bond_entropy.py",
        "--n_pts", str(args.n_pts),
        "--J", str(args.J),
        "--gamma_step", str(args.gamma_step),
        "--out_dir", args.out_dir,
    ])
    return np.load(f)


def ensure_otoc_arrays(args):
    out_tmin = Path(args.out_dir) / "otoc_tmin.npy"
    out_tmax = Path(args.out_dir) / "otoc_tmax.npy"

    if out_tmin.exists() and out_tmax.exists():
        print(f"[reuse] {out_tmin} and {out_tmax}")
        return np.load(out_tmin), np.load(out_tmax)

    # Accept pre-existing arrays in workspace root (from prior run).
    root_tmin = Path("otoc_tmin.npy")
    root_tmax = Path("otoc_tmax.npy")
    if root_tmin.exists() and root_tmax.exists():
        print("[reuse] root otoc_tmin.npy/otoc_tmax.npy -> copying into out_dir")
        os.makedirs(args.out_dir, exist_ok=True)
        shutil.copy2(root_tmin, out_tmin)
        shutil.copy2(root_tmax, out_tmax)
        return np.load(out_tmin), np.load(out_tmax)

    print("[gen] OTOC arrays via plot_otoc_lindbladian.py")
    run_cmd([
        sys.executable,
        "plot_otoc_lindbladian.py",
        "--n_qubits", "2",
        "--n_pts", str(args.n_pts),
        "--gamma_step", str(args.gamma_step),
        "--J", str(args.J),
        "--out_prefix", "otoc",
    ])

    # Script writes in workspace root by design; copy into out_dir cache.
    os.makedirs(args.out_dir, exist_ok=True)
    if Path("otoc_tmin.npy").exists() and Path("otoc_tmax.npy").exists():
        shutil.copy2("otoc_tmin.npy", out_tmin)
        shutil.copy2("otoc_tmax.npy", out_tmax)

    return np.load(out_tmin), np.load(out_tmax)


def compute_channel_stabilizer_grid(args):
    n = args.n_pts
    gs = args.gamma_step
    dt = args.dt_stabilizer
    out = Path(args.out_dir) / "channel_stabilizer_purity.npy"

    if out.exists():
        print(f"[reuse] {out}")
        return np.load(out)

    print("[gen] channel_stabilizer_purity.npy from exp(L*dt)")
    grid = np.zeros((n, n), dtype=float)
    for ig in range(n):
        gamma = gs * ig
        for igp in range(n):
            gp = gs * igp
            L = numeric_two_qubit_lindbladian(J=args.J, gamma=gamma, gamma_p=gp)
            E = expm(L * dt)
            diag = np.diag(E).real
            total = (4 ** 2) * np.sum(diag ** 2)  # d=4 for two qubits
            grid[ig, igp] = np.log2(total / (4 + 1))
        if ig % 5 == 0:
            print(f"  row {ig}/{n-1}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(out, grid)
    return grid


def ensure_product_chi30(args, scan_full):
    f = Path(args.out_dir) / "product_fra_schro_chi030.npy"
    if f.exists():
        print(f"[reuse] {f}")
        return np.load(f)

    # Reuse from scan_full if available (column 10 from point_extra pipeline).
    if scan_full.shape[2] > 10 and np.isfinite(scan_full[:, :, 10]).all():
        print("[reuse] extracting chi=30 product framability from scan_full[:, :, 10]")
        arr = scan_full[:, :, 10].copy()
        np.save(f, arr)
        return arr

    # If per-point chi30 files exist, assemble them.
    has_any_point = have_file(Path(args.out_dir) / "prod_schro_pt_0000_0000.npy")
    if has_any_point:
        print("[gen] assembling chi=30 grid via product_schro_collect.py --no_plot")
        run_cmd([
            sys.executable,
            "product_schro_collect.py",
            "--n_pts", str(args.n_pts),
            "--J", str(args.J),
            "--gamma_step", str(args.gamma_step),
            "--out_dir", args.out_dir,
            "--no_plot",
        ])
        return np.load(f)

    raise FileNotFoundError(
        "Missing chi=30 product-state framability. Neither scan_full col 10, "
        "nor product_fra_schro_chi030.npy, nor per-point prod_schro_pt files exist."
    )


def plot_full_scan(args, scan_full, operator_bond, otoc_small, otoc_large, stabilizer, chi30):
    # Column mapping from scan_full (shape expected >= (n,n,12))
    entropy = scan_full[:, :, 0]
    negativity = scan_full[:, :, 1]
    pauli_fra = scan_full[:, :, 2]
    min_fra = scan_full[:, :, 3]
    decay_rate = scan_full[:, :, 4]
    mag_x = scan_full[:, :, 6]
    dyadic_stab = scan_full[:, :, 7]
    max_bond_entropy = scan_full[:, :, 11]

    # Shared colour limits for all framability panels
    fra_arrays = [pauli_fra, min_fra, dyadic_stab, chi30]
    fra_vmin = min(np.nanmin(a) for a in fra_arrays)
    fra_vmax = max(np.nanmax(a) for a in fra_arrays)

    # Entropy/entanglement panels that get a zero-contour (use id() for identity check)
    entropy_panel_ids = {id(entropy), id(negativity), id(max_bond_entropy), id(operator_bond)}

    rows = [
        [
            (entropy, "Von Neumann entropy"),
            (negativity, "Negativity"),
            (max_bond_entropy, "Max LPDO bond entropy"),
            (operator_bond, "Operator bond entropy"),
        ],
        [
            (mag_x, r"$X$ magnetization"),
            (decay_rate, "Decay rate"),
            (otoc_small, r"Small t OTOC: $t=0.1\,\min(\gamma,\gamma')$"),
            (otoc_large, r"Large t OTOC: $t=10\,\max(\gamma,\gamma')$"),
            (stabilizer, r"Channel stabilizer purity $M(e^{L\,dt})$"),
        ],
        [
            (pauli_fra, "Pauli state framability"),
            (min_fra, "Optimized framability"),
            (dyadic_stab, "Dyadic stabilizer framability"),
            (chi30, r"Product-state framability ($\chi=30$)"),
        ],
    ]

    ncols = max(len(r) for r in rows)
    fig, axes = plt.subplots(3, ncols, figsize=(5.8 * ncols, 14))

    gammas = args.gamma_step * np.arange(args.n_pts)
    half = args.gamma_step / 2
    extent = [gammas[0] - half, gammas[-1] + half, gammas[0] - half, gammas[-1] + half]

    for r, row in enumerate(rows):
        for c in range(ncols):
            ax = axes[r, c]
            if c >= len(row):
                ax.set_visible(False)
                continue

            arr, title = row[c]
            is_fra = any(arr is a for a in fra_arrays)

            if is_fra:
                im = ax.imshow(arr, origin="lower", extent=extent, aspect="equal",
                               cmap="viridis", vmin=fra_vmin, vmax=fra_vmax)
                # White contour at framability = 1
                if np.nanmin(arr) < 1.0 < np.nanmax(arr):
                    ax.contour(arr, levels=[1.0], colors="white", linewidths=0.8,
                               extent=extent, origin="lower")
            else:
                im = ax.imshow(arr, origin="lower", extent=extent, aspect="equal", cmap="viridis")
                # White contour at 0 for entropy/entanglement panels
                if id(arr) in entropy_panel_ids:
                    if np.nanmin(arr) < 0.0 < np.nanmax(arr) or np.nanmax(arr) > 1e-10:
                        try:
                            ax.contour(arr, levels=[1e-10], colors="white", linewidths=0.8,
                                       extent=extent, origin="lower")
                        except Exception:
                            pass

            ax.set_title(title)
            ax.set_xlabel(r"$\gamma'$")
            ax.set_ylabel(r"$\gamma$")
            fig.colorbar(im, ax=ax)

    fig.suptitle(f"Two-qubit scan summary (J={args.J}, n_pts={args.n_pts}, step={args.gamma_step})")
    fig.tight_layout()

    out_png = Path(args.out_dir) / args.out_name
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"[saved] {out_png}")


def plot_bond_entropy_vs_framability(args, scan_full):
    """Side-by-side: max LPDO bond entropy | optimized framability."""
    max_bond_entropy = scan_full[:, :, 11]
    min_fra = scan_full[:, :, 3]

    gammas = args.gamma_step * np.arange(args.n_pts)
    half = args.gamma_step / 2
    extent = [gammas[0] - half, gammas[-1] + half, gammas[0] - half, gammas[-1] + half]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Max LPDO bond entropy
    ax = axes[0]
    im0 = ax.imshow(max_bond_entropy, origin="lower", extent=extent, aspect="equal", cmap="viridis")
    ax.set_title("Max LPDO bond entropy")
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    fig.colorbar(im0, ax=ax)
    if np.nanmax(max_bond_entropy) > 1e-10:
        try:
            ax.contour(max_bond_entropy, levels=[1e-10], colors="white", linewidths=0.8,
                       extent=extent, origin="lower")
        except Exception:
            pass

    # Optimized framability
    ax = axes[1]
    im1 = ax.imshow(min_fra, origin="lower", extent=extent, aspect="equal", cmap="viridis")
    ax.set_title("Optimized framability")
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    fig.colorbar(im1, ax=ax)
    if np.nanmin(min_fra) < 1.0 < np.nanmax(min_fra):
        ax.contour(min_fra, levels=[1.0], colors="white", linewidths=0.8,
                   extent=extent, origin="lower")

    fig.suptitle(f"Max LPDO bond entropy vs optimized framability  (J={args.J}, step={args.gamma_step})")
    fig.tight_layout()

    stem = Path(args.out_name).stem
    out_png = Path(args.out_dir) / f"{stem}_bond_vs_fra.png"
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    print(f"[saved] {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Build full two-qubit scan figure with reuse-first data generation.")
    parser.add_argument("--n_pts", type=int, default=41)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--gamma_step", type=float, default=0.2)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--out_name", type=str, default="two_qubit_scan_full.png")
    parser.add_argument("--dt_stabilizer", type=float, default=0.002,
                        help="dt used for channel stabilizer purity M(exp(L*dt)).")

    # Cluster submission knobs
    parser.add_argument("--submit_neighbor_twice", action="store_true",
                        help="Submit two neighbor-refine rounds on SLURM.")
    parser.add_argument("--after_job", type=str, default="",
                        help="Optional dependency job ID for round 1 array job.")
    parser.add_argument("--n_restarts", type=int, default=5)
    parser.add_argument("--maxfev", type=int, default=1000)
    parser.add_argument("--max_concurrent", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    maybe_submit_neighbor_twice(args)

    ensure_scan_full(args)
    scan_full = np.load(Path(args.out_dir) / "scan_full.npy")

    operator_bond = ensure_operator_bond_entropy(args)
    otoc_small, otoc_large = ensure_otoc_arrays(args)
    stabilizer = compute_channel_stabilizer_grid(args)
    chi30 = ensure_product_chi30(args, scan_full)

    plot_full_scan(
        args=args,
        scan_full=scan_full,
        operator_bond=operator_bond,
        otoc_small=otoc_small,
        otoc_large=otoc_large,
        stabilizer=stabilizer,
        chi30=chi30,
    )
    plot_bond_entropy_vs_framability(args, scan_full)


if __name__ == "__main__":
    main()
