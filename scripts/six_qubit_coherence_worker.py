"""
Worker: compute only the l1-coherence
    C(rho) = sum_{i!=j} |rho_{ij}|
of the 6-qubit steady-state density matrix at one (gamma, gamma') point.

Per-point output:
    <out_dir>/six_coh_<ig:04d>_<igp:04d>.npy   scalar float

The steady state is obtained via the trace-constrained dense linear solve
(the same routine as `six_qubit_negativity_worker._steady_state`); per-point
cost is ~15 s on a workstation core.

Usage
-----
    python six_qubit_coherence_worker.py --task_id 42 \
        --n_pts_g 51 --n_pts_gp 21 --J 1.0 --gamma_step 0.2 \
        --out_dir results_six_coh
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from six_qubit_lindbladian import build_lindbladian_comp, HILBERT_DIM
from six_qubit_negativity_worker import _steady_state


def coherence_l1(rho: np.ndarray) -> float:
    """l1-coherence: sum of |off-diagonal entries| in the computational basis."""
    a = np.abs(rho)
    return float(a.sum() - np.trace(a).real)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute 6-qubit l1-coherence for one (gamma, gamma') point."
    )
    p.add_argument("--task_id",    type=int, required=True)
    p.add_argument("--n_pts_g",    type=int, default=51)
    p.add_argument("--n_pts_gp",   type=int, default=21)
    p.add_argument("--J",          type=float, default=1.0)
    p.add_argument("--gamma_step", type=float, default=0.2)
    p.add_argument("--out_dir",    type=str, default="results_six_coh")
    args = p.parse_args()

    n_g, n_gp = args.n_pts_g, args.n_pts_gp
    tid = args.task_id
    total = n_g * n_gp

    if tid < 0 or tid >= total:
        print(f"ERROR: task_id {tid} out of range [0, {total-1}]", file=sys.stderr)
        sys.exit(1)

    ig = tid // n_gp
    igp = tid % n_gp

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"six_coh_{ig:04d}_{igp:04d}.npy")

    gamma = args.gamma_step * ig
    gp = args.gamma_step * igp

    try:
        print(f"[task {tid}] ig={ig} igp={igp}  gamma={gamma:.2f}  gamma'={gp:.2f}",
              flush=True)
        t0 = time.time()
        L_comp = build_lindbladian_comp(J=args.J, gamma=gamma, gamma_p=gp)
        print(f"  [built] L_comp  {time.time()-t0:.2f}s", flush=True)

        t0 = time.time()
        rho_ss = _steady_state(L_comp)
        print(f"  [ss] {time.time()-t0:.2f}s", flush=True)

        coh = coherence_l1(rho_ss)
        np.save(out_path, coh)
        print(f"  [saved] coh={coh:.6f}  -> {out_path}", flush=True)

    except Exception as e:
        print(f"ERROR in task {tid}: {e}", file=sys.stderr, flush=True)
        np.save(out_path, np.nan)
        sys.exit(1)


if __name__ == "__main__":
    main()
