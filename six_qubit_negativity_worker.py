"""
Worker: compute only entanglement negativity (3|3 row bipartition) for 6-qubit Lindbladian.

Per-point output:
    <out_dir>/six_neg_<ig:04d>_<igp:04d>.npy   scalar float
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigs

from six_qubit_lindbladian import (
    build_lindbladian_comp, LATTICE_SHAPE, HILBERT_DIM
)


def _row_bipartition_dim() -> int:
    rows, cols = LATTICE_SHAPE
    if rows != 2 or 2 * cols != 6:
        raise NotImplementedError(f"Two-row bipartition requires 2x3 lattice; got {LATTICE_SHAPE}.")
    return 2 ** cols


def _vec_to_rho(v: np.ndarray) -> np.ndarray:
    rho = v.reshape(HILBERT_DIM, HILBERT_DIM, order="F")
    return (rho + rho.conj().T) / 2.0


def _steady_state(L_comp) -> np.ndarray:
    """Compute steady state via sparse shift-invert eigensolver or dense fallback."""
    try:
        vals, vecs = eigs(L_comp, k=1, sigma=0.0, which="LM",
                          maxiter=5000, tol=1e-10)
        idx = int(np.argmin(np.abs(vals)))
        v = vecs[:, idx]
    except Exception:
        L_dense = L_comp.toarray()
        vals, vecs = np.linalg.eig(L_dense)
        idx = int(np.argmin(np.abs(vals)))
        v = vecs[:, idx]

    rho = _vec_to_rho(v)
    tr = np.trace(rho).real
    if abs(tr) < 1e-12:
        raise RuntimeError("Steady-state has zero trace.")
    rho = rho / tr
    return rho


def _row_negativity(rho: np.ndarray, d_site: int) -> float:
    """Negativity for 3|3 row bipartition."""
    rho_pt = (rho.reshape(d_site, d_site, d_site, d_site)
                  .transpose([0, 3, 2, 1])
                  .reshape(d_site * d_site, d_site * d_site))
    rho_pt = (rho_pt + rho_pt.conj().T) / 2.0
    evals = np.linalg.eigvalsh(rho_pt)
    return float(np.sum(np.abs(evals[evals < -1e-15])))


def main() -> None:
    p = argparse.ArgumentParser(description="Compute 6-qubit negativity for one (gamma, gamma') point.")
    p.add_argument("--task_id",    type=int, required=True)
    p.add_argument("--n_pts_g",    type=int, default=101)
    p.add_argument("--n_pts_gp",   type=int, default=7)
    p.add_argument("--J",          type=float, default=1.0)
    p.add_argument("--gamma_step", type=float, default=0.2)
    p.add_argument("--out_dir",    type=str, default="results_six_neg")
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
    out_path = os.path.join(args.out_dir, f"six_neg_{ig:04d}_{igp:04d}.npy")

    gamma = args.gamma_step * ig
    gp = args.gamma_step * igp

    try:
        print(f"[task {tid}] ig={ig} igp={igp}  gamma={gamma:.2f}  gamma'={gp:.2f}", flush=True)
        t0 = time.time()

        L_comp = build_lindbladian_comp(J=args.J, gamma=gamma, gamma_p=gp)
        print(f"  [built] L_comp  {time.time()-t0:.2f}s", flush=True)

        t0 = time.time()
        rho_ss = _steady_state(L_comp)
        print(f"  [ss] {time.time()-t0:.2f}s", flush=True)

        d_site = _row_bipartition_dim()
        neg = _row_negativity(rho_ss, d_site)

        np.save(out_path, neg)
        print(f"  [saved] neg={neg:.6f}", flush=True)

    except Exception as e:
        print(f"ERROR in task {tid}: {e}", file=sys.stderr, flush=True)
        np.save(out_path, np.nan)
        sys.exit(1)


if __name__ == "__main__":
    main()
