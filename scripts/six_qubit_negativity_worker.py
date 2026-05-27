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
    """Compute the steady state via a trace-constrained dense linear solve.

    The Lindbladian L has a one-dimensional kernel containing vec(rho_ss).
    ARPACK shift-invert at sigma=0 is ill-conditioned for singular L (gives
    ~1% errors at large gamma).  Sparse LU on this non-Hermitian operator
    suffers catastrophic fill-in.  We instead replace one row of L with the
    trace constraint vec(I)^T and solve the resulting full-rank system by
    dense LAPACK, which gives a unique vec(rho) satisfying both
    L vec(rho) = 0 (in all rows except the replaced one) and tr(rho) = 1.
    """
    d = HILBERT_DIM
    n = d * d

    # vec(rho)[i + d*j] = rho[i,j] (column-stacking), so trace row vec(I)^T
    # has 1's at indices i + d*i.
    diag_idx = np.arange(d) * (d + 1)

    M = L_comp.toarray().astype(np.complex128, copy=True)
    replace_row = int(diag_idx[0])  # = 0
    M[replace_row, :] = 0.0
    M[replace_row, diag_idx] = 1.0

    rhs = np.zeros(n, dtype=np.complex128)
    rhs[replace_row] = 1.0

    v = np.linalg.solve(M, rhs)

    rho = v.reshape(d, d, order="F")
    rho = (rho + rho.conj().T) / 2.0
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
