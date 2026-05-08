"""
Worker for the 6-qubit (2x3 lattice) parameter scan.

For each (gamma, gamma') point this script computes the quantities that
appear in the bond_vs_fra figure for the original 2-qubit problem,
adapted to the 2x3 lattice with bipartition between the two 3-qubit rows
({0,1,2} | {3,4,5}):

    * max entanglement negativity along the trajectory
      (partial transpose over the bottom row)
    * max LPDO bond dimension along the trajectory
    * max LPDO bond entropy   (kept for reference)
    * Pauli framability       (max row 1-norm of exp(dt * L_2q))
    * Optimised framability   (Kronecker / extended-Pauli frame)
      - per the user's instruction, the **operator stays a 2-qubit gate**

Each SLURM array task processes ONE (gamma, gamma') point:
    ig  = task_id // n_gp
    igp = task_id %  n_gp

Output:
    <out_dir>/six_point_<ig:04d>_<igp:04d>.npy   shape (5,)
        [0] max_negativity     (3|3 row bipartition, evolution)
        [1] max_bond_dim       (LPDO 3|3 row bipartition, evolution)
        [2] max_bond_entropy   (LPDO 3|3 row bipartition, evolution)
        [3] pauli_fra          (2-qubit gate)
        [4] min_fra            (2-qubit gate, optimised)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from framability import extended_pauli_D, heisenberg_framability
from optimize_framability import minimize_framability, DEFAULT_METHOD
from lpdo import (purification_sqrt, tensorize_to_lpdo,
                  truncate_and_validate, _bond_entropy, _bures_fidelity)
from six_qubit_lindbladian import (
    HILBERT_DIM, N_QUBITS, LATTICE_SHAPE,
    build_lindbladian_comp, steady_state, initial_yy_state,
)


# ---------------------------------------------------------------------------
# Bipartition between the two rows of the lattice
# ---------------------------------------------------------------------------

def _row_bipartition_dim() -> int:
    """
    Hilbert-space dimension of one row of the lattice.

    With row-major qubit numbering, the top row occupies the first
    `LATTICE_SHAPE[1]` (= cols) tensor factors of the kron product, and the
    bottom row occupies the next `cols` factors.  This is the only
    bipartition supported by the LPDO (and the negativity) helpers below.
    """
    rows, cols = LATTICE_SHAPE
    if rows != 2:
        raise NotImplementedError(
            f"Two-row bipartition only defined for a 2-row lattice; "
            f"got LATTICE_SHAPE={LATTICE_SHAPE}."
        )
    if 2 * cols != N_QUBITS:
        raise RuntimeError(
            f"Inconsistent lattice: rows*cols = {rows*cols} != N_QUBITS = {N_QUBITS}."
        )
    return 2 ** cols


# ---------------------------------------------------------------------------
# 6-qubit max-LPDO-bond-entropy along trajectory to steady state
# ---------------------------------------------------------------------------

def _vec_to_rho(v: np.ndarray) -> np.ndarray:
    """Inverse of column-stacking vec: vec(rho)[i + d*j] = rho[i,j]."""
    rho = v.reshape(HILBERT_DIM, HILBERT_DIM, order="F")
    return (rho + rho.conj().T) / 2.0


def _lpdo_bond_diagnostics(rho: np.ndarray, d_site: int):
    """
    LPDO bond dimension and bond entropy for the two-row bipartition
    (each row = one site of Hilbert dimension d_site = 2**cols).
    Returns (chi, entropy).  Raises on failure.
    """
    if rho.shape != (d_site * d_site, d_site * d_site):
        raise ValueError(
            f"_lpdo_bond_diagnostics: rho has shape {rho.shape}, "
            f"expected ({d_site*d_site}, {d_site*d_site})."
        )
    X = purification_sqrt(rho)
    A1, A2, _ = tensorize_to_lpdo(X, d_site)
    _, _, chi, _, info = truncate_and_validate(rho, A1, A2, d_site)
    return int(chi), float(_bond_entropy(info["singular_values"]))


def _row_negativity(rho: np.ndarray, d_site: int) -> float:
    """
    Entanglement negativity for the two-row bipartition.

    Computed as the sum of |negative eigenvalues| of the partial transpose
    over the bottom-row subsystem.  With row-major qubit numbering the
    matrix index factorises as i = i_top * d_site + i_bot, so reshaping rho
    as (d_site, d_site, d_site, d_site) gives axes
    (i_top, i_bot, j_top, j_bot); transpose([0, 3, 2, 1]) swaps i_bot and
    j_bot, i.e. partial-transposes the bottom subsystem.
    """
    if rho.shape != (d_site * d_site, d_site * d_site):
        raise ValueError(
            f"_row_negativity: rho has shape {rho.shape}, "
            f"expected ({d_site*d_site}, {d_site*d_site})."
        )
    rho_pt = (rho.reshape(d_site, d_site, d_site, d_site)
                  .transpose([0, 3, 2, 1])
                  .reshape(d_site * d_site, d_site * d_site))
    rho_pt = (rho_pt + rho_pt.conj().T) / 2.0
    evals  = np.linalg.eigvalsh(rho_pt)
    return float(np.sum(np.abs(evals[evals < -1e-15])))


def six_qubit_trajectory_diagnostics(L_comp, rho_ss: np.ndarray,
                                     gamma_step: float, *,
                                     max_steps: int = 100_000,
                                     fidelity_threshold: float = 0.9,
                                     verbose: bool = False):
    """
    Maxima along the trajectory  |+y><+y|^{⊗N}  ->  rho_ss  for the
    two-row bipartition of the lattice (top row | bottom row,
    d_site = 2**cols):

        max_negativity, max_bond_dim, max_bond_entropy

    Time evolution is performed in the column-stacking vectorisation of rho
    using sparse `expm_multiply`, with dt = 0.01 * gamma_step.
    """
    d_site = _row_bipartition_dim()
    dt     = 0.01 * gamma_step

    rho0 = initial_yy_state()
    v    = rho0.reshape(-1, order="F").astype(complex)

    max_neg = 0.0
    max_chi = 0
    max_S   = 0.0

    def _update(rho: np.ndarray) -> None:
        nonlocal max_neg, max_chi, max_S
        try:
            max_neg = max(max_neg, _row_negativity(rho, d_site))
        except Exception as exc:
            if verbose:
                print(f"  [warn] negativity: {exc}", flush=True)
        try:
            chi, S = _lpdo_bond_diagnostics(rho, d_site)
            if chi > max_chi:
                max_chi = chi
            if S   > max_S:
                max_S   = S
        except Exception as exc:
            if verbose:
                print(f"  [warn] LPDO: {exc}", flush=True)

    converged = False
    for step in range(max_steps):
        rho = _vec_to_rho(v)
        _update(rho)

        # Convergence check
        try:
            fid = _bures_fidelity(rho, rho_ss)
        except Exception:
            fid = 0.0
        if fid >= fidelity_threshold:
            converged = True
            break

        v_new = expm_multiply(dt * L_comp, v)
        if np.allclose(v, v_new, atol=1e-15):
            break
        v = v_new

    if verbose and not converged:
        print(f"  [info] trajectory did not reach fidelity "
              f"{fidelity_threshold} within {max_steps} steps", flush=True)

    # always include the steady state itself
    _update(rho_ss)

    return float(max_neg), int(max_chi), float(max_S)


# ---------------------------------------------------------------------------
# 2-qubit gate framability (operator unchanged per user instructions)
# ---------------------------------------------------------------------------

def two_qubit_gate_framability(J: float, gamma: float, gamma_p: float,
                               gamma_step: float):
    """
    Pauli- and optimised framability of the **two-qubit** propagator
    exp(dt * L_2q) — independent of the 6-qubit lattice.
    """
    L_2q   = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)
    dt     = 0.01 * gamma_step
    gate   = expm(dt * L_2q).real

    pauli_fra = float(np.max(np.sum(np.abs(gate), axis=1)))

    D_ext        = extended_pauli_D()
    d_ext_single = int(round(np.sqrt(D_ext.shape[1])))
    _, opt_fra = minimize_framability(
        gate, d_ext_single=d_ext_single, n_restarts=5,
        method=DEFAULT_METHOD, max_iter=200, maxfev=1000, verbose=False,
    )
    ext_fra  = heisenberg_framability(D_ext, gate)
    min_fra  = float(min(opt_fra, ext_fra, pauli_fra))
    return pauli_fra, min_fra


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute one (gamma, gamma') point of the 6-qubit (2x3) scan."
    )
    p.add_argument("--task_id",    type=int, required=True,
                   help="Flat grid index: ig = task_id // n_gp, igp = task_id %% n_gp.")
    p.add_argument("--n_pts",      type=int,   default=41,
                   help="Square grid size (used if --n_pts_g/--n_pts_gp not given).")
    p.add_argument("--n_pts_g",    type=int,   default=None,
                   help="Number of gamma  points (rows). Overrides --n_pts.")
    p.add_argument("--n_pts_gp",   type=int,   default=None,
                   help="Number of gamma' points (cols). Overrides --n_pts.")
    p.add_argument("--J",          type=float, default=1.0)
    p.add_argument("--gamma_step", type=float, default=0.2)
    p.add_argument("--out_dir",    type=str,   default="results_six")
    p.add_argument("--max_steps",  type=int,   default=100_000)
    p.add_argument("--fidelity_threshold", type=float, default=0.9)
    args = p.parse_args()

    n_g  = args.n_pts_g  if args.n_pts_g  is not None else args.n_pts
    n_gp = args.n_pts_gp if args.n_pts_gp is not None else args.n_pts
    total = n_g * n_gp
    tid   = args.task_id
    if tid < 0 or tid >= total:
        print(f"ERROR: task_id {tid} out of range [0, {total - 1}]", file=sys.stderr)
        sys.exit(1)

    ig  = tid // n_gp
    igp = tid %  n_gp

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"six_point_{ig:04d}_{igp:04d}.npy"
    if out_path.exists():
        print(f"[task {tid}] reuse {out_path}", flush=True)
        return

    gamma = args.gamma_step * ig
    gp    = args.gamma_step * igp

    print(f"[task {tid}] ig={ig} igp={igp}  gamma={gamma:.4f}  "
          f"gamma'={gp:.4f}", flush=True)

    t0 = time.time()
    L_comp = build_lindbladian_comp(J=args.J, gamma=gamma, gamma_p=gp)
    print(f"[task {tid}] L_comp built  shape={L_comp.shape}  nnz={L_comp.nnz}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    rho_ss = steady_state(L_comp)
    print(f"[task {tid}] steady state found  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    max_neg, max_chi, max_S = six_qubit_trajectory_diagnostics(
        L_comp, rho_ss, args.gamma_step,
        max_steps=args.max_steps,
        fidelity_threshold=args.fidelity_threshold,
    )
    print(f"[task {tid}] 3|3 bipartition:  max negativity = {max_neg:.6f}  "
          f"max LPDO chi = {max_chi}  max LPDO entropy = {max_S:.6f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    pauli_fra, min_fra = two_qubit_gate_framability(
        args.J, gamma, gp, args.gamma_step,
    )
    print(f"[task {tid}] 2q gate framability: pauli={pauli_fra:.6f}  "
          f"min={min_fra:.6f}  ({time.time()-t0:.1f}s)", flush=True)

    np.save(out_path, np.array(
        [max_neg, float(max_chi), max_S, pauli_fra, min_fra], dtype=float))
    print(f"[task {tid}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
