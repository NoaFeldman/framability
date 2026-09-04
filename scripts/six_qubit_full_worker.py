"""
Comprehensive worker for the 6-qubit (2x3 lattice) parameter scan.

For each (gamma, gamma') point this script computes every quantity that
appears in `results/two_qubit_scan_full.png`, adapted as follows:

    - *State-level* quantities (entropy, negativity, mag_x, max bond
      entropy along the trajectory) are computed from the 6-qubit
      Lindbladian / steady state.
    - *Operator-level* quantities (decay rate, OTOC, operator bond
      entropy, channel stabilizer purity, framability variants) are
      computed from the 2-qubit Lindbladian, per the user's instruction
      that "the operator stays a 2-qubit operator".

Output:
    <out_dir>/six_full_<ig:04d>_<igp:04d>.npy   shape (13,) float

        [ 0] ss_vn_entropy        Von Neumann entropy of rho_ss (6q)
        [ 1] ss_negativity        Negativity of rho_ss (3|3 row bipartition)
        [ 2] max_bond_entropy     Max LPDO bond entropy along trajectory (6q)
        [ 3] operator_bond        Operator bond entropy of exp(dt*L_2q)
        [ 4] mag_x                <X_1+...+X_6> in rho_ss
        [ 5] decay_rate           Liouvillian gap of L_2q (analysis.decay_rate)
        [ 6] otoc_small           OTOC at t = 0.1 * min(gamma, gamma') (2q)
        [ 7] otoc_large           OTOC at t = 10  * max(gamma, gamma') (2q)
        [ 8] channel_stab_purity  log2( d^2 sum_i E_ii^2 / (d+1) ) (2q)
        [ 9] pauli_fra            Pauli framability (2q)
        [10] min_fra              min(opt, ext-Pauli, Pauli) framability
        [11] dyadic_stab_fra      Dyadic-stabilizer framability (2q)
        [12] product_fra_chi30    Product-state framability, chi=30 (2q)

Usage
-----
    python six_qubit_full_worker.py --task_id 42 --n_pts_g 51 --n_pts_gp 21 \
                                    --J 1.0 --gamma_step 0.2 --out_dir results_six
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import decay_rate
from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from framability import (
    extended_pauli_D, heisenberg_framability, dyadic_stabilizer_framability,
    make_product_state_D,
)
from optimize_framability import minimize_framability, DEFAULT_METHOD
from lpdo import (purification_sqrt, tensorize_to_lpdo,
                  truncate_and_validate, _bond_entropy, _bures_fidelity)
from six_qubit_lindbladian import (
    HILBERT_DIM, N_QUBITS, LATTICE_SHAPE,
    build_lindbladian_comp, steady_state, initial_yy_state,
)


# ---------------------------------------------------------------------------
# Frozen product-state frame for chi=30 (matches scan_worker_extra.py)
# ---------------------------------------------------------------------------

PRODUCT_CHI = 30
np.random.seed(42)
PRODUCT_D = make_product_state_D(PRODUCT_CHI)


# ---------------------------------------------------------------------------
# 6-qubit helpers
# ---------------------------------------------------------------------------

def _row_bipartition_dim() -> int:
    rows, cols = LATTICE_SHAPE
    if rows != 2 or 2 * cols != N_QUBITS:
        raise NotImplementedError(
            f"Two-row bipartition only supported for a 2-row lattice; "
            f"got LATTICE_SHAPE={LATTICE_SHAPE}.")
    return 2 ** cols


def _vec_to_rho(v: np.ndarray) -> np.ndarray:
    rho = v.reshape(HILBERT_DIM, HILBERT_DIM, order="F")
    return (rho + rho.conj().T) / 2.0


def _row_negativity(rho: np.ndarray, d_site: int) -> float:
    rho_pt = (rho.reshape(d_site, d_site, d_site, d_site)
                  .transpose([0, 3, 2, 1])
                  .reshape(d_site * d_site, d_site * d_site))
    rho_pt = (rho_pt + rho_pt.conj().T) / 2.0
    evals = np.linalg.eigvalsh(rho_pt)
    return float(np.sum(np.abs(evals[evals < -1e-15])))


def _vn_entropy(rho: np.ndarray) -> float:
    rho_h = (rho + rho.conj().T) / 2.0
    evals = np.linalg.eigvalsh(rho_h)
    evals = evals[evals > 1e-15]
    return float(-np.sum(evals * np.log(evals)))


def _mag_x(rho: np.ndarray, n: int = N_QUBITS) -> float:
    """Sum_i Tr(rho X_i)."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    total = 0.0
    for site in range(n):
        ops = [sx if k == site else I2 for k in range(n)]
        op = ops[0]
        for m in ops[1:]:
            op = np.kron(op, m)
        total += np.trace(rho @ op).real
    return float(total)


def _lpdo_bond_diagnostics(rho: np.ndarray, d_site: int):
    X = purification_sqrt(rho)
    A1, A2, _ = tensorize_to_lpdo(X, d_site)
    _, _, chi, _, info = truncate_and_validate(rho, A1, A2, d_site)
    return int(chi), float(_bond_entropy(info["singular_values"]))


def six_qubit_trajectory_max_bond_entropy(L_comp, rho_ss, gamma_step, *,
                                          max_steps=100_000,
                                          fidelity_threshold=0.9):
    """Max LPDO bond entropy along trajectory |+y><+y|^N -> rho_ss."""
    d_site = _row_bipartition_dim()
    dt = 0.01 * gamma_step

    rho0 = initial_yy_state()
    v = rho0.reshape(-1, order="F").astype(complex)

    max_S = 0.0
    for _ in range(max_steps):
        rho = _vec_to_rho(v)
        try:
            _, S = _lpdo_bond_diagnostics(rho, d_site)
            if S > max_S:
                max_S = S
        except Exception:
            pass
        try:
            fid = _bures_fidelity(rho, rho_ss)
        except Exception:
            fid = 0.0
        if fid >= fidelity_threshold:
            break
        v_new = expm_multiply(dt * L_comp, v)
        if np.allclose(v, v_new, atol=1e-15):
            break
        v = v_new

    try:
        _, S = _lpdo_bond_diagnostics(rho_ss, d_site)
        if S > max_S:
            max_S = S
    except Exception:
        pass
    return float(max_S)


# ---------------------------------------------------------------------------
# 2-qubit operator quantities
# ---------------------------------------------------------------------------

def _operator_bond_entropy(gate16: np.ndarray) -> float:
    T = gate16.reshape(4, 4, 4, 4).transpose(0, 2, 1, 3).reshape(16, 16)
    sv = np.linalg.svd(T, compute_uv=False)
    p = sv ** 2
    s = p.sum()
    if s < 1e-30:
        return 0.0
    p = p / s
    p = p[p > 1e-30]
    return float(-np.sum(p * np.log(p)))


def _otoc_2q(L_2q: np.ndarray, gamma: float, gamma_p: float):
    """Return (otoc_small, otoc_large) for the 2-qubit Lindbladian.

    Mirrors plot_otoc_lindbladian.py for n=2:
    psi_0 = |00>, V_0 = X_1, W_0 = X_2,
    OTOC(t) = <psi_0| W(t)^dag V_0^dag W(t) V_0 |psi_0>
    with W(t) = E_t^dag(W_0) computed in the Pauli-string basis.
    """
    d = 4
    # 2-qubit Pauli basis (16 entries) in lex order matching numeric_two_qubit_lindbladian
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis_1q = [I2, X, Y, Z]
    basis = [np.kron(a, b) for a in paulis_1q for b in paulis_1q]

    # X_1 = X (x) I,  X_2 = I (x) X
    V0 = np.kron(X, I2)
    W0 = np.kron(I2, X)
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0

    def coeffs(O):
        return np.array([np.trace(P @ O) / d for P in basis], dtype=complex)

    def to_op(c):
        out = np.zeros_like(basis[0], dtype=complex)
        for ci, P in zip(c, basis):
            out = out + ci * P
        return out

    coeffs_w0 = coeffs(W0)

    def otoc_at(t):
        E_t = expm(L_2q * t)
        coeffs_wt = E_t.conj().T @ coeffs_w0
        Wt = to_op(coeffs_wt)
        op = Wt.conj().T @ V0.conj().T @ Wt @ V0
        return float(np.real(np.vdot(psi0, op @ psi0)))

    t_min = 0.1 * min(gamma, gamma_p)
    t_max = 10.0 * max(gamma, gamma_p)
    return otoc_at(t_min), otoc_at(t_max)


def _channel_stabilizer_purity(L_2q: np.ndarray, dt_stab: float = 0.1) -> float:
    """log2( d^2 * sum_i E_ii^2 / (d+1) ) for E = exp(L_2q * dt_stab), d=4."""
    E = expm(L_2q * dt_stab)
    diag = np.diag(E).real
    total = (4 ** 2) * np.sum(diag ** 2)
    return float(np.log2(total / (4 + 1)))


def _two_qubit_framabilities(gate16: np.ndarray):
    """Return (pauli_fra, min_fra, dyadic_stab, product_chi30)."""
    pauli_fra = float(np.max(np.sum(np.abs(gate16), axis=1)))

    D_ext = extended_pauli_D()
    d_ext_single = int(round(np.sqrt(D_ext.shape[1])))
    _, opt_fra = minimize_framability(
        gate16, d_ext_single=d_ext_single, n_restarts=5,
        method=DEFAULT_METHOD, max_iter=200, maxfev=1000, verbose=False,
    )
    ext_fra = heisenberg_framability(D_ext, gate16)
    min_fra = float(min(opt_fra, ext_fra, pauli_fra))

    dyadic = float(dyadic_stabilizer_framability(gate16, n_qubits=2))
    product = float(heisenberg_framability(PRODUCT_D, gate16))
    return pauli_fra, min_fra, dyadic, product


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute one (gamma, gamma') point of the comprehensive "
                    "6-qubit scan (state-level @ 6q + operator-level @ 2q)."
    )
    p.add_argument("--task_id",   type=int, required=True)
    p.add_argument("--n_pts",     type=int, default=51)
    p.add_argument("--n_pts_g",   type=int, default=None)
    p.add_argument("--n_pts_gp",  type=int, default=None)
    p.add_argument("--J",         type=float, default=1.0)
    p.add_argument("--gamma_step", type=float, default=0.2)
    p.add_argument("--out_dir",   type=str, default="results_six")
    p.add_argument("--max_steps", type=int, default=100_000)
    p.add_argument("--fidelity_threshold", type=float, default=0.9)
    p.add_argument("--dt_stabilizer", type=float, default=0.1)
    args = p.parse_args()

    n_g  = args.n_pts_g  if args.n_pts_g  is not None else args.n_pts
    n_gp = args.n_pts_gp if args.n_pts_gp is not None else args.n_pts
    total = n_g * n_gp
    tid = args.task_id
    if tid < 0 or tid >= total:
        print(f"ERROR: task_id {tid} out of range [0, {total - 1}]", file=sys.stderr)
        sys.exit(1)

    ig  = tid // n_gp
    igp = tid %  n_gp

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"six_full_{ig:04d}_{igp:04d}.npy"
    if out_path.exists():
        print(f"[task {tid}] reuse {out_path}", flush=True)
        return

    gamma = args.gamma_step * ig
    gp    = args.gamma_step * igp

    print(f"[task {tid}] ig={ig} igp={igp}  gamma={gamma:.4f}  "
          f"gamma'={gp:.4f}", flush=True)

    # ---- 6-qubit Lindbladian + steady state ---------------------------------
    t0 = time.time()
    L_comp = build_lindbladian_comp(J=args.J, gamma=gamma, gamma_p=gp)
    print(f"[task {tid}] L_comp built  shape={L_comp.shape}  nnz={L_comp.nnz}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    rho_ss = steady_state(L_comp)
    d_site = _row_bipartition_dim()
    print(f"[task {tid}] steady state ({time.time()-t0:.1f}s)", flush=True)

    ss_vn  = _vn_entropy(rho_ss)
    ss_neg = _row_negativity(rho_ss, d_site)
    mag_x  = _mag_x(rho_ss)

    t0 = time.time()
    max_bond_S = six_qubit_trajectory_max_bond_entropy(
        L_comp, rho_ss, args.gamma_step,
        max_steps=args.max_steps,
        fidelity_threshold=args.fidelity_threshold,
    )
    print(f"[task {tid}] trajectory max LPDO entropy = {max_bond_S:.6f}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ---- 2-qubit operator-level quantities ---------------------------------
    t0 = time.time()
    L_2q = numeric_two_qubit_lindbladian(J=args.J, gamma=gamma, gamma_p=gp)
    dt = 0.01 * args.gamma_step
    gate = expm(dt * L_2q).real

    op_bond = _operator_bond_entropy(gate)
    decay   = decay_rate(L_2q)
    otoc_s, otoc_l = _otoc_2q(L_2q, gamma, gp)
    chan_M  = _channel_stabilizer_purity(L_2q, dt_stab=args.dt_stabilizer)
    pauli_fra, min_fra, dyadic, product30 = _two_qubit_framabilities(gate)
    print(f"[task {tid}] 2q operator-level  "
          f"({time.time()-t0:.1f}s)", flush=True)

    arr = np.array([
        ss_vn, ss_neg, max_bond_S, op_bond,
        mag_x, decay, otoc_s, otoc_l, chan_M,
        pauli_fra, min_fra, dyadic, product30,
    ], dtype=float)

    np.save(out_path, arr)
    print(f"[task {tid}] wrote {out_path}", flush=True)
    print(f"  ss_vn={ss_vn:.4f}  ss_neg={ss_neg:.4f}  maxS={max_bond_S:.4f}  "
          f"opbond={op_bond:.4f}", flush=True)
    print(f"  mag_x={mag_x:.4f}  decay={decay:.4e}  otoc_s={otoc_s:.4f}  "
          f"otoc_l={otoc_l:.4f}  chanM={chan_M:.4f}", flush=True)
    print(f"  pauli={pauli_fra:.4f}  min={min_fra:.4f}  dyadic={dyadic:.4f}  "
          f"product30={product30:.4f}", flush=True)


if __name__ == "__main__":
    main()
