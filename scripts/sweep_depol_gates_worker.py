"""
Worker for the depolarised-gate sweep.

For one (gate, p) pair compute:
    - Framability for 5 frame choices
        0: Pauli frame                  (D = I_16)
        1: Extended-Pauli frame         (D = extended_pauli_D(),  16x36)
        2: Optimized Kron frame, d_ext_single=4
        3: Optimized Kron frame, d_ext_single=6
        4: Optimized Kron frame, d_ext_single=8
    - OTOC of the depolarised channel
    - Channel stabilizer purity
    - Operator bond entropy

Gates: CNOT (2-qubit), H, T (1-qubit, lifted to G(x)I).
Depolarisation: N_p(x)N_p in the 2q Pauli basis (paper Eq. 38).

task_id = gate_idx * N_P + p_idx   with  P_VALUES = [0.01*i for i in range(8)]

Output: <out_dir>/sweep_<g>_<p:02d>.npz
        keys: framability(5,), otoc(), stab(), obe(),
              gate (str), p (float)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framability import extended_pauli_D, heisenberg_framability
from optimize_framability import minimize_framability, DEFAULT_METHOD


# ── parameter grid ───────────────────────────────────────────────────────────
GATES = ['CNOT', 'H', 'T']
P_VALUES = [0.01 * i for i in range(11)]
N_FRAMES = 5
N_P = len(P_VALUES)


# ── Pauli basis ──────────────────────────────────────────────────────────────
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS_1Q = [_I, _X, _Y, _Z]
PAULIS_2Q = [np.kron(a, b) for a in PAULIS_1Q for b in PAULIS_1Q]


def _superop_2q(U):
    n = 16
    L = np.zeros((n, n), dtype=float)
    for j, Bj in enumerate(PAULIS_2Q):
        img = U @ Bj @ U.conj().T
        for i, Bi in enumerate(PAULIS_2Q):
            L[i, j] = (np.trace(Bi.conj().T @ img) / 4).real
    return L


def _depol_2q(p):
    diag = np.array(
        [(1.0 - 4 * p) ** ((a != 0) + (b != 0))
         for a in range(4) for b in range(4)],
        dtype=float,
    )
    return np.diag(diag)


def build_channel(gate_label, p):
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
    T = np.diag([1.0, np.exp(1j * np.pi / 4)]).astype(complex)
    sqrtT = np.diag([1.0, np.exp(1j * np.pi / 8)]).astype(complex)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)
    if gate_label == 'H':
        U = np.kron(H, _I)
    elif gate_label == 'T':
        U = np.kron(T, _I)
    elif gate_label == 'sqrtT':
        U = np.kron(sqrtT, _I)
    elif gate_label == 'CNOT':
        U = CNOT
    else:
        raise ValueError(gate_label)
    return _depol_2q(p) @ _superop_2q(U)


# ── channel-level quantities ─────────────────────────────────────────────────
def pauli_framability(channel):
    return float(np.max(np.sum(np.abs(channel), axis=1)))


def ext_pauli_framability(channel):
    return float(heisenberg_framability(extended_pauli_D(), channel))


def ext_pauli_framability_scaled(channel, scale=0.84):
    """Extended Pauli frame with single-qubit scale a passed to
    extended_pauli_D, so the extra-column entries equal a/sqrt(2).
    Default a=0.84 -> entries = sqrt(1/2)*0.84."""
    return float(heisenberg_framability(extended_pauli_D(scale), channel))


def optimized_framability(channel, d_ext_single, n_restarts):
    _, f = minimize_framability(
        channel, d_ext_single=d_ext_single, n_restarts=n_restarts,
        method=DEFAULT_METHOD, max_iter=200, maxfev=1000, verbose=False,
    )
    return float(f)


def operator_bond_entropy(channel):
    T = channel.reshape(4, 4, 4, 4).transpose(0, 2, 1, 3).reshape(16, 16)
    sv = np.linalg.svd(T, compute_uv=False)
    p = sv ** 2
    s = p.sum()
    if s < 1e-30:
        return 0.0
    p = p / s
    p = p[p > 1e-30]
    return float(-np.sum(p * np.log(p)))


def channel_stabilizer_purity(channel):
    diag = np.diag(channel).real
    total = (4 ** 2) * np.sum(diag ** 2)
    return float(np.log2(total / (4 + 1)))


def channel_otoc(channel):
    d = 4
    V0 = np.kron(_X, _I)
    W0 = np.kron(_I, _X)
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    coeffs_w0 = np.array([np.trace(P @ W0) / d for P in PAULIS_2Q],
                         dtype=complex)
    coeffs_wt = channel.conj().T @ coeffs_w0
    Wt = np.zeros_like(PAULIS_2Q[0], dtype=complex)
    for ci, P in zip(coeffs_wt, PAULIS_2Q):
        Wt = Wt + ci * P
    op = Wt.conj().T @ V0.conj().T @ Wt @ V0
    return float(np.real(np.vdot(psi0, op @ psi0)))


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help=f'0..{len(GATES) * N_P - 1}; '
                             'task_id = gate_idx * N_P + p_idx.')
    parser.add_argument('--out_dir', type=str, default='results_depol_sweep')
    parser.add_argument('--n_restarts', type=int, default=5)
    args = parser.parse_args()

    g = args.task_id // N_P
    pi = args.task_id % N_P
    if g >= len(GATES):
        print(f'ERROR: task_id {args.task_id} out of range '
              f'(max {len(GATES) * N_P - 1})', file=sys.stderr)
        sys.exit(1)

    gate = GATES[g]
    p = P_VALUES[pi]
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'sweep_{g}_{pi:02d}.npz')
    if os.path.exists(out_path):
        print(f'Skip: {out_path} already exists', flush=True)
        return

    print(f'[task {args.task_id}] gate={gate}  p={p:.3f}', flush=True)
    channel = build_channel(gate, p)

    fra = np.full(N_FRAMES, np.nan)
    print('  computing Pauli framability...', flush=True)
    fra[0] = pauli_framability(channel)
    print('  computing extended-Pauli framability...', flush=True)
    fra[1] = ext_pauli_framability(channel)
    for k, d_ext_single in enumerate([4, 6, 8], start=2):
        print(f'  computing optimized framability (d_ext_single={d_ext_single})...',
              flush=True)
        fra[k] = optimized_framability(channel, d_ext_single, args.n_restarts)

    otoc = channel_otoc(channel)
    stab = channel_stabilizer_purity(channel)
    obe = operator_bond_entropy(channel)

    np.savez(out_path,
             framability=fra, otoc=otoc, stab=stab, obe=obe,
             gate=np.array(gate), p=np.array(p))
    print(f'[task {args.task_id}] saved {out_path}  fra={fra}  '
          f'otoc={otoc:.4f}  stab={stab:.4f}  obe={obe:.4f}',
          flush=True)


if __name__ == '__main__':
    main()
