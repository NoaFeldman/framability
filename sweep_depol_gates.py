"""
Sweep p over [0, 0.07] in steps of 0.01 for three gates (CNOT, H, T) and
five frame choices (Pauli, extended-Pauli, optimized Kron with d_ext_single
in {4, 6, 8}).  For every (gate, p) combination compute:

    - Framability (one value per frame choice)
    - OTOC of the depolarised channel
    - Channel stabilizer purity
    - Operator bond entropy

Single-qubit gates (H, T) are lifted to two qubits as G(x)I so every gate
is described by a 16x16 superoperator in the two-qubit Pauli-string basis.
The depolarising channel applied after the gate is N_p (x) N_p, matching
the convention in depol_fra_worker.py.

Output:
    results_depol_sweep/depol_sweep.npz       (raw arrays)
    results_plots/depol_sweep.png             (composite figure)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from framability import extended_pauli_D, heisenberg_framability
from optimize_framability import minimize_framability, DEFAULT_METHOD


# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS_1Q = [_I, _X, _Y, _Z]
PAULIS_2Q = [np.kron(a, b) for a in PAULIS_1Q for b in PAULIS_1Q]


# ---------------------------------------------------------------------------
# Superoperator helpers
# ---------------------------------------------------------------------------
def _superop_2q(U: np.ndarray) -> np.ndarray:
    """16x16 real superoperator for rho -> U rho U^dag in the 2q Pauli basis."""
    n = 16
    L = np.zeros((n, n), dtype=float)
    for j, Bj in enumerate(PAULIS_2Q):
        img = U @ Bj @ U.conj().T
        for i, Bi in enumerate(PAULIS_2Q):
            L[i, j] = (np.trace(Bi.conj().T @ img) / 4).real
    return L


def _depol_2q(p: float) -> np.ndarray:
    """N_p (x) N_p in the 2q Pauli basis (paper Eq. 38)."""
    diag = np.array(
        [(1.0 - 4 * p) ** ((a != 0) + (b != 0))
         for a in range(4) for b in range(4)],
        dtype=float,
    )
    return np.diag(diag)


def build_channel(gate_label: str, p: float) -> np.ndarray:
    """Return the 16x16 depolarised channel matrix for the given gate."""
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
    T = np.diag([1.0, np.exp(1j * np.pi / 4)]).astype(complex)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)

    if gate_label == 'H':
        U = np.kron(H, _I)
    elif gate_label == 'T':
        U = np.kron(T, _I)
    elif gate_label == 'CNOT':
        U = CNOT
    else:
        raise ValueError(f"Unknown gate {gate_label!r}")

    return _depol_2q(p) @ _superop_2q(U)


# ---------------------------------------------------------------------------
# Framability for the five frame choices
# ---------------------------------------------------------------------------
def pauli_framability(channel: np.ndarray) -> float:
    """Framability w.r.t. the Pauli frame D = I_16: max row 1-norm of channel
    (matches scan_worker.py convention)."""
    return float(np.max(np.sum(np.abs(channel), axis=1)))


def ext_pauli_framability(channel: np.ndarray) -> float:
    D_ext = extended_pauli_D()
    return float(heisenberg_framability(D_ext, channel))


def optimized_framability(channel: np.ndarray, d_ext_single: int,
                          n_restarts: int = 5) -> float:
    _, f = minimize_framability(
        channel, d_ext_single=d_ext_single, n_restarts=n_restarts,
        method=DEFAULT_METHOD, max_iter=200, maxfev=1000, verbose=False,
    )
    return float(f)


# ---------------------------------------------------------------------------
# Other quantities (channel-level versions; work directly on the channel matrix
# in the 2q Pauli basis, no Lindbladian / time evolution required)
# ---------------------------------------------------------------------------
def operator_bond_entropy(channel: np.ndarray) -> float:
    T = channel.reshape(4, 4, 4, 4).transpose(0, 2, 1, 3).reshape(16, 16)
    sv = np.linalg.svd(T, compute_uv=False)
    p = sv ** 2
    s = p.sum()
    if s < 1e-30:
        return 0.0
    p = p / s
    p = p[p > 1e-30]
    return float(-np.sum(p * np.log(p)))


def channel_stabilizer_purity(channel: np.ndarray) -> float:
    """log2( d^2 * sum_i E_ii^2 / (d+1) ) with d=4."""
    diag = np.diag(channel).real
    total = (4 ** 2) * np.sum(diag ** 2)
    return float(np.log2(total / (4 + 1)))


def channel_otoc(channel: np.ndarray) -> float:
    """OTOC of a 2q channel (Pauli basis) at one application of the channel.

    Mirrors `six_qubit_full_worker._otoc_2q` but with the discrete channel
    replacing exp(L*t).  psi_0=|00>, V_0=X_1, W_0=X_2.
    """
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


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
GATES = ['CNOT', 'H', 'T']
FRAME_LABELS = [
    'Pauli frame',
    'Extended-Pauli frame',
    'Optimized (d_ext_single=4)',
    'Optimized (d_ext_single=6)',
    'Optimized (d_ext_single=8)',
]
FRAME_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']


def compute_one_frame(channel: np.ndarray, label: str,
                      n_restarts: int = 5) -> float:
    if label == 'Pauli frame':
        return pauli_framability(channel)
    if label == 'Extended-Pauli frame':
        return ext_pauli_framability(channel)
    if label.startswith('Optimized'):
        d_ext_single = int(label.split('=')[-1].rstrip(')'))
        return optimized_framability(channel, d_ext_single,
                                     n_restarts=n_restarts)
    raise ValueError(label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_max',      type=int,   default=8,
                        help='Number of p values: p = 0.01 * i for i in range(p_max).')
    parser.add_argument('--n_restarts', type=int,   default=5,
                        help='Restarts for the optimized framability searches.')
    parser.add_argument('--out_dir',    type=str,
                        default='results_depol_sweep')
    args = parser.parse_args()

    p_values = np.array([0.01 * i for i in range(args.p_max)])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_g = len(GATES)
    n_p = len(p_values)
    n_f = len(FRAME_LABELS)

    fra = np.full((n_g, n_p, n_f), np.nan)
    otoc = np.full((n_g, n_p), np.nan)
    stab = np.full((n_g, n_p), np.nan)
    obe = np.full((n_g, n_p), np.nan)

    for ig, gate in enumerate(GATES):
        for ip, p in enumerate(p_values):
            channel = build_channel(gate, float(p))

            otoc[ig, ip] = channel_otoc(channel)
            stab[ig, ip] = channel_stabilizer_purity(channel)
            obe[ig, ip] = operator_bond_entropy(channel)

            for jf, lbl in enumerate(FRAME_LABELS):
                fra[ig, ip, jf] = compute_one_frame(
                    channel, lbl, n_restarts=args.n_restarts,
                )

            print(f'[{gate:>4}  p={p:.2f}]  '
                  f'OTOC={otoc[ig,ip]:.4f}  '
                  f'stab={stab[ig,ip]:.4f}  '
                  f'obe={obe[ig,ip]:.4f}  '
                  f'fra={[f"{x:.3f}" for x in fra[ig,ip]]}',
                  flush=True)

    npz_path = out_dir / 'depol_sweep.npz'
    np.savez(npz_path,
             p_values=p_values, gates=np.array(GATES),
             frame_labels=np.array(FRAME_LABELS),
             framability=fra, otoc=otoc,
             channel_stabilizer_purity=stab,
             operator_bond_entropy=obe)
    print(f'[saved] {npz_path}')

    # ---- Plot: rows = gates, cols = [framability, OTOC, stab, op-bond] -----
    fig, axes = plt.subplots(n_g, 4, figsize=(20, 4.0 * n_g), sharex=True)
    if n_g == 1:
        axes = axes[np.newaxis, :]

    for ig, gate in enumerate(GATES):
        ax = axes[ig, 0]
        for jf, lbl in enumerate(FRAME_LABELS):
            ax.plot(p_values, fra[ig, :, jf], 'o-',
                    color=FRAME_COLORS[jf], label=lbl)
        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
        ax.set_ylabel('Framability')
        ax.set_title(f'{gate}: framability')
        ax.grid(alpha=0.3)
        if ig == 0:
            ax.legend(fontsize=8)

        axes[ig, 1].plot(p_values, otoc[ig], 'o-', color='tab:cyan')
        axes[ig, 1].set_ylabel('OTOC')
        axes[ig, 1].set_title(f'{gate}: OTOC')
        axes[ig, 1].grid(alpha=0.3)

        axes[ig, 2].plot(p_values, stab[ig], 'o-', color='tab:olive')
        axes[ig, 2].set_ylabel('Channel stabilizer purity')
        axes[ig, 2].set_title(f'{gate}: stabilizer purity')
        axes[ig, 2].grid(alpha=0.3)

        axes[ig, 3].plot(p_values, obe[ig], 'o-', color='tab:brown')
        axes[ig, 3].set_ylabel('Operator bond entropy')
        axes[ig, 3].set_title(f'{gate}: op. bond entropy')
        axes[ig, 3].grid(alpha=0.3)

        for ax_ in axes[ig]:
            if ig == n_g - 1:
                ax_.set_xlabel(r'depolarisation $p$')

    fig.suptitle('Depolarised gates: framability and channel quantities vs. p '
                 '(H, T lifted to 2 qubits as G(x)I)')
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    png_path = Path('results_plots') / 'depol_sweep.png'
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    print(f'[saved] {png_path}')


if __name__ == '__main__':
    main()
