"""
Evaluate the framability of the fixed frame
    S = [I, X, Y, Z, a*(X+Y)/√2, a*(X-Y)/√2]
for gate sets {H, CNOT, T} and {H, Toffoli} across all p values,
and compare to the current minimax optimisation results.

No optimisation is performed — this is a direct upper-bound check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from optimize_framability import (
    N_FIXED_COLS, _FIXED_COLS, _kron_power, _get_framability_fast,
    _project_columns_bloch,
)
from sweep_depol_gates_worker import build_channel
from minimax_toffoli_worker import build_channel_3q

N_S_ROWS = 4
P_VALUES = [0.01 * i for i in range(11)]


def build_S(a: float, d_ext_single: int) -> np.ndarray:
    """S = [I | X, Y, Z, a(X+Y)/√2, a(X-Y)/√2, zeros...]  shape (4, d_ext_single)."""
    n_free = d_ext_single - N_FIXED_COLS
    base = np.array([
        [0.0, 0.0, 0.0, 0.0,           0.0          ],
        [1.0, 0.0, 0.0, a/np.sqrt(2),  a/np.sqrt(2) ],
        [0.0, 1.0, 0.0, a/np.sqrt(2), -a/np.sqrt(2) ],
        [0.0, 0.0, 1.0, 0.0,           0.0          ],
    ])
    free = np.zeros((N_S_ROWS, n_free))
    k = min(n_free, base.shape[1])
    free[:, :k] = base[:, :k]
    free = _project_columns_bloch(free)
    return np.hstack([_FIXED_COLS, free])


def minimax_at_S(S: np.ndarray, channels: list[np.ndarray], n_qubits: int) -> float:
    D = _kron_power(S, n_qubits)
    return float(np.max([_get_framability_fast(D, ch) for ch in channels]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--a',           type=float, default=1.0,
                        help='Scale for the XY columns (default 1.0).')
    parser.add_argument('--hcnott_dir',  type=str,
                        default='results_minimax_H_CNOT_T')
    parser.add_argument('--toffoli_dir', type=str,
                        default='results_minimax_toffoli')
    parser.add_argument('--out',         type=str,
                        default='results_plots/eval_ixyz_frame.png')
    args = parser.parse_args()

    p_values = np.array(P_VALUES)

    # ── {H, CNOT, T} ─────────────────────────────────────────────────────────
    gates_hct    = ['H', 'CNOT', 'T']
    d_ext_hct    = [4, 6, 8]
    n_qubits_hct = 2

    fixed_hct = {d: np.full(len(P_VALUES), np.nan) for d in d_ext_hct}
    for d in d_ext_hct:
        S = build_S(args.a, d)
        for pi, p in enumerate(P_VALUES):
            channels = [build_channel(g, float(p)) for g in gates_hct]
            fixed_hct[d][pi] = minimax_at_S(S, channels, n_qubits_hct)
        print(f'[H,CNOT,T] d={d}  worst at fixed S: '
              f'{fixed_hct[d]}', flush=True)

    # Load saved minimax for comparison
    saved_hct = None
    summary_hct = Path(args.hcnott_dir) / 'minimax_frame.npz'
    if summary_hct.exists():
        d_saved = np.load(summary_hct)
        saved_hct = d_saved['worst_mono']   # (N_D, N_P)
        d_ext_saved_hct = list(d_saved['d_ext_singles'])

    # ── {H, Toffoli} ─────────────────────────────────────────────────────────
    gates_toff    = ['H', 'Toffoli']
    d_ext_toff    = [4, 6]
    n_qubits_toff = 3

    fixed_toff = {d: np.full(len(P_VALUES), np.nan) for d in d_ext_toff}
    for d in d_ext_toff:
        S = build_S(args.a, d)
        for pi, p in enumerate(P_VALUES):
            channels = [build_channel_3q(g, float(p)) for g in gates_toff]
            fixed_toff[d][pi] = minimax_at_S(S, channels, n_qubits_toff)
        print(f'[H,Toffoli] d={d}  worst at fixed S: '
              f'{fixed_toff[d]}', flush=True)

    saved_toff = None
    summary_toff = Path(args.toffoli_dir) / 'minimax_toffoli_summary.npz'
    if summary_toff.exists():
        d_saved_t = np.load(summary_toff)
        saved_toff = d_saved_t['worst']
        d_ext_saved_toff = list(d_saved_t['d_ext_singles'])

    # ── Print comparison table ─────────────────────────────────────────────
    print('\n=== {H, CNOT, T}: fixed S vs optimised ===')
    for d in d_ext_hct:
        print(f'  d_ext_single={d}  fixed: {fixed_hct[d].round(4)}')
        if saved_hct is not None and d in d_ext_saved_hct:
            di = d_ext_saved_hct.index(d)
            print(f'          optim: {saved_hct[di].round(4)}')
            delta = fixed_hct[d] - saved_hct[di]
            print(f'          delta: {delta.round(4)}  (>0 means fixed is worse)')

    print('\n=== {H, Toffoli}: fixed S vs optimised ===')
    for d in d_ext_toff:
        print(f'  d_ext_single={d}  fixed: {fixed_toff[d].round(4)}')
        if saved_toff is not None and d in d_ext_saved_toff:
            di = d_ext_saved_toff.index(d)
            print(f'          optim: {saved_toff[di].round(4)}')
            delta = fixed_toff[d] - saved_toff[di]
            print(f'          delta: {delta.round(4)}  (>0 means fixed is worse)')

    # ── Plot ─────────────────────────────────────────────────────────────────
    D_COLORS  = {4: 'tab:green', 6: 'tab:red', 8: 'tab:purple'}
    D_MARKERS = {4: '^', 6: 'D', 8: 'v'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (gate_label, fixed_data, saved_data, d_list, d_ext_saved_list) in zip(
        axes,
        [
            (r'$\{H,CNOT,T\}$', fixed_hct, saved_hct,
             d_ext_hct, d_ext_saved_hct if saved_hct is not None else []),
            (r'$\{H,\mathrm{Toffoli}\}$', fixed_toff, saved_toff,
             d_ext_toff, d_ext_saved_toff if saved_toff is not None else []),
        ],
    ):
        for d in d_list:
            c, m = D_COLORS[d], D_MARKERS[d]
            ax.plot(p_values, fixed_data[d], color=c, marker=m,
                    linestyle='-', linewidth=1.8, markersize=6,
                    label=fr'fixed S, $d={d}$')
            if saved_data is not None and d in d_ext_saved_list:
                di = d_ext_saved_list.index(d)
                ax.plot(p_values, saved_data[di], color=c, marker=m,
                        linestyle=':', linewidth=1.4, markersize=4, alpha=0.7,
                        label=fr'optimised, $d={d}$')
        ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
        ax.set_xlabel(r'depolarisation $p$')
        ax.set_ylabel(r'$\max_g$ framability')
        ax.set_title(gate_label)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

    fig.suptitle(fr'Fixed frame $S=[I,X,Y,Z,a(X+Y)/\sqrt{{2}},a(X-Y)/\sqrt{{2}}]$ '
                 fr'with $a={args.a}$ vs optimised')
    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    print(f'\n[saved] {args.out}')


if __name__ == '__main__':
    main()
