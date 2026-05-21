"""
Compare minimax framability vs depolarisation p for two gate sets:
    {H, CNOT, T}   (2-qubit frame, d_ext_single in [4, 6, 8])
    {H, Toffoli}   (3-qubit frame, d_ext_single in [4, 6])

Reads:
    results_minimax_H_CNOT_T/minimax_frame.npz    (from minimax_frame_collect.py)
    results_minimax_toffoli/minimax_toffoli_summary.npz  (from minimax_toffoli_collect.py)

Falls back to per-task npz files if either summary is absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def _monotonise(arr: np.ndarray) -> np.ndarray:
    """Enforce non-increasing in d_ext_single dimension (axis 0), ignoring NaN."""
    out = arr.copy()
    running = np.full(arr.shape[1], np.inf)
    for di in range(arr.shape[0]):
        mask = ~np.isnan(out[di])
        running[mask] = np.minimum(running[mask], out[di, mask])
        out[di, mask] = running[mask]
    return out


def _load_hcnott(in_dir: Path):
    """Return (p_values, d_ext_singles, worst_mono) for {H, CNOT, T}."""
    summary = in_dir / 'minimax_frame.npz'
    if summary.exists():
        d = np.load(summary)
        return d['p_values'], d['d_ext_singles'], d['worst_mono']

    # Fallback: read per-task files
    print(f'[warn] {summary} not found — aggregating per-task files')
    D_EXT_SINGLES = [4, 6, 8]
    P_VALUES      = [0.01 * i for i in range(11)]
    worst = np.full((len(D_EXT_SINGLES), len(P_VALUES)), np.nan)
    for di, d in enumerate(D_EXT_SINGLES):
        for pi in range(len(P_VALUES)):
            f = in_dir / f'minimax_{d}_{pi:02d}.npz'
            if f.exists():
                worst[di, pi] = float(np.load(f)['worst'])
    return np.array(P_VALUES), np.array(D_EXT_SINGLES), _monotonise(worst)


def _load_toffoli(in_dir: Path):
    """Return (p_values, d_ext_singles, worst_mono) for {H, Toffoli}."""
    summary = in_dir / 'minimax_toffoli_summary.npz'
    if summary.exists():
        d = np.load(summary)
        return d['p_values'], d['d_ext_singles'], _monotonise(d['worst'])

    # Fallback: read per-task files
    print(f'[warn] {summary} not found — aggregating per-task files')
    D_EXT_SINGLES = [4, 6]
    P_VALUES      = [0.01 * i for i in range(11)]
    worst = np.full((len(D_EXT_SINGLES), len(P_VALUES)), np.nan)
    for di, d in enumerate(D_EXT_SINGLES):
        for pi in range(len(P_VALUES)):
            f = in_dir / f'minimax_toffoli_{d}_{pi:02d}.npz'
            if f.exists():
                worst[di, pi] = float(np.load(f)['worst'])
    return np.array(P_VALUES), np.array(D_EXT_SINGLES), _monotonise(worst)


# ── colours / styles ─────────────────────────────────────────────────────────

D_COLORS  = {4: 'tab:green', 6: 'tab:red', 8: 'tab:purple'}
D_MARKERS = {4: '^', 6: 'D', 8: 'v'}


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hcnott_dir', type=str,
                        default='results_minimax_H_CNOT_T')
    parser.add_argument('--toffoli_dir', type=str,
                        default='results_minimax_toffoli')
    parser.add_argument('--out',  type=str,
                        default='minimax_gate_sets.png')
    args = parser.parse_args()

    p_hct,  d_hct,  w_hct  = _load_hcnott( Path(args.hcnott_dir))
    p_toff, d_toff, w_toff = _load_toffoli(Path(args.toffoli_dir))

    fig, ax = plt.subplots(figsize=(8, 5))

    for di, d in enumerate(d_hct):
        ax.plot(p_hct, w_hct[di],
                color=D_COLORS[int(d)], marker=D_MARKERS[int(d)],
                linestyle='-', linewidth=1.8, markersize=6,
                label=fr'$\{{H,CNOT,T\}}$  $d_{{\rm ext}}={d}^2={d**2}$')

    for di, d in enumerate(d_toff):
        ax.plot(p_toff, w_toff[di],
                color=D_COLORS[int(d)], marker=D_MARKERS[int(d)],
                linestyle='--', linewidth=1.8, markersize=6,
                label=fr'$\{{H,\rm Toffoli\}}$  $d_{{\rm ext}}={d}^3={d**3}$')

    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.9)
    ax.set_xlabel(r'depolarisation $p$', fontsize=12)
    ax.set_ylabel(r'$\min_S\,\max_g\,$ framability', fontsize=12)
    ax.set_title(r'Minimax framability: $\{H,CNOT,T\}$ (solid) vs '
                 r'$\{H,\mathrm{Toffoli}\}$ (dashed)', fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=170)
    plt.close(fig)
    print(f'[saved] {args.out}')


if __name__ == '__main__':
    main()
