"""
Collect per-task results from depol_kron_worker.py and produce:
  1. results_depol_kron/depol_kron_summary.npz
       framability  (N_SAMPLES, N_P, N_D)
       p_values     (N_P,)
       d_ext_singles (N_D,)
       angles       (N_SAMPLES, 3)  -- (alpha, beta, gamma)
  2. depol_kron.png  -- avg framability vs p, one line per d_ext
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MASTER_SEED   = 42
N_SAMPLES     = 10
D_EXT_SINGLES = [4, 6, 8]
P_VALUES      = [0.01 * i for i in range(10)]
N_D, N_P      = len(D_EXT_SINGLES), len(P_VALUES)


def _load_summary(in_dir: Path):
    fra = np.full((N_SAMPLES, N_P, N_D), np.nan)
    for d_idx, d in enumerate(D_EXT_SINGLES):
        for s_idx in range(N_SAMPLES):
            for p_idx in range(N_P):
                f = in_dir / f'depol_kron_{d}_{s_idx:02d}_{p_idx:02d}.npz'
                if f.exists():
                    fra[s_idx, p_idx, d_idx] = float(np.load(f)['framability'])
                else:
                    print(f'  missing: {f.name}')
    return fra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  default='results_depol_kron')
    parser.add_argument('--out_dir', default='results_depol_kron')
    parser.add_argument('--plot',    default='depol_kron.png')
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    fra = _load_summary(in_dir)

    rng    = np.random.default_rng(MASTER_SEED)
    angles = rng.uniform(0.0, np.pi / 2.0, size=(N_SAMPLES, 3))

    p_values = np.array(P_VALUES)
    np.savez(out_dir / 'depol_kron_summary.npz',
             framability   = fra,
             p_values      = p_values,
             d_ext_singles = np.array(D_EXT_SINGLES),
             angles        = angles)
    print(f'Saved {out_dir / "depol_kron_summary.npz"}')

    n_missing = np.sum(np.isnan(fra))
    if n_missing:
        print(f'Warning: {n_missing} missing tasks out of {N_SAMPLES * N_P * N_D}')

    # ── plot ──────────────────────────────────────────────────────────────────
    D_COLORS  = {4: 'tab:blue', 6: 'tab:orange', 8: 'tab:green'}
    D_MARKERS = {4: 'o', 6: 's', 8: '^'}

    fig, ax = plt.subplots(figsize=(7, 5))

    for d_idx, d in enumerate(D_EXT_SINGLES):
        col = D_COLORS[d]
        mrk = D_MARKERS[d]
        slice_dp = fra[:, :, d_idx]          # (N_SAMPLES, N_P)

        # individual sample curves (thin, transparent)
        for s in range(N_SAMPLES):
            row = slice_dp[s]
            if not np.all(np.isnan(row)):
                ax.plot(p_values, row, color=col, alpha=0.15, lw=0.8)

        # mean ± std
        mean = np.nanmean(slice_dp, axis=0)
        std  = np.nanstd(slice_dp,  axis=0)
        ax.plot(p_values, mean, color=col, marker=mrk, lw=2,
                markersize=6, label=fr'$d={d}$  (mean)')
        ax.fill_between(p_values, mean - std, mean + std,
                        color=col, alpha=0.2)

    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_xlabel(r'depolarisation $p$', fontsize=12)
    ax.set_ylabel(r'framability', fontsize=12)
    ax.set_title(
        r'Optimised framability of $e^{i(\alpha XX+\beta YY+\gamma ZZ)}$'
        '\nunder 2-qubit depolarisation  —  10 random gates',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=170)
    plt.close(fig)
    print(f'Saved {args.plot}')

    # summary table
    print('\np     ' + '  '.join(f'd={d:2d}(mean)  std' for d in D_EXT_SINGLES))
    for p_idx, p in enumerate(P_VALUES):
        row = f'{p:.2f}  '
        for d_idx in range(N_D):
            m = np.nanmean(fra[:, p_idx, d_idx])
            s = np.nanstd(fra[:, p_idx, d_idx])
            row += f'{m:8.4f}  {s:6.4f}  '
        print(row)


if __name__ == '__main__':
    main()
