"""
Collect the heavy depol_kron optimisation (depol_kron_opt_worker.py):
keep, for every (gate_label, d_ext_single) cell, the batch that reached the
lowest Heisenberg framability, and write a summary + a per-cell best frame.

Outputs
-------
  <out_dir>/depol_kron_opt_summary.npz
      gate_labels    (N_GATES,)               str
      alpha,beta,gamma,p   (N_GATES,)         float   -- per gate
      d_ext_singles  (N_D,)                   int
      framability    (N_GATES, N_D)           float   -- best over batches
      floor          (N_GATES, N_D)           float   -- spectral radius
      gap            (N_GATES, N_D)           float   -- framability - floor
      best_batch     (N_GATES, N_D)           int
      stage          (N_GATES, N_D)           str     -- winning search stage
      use_complex    (N_GATES, N_D)           bool
      n_found        (N_GATES, N_D)           int     -- batches present
  <out_dir>/best_frames/<label>_d<d>.npz       -- D, S, x, framability, floor
  console table of framability / floor / gap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GATES = [
    ('g1_p0.00', (float(np.sqrt(0.5)), float(np.exp(-1.0)), float(np.pi)), 0.00),
    ('g1_p0.08', (float(np.sqrt(0.5)), float(np.exp(-1.0)), float(np.pi)), 0.08),
    ('g2_p0.00', (0.3, 0.3, 0.0), 0.00),
    ('g2_p0.08', (0.3, 0.3, 0.0), 0.08),
]
D_EXT_SINGLES = [4, 6, 8]
N_BATCHES     = 16

N_GATES = len(GATES)
N_D     = len(D_EXT_SINGLES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  default='results_depol_kron_opt')
    parser.add_argument('--out_dir', default='results_depol_kron_opt')
    parser.add_argument('--n_batches', type=int, default=N_BATCHES)
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    (out_dir / 'best_frames').mkdir(parents=True, exist_ok=True)

    fra   = np.full((N_GATES, N_D), np.nan)
    floor = np.full((N_GATES, N_D), np.nan)
    gap   = np.full((N_GATES, N_D), np.nan)
    best_batch  = np.full((N_GATES, N_D), -1, dtype=int)
    n_found     = np.zeros((N_GATES, N_D), dtype=int)
    stage       = np.empty((N_GATES, N_D), dtype=object)
    use_complex = np.zeros((N_GATES, N_D), dtype=bool)

    for g_idx, (label, _, _) in enumerate(GATES):
        for d_idx, d in enumerate(D_EXT_SINGLES):
            best = None
            for b in range(args.n_batches):
                f = in_dir / f'{label}_d{d}_b{b:02d}.npz'
                if not f.exists():
                    continue
                data = dict(np.load(f, allow_pickle=True))
                if 'framability' not in data:
                    continue
                val = float(data['framability'])
                n_found[g_idx, d_idx] += 1
                if not np.isfinite(val):
                    continue
                if best is None or val < best[0]:
                    best = (val, b, data)

            if best is None:
                print(f'  missing: no valid batch for {label} d={d}')
                continue

            val, b, data = best
            fra[g_idx, d_idx]   = val
            floor[g_idx, d_idx] = float(data['floor'])
            gap[g_idx, d_idx]   = float(data.get('gap', val - float(data['floor'])))
            best_batch[g_idx, d_idx]  = b
            stage[g_idx, d_idx]       = str(data.get('stage', ''))
            use_complex[g_idx, d_idx] = bool(data.get('use_complex', False))

            np.savez(
                out_dir / 'best_frames' / f'{label}_d{d}.npz',
                framability = np.array(val),
                floor       = data['floor'],
                gap         = np.array(gap[g_idx, d_idx]),
                D           = data['D'],
                S           = data['S'],
                x           = data['x'],
                use_complex = data['use_complex'],
                stage       = data['stage'],
                alpha       = data['alpha'],
                beta        = data['beta'],
                gamma       = data['gamma'],
                p           = data['p'],
                d_ext_single = data['d_ext_single'],
                gate_label  = data['gate_label'],
                best_batch  = np.array(b),
                code_version = data['code_version'],
            )

    alphas = np.array([g[1][0] for g in GATES])
    betas  = np.array([g[1][1] for g in GATES])
    gammas = np.array([g[1][2] for g in GATES])
    ps     = np.array([g[2]    for g in GATES])

    np.savez(
        out_dir / 'depol_kron_opt_summary.npz',
        gate_labels   = np.array([g[0] for g in GATES]),
        alpha         = alphas,
        beta          = betas,
        gamma         = gammas,
        p             = ps,
        d_ext_singles = np.array(D_EXT_SINGLES),
        framability   = fra,
        floor         = floor,
        gap           = gap,
        best_batch    = best_batch,
        stage         = stage.astype(str),
        use_complex   = use_complex,
        n_found       = n_found,
    )
    print(f'Saved {out_dir / "depol_kron_opt_summary.npz"}')

    # ── console table ─────────────────────────────────────────────────────────
    print('\nBest Heisenberg framability (lower = better)')
    hdr = 'gate            p      ' + '  '.join(
        f'd={d}: fra   (gap)   ' for d in D_EXT_SINGLES)
    print(hdr)
    for g_idx, (label, (a, b_, c), p) in enumerate(GATES):
        row = f'{label:12s}  {p:4.2f}  '
        for d_idx in range(N_D):
            f_ = fra[g_idx, d_idx]
            gp = gap[g_idx, d_idx]
            if np.isnan(f_):
                row += '   --                '
            else:
                row += f'{f_:9.6f} ({gp:8.2e})  '
        print(row)

    n_missing = int(np.sum(np.isnan(fra)))
    if n_missing:
        print(f'\nWarning: {n_missing}/{N_GATES * N_D} cells have no result yet')


if __name__ == '__main__':
    main()
