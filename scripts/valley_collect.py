"""
Collect per-task valley results into a single npz summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from valley_worker import POINTS, N_TASKS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_valley')
    parser.add_argument('--out_dir', type=str, default='results_valley')
    parser.add_argument('--d_ext_single', type=int, default=6)
    parser.add_argument('--tag_suffix', type=str, default='',
                        help='Suffix used by valley_worker (e.g. "long").')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_part = f'_{args.tag_suffix}' if args.tag_suffix else ''

    gammas    = np.full(N_TASKS, np.nan)
    gamma_ps  = np.full(N_TASKS, np.nan)
    f_opt     = np.full(N_TASKS, np.nan)
    alpha_min = np.full(N_TASKS, np.nan)
    alpha_max = np.full(N_TASKS, np.nan)
    alpha_mean = np.full(N_TASKS, np.nan)
    n_edges = np.zeros(N_TASKS, dtype=int)

    missing = []
    for tid, (g, gp) in enumerate(POINTS):
        f = in_dir / f'valley_task{tid:02d}_d{args.d_ext_single}{suffix_part}.npz'
        if not f.exists():
            missing.append(str(f))
            continue
        d = np.load(f, allow_pickle=True)
        gammas[tid]   = float(d['gamma'])
        gamma_ps[tid] = float(d['gamma_p'])
        f_opt[tid]    = float(d['f_opt'])
        alphas = d['edge_alphas']
        n_edges[tid]    = alphas.size
        alpha_min[tid]  = float(np.min(alphas))
        alpha_max[tid]  = float(np.max(alphas))
        alpha_mean[tid] = float(np.mean(alphas))
        print(f'  task {tid:02d}  (gamma={g}, gamma_p={gp}):  '
              f'f_opt={f_opt[tid]:.6f}  alphas in '
              f'[{alpha_min[tid]:.3f}, {alpha_max[tid]:.3f}]  '
              f'mean={alpha_mean[tid]:.3f}')
    if missing:
        print(f'[warn] {len(missing)} missing files:')
        for m in missing:
            print('   ', m)

    npz_path = out_dir / f'valley_summary_d{args.d_ext_single}{suffix_part}.npz'
    np.savez(npz_path,
             gammas=gammas, gamma_ps=gamma_ps,
             d_ext_single=np.array(args.d_ext_single),
             f_opt=f_opt,
             alpha_min=alpha_min, alpha_max=alpha_max, alpha_mean=alpha_mean,
             n_edges=n_edges)
    print(f'[saved] {npz_path}')


if __name__ == '__main__':
    main()
