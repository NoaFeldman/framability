"""
Collect random_kron_<d>_<sample:02d>.npz files from random_kron_worker.py
and compute the average minimal framability over all samples.

Output: <out_dir>/random_kron_summary.npz
  keys: framability (N_D, N_SAMPLES), mean_framability (N_D,),
        angles (N_SAMPLES, 3)  -- [alpha, beta, gamma] for each sample,
        d_ext_singles (N_D,)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

D_EXT_SINGLES = [4, 6]
N_D           = len(D_EXT_SINGLES)
N_SAMPLES     = 10
MASTER_SEED   = 42


def _generate_angles() -> np.ndarray:
    rng = np.random.default_rng(MASTER_SEED)
    return rng.uniform(0.0, np.pi / 2.0, size=(N_SAMPLES, 3))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  type=str, default='results_random_kron')
    parser.add_argument('--out_dir', type=str, default='results_random_kron')
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    angles = _generate_angles()
    fra    = np.full((N_D, N_SAMPLES), np.nan)

    for di, d in enumerate(D_EXT_SINGLES):
        for si in range(N_SAMPLES):
            f = in_dir / f'random_kron_{d}_{si:02d}.npz'
            if not f.exists():
                print(f'[missing] {f}')
                continue
            data = np.load(f, allow_pickle=True)
            fra[di, si] = float(data['framability'])

    mean_fra = np.nanmean(fra, axis=1)

    out_path = out_dir / 'random_kron_summary.npz'
    np.savez(out_path,
             framability=fra,
             mean_framability=mean_fra,
             angles=angles,
             d_ext_singles=np.array(D_EXT_SINGLES))
    print(f'[saved] {out_path}')

    for di, d in enumerate(D_EXT_SINGLES):
        n_ok = int(np.sum(np.isfinite(fra[di])))
        print(f'  d_ext_single={d}: mean={mean_fra[di]:.6f}  '
              f'({n_ok}/{N_SAMPLES} tasks done)')

    print('\nAngle table (alpha, beta, gamma):')
    for si, (a, b, g) in enumerate(angles):
        print(f'  [{si:2d}]  alpha={a:.6f}  beta={b:.6f}  gamma={g:.6f}')


if __name__ == '__main__':
    main()
