"""
Collect dissipative-PT refinement results and regenerate the figure.

Reads dpt_refine_<ih:02d>_<ig:02d>.npz point files produced by
dissipative_PT_refine_worker.py.  For each point, the refined opt_fra_4 /
opt_fra_6 are merged into the base scan npz (dpt_<ih>_<ig>.npz) by taking the
elementwise minimum, then dissipative_PT_collect.py is re-run to rebuild the
summary npz and the colormap figure.

Usage:
    python scripts/dissipative_PT_refine_collect.py \\
        --in_dir results_dpt --out_png results_plots/dissipative_PT.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dissipative_PT import N_H, N_G

# Framability value key -> matching optimal-frame (S matrix) key.
FRA_KEYS = {'opt_fra_4': 'opt_S_4', 'opt_fra_6': 'opt_S_6'}
TOL = 1e-9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir',  type=str, default='results_dpt')
    p.add_argument('--out_png', type=str, default='results_plots/dissipative_PT.png')
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    n_improved = 0
    n_missing  = 0

    for ih in range(N_H):
        for ig in range(N_G):
            base = in_dir / f'dpt_{ih:02d}_{ig:02d}.npz'
            ref  = in_dir / f'dpt_refine_{ih:02d}_{ig:02d}.npz'
            if not base.exists():
                continue
            if not ref.exists():
                n_missing += 1
                continue

            b = dict(np.load(base))
            r = np.load(ref)
            changed = False
            for key, s_key in FRA_KEYS.items():
                if key not in r or key not in b:
                    continue
                rv = float(r[key])
                if np.isfinite(rv) and rv < float(b[key]) - TOL:
                    old = float(b[key])
                    b[key] = np.array(rv)
                    if s_key in r:                       # carry the matching frame
                        b[s_key] = np.asarray(r[s_key])
                    changed = True
                    n_improved += 1
                    print(f'  ({ih:2d},{ig:2d}) {key}: {old:.6f} → {rv:.6f}  '
                          f'(Δ = {old - rv:.6f})', flush=True)
            if changed:
                np.savez(base, **b)

    print(f'\nUpdated {n_improved} framability value(s).', flush=True)
    if n_missing:
        print(f'({n_missing} grid point(s) had no refine file — left unchanged)',
              flush=True)

    print('\nRegenerating summary npz and figure ...', flush=True)
    script = Path(__file__).resolve().parent / 'dissipative_PT_collect.py'
    subprocess.check_call([
        sys.executable, str(script),
        '--in_dir',  args.in_dir,
        '--out_png', args.out_png,
        '--save_npz',
    ])
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
