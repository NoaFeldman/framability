"""
Merge the sign-problem refinement rounds back into the base Trotter-scan npz
files and regenerate the figure.

For each point, sign_opt in the base npz is replaced by the maximum over the
base value and every sign-refine round 1..max_round (the winning rotation
vector is stored alongside as sign_n), then trotter_scan_collect.py is re-run.

Usage (after the sign-refine rounds finished):
    python scripts/trotter_sign_refine_collect.py --model model1 \
        --in_dir results_trotter --max_round 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS

TOL = 1e-9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',     type=str, required=True, choices=list(MODELS))
    p.add_argument('--in_dir',    type=str, default='results_trotter')
    p.add_argument('--out_png',   type=str, default=None)
    p.add_argument('--max_round', type=int, default=3)
    args = p.parse_args()

    model = MODELS[args.model]
    mdir = Path(args.in_dir) / model.name
    n_improved = 0

    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            base = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not base.exists():
                continue
            b = dict(np.load(base, allow_pickle=True))
            if 'sign_opt' not in b:
                continue
            changed = False
            for rnd in range(1, args.max_round + 1):
                ref = mdir / f'pt_sign_r{rnd:02d}_{ix:03d}_{iy:03d}.npz'
                if not ref.exists():
                    continue
                r = np.load(ref)
                if 'sign_opt' not in r:
                    continue
                rv = float(r['sign_opt'])
                if np.isfinite(rv) and rv > float(b['sign_opt']) + TOL:
                    old = float(b['sign_opt'])
                    b['sign_opt'] = np.array(rv)
                    if 'sign_n' in r:
                        b['sign_n'] = np.asarray(r['sign_n'])
                    changed = True
                    n_improved += 1
                    print(f'  ({ix:3d},{iy:3d}) r{rnd} sign_opt: '
                          f'{old:.6f} -> {rv:.6f}', flush=True)
            if changed:
                np.savez(base, **b)

    print(f'\nUpdated {n_improved} sign_opt value(s).', flush=True)

    print('\nRegenerating summary npz and figure ...', flush=True)
    cmd = [sys.executable,
           str(Path(__file__).resolve().parent / 'trotter_scan_collect.py'),
           '--model', model.name, '--in_dir', args.in_dir, '--save_npz']
    if args.out_png:
        cmd += ['--out_png', args.out_png]
    subprocess.check_call(cmd)
    print('Done.', flush=True)


if __name__ == '__main__':
    main()
