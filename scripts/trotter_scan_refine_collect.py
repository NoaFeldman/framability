"""
Merge Trotter-scan refinement rounds back into the base scan and regenerate the
figure.

For each point and each refined framability key (opt_fra_4 / opt_fra_6), the
elementwise minimum over the base value and every refine round 1..max_round is
written back into the base npz (carrying the matching optimal frame), then
trotter_scan_collect.py is re-run to rebuild the summary and the colormap.

Usage (after all refine rounds finished):
    python scripts/trotter_scan_refine_collect.py --model model1 \
        --in_dir results_trotter_v3 --max_round 6
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, FRA_REFINE_KEYS

TOL = 1e-9


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',     type=str, required=True, choices=list(MODELS))
    p.add_argument('--in_dir',    type=str, default='results_trotter_v3')
    p.add_argument('--out_png',   type=str, default=None)
    p.add_argument('--max_round', type=int, default=6)
    args = p.parse_args()

    model = MODELS[args.model]
    mdir = Path(args.in_dir) / model.name
    n_improved = 0

    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            base = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not base.exists():
                continue
            b = dict(np.load(base))
            changed = False
            for rnd in range(1, args.max_round + 1):
                ref = mdir / f'pt_refine_r{rnd:02d}_{ix:03d}_{iy:03d}.npz'
                if not ref.exists():
                    continue
                r = np.load(ref)
                for key, s_key in FRA_REFINE_KEYS.items():
                    if key not in r or key not in b:
                        continue
                    rv = float(r[key])
                    if np.isfinite(rv) and rv < float(b[key]) - TOL:
                        old = float(b[key])
                        b[key] = np.array(rv)
                        if s_key in r:
                            b[s_key] = np.asarray(r[s_key])
                        changed = True
                        n_improved += 1
                        print(f'  ({ix:3d},{iy:3d}) r{rnd} {key}: '
                              f'{old:.6f} -> {rv:.6f}', flush=True)
            if changed:
                np.savez(base, **b)

    print(f'\nUpdated {n_improved} framability value(s).', flush=True)

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
