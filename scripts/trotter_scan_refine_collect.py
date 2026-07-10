"""
Merge Trotter-scan refinement rounds back into the base scan and regenerate the
figure.

For each point and each refined framability key (opt_fra_4 / opt_fra_6), the
elementwise minimum over the base value and every refine file (full rounds
pt_refine_r*, quick rounds pt_qrefine_r*, cross-eval sweeps pt_xeval_r*,
island polishes pt_polish_r* and floor hunts pt_fhunt_r*) is written back
into the base npz (carrying the matching optimal frame), then
trotter_scan_collect.py is re-run to rebuild the summary and the colormap.

Usage (after all refine rounds finished):
    python scripts/trotter_scan_refine_collect.py --model model1 \
        --in_dir results_trotter_v3 --max_round 6
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, FRA_REFINE_KEYS
from trotter_refine_common import REFINE_PATTERNS

TOL = 1e-9

# 'pt_<kind>_r*' -> '<kind>', in registry (= merge) order
_KINDS = [p[len('pt_'):-len('_r*')] for p in REFINE_PATTERNS]
_FILE_RE = re.compile(
    r'^pt_(' + '|'.join(_KINDS) + r')_r.*_(\d{3})_(\d{3})\.npz$')


def _refine_files_by_point(mdir: Path) -> dict:
    """One directory listing -> {(ix, iy): [refinement files]} in the same
    order the old per-point globs produced (REFINE_PATTERNS order, each
    flavour sorted by name).  A single scan replaces
    n_points * len(REFINE_PATTERNS) whole-directory globs, which dominate
    the runtime on parallel filesystems."""
    by_point: dict = {}
    for p in mdir.iterdir():
        m = _FILE_RE.match(p.name)
        if m:
            key = (int(m.group(2)), int(m.group(3)))
            by_point.setdefault(key, []).append(
                (_KINDS.index(m.group(1)), p.name, p))
    return {k: [p for _, _, p in sorted(v)] for k, v in by_point.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',     type=str, required=True, choices=list(MODELS))
    p.add_argument('--in_dir',    type=str, default='results_trotter_v3')
    p.add_argument('--out_png',   type=str, default=None)
    p.add_argument('--max_round', type=int, default=None,
                   help='unused (kept for compatibility; all rounds are globbed)')
    args = p.parse_args()

    model = MODELS[args.model]
    mdir = Path(args.in_dir) / model.name
    n_improved = 0

    print(f'Indexing refinement files in {mdir}/ ...', flush=True)
    refine_map = _refine_files_by_point(mdir)
    print(f'{sum(len(v) for v in refine_map.values())} refinement file(s) '
          f'for {len(refine_map)} point(s).', flush=True)

    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            refs = refine_map.get((ix, iy), [])
            if not refs:
                continue
            base = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not base.exists():
                continue
            b = dict(np.load(base))
            changed = False
            for ref in refs:
                try:
                    r = np.load(ref)
                except Exception:
                    print(f'  WARNING: unreadable {ref.name} — skipped', flush=True)
                    continue
                for key, s_key in FRA_REFINE_KEYS.items():
                    if key not in r:
                        continue
                    rv = float(r[key])
                    if not np.isfinite(rv):
                        continue
                    old = float(b[key]) if key in b else float('nan')
                    # Missing/NaN base value (failed base-scan point): any
                    # finite refinement wins, matching the progress plot.
                    bv = old if np.isfinite(old) else np.inf
                    if rv < bv - TOL:
                        b[key] = np.array(rv)
                        if s_key in r:
                            b[s_key] = np.asarray(r[s_key])
                        changed = True
                        n_improved += 1
                        print(f'  ({ix:3d},{iy:3d}) {ref.name} {key}: '
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
