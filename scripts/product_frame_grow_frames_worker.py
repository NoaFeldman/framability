"""
Stage 1 of the product-frame growth pipeline: build the frame LADDER.

ONE ARRAY TASK = ONE PARAMETER SET (7 of them, see product_frame_grow.CASES),
because the rounds of a ladder are strictly sequential -- round r+1 draws its
candidates from the frame of round r.  The expensive, embarrassingly parallel
part (one framability LP per (case, round, gamma', field)) is stage 2,
scripts/product_frame_grow_worker.py.

The growth step always runs at gamma' = J, where continuous_simulation.tex
guarantees the free-local-unitary Euler step of every product frame element is
separable, so the ladder is shared by both evaluated parameter sets
(gamma' = J and gamma' = GP_FACTOR * J).  See product_frame_grow for the
extraction itself.

Output: <out_dir>/<tag>/frames.npz
    d_exts          (n_round+1,)  the d_ext ladder, starting at 6
    frame_<r:03d>   (3, d_ext)    Bloch vectors of the frame after round r
    rounds_json     JSON string with the per-round diagnostics (candidate
                    counts, which filter was used, gauge_max, lambda_-, and the
                    frame_element_criterion verdicts)
    plus the parameters and GROW_VERSION.

Idempotent: an existing file with the same version and the same growth
parameters is kept unless --force is given.

Usage:
    python scripts/product_frame_grow_frames_worker.py --task_id 0
    python scripts/product_frame_grow_frames_worker.py --task_id 3 --filter gauge
"""

from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from product_frame_grow import (CASES, CASE_BY_TAG, CRITERION_MAX_DEXT_DEFAULT,  # noqa: E402
                                DT_DEFAULT, D_EXT_MAX_DEFAULT, GROW_VERSION,
                                grow_frames)

OUT_DIR_DEFAULT = 'results_product_frame_grow'

# The growth parameters an existing frames file must agree on to be reused.
_KEYS = ('dt', 'd_ext_max', 'max_new_per_round', 'filter', 'criterion_max_dext',
         'accept', 'gate_kind', 'version')


def frames_path(out_dir, tag: str) -> Path:
    return Path(out_dir) / tag / 'frames.npz'


def save_frames(path: Path, grown: dict, meta: dict, task_id: int = 0) -> None:
    """Write the ladder atomically (temp name carries the task id and pid)."""
    data = {f'frame_{r:03d}': f for r, f in enumerate(grown['frames'])}
    data['d_exts'] = np.array(grown['d_exts'], dtype=int)
    data['rounds_json'] = np.array(json.dumps(grown['rounds'], default=float))
    for k, v in meta.items():
        data[k] = np.array(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.tmp_{path.stem}_{task_id}_{os.getpid()}.npz')
    np.savez(tmp, **data)
    os.replace(tmp, path)


def load_frames(out_dir, tag: str) -> dict | None:
    """Read a ladder written by save_frames, or None if it is missing/unreadable."""
    path = frames_path(out_dir, tag)
    if not path.exists():
        return None
    try:
        d = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f'  warning: unreadable {path} ({e})', flush=True)
        return None
    d_exts = [int(x) for x in np.asarray(d['d_exts']).ravel()]
    out = {k: d[k] for k in d.files if not k.startswith('frame_')}
    out['d_exts'] = d_exts
    out['frames'] = [np.asarray(d[f'frame_{r:03d}'], dtype=float)
                     for r in range(len(d_exts))]
    out['rounds'] = json.loads(str(d['rounds_json'])) if 'rounds_json' in d.files else []
    return out


def _meta(args) -> dict:
    return dict(dt=args.dt, d_ext_max=args.d_ext_max,
                max_new_per_round=args.max_new_per_round, filter=args.filter,
                criterion_max_dext=args.criterion_max_dext, accept=args.accept,
                gate_kind=args.gate, version=GROW_VERSION)


def _up_to_date(existing: dict | None, meta: dict) -> bool:
    if existing is None:
        return False
    for k in _KEYS:
        if k not in existing:
            return False
        old, new = np.asarray(existing[k]).ravel()[0], meta[k]
        if isinstance(new, str):
            if str(old) != new:
                return False
        elif not np.isclose(float(old), float(new)):
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, required=True,
                   help=f'index into product_frame_grow.CASES (0..{len(CASES) - 1})')
    p.add_argument('--out_dir', type=str, default=OUT_DIR_DEFAULT)
    p.add_argument('--dt', type=float, default=DT_DEFAULT)
    p.add_argument('--d_ext_max', type=int, default=D_EXT_MAX_DEFAULT)
    p.add_argument('--max_new_per_round', type=int, default=12,
                   help='per-round budget of new frame elements; a group is six '
                        'states and is atomic, so a round may overshoot slightly')
    p.add_argument('--filter', type=str, default='criterion',
                   choices=('criterion', 'gauge'),
                   help="'criterion' = product_frame_trick.frame_element_criterion "
                        "on the shortlist (exact, but only up to "
                        "--criterion_max_dext); 'gauge' = the cheap "
                        'gate-independent redundancy test only')
    p.add_argument('--criterion_max_dext', type=int,
                   default=CRITERION_MAX_DEXT_DEFAULT,
                   help='above this d_ext the criterion filter degrades to gauge '
                        "(product_frame_trick._min_l1's dense epigraph block is "
                        'O(d_ext^4) in memory); the choice is logged per round')
    p.add_argument('--accept', type=str, default='nonharmful',
                   choices=('nonharmful', 'useful'))
    p.add_argument('--gate', type=str, default='euler', choices=('euler', 'expm'))
    p.add_argument('--max_rounds', type=int, default=60)
    p.add_argument('--force', action='store_true',
                   help='regrow even if an up-to-date frames file exists')
    args = p.parse_args()

    if not (0 <= args.task_id < len(CASES)):
        print(f'ERROR: task_id must be in [0, {len(CASES)})', file=sys.stderr)
        sys.exit(1)
    case = CASES[args.task_id]
    tag = case['tag']
    path = frames_path(args.out_dir, tag)
    meta = _meta(args)

    if not args.force and _up_to_date(load_frames(args.out_dir, tag), meta):
        existing = load_frames(args.out_dir, tag)
        print(f'[{tag}] up-to-date ladder already present: '
              f'd_exts={existing["d_exts"]} -- nothing to do', flush=True)
        return

    print(f'[{tag}] growing: model={case["model"]} J={case["J"]} '
          f'gamma={case["gamma"]} gamma_p=J dt={args.dt} '
          f'filter={args.filter} (criterion up to d_ext {args.criterion_max_dext}) '
          f'budget={args.max_new_per_round}/round -> d_ext >= {args.d_ext_max}',
          flush=True)
    t0 = time.perf_counter()
    grown = grow_frames(case, dt=args.dt, d_ext_max=args.d_ext_max,
                        max_new_per_round=args.max_new_per_round,
                        filter_mode=args.filter,
                        criterion_max_dext=args.criterion_max_dext,
                        accept=args.accept, gate_kind=args.gate,
                        max_rounds=args.max_rounds)
    save_frames(path, grown, dict(meta, tag=tag, model=case['model'], J=case['J'],
                                  gamma=case['gamma'], h=grown['h'],
                                  gamma_p_grow=grown['gamma_p_grow']),
                task_id=args.task_id)
    print(f'[{tag}] done in {time.perf_counter() - t0:.0f}s: '
          f'{len(grown["frames"])} frames, d_exts={grown["d_exts"]} -> {path}',
          flush=True)


if __name__ == '__main__':
    main()
