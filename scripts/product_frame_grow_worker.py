"""
Stage 2 of the product-frame growth pipeline: the framability of every rung.

ONE WORK UNIT = ONE (case, round, gamma' variant, target variant), i.e. one
Schroedinger framability LP over D = kron(S, S) for the frame that stage 1
(scripts/product_frame_grow_frames_worker.py) grew.  The units are completely
independent, so the flat list is strided over the array: with 7 cases x ~9
rounds x 2 gamma' = ~130 units (x2 if the 'free' reference is asked for too),
a 200-task array runs 1-2 units per task.  The list is sorted by d_ext
DESCENDING before striding, so each task pairs one expensive rung with one
cheap one and the array load balances.

The two gamma' variants are
    at : gamma' = J             -- the continuous_simulation.tex threshold
    lo : gamma' = GP_FACTOR * J -- the detuned set (0.99 J by default)
and the target variants are 'plain' (the default: the bare Euler gate
rho -> rho + dt L(rho), which the U-rotated frame elements of stage 1 are built
for) and 'free' (an optional dt -> 0 reference; see product_frame_grow).

Every unit also records negativity_floor(Y): a certified lower bound on the
framability that no frame can beat, because the Euler step is not a positive
map.  A value sitting on its floor means dt is too coarse.

Output: <out_dir>/<tag>/fra_r<r:03d>_<gp>_<field>.npz
Idempotent: a unit whose file already holds a finite framability at this
GROW_VERSION is skipped, so resubmitting the array fills exactly the holes.

Usage:
    python scripts/product_frame_grow_worker.py --task_id 0 --n_chunks 200
    python scripts/product_frame_grow_worker.py --task_id 0 --n_chunks 1   # all
    python scripts/product_frame_grow_worker.py --task_id 0 --n_chunks 1 \
        --fields plain free                      # with the reference variant
"""

from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from product_frame_grow import (CASES, CASE_BY_TAG, FIELDS, GP_FACTOR_DEFAULT,  # noqa: E402
                                GROW_VERSION, build_targets, frame_matrix,
                                framability_targets, model_h, negativity_floor)
from product_frame_grow_frames_worker import (OUT_DIR_DEFAULT,               # noqa: E402
                                              load_frames)

GP_VARIANTS = ('at', 'lo')       # gamma' = J  and  gamma' = GP_FACTOR * J
STORE_COLUMNS_MAX_DEXT = 32      # keep the per-column gauges for small frames only


def unit_path(out_dir, tag: str, r: int, gp: str, field: str) -> Path:
    return Path(out_dir) / tag / f'fra_r{r:03d}_{gp}_{field}.npz'


def gamma_p_of(variant: str, J: float, gp_factor: float) -> float:
    if variant == 'at':
        return float(J)
    if variant == 'lo':
        return float(gp_factor) * float(J)
    raise ValueError(f'gamma_p variant must be one of {GP_VARIANTS}, got {variant!r}')


def work_list(out_dir, tags=None, fields=('plain',), gps=GP_VARIANTS) -> list:
    """Flat unit list [(tag, round, d_ext, gp, field)], most expensive first.

    Only cases whose stage-1 ladder exists contribute; a missing ladder is
    reported and skipped (rerun stage 1 for it).
    """
    units = []
    for case in CASES:
        tag = case['tag']
        if tags and tag not in tags:
            continue
        lad = load_frames(out_dir, tag)
        if lad is None:
            print(f'  warning: no ladder for {tag} -- run stage 1 first', flush=True)
            continue
        for r, d_ext in enumerate(lad['d_exts']):
            for gp in gps:
                for field in fields:
                    units.append((tag, r, int(d_ext), gp, field))
    units.sort(key=lambda u: (-u[2], u[0], u[3], u[4]))
    return units


def _done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        d = np.load(path, allow_pickle=True)
    except Exception:
        return False
    if 'framability' not in d.files or 'version' not in d.files:
        return False
    if str(np.asarray(d['version']).ravel()[0]) != GROW_VERSION:
        return False
    return bool(np.isfinite(float(np.asarray(d['framability']).ravel()[0])))


def _save_atomic(path: Path, data: dict, task_id: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.tmp_{path.stem}_{task_id}_{os.getpid()}.npz')
    np.savez(tmp, **data)
    os.replace(tmp, path)


def run_unit(out_dir, tag: str, r: int, gp: str, field: str, gp_factor: float,
             task_id: int = 0, force: bool = False) -> None:
    path = unit_path(out_dir, tag, r, gp, field)
    if not force and _done(path):
        print(f'  [{tag} r{r:03d} {gp} {field}] already done -- skipping', flush=True)
        return

    lad = load_frames(out_dir, tag)
    if lad is None:
        raise RuntimeError(f'no stage-1 ladder for {tag} in {out_dir}')
    if r >= len(lad['frames']):
        raise RuntimeError(f'{tag}: round {r} beyond the ladder '
                           f'({len(lad["frames"])} frames)')

    case = CASE_BY_TAG[tag]
    J, gamma = float(case['J']), float(case['gamma'])
    h = model_h(case['model'])
    gamma_p = gamma_p_of(gp, J, gp_factor)
    dt = float(np.asarray(lad['dt']).ravel()[0])
    gate_kind = str(np.asarray(lad['gate_kind']).ravel()[0])
    blochs = list(np.asarray(lad['frames'][r], dtype=float).T)
    d_ext = len(blochs)

    print(f'  [{tag} r{r:03d} {gp} {field}] d_ext={d_ext} '
          f"gamma'={gamma_p:g} dt={dt:g} gate={gate_kind}: {d_ext ** 2} column LPs",
          flush=True)
    t0 = time.perf_counter()
    Y = build_targets(blochs, J, gamma, h, gamma_p, dt, field=field,
                      gate_kind=gate_kind)
    # Certified lower bound on f for ANY frame (the Euler step is not positive):
    # cheap, so it is recorded next to every value and drawn by the collect
    # script.  f sitting on the floor means dt is too coarse, not the frame.
    floor, floor_col = negativity_floor(Y)
    f, cols = framability_targets(np.kron(frame_matrix(blochs),
                                          frame_matrix(blochs)), Y)
    el = time.perf_counter() - t0
    if np.isfinite(f) and f < floor - 1e-7 * max(1.0, floor):
        print(f'  WARNING framability {f:.12f} below its certified floor '
              f'{floor:.12f} -- LP tolerance or a bug', flush=True)

    data = dict(framability=np.array(f), rate=np.array((f - 1.0) / dt),
                floor=np.array(floor), floor_rate=np.array((floor - 1.0) / dt),
                floor_col=np.array(int(floor_col)),
                tag=np.array(tag), model=np.array(case['model']),
                round=np.array(r), d_ext=np.array(d_ext), gp_variant=np.array(gp),
                field=np.array(field), J=np.array(J), gamma=np.array(gamma),
                h=np.array(h), gamma_p=np.array(gamma_p), gp_factor=np.array(gp_factor),
                dt=np.array(dt), gate_kind=np.array(gate_kind),
                bloch=np.asarray(lad['frames'][r], dtype=float),
                col_argmax=np.array(int(np.argmax(cols))),
                col_min=np.array(float(np.min(cols))),
                col_mean=np.array(float(np.mean(cols))),
                col_median=np.array(float(np.median(cols))),
                elapsed=np.array(el), version=np.array(GROW_VERSION))
    if d_ext <= STORE_COLUMNS_MAX_DEXT:
        data['col_vals'] = np.asarray(cols, dtype=float)
    _save_atomic(path, data, task_id)
    print(f'  [{tag} r{r:03d} {gp} {field}] f = {f:.12f}  rate = {(f - 1.0) / dt:.6e}'
          f'  (floor rate {(floor - 1.0) / dt:.6e})  ({el:.1f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=200,
                   help='number of array tasks the unit list is strided over')
    p.add_argument('--out_dir', type=str, default=OUT_DIR_DEFAULT)
    p.add_argument('--tags', type=str, nargs='*', default=None,
                   help='restrict to these case tags (default: all seven)')
    p.add_argument('--fields', type=str, nargs='*', default=['plain'],
                   choices=list(FIELDS),
                   help="'plain' (default) is the requested Euler-gate "
                        "framability; add 'free' for the dt -> 0 reference")
    p.add_argument('--gps', type=str, nargs='*', default=list(GP_VARIANTS),
                   choices=list(GP_VARIANTS))
    p.add_argument('--gp_factor', type=float, default=GP_FACTOR_DEFAULT,
                   help="gamma'/J of the detuned ('lo') parameter set")
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    n = max(args.n_chunks, 1)
    if not (0 <= args.task_id < n):
        print(f'ERROR: task_id must be in [0, {n})', file=sys.stderr)
        sys.exit(1)

    units = work_list(args.out_dir, args.tags, tuple(args.fields), tuple(args.gps))
    mine = units[args.task_id::n]
    print(f'[chunk {args.task_id}/{n}] {len(units)} units total, {len(mine)} mine: '
          f'{[(t, r, d, g, f) for t, r, d, g, f in mine]}', flush=True)
    t0 = time.perf_counter()
    for i, (tag, r, _d, gp, field) in enumerate(mine, start=1):
        run_unit(args.out_dir, tag, r, gp, field, args.gp_factor,
                 task_id=args.task_id, force=args.force)
        print(f'[chunk {args.task_id}] {i}/{len(mine)} units, '
              f'elapsed {time.perf_counter() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
