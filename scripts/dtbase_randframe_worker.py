"""
Random-frame framabilities along the DT_BASE line, for the dt->0 colormaps.

Adds six fixed-random-frame measures to results_dtbase_line/<model>_dtbase_extrap.png,
evaluated on the same two-qubit bond Trotter gate and the same DT_BASE line as
scripts/trotter_dtbase_line_worker.py (bottom N_BASE = 10 DT_BASE values,
dt = DT_BASE / max(||H||_1, {gamma_k})):

    prod_mix_{chi}    Schroedinger product-state framability, D = kron(D_1, D_1),
                      D_1 = chi random MIXED single-qubit states (Bloch vector
                      uniform in the ball, framability.make_product_state_D
                      mixed=True).  Evaluator: framability.product_state_framability.
    heis_unit_{chi}   Heisenberg framability for D = kron(S, S), S (4 x chi):
                      column 0 = identity (1,0,0,0), columns 1..chi-1 =
                      (0, n_j) with n_j uniform on the unit sphere, i.e. the
                      operator-norm constraint |c_I| + ||b||_2 <= 1 saturated.
    heis_rnorm_{chi}  Same S, identity still (1,0,0,0), but every non-identity
                      column is (0, r_j n_j) with r_j ~ U[SUPPORT_EPS, 1]
                      (optimize_framability.SUPPORT_EPS = 1e-2).  The directions
                      n_j are THE SAME as in heis_unit_{chi}, so the two differ
                      only by the norms.
                      Evaluator (both Heisenberg keys): dissipative_PT._framability_lp
                      (the certified reference LP, HiGHS ladder + full-support guard).

for chi in CHIS = (10, 40); chi counts every column of S, the identity included
(the d_ext_single convention of opt_fra_4/opt_fra_6).  Nothing is optimised: each
frame is drawn ONCE from a fixed seed and reused at every grid point and every
DT_BASE, as for prod_fra_10/prod_fra_40.

Only gamma' <= P2_MAX (default 4.2) is swept; the rest of the grid stays empty.

Work unit = one (grid point, DT_BASE index) pair.  The flat pair list is strided
over --n_chunks array tasks (task t gets pairs t, t+n, t+2n, ...), so every task
samples the whole grid and loads balance.  Each pair is owned by exactly one
task, so no two tasks ever write the same file.

Output: <out_dir>/<tag>/base_<idx:03d>.npz   tag = point_tag(model, p1, p2)
Idempotent per key: an existing file is loaded and only missing / non-finite
keys are computed, and the file is rewritten atomically after EVERY key (a task
killed by the time limit loses at most one measure) -- resubmitting the array
fills exactly the holes.

Usage:
    python scripts/dtbase_randframe_worker.py --task_id 0 --n_chunks 200
    python scripts/dtbase_randframe_worker.py --task_id 0 --n_chunks 1 --stride 5
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
from trotter_lindbladian_scan import (MODELS, bond_trotter_gate, choose_dt,  # noqa: E402
                                      DIM_DEFAULT, PROD_FRAME_SEED)
from framability import make_product_state_D, product_state_framability      # noqa: E402
from dissipative_PT import _framability_lp                                   # noqa: E402
from optimize_framability import SUPPORT_EPS                                 # noqa: E402
from trotter_dtbase_line_worker import base_grid, point_tag, N_BASE          # noqa: E402

RANDFRAME_VERSION = '1.0'
CHIS = (10, 40)
P2_MAX_DEFAULT = 4.2
OUT_DIR_DEFAULT = 'results_dtbase_randframe'

# (key, panel label) in figure order.  Shared with scripts/dtbase_randframe_collect.py.
MEASURES = (
    [(f'prod_mix_{chi}', rf'Mixed product-state framability ($\chi={chi}$)')
     for chi in CHIS]
    + [(f'heis_unit_{chi}',
        rf'Random Heisenberg framability, $\|O\|=1$ ($\chi={chi}$)')
       for chi in CHIS]
    + [(f'heis_rnorm_{chi}',
        rf'Random Heisenberg framability, $\|O\|\in[\epsilon,1]$ ($\chi={chi}$)')
       for chi in CHIS]
)


# ---------------------------------------------------------------------------
#  Fixed random frames
# ---------------------------------------------------------------------------
def _rng(kind: int, chi: int, seed: int) -> np.random.Generator:
    """Independent, reproducible stream per (frame family, chi)."""
    return np.random.default_rng([seed, kind, chi])


def random_heisenberg_S(chi: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(S_unit, S_rnorm), both 4 x chi.  Column 0 is the identity (1,0,0,0) in
    both; the chi-1 other columns are (0, n_j) resp. (0, r_j n_j), with the SAME
    isotropic unit directions n_j and r_j ~ U[SUPPORT_EPS, 1]."""
    rng = _rng(1, chi, seed)
    dirs = rng.standard_normal((3, chi - 1))
    dirs /= np.linalg.norm(dirs, axis=0, keepdims=True)
    radii = rng.uniform(SUPPORT_EPS, 1.0, size=chi - 1)
    ident = np.array([[1.0], [0.0], [0.0], [0.0]])
    S_unit = np.hstack([ident, np.vstack([np.zeros((1, chi - 1)), dirs])])
    S_rnorm = np.hstack([ident, np.vstack([np.zeros((1, chi - 1)), dirs * radii])])
    return S_unit, S_rnorm


def build_frames(seed: int) -> dict:
    """{key: 16 x chi^2 frame matrix} for every key of MEASURES."""
    frames = {}
    for chi in CHIS:
        frames[f'prod_mix_{chi}'] = make_product_state_D(chi, mixed=True,
                                                         rng=_rng(0, chi, seed))
        S_unit, S_rnorm = random_heisenberg_S(chi, seed)
        frames[f'heis_unit_{chi}'] = np.kron(S_unit, S_unit)
        frames[f'heis_rnorm_{chi}'] = np.kron(S_rnorm, S_rnorm)
    return frames


def evaluate(key: str, D: np.ndarray, gate: np.ndarray) -> float:
    if key.startswith('prod_mix_'):
        return float(product_state_framability(None, gate, D=D))
    return float(_framability_lp(D, gate))                 # Heisenberg: gate^T D


# ---------------------------------------------------------------------------
#  Work list
# ---------------------------------------------------------------------------
def grid_vals(model: str, stride: int, p2_max: float):
    m = MODELS[model]
    p1_vals = np.asarray(m.p1_vals[::stride], float)
    p2_vals = np.asarray(m.p2_vals[::stride], float)
    return p1_vals, p2_vals[p2_vals <= p2_max + 1e-9]


def work_list(model: str, stride: int, p2_max: float) -> list:
    """Flat [(p1, p2, base_idx)], ordered point by point."""
    p1_vals, p2_vals = grid_vals(model, stride, p2_max)
    return [(float(p1), float(p2), b)
            for p1 in p1_vals for p2 in p2_vals for b in range(N_BASE)]


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        d = np.load(path, allow_pickle=True)
        return {k: d[k] for k in d.files}
    except Exception as e:                   # truncated file: recompute from scratch
        print(f'  warning: unreadable {path} ({e}); recomputing', flush=True)
        return {}


def _missing(existing: dict) -> list:
    return [k for k, _ in MEASURES
            if k not in existing or not np.isfinite(np.asarray(existing[k], float))]


def _save_atomic(path: Path, data: dict, task_id: int) -> None:
    # Temp name does not match base_*.npz and carries the task id + pid.
    tmp = path.with_name(f'.tmp_{path.stem}_{task_id}_{os.getpid()}.npz')
    np.savez(tmp, **data)
    os.replace(tmp, path)


def run_pair(model: str, p1: float, p2: float, base_idx: int, frames: dict,
             args) -> None:
    tag = point_tag(model, p1, p2)
    out = Path(args.out_dir) / tag / f'base_{base_idx:03d}.npz'
    existing = _load(out)
    need = _missing(existing)
    if not need:
        return

    m = MODELS[model]
    base = float(base_grid()[base_idx])
    H1, H2, j1, j2 = m.build(p1, p2)
    dt = choose_dt(H1, H2, j1, j2, base=base)     # identical to the dtbase-line worker
    gate = bond_trotter_gate(H1, H2, j1, j2, args.dim, dt)

    out.parent.mkdir(parents=True, exist_ok=True)
    save = dict(existing)
    save.update(model=np.array(model), p1=np.array(p1), p2=np.array(p2),
                base_idx=np.array(base_idx), base=np.array(base),
                dt=np.array(dt), dim=np.array(args.dim),
                seed=np.array(args.seed), support_eps=np.array(SUPPORT_EPS),
                code_version=np.array(RANDFRAME_VERSION))
    print(f'[{tag} base {base_idx}] dt={dt:.5g}  need={need}', flush=True)
    for key in need:
        t0 = time.perf_counter()
        try:
            val = evaluate(key, frames[key], gate)
        except Exception as e:
            print(f'  ERROR {key}: {type(e).__name__}: {e}', flush=True)
            continue
        save[key] = np.array(val)
        save[f't_{key}'] = np.array(time.perf_counter() - t0)
        _save_atomic(out, save, args.task_id)
        print(f'  {key} = {val:.6f}  ({time.perf_counter() - t0:.1f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',  type=int, required=True)
    p.add_argument('--n_chunks', type=int, default=200,
                   help='number of array tasks the pair list is strided over')
    p.add_argument('--model',    type=str, default='model3', choices=list(MODELS))
    p.add_argument('--stride',   type=int, default=1,
                   help='grid stride (must match the collect step)')
    p.add_argument('--p2_max',   type=float, default=P2_MAX_DEFAULT,
                   help="largest gamma' (p2) value swept")
    p.add_argument('--out_dir',  type=str, default=OUT_DIR_DEFAULT)
    p.add_argument('--dim',      type=int, default=DIM_DEFAULT, choices=(1, 2, 3))
    p.add_argument('--seed',     type=int, default=PROD_FRAME_SEED,
                   help='seed of the fixed random frames (same for all tasks)')
    args = p.parse_args()

    if not (0 <= args.task_id < max(args.n_chunks, 1)):
        print(f'ERROR: task_id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    pairs = work_list(args.model, args.stride, args.p2_max)[args.task_id::max(args.n_chunks, 1)]
    frames = build_frames(args.seed)
    print(f'[chunk {args.task_id}/{args.n_chunks}] {args.model} stride={args.stride} '
          f"gamma'<={args.p2_max}: {len(pairs)} (point, DT_BASE) pairs, "
          f'keys={[k for k, _ in MEASURES]}', flush=True)
    t0 = time.perf_counter()
    for i, (p1, p2, b) in enumerate(pairs, start=1):
        run_pair(args.model, p1, p2, b, frames, args)
        if i % 20 == 0 or i == len(pairs):
            el = time.perf_counter() - t0
            print(f'[chunk {args.task_id}] {i}/{len(pairs)} pairs  elapsed {el:.0f}s  '
                  f'eta {el / i * (len(pairs) - i):.0f}s', flush=True)


if __name__ == '__main__':
    main()
