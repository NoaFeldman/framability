"""
Per-base cluster worker: framability vs Trotter DT_BASE for one model point.

For a fixed model and (p1, p2) parameter point (p1 the x-axis parameter, p2 the
y-axis parameter of trotter_lindbladian_scan.MODELS), the Trotter-step control
`base` is swept over

    base in [1e-2 * i for i in range(1, 100)]        # 0.01 .. 0.99  (99 points)

For each base the two-qubit bond Trotter gate is expm(L_bond * dt) with the
per-point adaptive step

    dt = base / max(||H||_1, {gamma_k})              # choose_dt(..., base=base)

(the edited line DT_BASE / max(||H||_1, {gamma_k}) with DT_BASE = base), and the
following five framabilities are evaluated on that 16x16 gate:

    stab_fra   stabilizer-3 framability          framability.stabilizer_3_framability
    pauli_fra  Pauli framability                 dissipative_PT.pauli_framability
    opt_fra_4  opt Heisenberg framability, d_ext_single=4  dissipative_PT.optimise_framability
    opt_fra_6  opt Heisenberg framability, d_ext_single=6  dissipative_PT.optimise_framability
    gamma_ch1  max Janek (product-frame gauge)   trotter_lindbladian_scan.gamma_ch1_framability

The optimiser seed is held fixed across the whole base sweep (identical restart
set at every base) so the resulting curves are smooth.

Output: <out_dir>/<tag>/base_<idx:03d>.npz     tag = point_tag(model, p1, p2)

Usage (single base point):
    python scripts/trotter_dtbase_line_worker.py --model model3 --p1 3 --p2 3 --task_id 0

Usage (strided across an array):
    python scripts/trotter_dtbase_line_worker.py --model model3 --p1 3 --p2 3 \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 99
"""

from __future__ import annotations

import os
# Single-thread the BLAS/LP backend so the SLURM array (not nested threads) owns
# the parallelism (one cpu-per-task; avoids oversubscription).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (
    MODELS, bond_trotter_gate, choose_dt, gamma_ch1_framability, DIM_DEFAULT,
)
from framability import stabilizer_3_framability
from dissipative_PT import pauli_framability, optimise_framability

# Bump when the base grid or the set/definition of stored quantities changes so
# workers re-run stale npz files.
DTBASE_LINE_VERSION = '1.0'

# (key, human label) in figure order.  Shared by the collect/plot script.
MEASURES = [
    ('stab_fra',  'Stabilizer-3 framability'),
    ('pauli_fra', 'Pauli framability'),
    ('opt_fra_4', 'Opt Heisenberg framability (d=4)'),
    ('opt_fra_6', 'Opt Heisenberg framability (d=6)'),
    ('gamma_ch1', 'max Janek'),
]


def base_grid() -> np.ndarray:
    """The scanned DT_BASE values: [1e-2 * i for i in range(1, 100)]."""
    return np.array([1e-2 * i for i in range(1, 100)], dtype=float)


N_BASE = len(base_grid())          # 99


def point_tag(model: str, p1: float, p2: float) -> str:
    """Filesystem-safe tag identifying a (model, p1, p2) point (collision-free
    across the parameter values a user may pick)."""
    def f(v: float) -> str:
        return format(float(v), '.4f').replace('-', 'm').replace('.', 'p')
    return f'{model}_p1_{f(p1)}_p2_{f(p2)}'


def _is_current(out: Path) -> bool:
    """True iff `out` exists and carries the current code version."""
    if not out.exists():
        return False
    try:
        d = np.load(out, allow_pickle=True)
        return 'code_version' in d and str(d['code_version']) == DTBASE_LINE_VERSION
    except Exception:
        return False


def compute_base(model_name: str, p1: float, p2: float, base: float, *,
                 dim: int, fra_restarts: int, fra_maxfev_4: int,
                 fra_maxfev_6: int, ch1_restarts: int, seed: int) -> dict:
    """The five framabilities of the model's bond Trotter gate at DT_BASE=base."""
    m = MODELS[model_name]
    H1, H2, j1, j2 = m.build(p1, p2)
    dt = choose_dt(H1, H2, j1, j2, base=base)      # base / max(||H||_1, {gamma_k})
    gate = bond_trotter_gate(H1, H2, j1, j2, dim, dt)

    out: dict = dict(base=base, dt=dt)
    out['stab_fra'] = stabilizer_3_framability(gate)                         # c1
    out['pauli_fra'] = pauli_framability(gate)                              # c2
    out['opt_fra_4'] = optimise_framability(gate, 4, fra_restarts,
                                            fra_maxfev_4, seed)             # d (H, d=4)
    out['opt_fra_6'] = optimise_framability(gate, 6, fra_restarts,
                                            fra_maxfev_6, seed + 1)         # d (H, d=6)
    out['gamma_ch1'] = gamma_ch1_framability(gate, ch1_restarts, seed)     # d3 max Janek
    return out


def run_point(args, base_idx: int) -> None:
    """Compute and save one base point (skips if already current)."""
    base_vals = base_grid()
    base = float(base_vals[base_idx])
    tag = point_tag(args.model, args.p1, args.p2)
    out_dir = Path(args.out_dir) / tag
    out = out_dir / f'base_{base_idx:03d}.npz'

    if _is_current(out):
        print(f'[skip] {tag}/{out.name} already at version {DTBASE_LINE_VERSION}',
              flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'[base {base_idx}/{N_BASE}] {args.model} '
          f'{MODELS[args.model].p1_name}={args.p1:.4f} '
          f'{MODELS[args.model].p2_name}={args.p2:.4f}  base={base:.4f}  '
          f'dim={args.dim}', flush=True)

    res = compute_base(args.model, args.p1, args.p2, base, dim=args.dim,
                       fra_restarts=args.fra_restarts,
                       fra_maxfev_4=args.fra_maxfev_4,
                       fra_maxfev_6=args.fra_maxfev_6,
                       ch1_restarts=args.ch1_restarts, seed=args.seed)

    save = {k: np.array(res[k]) for k, _ in MEASURES}
    save.update(
        base=np.array(base), dt=np.array(res['dt']),
        base_idx=np.array(base_idx),
        p1=np.array(args.p1), p2=np.array(args.p2),
        dim=np.array(args.dim), model=np.array(args.model),
        seed=np.array(args.seed),
        code_version=np.array(DTBASE_LINE_VERSION),
    )
    np.savez(out, **save)
    print(f'  saved {tag}/{out.name}  '
          f'stab={res["stab_fra"]:.4f} pauli={res["pauli_fra"]:.4f} '
          f'd4={res["opt_fra_4"]:.4f} d6={res["opt_fra_6"]:.4f} '
          f'janek={res["gamma_ch1"]:.4f}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, required=True, choices=list(MODELS))
    p.add_argument('--p1',       type=float, required=True,
                   help='x-axis parameter value (model.p1_name)')
    p.add_argument('--p2',       type=float, required=True,
                   help='y-axis parameter value (model.p2_name)')
    p.add_argument('--task_id',  type=int, required=True,
                   help='base index when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the 99 base points into this many strided array tasks')
    p.add_argument('--out_dir',  type=str, default='results_dtbase_line')
    p.add_argument('--dim',      type=int, default=DIM_DEFAULT, choices=(1, 2, 3))
    p.add_argument('--fra_restarts', type=int, default=5)
    p.add_argument('--fra_maxfev_4', type=int, default=1000)
    p.add_argument('--fra_maxfev_6', type=int, default=500)
    p.add_argument('--ch1_restarts', type=int, default=15)
    p.add_argument('--seed',     type=int, default=0,
                   help='optimiser seed, held fixed across the base sweep')
    args = p.parse_args()

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N_BASE):
            print(f'ERROR: task_id must be in [0, {N_BASE})', file=sys.stderr)
            sys.exit(1)
        run_point(args, args.task_id)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    base_ids = list(range(args.task_id, N_BASE, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {args.model}: '
          f'{len(base_ids)} base points', flush=True)
    for bid in base_ids:
        run_point(args, bid)


if __name__ == '__main__':
    main()
