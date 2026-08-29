"""
Per-base cluster worker: framability vs Trotter DT_BASE for one model point.

For a fixed model and (p1, p2) parameter point (p1 the x-axis parameter, p2 the
y-axis parameter of trotter_lindbladian_scan.MODELS), the Trotter-step control
`base` is swept over the bottom N_BASE_KEEP values of the historical grid

    base_grid_full() = [1e-2 * i for i in range(1, 100)]   # 0.01 .. 0.99  (99 pts)
    base_grid()       = base_grid_full()[:N_BASE_KEEP]     # 0.01 .. 0.10  (10 pts)

(the DT_BASE range used by every dt->0 extrapolation plot in this pipeline is
restricted to these bottom 10 values -- large DT_BASE only ever entered the
figures through the discarded tail of the fit anyway).  For each base the
two-qubit bond Trotter gate is expm(L_bond * dt) with the per-point adaptive step

    dt = base / max(||H||_1, {gamma_k})              # choose_dt(..., base=base)

and the following seven framabilities are evaluated on that 16x16 gate:

    stab_fra    stabilizer-3 framability                framability.stabilizer_3_framability
    pauli_fra   Pauli framability                        dissipative_PT.pauli_framability
    opt_fra_4   opt Heisenberg framability, d_ext=4       dissipative_PT.optimise_framability
    opt_fra_6   opt Heisenberg framability, d_ext=6       dissipative_PT.optimise_framability
    gamma_ch1   max Janek (product-frame gauge)          trotter_lindbladian_scan.gamma_ch1_framability
    sch_fra_6   optimal-Schroedinger framability, d_ext=6 optimize_framability.minimize_schroedinger_framability
    prod_fra_10 product-state framability, chi=10        framability.product_state_framability

The optimiser seed is held fixed across the whole base sweep (identical restart
set at every base) so the resulting curves are smooth.

Idempotency is per-KEY, not per-file: a worker run loads any existing
base_<idx>.npz, computes only the MEASURES keys still missing from it, and
merges the result back in (_missing_keys/_load_existing).  So re-running this
worker after MEASURES grows (as it just did, +sch_fra_6/+prod_fra_10) backfills
only the new keys everywhere -- it never recomputes stab_fra/pauli_fra/
opt_fra_4/opt_fra_6/gamma_ch1 for a point that already has them.

Output: <out_dir>/<tag>/base_<idx:03d>.npz     tag = point_tag(model, p1, p2)

Usage (single base point):
    python scripts/trotter_dtbase_line_worker.py --model model3 --p1 3 --p2 3 --task_id 0

Usage (strided across an array):
    python scripts/trotter_dtbase_line_worker.py --model model3 --p1 3 --p2 3 \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 10
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
    PROD_FRAME_SEED,
)
from framability import stabilizer_3_framability, product_state_framability
from dissipative_PT import (
    pauli_framability, optimise_framability, embed_frame_params,
)
from optimize_framability import minimize_schroedinger_framability

# Bump for provenance whenever the *definition* of an existing key changes (a
# key's presence/absence in a saved npz is what actually gates recomputation --
# see _missing_keys -- so bumping this alone does not force a re-run; it is
# metadata only).
DTBASE_LINE_VERSION = '1.1'

# d_ext for "optimal-schroedinger-framability" (item 2) and chi for
# "product-state-framability" (item 3), per spec.
SCH_D_EXT = 6
PROD_CHI = 10

# (key, human label) in figure order.  Shared by the collect/plot script.
# sch_fra_6 and prod_fra_10 were added alongside the original five; existing
# base_*.npz files simply lack those keys until a worker run backfills them
# (see _missing_keys / run_point), which never touches the five original keys.
MEASURES = [
    ('stab_fra',   'Stabilizer-3 framability'),
    ('pauli_fra',  'Pauli framability'),
    ('opt_fra_4',  'Opt Heisenberg framability (d=4)'),
    ('opt_fra_6',  'Opt Heisenberg framability (d=6)'),
    ('gamma_ch1',  'max Janek'),
    ('sch_fra_6',  'Optimal Schrödinger framability (d=6)'),
    ('prod_fra_10', r'Product-state framability ($\chi=10$)'),
]

# Keys whose optimal frame (as a flat param vector, see dissipative_PT.
# params_from_frame/frame_from_params) is also stored, so a later
# neighbor-seeded refine pass (scripts/trotter_dtbase_line_quick_refine_worker.py)
# can warm-start from it instead of re-optimising from scratch.
FRAME_KEYS = {'opt_fra_4': ('opt_x_4', 4), 'opt_fra_6': ('opt_x_6', 6)}

N_BASE_FULL = 99                   # historical full sweep length (0.01 .. 0.99)
N_BASE_KEEP = 10                   # item 1: keep only the bottom 10 DT_BASE values


def base_grid_full() -> np.ndarray:
    """The historical scanned DT_BASE values: [1e-2 * i for i in range(1, 100)]."""
    return np.array([1e-2 * i for i in range(1, N_BASE_FULL + 1)], dtype=float)


def base_grid() -> np.ndarray:
    """DT_BASE values actually swept/plotted: the bottom N_BASE_KEEP values of
    base_grid_full() (0.01 .. 0.10 by default), per item 1 of the dt-extrapolation
    pipeline redesign.  base_idx keeps the same meaning as before (index into the
    full 99-point grid), just truncated -- existing base_NNN.npz files for
    idx >= N_BASE_KEEP are simply not read by this range, not deleted."""
    return base_grid_full()[:N_BASE_KEEP]


N_BASE = len(base_grid())          # 10


def point_tag(model: str, p1: float, p2: float) -> str:
    """Filesystem-safe tag identifying a (model, p1, p2) point (collision-free
    across the parameter values a user may pick)."""
    def f(v: float) -> str:
        return format(float(v), '.4f').replace('-', 'm').replace('.', 'p')
    return f'{model}_p1_{f(p1)}_p2_{f(p2)}'


def _load_existing(out: Path) -> dict | None:
    """Return {key: value} of everything stored in `out`, or None if absent/
    unreadable."""
    if not out.exists():
        return None
    try:
        d = np.load(out, allow_pickle=True)
        return {k: d[k] for k in d.files}
    except Exception:
        return None


def _missing_keys(existing: dict | None) -> list[str]:
    """MEASURES keys not yet present (or non-finite) in `existing` -- i.e. the
    keys a worker run still needs to compute.  This (not a version stamp) is
    what gates recomputation, so adding a new measure only ever computes that
    new measure for already-processed points and never touches the keys that
    were already there (item: "do not regenerate data that already exists")."""
    if existing is None:
        return [k for k, _ in MEASURES]
    missing = []
    for k, _ in MEASURES:
        if k not in existing or not np.isfinite(np.asarray(existing[k])):
            missing.append(k)
    return missing


def compute_base(model_name: str, p1: float, p2: float, base: float, *,
                 dim: int, needed_keys: list[str], fra_restarts: int,
                 fra_maxfev_4: int, fra_maxfev_6: int, ch1_restarts: int,
                 sch_restarts: int, sch_maxfev_6: int, seed: int,
                 existing_x: dict | None = None) -> dict:
    """Only the requested `needed_keys` subset of the seven framabilities of the
    model's bond Trotter gate at DT_BASE=base (plus dt, always).

    existing_x : optional {opt_x_4/opt_x_6: flat params} already stored for this
        point, used to warm-start (only read, never recomputed) -- notably so a
        d=6 recompute can still embed an earlier d=4 frame when opt_fra_4 is not
        itself being recomputed this run."""
    m = MODELS[model_name]
    H1, H2, j1, j2 = m.build(p1, p2)
    dt = choose_dt(H1, H2, j1, j2, base=base)      # base / max(||H||_1, {gamma_k})
    gate = bond_trotter_gate(H1, H2, j1, j2, dim, dt)

    out: dict = dict(dt=dt)
    need = set(needed_keys)
    if 'stab_fra' in need:
        out['stab_fra'] = stabilizer_3_framability(gate)                     # c1
    if 'pauli_fra' in need:
        out['pauli_fra'] = pauli_framability(gate)                          # c2
    if 'opt_fra_4' in need:
        f4, x4 = optimise_framability(gate, 4, fra_restarts, fra_maxfev_4,
                                      seed, return_x=True)                  # d (H, d=4)
        out['opt_fra_4'] = f4
        out['opt_x_4'] = x4
    if 'opt_fra_6' in need:
        # Seed the d=6 search with the d=4 optimum embedded into d=6.
        # embed_frame_params pads by replicating the last column, so
        # kron(S6,S6) contains every column of kron(S4,S4) and the duplicates
        # add no new LP targets -- the embedded frame therefore evaluates to
        # exactly the d=4 value, making the d=6 optimum provably <= the d=4 one.
        # Without this the only structured seed is _ixyz_init(6) (the extended-
        # Pauli frame), whose Powell basin can sit ABOVE the plain Pauli value,
        # which is how opt_fra_6 ended up looking worse than pauli_fra.
        seeds_6 = []
        x4_seed = out.get('opt_x_4')
        if x4_seed is None and 'opt_x_4' in (existing_x or {}):
            x4_seed = np.asarray((existing_x or {})['opt_x_4'], dtype=float)
        if x4_seed is not None:
            seeds_6.append(embed_frame_params(np.asarray(x4_seed, float), 4, 6))
        f6, x6 = optimise_framability(gate, 6, fra_restarts, fra_maxfev_6,
                                      seed + 1,
                                      extra_init_xs=seeds_6 or None,
                                      return_x=True)                        # d (H, d=6)
        out['opt_fra_6'] = f6
        out['opt_x_6'] = x6
    if 'gamma_ch1' in need:
        out['gamma_ch1'] = gamma_ch1_framability(gate, ch1_restarts, seed) # d3 max Janek
    if 'sch_fra_6' in need:
        # "optimal-schroedinger-framability", d_ext=6 (item 2)
        _, out['sch_fra_6'] = minimize_schroedinger_framability(
            gate, SCH_D_EXT, n_restarts=sch_restarts, maxfev=sch_maxfev_6,
            seed=seed + 2, verbose=False)
    if 'prod_fra_10' in need:
        # "product-state-framability", chi=10 (item 3); same reseed-then-call
        # convention as trotter_lindbladian_scan.compute_point / dtbase_gamma_scan
        # so the same Haar-random product frame is drawn at every point.
        np.random.seed(PROD_FRAME_SEED)
        out['prod_fra_10'] = product_state_framability(PROD_CHI, gate)
    return out


def run_point(args, base_idx: int) -> None:
    """Compute and save the still-missing measures of one base point, merging
    into any existing file so already-computed measures are left untouched."""
    base_vals = base_grid()
    base = float(base_vals[base_idx])
    tag = point_tag(args.model, args.p1, args.p2)
    out_dir = Path(args.out_dir) / tag
    out = out_dir / f'base_{base_idx:03d}.npz'

    existing = _load_existing(out)
    needed = _missing_keys(existing)
    if not needed:
        print(f'[skip] {tag}/{out.name} already has all {len(MEASURES)} measures',
              flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f'[base {base_idx}/{N_BASE}] {args.model} '
          f'{MODELS[args.model].p1_name}={args.p1:.4f} '
          f'{MODELS[args.model].p2_name}={args.p2:.4f}  base={base:.4f}  '
          f'dim={args.dim}  needed={needed}', flush=True)

    res = compute_base(args.model, args.p1, args.p2, base, dim=args.dim,
                       needed_keys=needed,
                       fra_restarts=args.fra_restarts,
                       fra_maxfev_4=args.fra_maxfev_4,
                       fra_maxfev_6=args.fra_maxfev_6,
                       ch1_restarts=args.ch1_restarts,
                       sch_restarts=args.sch_restarts,
                       sch_maxfev_6=args.sch_maxfev_6,
                       seed=args.seed,
                       existing_x=existing)

    save = dict(existing) if existing is not None else {}
    for k in needed:
        save[k] = np.array(res[k])
    # Persist the optimal frames too.  These are NOT in MEASURES (so never in
    # `needed`), and an earlier version dropped them here -- which silently
    # disabled the neighbour seeding in
    # scripts/trotter_dtbase_line_quick_refine_worker.py, since _best_known
    # only returns a warm-start frame when its x_key is present in the npz.
    for _key, (_x_key, _d) in FRAME_KEYS.items():
        if _x_key in res:
            save[_x_key] = np.asarray(res[_x_key], dtype=float)
    save.update(
        base=np.array(base), dt=np.array(res['dt']),
        base_idx=np.array(base_idx),
        p1=np.array(args.p1), p2=np.array(args.p2),
        dim=np.array(args.dim), model=np.array(args.model),
        seed=np.array(args.seed),
        code_version=np.array(DTBASE_LINE_VERSION),
    )
    np.savez(out, **save)
    summary = '  '.join(f'{k}={float(save[k]):.4f}' for k, _ in MEASURES if k in save)
    print(f'  saved {tag}/{out.name}  {summary}  '
          f'({time.perf_counter() - t0:.0f}s)', flush=True)


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
    p.add_argument('--sch_restarts', type=int, default=5,
                   help='restarts for minimize_schroedinger_framability (sch_fra_6)')
    p.add_argument('--sch_maxfev_6', type=int, default=500)
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
