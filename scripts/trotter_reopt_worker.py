"""
Re-optimise ONLY the optimised-framability quantities of an existing Trotter scan,
in place, with a *different* (stronger) optimisation method.

Motivation
----------
The main scan (trotter_lindbladian_scan.compute_point) optimises its framability
quantities with local, derivative-free methods that stall above the true optimum:

    opt_fra_4 / opt_fra_6   (Heisenberg)   -- Powell        (minimise)
    sch_fra_4 / 6 / 8       (Schrodinger)  -- Nelder-Mead   (minimise)
    gamma_ch1               (product frame)-- Powell        (maximise)
    deph_heis_fra_4         (Heisenberg,   -- Powell        (minimise)
    deph_schro_fra_4         dephased gate) -- Nelder-Mead   (minimise)

This worker leaves every other stored quantity untouched and recomputes only
those, each with a genuinely different optimiser:

    Heisenberg (opt_fra_*, deph_heis_fra_4)
        optimize_framability.minimize_framability(method='alternating')  --
        the certificate-based alternating minimisation -- warm-started from the
        point's own stored opt_S frame plus its four grid-neighbour opt_S frames
        (the recompute pipeline's escape from Powell branch/stall artefacts),
        then a Polyak floor-polish.  Every candidate is re-checked with the
        reference LP (dissipative_PT._framability_lp) before it is accepted.
    Schrodinger (sch_fra_*, deph_schro_fra_4)
        minimize_schroedinger_framability(method='dual_annealing') -- a global
        search in place of the local simplex (no stored state frame to seed).
    gamma_ch1
        a global dual_annealing maximisation over the single-qubit product frame
        (the product-frame gauge is a MAXimum, so a better optimiser finds a
        LARGER value), seeded with the deterministic axis corners.

The stored value is only overwritten when the new one is a genuine improvement in
the correct direction (never degraded): smaller for the minimised framabilities,
larger for gamma_ch1.  The rebuilt gate is taken from the point's own stored
(p1, p2, dim, dt) so no scan parameter is re-derived.  reopt_<key> values and a
reopt_version stamp are written for provenance; a point already at the current
stamp is skipped (so the array is safely resubmittable) unless --force.

Grid layout mirrors trotter_scan_worker: point_id = ix * N_Y + iy, strided across
--n_chunks array tasks; one read/write per pt file (no cross-task collision).

    python scripts/trotter_reopt_worker.py --model model7a \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (
    MODELS, bond_trotter_gate, DEPHASING_DT_FACTOR, _has_dephasing,
    gamma_ch1_framability, _CH1_SEEDS,
)
from dissipative_PT import (
    _SZ, _framability_lp, _kron_power, frame_from_params, _ixyz_init,
)
from optimize_framability import (
    minimize_framability, minimize_schroedinger_framability,
    polyak_floor_polish, spectral_floor, OPT_VERSION, SUPPORT_EPS,
)
from gamma_ch1_sphere import gamma_CH1, frame_op_1q, pauli_coeffs

# -reopt2: the Heisenberg re-optimisation now enforces genuine per-Pauli support
# (SUPPORT_EPS) on the returned frame, rejecting the collapsed all-identity frames
# the rank-only guard admitted (vacuous framability = spectral floor).
REOPT_VERSION = f'{OPT_VERSION}-reopt2'
FRA_ONE_TOL = 1e-6

# The optimised-framability quantities this worker recomputes, with their
# optimiser family, single-qubit frame size (None where not applicable) and the
# never-degrade direction ('min' for a minimised framability, 'max' for the
# product-frame gauge gamma_ch1).
#   key, kind, d_ext_single, direction
REOPT_SPECS = [
    ('opt_fra_4',       'heis',       4, 'min'),
    ('opt_fra_6',       'heis',       6, 'min'),
    ('sch_fra_4',       'schro',      4, 'min'),
    ('sch_fra_6',       'schro',      6, 'min'),
    ('sch_fra_8',       'schro',      8, 'min'),
    ('gamma_ch1',       'ch1',     None, 'max'),
    ('deph_heis_fra_4', 'heis_deph',  4, 'min'),
    ('deph_schro_fra_4','schro_deph', 4, 'min'),
]


# ---------------------------------------------------------------------------
#  Frame-seed helpers (shared with the recompute worker's conventions)
# ---------------------------------------------------------------------------
def _seed_x(S) -> np.ndarray | None:
    """Flat parameter seed (free columns raveled) from a real frame S."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[1] < 2 or not np.all(np.isfinite(S)):
        return None
    return S[:, 1:].ravel()


def _pauli_support_ok(S) -> bool:
    """True iff every Pauli X, Y, Z has support >= SUPPORT_EPS on some column of
    the single-qubit frame S:  max_j |S[a, j]| >= SUPPORT_EPS  for a in {X,Y,Z}.

    The rank-based _has_full_support (rtol 1e-6) used by the framability LPs is
    too lenient -- it admits near-collapsed frames whose whole Bloch part is
    ~1e-3, so D = S(x)S is technically full rank but the framability certificate
    is vacuous (all mass on the identity column; framability trivially = the
    spectral floor).  This is the same per-Pauli support requirement the
    Schrodinger optimiser enforces, and is what "the frame has support on all
    Pauli operators" means here."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] < 4 or S.shape[1] < 1 or not np.all(np.isfinite(S)):
        return False
    return bool(np.min(np.max(np.abs(S[1:4, :]), axis=1)) >= SUPPORT_EPS)


def _confirm_heis(S, gate) -> float:
    """Reference-LP framability of the Heisenberg frame S, or +inf when the frame
    lacks genuine support on every Pauli (a collapsed frame is rejected so it can
    never win the never-degrade comparison)."""
    S = np.asarray(S, dtype=float)
    if not _pauli_support_ok(S):
        return float('inf')
    return _framability_lp(_kron_power(S, 2), gate)


def _neighbour_heis_seeds(in_dir: Path, name: str, ix: int, iy: int,
                          de: int) -> list:
    """Stored opt_S_<de> frames of the four grid neighbours (as flat seeds)."""
    seeds = []
    model = MODELS[name]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        jx, jy = ix + dx, iy + dy
        if not (0 <= jx < model.N_X and 0 <= jy < model.N_Y):
            continue
        nf = in_dir / name / f'pt_{jx:03d}_{jy:03d}.npz'
        if not nf.exists():
            continue
        try:
            nz = np.load(nf, allow_pickle=True)
            if f'opt_S_{de}' in nz.files:
                s = _seed_x(nz[f'opt_S_{de}'])
                if s is not None:
                    seeds.append(s)
            nz.close()
        except Exception:
            pass
    return seeds


# ---------------------------------------------------------------------------
#  Per-kind re-optimisers.  Each returns (new_value, new_frame_or_None).
# ---------------------------------------------------------------------------
def _reopt_heis(gate, de, stored_S, in_dir, name, ix, iy, args):
    """Alternating-certificate re-optimisation of the Heisenberg framability,
    constrained to frames with genuine support on every Pauli.

    Warm-started from the stored opt_S frame, the grid-neighbour opt_S frames
    and -- always -- the extended-Pauli (ixyz) frame, which is full-support by
    construction so a valid answer exists even when the local optimiser drifts
    into a collapsed frame.  The alternating result is Polyak floor-polished, and
    among {optimised, polished, ixyz baseline} only the frames that pass the
    per-Pauli support test are kept; the one with the smallest reference-LP
    framability is returned.  Returns (value, frame) with value LP-confirmed and
    the frame guaranteed to have full Pauli support (never a vacuous certificate).
    """
    ixyz_x = _ixyz_init(de)
    seeds = [_seed_x(stored_S)] if stored_S is not None else []
    seeds += _neighbour_heis_seeds(in_dir, name, ix, iy, de)
    seeds.append(ixyz_x)                       # guaranteed full-support seed
    extra = [s for s in seeds if s is not None] or None
    _, _f, x_opt = minimize_framability(
        gate, de, n_restarts=args.n_restarts, method='alternating',
        maxfev=args.maxfev, seed=args.seed, verbose=False, return_x=True,
        extra_init_xs=extra)
    S_opt = frame_from_params(x_opt, de)

    # Candidate full-support frames.  The optimiser can return a collapsed frame
    # (its internal LP only checks rank, not per-Pauli support); polishing may
    # collapse a good frame too.  Keep every candidate but score each through
    # _confirm_heis, which rejects support-poor frames with +inf -- so the ixyz
    # extended-Pauli frame (always full-support) is the guaranteed fallback.
    cands = [S_opt]
    if args.polish_iters > 0 and _pauli_support_ok(S_opt):
        _, S_pol = polyak_floor_polish(S_opt, gate, n_iter=args.polish_iters)
        cands.append(S_pol)
    cands.append(frame_from_params(ixyz_x, de))

    best_val, best_S = float('inf'), None
    for S in cands:
        v = _confirm_heis(S, gate)             # +inf unless full Pauli support
        if np.isfinite(v) and v < best_val:
            best_val, best_S = v, np.asarray(S)
    return float(best_val), best_S


def _reopt_schro(gate, de, args):
    """Global (dual_annealing) re-optimisation of the Schrodinger framability.
    No state frame is stored by the scan, so this is a fresh global search."""
    _, f_opt = minimize_schroedinger_framability(
        gate, de, method='dual_annealing', n_restarts=args.n_restarts,
        maxfev=args.sch_maxfev, seed=args.seed, verbose=False)
    return float(f_opt), None


def _reopt_ch1(gate, args):
    """Global dual_annealing MAXimisation of gamma_ch1 over the product frame.

    gamma_ch1_framability is a maximum of the product-frame gauge, so a stronger
    optimiser finds a LARGER value.  The deterministic axis corners (_CH1_SEEDS)
    and a Powell-seeded baseline give a floor the global search can only raise."""
    from scipy.optimize import dual_annealing
    gate = np.asarray(gate, dtype=float)

    def val(b1, b2):
        op = np.kron(frame_op_1q(b1), frame_op_1q(b2))
        return gamma_CH1(gate @ pauli_coeffs(op))

    # Baseline: the scan's own local optimiser (Powell restarts) — never lose to it.
    best = gamma_ch1_framability(gate, n_restarts=args.n_restarts, seed=args.seed)
    for b1 in _CH1_SEEDS:
        for b2 in _CH1_SEEDS:
            best = max(best, val(b1, b2))

    def neg(p):
        b1, b2 = p[:3], p[3:]
        r1, r2 = np.linalg.norm(b1), np.linalg.norm(b2)
        if r1 > 1.0:
            b1 = b1 / r1
        if r2 > 1.0:
            b2 = b2 / r2
        return -val(b1, b2)

    res = dual_annealing(
        neg, [(-1.0, 1.0)] * 6, maxfun=args.ch1_maxfev,
        maxiter=max(200, args.ch1_maxfev // 3), seed=args.seed)
    best = max(best, float(-res.fun))
    return float(best), None


# ---------------------------------------------------------------------------
#  Per-point driver
# ---------------------------------------------------------------------------
def process_point(in_dir: Path, name: str, ix: int, iy: int, args) -> None:
    f = in_dir / name / f'pt_{ix:03d}_{iy:03d}.npz'
    if not f.exists():
        print(f'[miss] {name}/pt_{ix:03d}_{iy:03d}.npz', flush=True)
        return
    z = np.load(f, allow_pickle=True)
    if (not args.force and 'reopt_version' in z.files
            and str(z['reopt_version']) == REOPT_VERSION):
        print(f'[skip] {name}/{f.name} already at {REOPT_VERSION}', flush=True)
        z.close()
        return
    data = {k: z[k] for k in z.files}
    z.close()

    p1, p2 = float(data['p1']), float(data['p2'])
    dt, dim = float(data['dt']), int(data['dim'])
    H1, H2, j1, j2 = MODELS[name].build(p1, p2)
    gate = bond_trotter_gate(H1, H2, j1, j2, dim, dt)
    if abs(spectral_floor(gate) - float(data.get('floor', spectral_floor(gate)))) > 1e-6:
        print(f'[WARN] {name}/{f.name}: gate/floor mismatch', flush=True)

    # The dephasing-augmented gate (group g) — only defined for models whose
    # jumps carry no dephasing operator, exactly as in compute_point.
    gate_deph = None
    if not _has_dephasing(j1, j2):
        j1_deph = list(j1 or []) + [np.sqrt(DEPHASING_DT_FACTOR * dt) * _SZ]
        gate_deph = bond_trotter_gate(H1, H2, j1_deph, j2, dim, dt)

    which = set(args.quantities) if args.quantities else \
        {k for k, *_ in REOPT_SPECS}

    t0 = time.perf_counter()
    line = [f'{name}/pt_{ix:03d}_{iy:03d} ({p1:.3f},{p2:.3f})']
    for key, kind, de, direction in REOPT_SPECS:
        if key not in which or key not in data:
            continue
        stored = float(data[key])
        if kind == 'heis':
            new_val, new_S = _reopt_heis(
                gate, de, data.get(f'opt_S_{de}'), in_dir, name, ix, iy, args)
        elif kind == 'heis_deph':
            if gate_deph is None:
                continue
            new_val, new_S = _reopt_heis(
                gate_deph, de, None, in_dir, name, ix, iy, args)
        elif kind == 'schro':
            new_val, new_S = _reopt_schro(gate, de, args)
        elif kind == 'schro_deph':
            if gate_deph is None:
                continue
            new_val, new_S = _reopt_schro(gate_deph, de, args)
        elif kind == 'ch1':
            new_val, new_S = _reopt_ch1(gate, args)
        else:                                                    # pragma: no cover
            continue

        data[f'reopt_{key}'] = np.array(new_val)

        # A stored Heisenberg value whose frame lacks genuine Pauli support is a
        # vacuous certificate (the very bug this fixes): treat it as +inf so the
        # honest full-support result always replaces it, even when numerically
        # larger.  Every _reopt_heis result is full-support by construction.
        stored_invalid = (kind == 'heis'
                          and not _pauli_support_ok(data.get(f'opt_S_{de}')))
        eff_stored = float('inf') if stored_invalid else stored
        if direction == 'min':
            improved = np.isfinite(new_val) and new_val < eff_stored - 1e-12
        else:                                                    # 'max'
            improved = np.isfinite(new_val) and new_val > eff_stored + 1e-12
        if improved:
            data[key] = np.array(new_val)
            if new_S is not None and de is not None and f'opt_S_{de}' in data \
                    and kind == 'heis':
                data[f'opt_S_{de}'] = new_S      # keep the stored frame consistent
        tag = 'IMPROVED' + ('(stored-collapsed)' if stored_invalid else '') \
            if improved else 'kept'
        line.append(f'{key}: {stored:.6f}->{new_val:.6f} {tag}')

    data['reopt_version'] = np.array(REOPT_VERSION)
    tmp = f.with_name(f'{f.stem}.tmp{os.getpid()}.npz')
    np.savez(tmp, **data)
    os.replace(tmp, f)
    line.append(f'({time.perf_counter() - t0:.0f}s)')
    print('  '.join(line), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, required=True, choices=list(MODELS))
    p.add_argument('--task_id',  type=int, required=True,
                   help='point id when --n_chunks=1, else chunk id 0..n_chunks-1')
    p.add_argument('--n_chunks', type=int, default=1,
                   help='split the grid into this many strided array tasks')
    p.add_argument('--in_dir',   type=str, default='results_trotter_v3')
    p.add_argument('--quantities', type=str, nargs='*', default=None,
                   help='subset of keys to re-optimise (default: all optimised '
                        'framabilities)')
    p.add_argument('--n_restarts',   type=int, default=12)
    p.add_argument('--maxfev',       type=int, default=6000,
                   help='budget for the Heisenberg alternating method')
    p.add_argument('--sch_maxfev',   type=int, default=3000,
                   help='budget for the Schrodinger dual_annealing search')
    p.add_argument('--ch1_maxfev',   type=int, default=2000,
                   help='budget for the gamma_ch1 dual_annealing search')
    p.add_argument('--polish_iters', type=int, default=300)
    p.add_argument('--seed',   type=int, default=0)
    p.add_argument('--force',  action='store_true')
    args = p.parse_args()

    model = MODELS[args.model]
    N = model.N_TOTAL
    in_dir = Path(args.in_dir)

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        ix, iy = args.task_id // model.N_Y, args.task_id % model.N_Y
        process_point(in_dir, model.name, ix, iy, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[task {args.task_id}/{args.n_chunks}] {model.name}: '
          f'{len(point_ids)} points  ({REOPT_VERSION})', flush=True)
    for pid in point_ids:
        ix, iy = pid // model.N_Y, pid % model.N_Y
        process_point(in_dir, model.name, ix, iy, args)


if __name__ == '__main__':
    main()
