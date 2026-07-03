"""
Build the model5 (dissipative-PT) Trotter-scan results from the existing
results_dpt scan, computing only the quantities the dpt pipeline never stored.

model5's bond gate (dim=1, dt=0.05) is exactly dissipative_PT.bond_trotter_gate
and its full-lattice NESS is exactly dissipative_PT.build_full_lindbladian, so
all gate/NESS quantities already computed by scripts/dissipative_PT_worker.py
are reused as is:

    reused    : sign_init, sign_opt, floor, pauli_fra,
                lind_rate (= decay_rate), mag_z (= mean_mag), ss_vn,
                neg (= neg_half),
                opt_fra_4 / opt_fra_6 + opt_S_4 / opt_S_6
                (best over the dpt base scan and every dpt refine round)
    computed  : lpdo (NESS LPDO bond entropy, d_A=2),
                lpdo_max (path from the model's lpdo_init state),
                mag_x, stab_fra, gamma_ch1

If a point has no dpt base file at all, everything is computed from scratch
with trotter_lindbladian_scan.compute_point.

Grid: model5's (h, gamma) = dissipative_PT (H_LIST x GAMMA_LIST), point_id =
ih * N_G + ig.  Output: <out_dir>/model5/pt_<ih:03d>_<ig:03d>.npz (same format
and version stamp as trotter_scan_worker.py, so collect / refine / plotting
work unchanged).

Usage (strided across a 200-task array):
    python scripts/trotter_model5_import_worker.py \
        --task_id $SLURM_ARRAY_TASK_ID --n_chunks 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (
    MODELS, QUANTITIES, TLS_VERSION, compute_point,
    bond_trotter_gate, build_full_lindbladian_model, lpdo_init_vector,
    gamma_ch1_framability, _site_mean,
    LATTICE_LX, LATTICE_LY, LPDO_PATH_DT, LPDO_PATH_FIDELITY,
)
from dissipative_PT import (
    _SX, steady_state_and_decay, pauli_to_rho, lpdo_bond_entropy,
)
from analysis import compute_max_bond_dim
from framability import dyadic_stabilizer_framability

MODEL = MODELS['model5']

# dpt key -> trotter-scan key for the directly reusable scalars
REUSE_MAP = {
    'sign_init':  'sign_init',
    'sign_opt':   'sign_opt',
    'floor':      'floor',
    'pauli_fra':  'pauli_fra',
    'decay_rate': 'lind_rate',
    'mean_mag':   'mag_z',
    'ss_vn':      'ss_vn',
    'neg_half':   'neg',
}


def _is_current(out: Path) -> bool:
    if not out.exists():
        return False
    try:
        d = np.load(out, allow_pickle=True)
        return 'code_version' in d and str(d['code_version']) == TLS_VERSION
    except Exception:
        return False


def _best_opt_fra(dpt_dir: Path, ih: int, ig: int, key: str, s_key: str):
    """Lowest (value, frame) over the dpt base file and every dpt refine round."""
    files = [dpt_dir / f'dpt_{ih:02d}_{ig:02d}.npz']
    files += sorted(dpt_dir.glob(f'dpt_refine*_{ih:02d}_{ig:02d}.npz'))
    best_val, best_S = np.inf, None
    for f in files:
        if not f.exists():
            continue
        try:
            d = np.load(f)
        except Exception:
            continue
        if key not in d:
            continue
        v = float(d[key])
        if np.isfinite(v) and v < best_val:
            S = np.asarray(d[s_key], dtype=float) if s_key in d else None
            if S is not None and np.all(np.isfinite(S)):
                best_val, best_S = v, S
    return best_val, best_S


def run_point(point_id: int, args) -> None:
    ih = point_id // MODEL.N_Y
    ig = point_id %  MODEL.N_Y
    h     = float(MODEL.p1_vals[ih])
    gamma = float(MODEL.p2_vals[ig])
    out_dir = Path(args.out_dir) / MODEL.name
    out = out_dir / f'pt_{ih:03d}_{ig:03d}.npz'
    dpt_dir = Path(args.dpt_dir)
    dpt_base = dpt_dir / f'dpt_{ih:02d}_{ig:02d}.npz'

    if _is_current(out):
        print(f'[skip] {MODEL.name}/{out.name} already at version {TLS_VERSION}',
              flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    N = LATTICE_LX * LATTICE_LY

    if not dpt_base.exists():
        # No dpt result to import: compute the full point from scratch.
        print(f'[full] {MODEL.name} pt ({ih},{ig}) h={h:.2f} gamma={gamma:.2f}: '
              f'no {dpt_base.name}, running compute_point', flush=True)
        res = compute_point(MODEL, h, gamma, dim=MODEL.dim, dt=MODEL.dt,
                            seed=args.seed + point_id, verbose=True)
        save = {k: np.array(res[k]) for k, _, _, _ in QUANTITIES}
        save.update(sign_init=np.array(res['sign_init']),
                    floor=np.array(res['floor']),
                    opt_S_4=np.asarray(res['opt_S_4']),
                    opt_S_6=np.asarray(res['opt_S_6']))
    else:
        dpt = np.load(dpt_base)
        print(f'[import {point_id}/{MODEL.N_TOTAL}] h={h:.2f} gamma={gamma:.2f} '
              f'from {dpt_base.name}', flush=True)

        save = {}
        for src, dst in REUSE_MAP.items():
            save[dst] = np.array(float(dpt[src])) if src in dpt else np.array(np.nan)

        f4, S4 = _best_opt_fra(dpt_dir, ih, ig, 'opt_fra_4', 'opt_S_4')
        f6, S6 = _best_opt_fra(dpt_dir, ih, ig, 'opt_fra_6', 'opt_S_6')
        save['opt_fra_4'] = np.array(f4 if np.isfinite(f4) else np.nan)
        save['opt_fra_6'] = np.array(f6 if np.isfinite(f6) else np.nan)
        save['opt_S_4'] = S4 if S4 is not None else np.full((4, 4), np.nan)
        save['opt_S_6'] = S6 if S6 is not None else np.full((4, 6), np.nan)

        # ── quantities the dpt pipeline never computed ────────────────────────
        H1, H2, jumps1, jumps2 = MODEL.build(h, gamma)
        gate = bond_trotter_gate(H1, H2, jumps1, jumps2, MODEL.dim, MODEL.dt)
        save['stab_fra']  = np.array(dyadic_stabilizer_framability(gate))
        save['gamma_ch1'] = np.array(gamma_ch1_framability(
            gate, args.ch1_restarts, args.seed + point_id))

        L_full = build_full_lindbladian_model(H1, H2, jumps1, jumps2)
        c_ss, decay = steady_state_and_decay(L_full, N=N)
        if 'decay_rate' not in dpt:
            save['lind_rate'] = np.array(decay)
        if c_ss is not None:
            rho = pauli_to_rho(c_ss, N=N)
            try:
                save['lpdo'] = np.array(lpdo_bond_entropy(rho, d_A=2))
            except Exception:
                save['lpdo'] = np.array(np.nan)
            try:
                _, lpdo_max = compute_max_bond_dim(
                    L_full, rho, None, N=N, init=lpdo_init_vector(MODEL, N),
                    dt=LPDO_PATH_DT, fidelity_threshold=LPDO_PATH_FIDELITY)
                save['lpdo_max'] = np.array(float(lpdo_max))
            except Exception:
                save['lpdo_max'] = np.array(np.nan)
            save['mag_x'] = np.array(_site_mean(rho, _SX, N))
            # backfill any NESS scalar the dpt file happened to miss
            if not np.isfinite(float(save['ss_vn'])):
                from dissipative_PT import vn_entropy
                save['ss_vn'] = np.array(vn_entropy(rho))
        else:
            save['lpdo'] = save['lpdo_max'] = save['mag_x'] = np.array(np.nan)

    save.update(p1=np.array(h), p2=np.array(gamma),
                ix=np.array(ih), iy=np.array(ig),
                dim=np.array(MODEL.dim), dt=np.array(MODEL.dt),
                model=np.array(MODEL.name),
                code_version=np.array(TLS_VERSION))
    np.savez(out, **save)
    print(f'  saved {MODEL.name}/{out.name}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--task_id',      type=int, required=True)
    p.add_argument('--n_chunks',     type=int, default=1)
    p.add_argument('--out_dir',      type=str, default='results_trotter')
    p.add_argument('--dpt_dir',      type=str, default='results_dpt')
    p.add_argument('--ch1_restarts', type=int, default=15)
    p.add_argument('--seed',         type=int, default=0)
    args = p.parse_args()

    N = MODEL.N_TOTAL

    if args.n_chunks <= 1:
        if not (0 <= args.task_id < N):
            print(f'ERROR: task_id must be in [0, {N})', file=sys.stderr)
            sys.exit(1)
        run_point(args.task_id, args)
        return

    if not (0 <= args.task_id < args.n_chunks):
        print(f'ERROR: chunk id must be in [0, {args.n_chunks})', file=sys.stderr)
        sys.exit(1)

    point_ids = list(range(args.task_id, N, args.n_chunks))
    print(f'[chunk {args.task_id}/{args.n_chunks}] {MODEL.name}: '
          f'{len(point_ids)} points', flush=True)
    for pid in point_ids:
        run_point(pid, args)


if __name__ == '__main__':
    main()
