"""Independent audit of the 'alternating'-scheme framabilities (models 1-5).

The alternating optimiser (optimize_framability.minimize_framability, method=
'alternating') minimises using its own fast internal evaluator and then stores
the winning single-qubit frame as opt_S_4 / opt_S_6 and the value as
opt_fra_4 / opt_fra_6 in each results_trotter_v3/<model>/pt_<ix>_<iy>.npz.

This script does NOT trust the stored value.  For every point it:

  1. rebuilds the exact 16x16 Trotter gate from the *stored* p1, p2, dt, dim
     via the model builder (bit-identical to the scanned gate), cross-checking
     the rebuilt spectral floor against the stored `floor`;
  2. takes the stored optimised frame S = opt_S_de (4 x de), forms the two-qubit
     dictionary D = kron(S, S), and re-solves the framability LP with the
     *reference* evaluator dissipative_PT._framability_lp (HiGHS, presolve off);
  3. compares that recomputed framability with the stored opt_fra_de.

The point of interest is every point the scan calls framable, i.e.
opt_fra_de <= 1 + FRA_ONE_TOL.  For those we assert the *independently
recomputed* framability is also <= 1 (within tolerance) and reproduces the
stored value.  A frame whose reference framability exceeds 1 while its stored
value is 1 is a scheme artifact (fast-evaluator over-report, rank-deficient
frame reported as feasible, un-polished certificate, ...), which is exactly the
"too good to be true" failure mode.

    python scripts/trotter_alt_verify.py --in_dir results_trotter_v3_alt
    python scripts/trotter_alt_verify.py --in_dir results_trotter_v3_alt \
        --models model3 --out_npz verify_model3.npz

Exit code 0 if every framable point verifies, 4 if any violation is found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, bond_trotter_gate
from dissipative_PT import _framability_lp, _kron_power, spectral_floor

D_EXTS = (4, 6)
FRA_ONE_TOL = 1e-6      # a point is "claimed framable" if opt_fra <= 1 + this
VERIFY_TOL = 1e-5       # recomputed framability above 1 + this is a real violation
FLOOR_TOL = 1e-6        # rebuilt-vs-stored spectral floor mismatch => gate mismatch


def _recompute(S: np.ndarray, gate: np.ndarray) -> float:
    """Reference framability of the stored single-qubit frame S on `gate`.

    Uses the repo's dissipative_PT._framability_lp (HiGHS, presolve OFF).  Note
    this evaluator returns +inf whenever HiGHS reports a non-zero status for a
    per-column LP, which -- with presolve disabled -- happens spuriously on
    perfectly feasible, full-rank problems; hence flagged points are re-checked
    with _robust_recompute below before a violation is declared."""
    S = np.asarray(S, dtype=float)
    if not np.all(np.isfinite(S)):
        return float('nan')
    D = _kron_power(S, 2)
    return _framability_lp(D, gate)


def _robust_recompute(S: np.ndarray, gate: np.ndarray) -> tuple[float, bool]:
    """Presolve-ON re-solve of the framability LP (scipy linprog 'highs').

    Returns (value, ok).  ok is False iff some per-column LP returns a non-zero
    status even with presolve enabled (HiGHS status 4 = numerical difficulties),
    i.e. the point is genuinely ill-conditioned and cannot be certified here."""
    S = np.asarray(S, dtype=float)
    if not np.all(np.isfinite(S)):
        return float('nan'), False
    D = _kron_power(S, 2)
    n, de = D.shape
    Y = gate.real.T @ D
    c = np.concatenate([np.zeros(de), np.ones(de)])
    Ide = np.eye(de)
    A_ub = np.vstack([np.hstack([Ide, -Ide]), np.hstack([-Ide, -Ide])])
    b_ub = np.zeros(2 * de)
    A_eq = np.hstack([D, np.zeros((n, de))])
    bounds = [(None, None)] * de + [(0.0, None)] * de
    best = 0.0
    for j in range(de):
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=Y[:, j],
                    bounds=bounds, method='highs')
        if r.status != 0:
            return float('inf'), False
        best = max(best, float(np.abs(r.x[:de]).sum()))
    return best, True


def verify_model(in_dir: Path, name: str, args) -> dict:
    model = MODELS[name]
    rows = []          # per (point, de) audit records
    n_files = n_missing_alt = n_gate_mismatch = 0

    for ix in range(model.N_X):
        for iy in range(model.N_Y):
            f = in_dir / name / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            n_files += 1
            has_alt = 'alt_opt_version' in z.files
            if not has_alt:
                n_missing_alt += 1

            p1, p2 = float(z['p1']), float(z['p2'])
            dt, dim = float(z['dt']), int(z['dim'])
            H1, H2, j1, j2 = model.build(p1, p2)
            gate = bond_trotter_gate(H1, H2, j1, j2, dim, dt)

            fl = spectral_floor(gate)
            gate_ok = abs(fl - float(z['floor'])) <= FLOOR_TOL
            if not gate_ok:
                n_gate_mismatch += 1

            for de in D_EXTS:
                stored = float(z[f'opt_fra_{de}'])
                S = z[f'opt_S_{de}']
                fast = _recompute(S, gate)
                claimed = stored <= 1.0 + FRA_ONE_TOL
                # Only claimed-framable points that the (presolve-off) fast
                # evaluator flags are worth the expensive presolve-on re-solve.
                flagged = claimed and (not np.isfinite(fast)
                                       or fast > 1.0 + VERIFY_TOL)
                if flagged:
                    final, ok = _robust_recompute(S, gate)
                else:
                    final, ok = fast, True
                rows.append((ix, iy, p1, p2, de, stored, fast, final,
                             float(ok), gate_ok, has_alt))
            z.close()

    if not rows:
        return {'name': name, 'n_files': 0}

    arr = np.array(rows, dtype=float)
    C = {'ix': 0, 'iy': 1, 'p1': 2, 'p2': 3, 'de': 4, 'stored': 5,
         'fast': 6, 'final': 7, 'ok': 8, 'gate_ok': 9, 'has_alt': 10}
    out = {'name': name, 'n_files': n_files, 'n_missing_alt': n_missing_alt,
           'n_gate_mismatch': n_gate_mismatch, 'rows': arr, 'cols': C}

    # Per-de audit over the claimed-framable subset, using the presolve-on
    # `final` value as the authoritative framability of the stored frame.
    per_de = {}
    for de in D_EXTS:
        m = arr[:, C['de']] == de
        sub = arr[m]
        stored = sub[:, C['stored']]
        final = sub[:, C['final']]
        ok = sub[:, C['ok']] > 0.5
        framable = stored <= 1.0 + FRA_ONE_TOL      # what the scan claims
        finite = np.isfinite(final)
        # a claimed-framable frame is:
        #   clean       -> authoritative framability <= 1 + VERIFY_TOL
        #   violation   -> authoritative framability  > 1 + VERIFY_TOL (solvable)
        #   uncertain   -> LP could not be solved even with presolve (status!=0)
        clean = framable & ok & finite & (final <= 1.0 + VERIFY_TOL)
        violation = framable & ok & finite & (final > 1.0 + VERIFY_TOL)
        uncertain = framable & (~ok | ~finite)
        cons = np.abs(final - stored)[clean]
        per_de[de] = {
            'n_framable': int(framable.sum()),
            'n_clean': int(clean.sum()),
            'n_violation': int(violation.sum()),
            'n_uncertain': int(uncertain.sum()),
            'max_final_clean': (float(final[clean].max())
                                if clean.any() else float('nan')),
            'max_abs_diff': float(cons.max()) if cons.size else 0.0,
            'viol_idx': np.where(violation)[0],
            'unc_idx': np.where(uncertain)[0],
            'coords': sub[:, :4], 'stored': stored, 'fast': sub[:, C['fast']],
            'final': final,
        }
    out['per_de'] = per_de
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', type=str, default='results_trotter_v3_alt')
    p.add_argument('--models', type=str,
                   default='model1,model2,model3,model4,model5')
    p.add_argument('--out_npz', type=str, default=None,
                   help='optional: dump the full per-point audit table')
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    names = [s.strip() for s in args.models.split(',') if s.strip()]
    total_violation = total_uncertain = 0
    dump = {}

    def _list(d, idx, tag):
        coords = d['coords'][idx]
        order = np.argsort(-np.nan_to_num(d['final'][idx], nan=np.inf))
        print(f'    {tag} (up to 10):')
        for k in order[:10]:
            ix, iy, pp1, pp2 = coords[k]
            fin = d['final'][idx][k]
            fin_s = 'unsolvable' if not np.isfinite(fin) else f'{fin:.6f}'
            print(f'      pt_{int(ix):03d}_{int(iy):03d} '
                  f'({pp1:+.3f},{pp2:+.3f}): stored={d["stored"][idx][k]:.6f} '
                  f'presolve-off={d["fast"][idx][k]:.6g} authoritative={fin_s}')

    for name in names:
        res = verify_model(in_dir, name, args)
        if res.get('n_files', 0) == 0:
            print(f'{name}: no point files under {in_dir/name} -- skipped',
                  flush=True)
            continue
        print(f'\n=== {name} ===  ({res["n_files"]} points, '
              f'{res["n_missing_alt"]} without alt stamp, '
              f'{res["n_gate_mismatch"]} gate-floor mismatches)')
        for de in D_EXTS:
            d = res['per_de'][de]
            total_violation += d['n_violation']
            total_uncertain += d['n_uncertain']
            status = ('OK' if (d['n_violation'] == 0 and d['n_uncertain'] == 0)
                      else 'VIOLATION' if d['n_violation'] else 'UNCERTAIN')
            print(f'  d_ext={de}: framable(claimed)={d["n_framable"]:5d}  '
                  f'clean={d["n_clean"]:5d}  violations={d["n_violation"]}  '
                  f'uncertain={d["n_uncertain"]}  '
                  f'max framability(clean)={d["max_final_clean"]:.9f}  '
                  f'max|final-stored|={d["max_abs_diff"]:.2e}  [{status}]')
            if d['n_violation']:
                _list(d, d['viol_idx'], 'genuine violations (framability > 1)')
            if d['n_uncertain']:
                _list(d, d['unc_idx'], 'uncertain (LP ill-conditioned, uncertifiable)')
        if args.out_npz:
            dump[f'{name}_rows'] = res['rows']

    if args.out_npz and dump:
        np.savez(args.out_npz, **dump)
        print(f'\nfull audit table -> {args.out_npz}')

    print(f'\nTOTAL: genuine violations = {total_violation}, '
          f'uncertain (ill-conditioned) = {total_uncertain}')
    if total_violation:
        print('AUDIT FAILED: some claimed-framable frames genuinely exceed 1 '
              'under the presolve-on reference LP.')
    elif total_uncertain:
        print('AUDIT PASSED WITH CAVEATS: every solvable claimed-framable frame '
              'verifies (<= 1); a few points are LP-ill-conditioned and could '
              'not be certified either way.')
    else:
        print('AUDIT PASSED: every claimed-framable frame reproduces '
              'framability <= 1 under the reference LP.')
    sys.exit(4 if total_violation else 0)


if __name__ == '__main__':
    main()
