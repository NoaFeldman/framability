"""
Stage 1 of the alternating-scheme cleanup: LOCAL CONFIRMATION, no optimisation.

The alternating worker stored opt_fra_4/6 from optimize_framability.
_get_framability_fast, which is a *cheap batched scanner* that (a) is documented
"confirm with _framability_lp before trusting" and (b) was found to genuinely
UNDER-report the framability on some frames (e.g. model5/pt_010_024: scanner 1.0
vs true 1.0447).  This script re-evaluates the *stored frames* with the reference
evaluator dissipative_PT._framability_lp (now hardened with a presolve-on HiGHS
ladder) -- it does NOT re-optimise anything, so it is cheap and runs locally.

For every point and d_ext in {4, 6} it:
  1. rebuilds the exact gate from the stored p1, p2, dt, dim (spectral-floor
     cross-check), and evaluates BOTH stored frames -- opt_S_de (the scheme's
     winner) and prev_S_de (the pre-alternating Powell frame, if present) --
     with _framability_lp;
  2. keeps whichever stored frame has the lower *true* framability (this alone
     rescues points whose Powell frame was actually better than the buggy
     alternating winner, with zero optimisation);
  3. writes the confirmed value/frame back in place:
        conf_fra_de     true framability of the best stored frame (authoritative)
        opt_fra_de      <- conf_fra_de  (corrected; never below the true value)
        opt_S_de        <- the best stored frame
        unconf_fra_de   the pre-confirmation opt_fra_de (provenance, written once)
        conf_version    stamp; a point already at CONF_VERSION is skipped
  4. if the confirmed value is still > 1 + FRA_ONE_TOL it is added to the
     recompute manifest -- the ONLY points Stage 2 (cluster) needs to touch.

Everything else in each npz is copied through untouched.  Points not previously
claimed framable are still confirmed (so opt_fra becomes trustworthy everywhere)
but only *corrected false positives / newly-uncertifiable* points enter the
manifest, keeping the cluster job as small as possible.

    python scripts/trotter_alt_confirm.py --in_dir results_trotter_v3 \
        --out_manifest recompute_manifest

Add --dry_run to confirm and report WITHOUT modifying any npz.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, bond_trotter_gate
import dissipative_PT as dp

D_EXTS = (4, 6)
FRA_ONE_TOL = 1e-6
FLOOR_TOL = 1e-6
CONF_VERSION = '1.0-lp-confirm'


def _conf(S, gate) -> float:
    """True framability of stored frame S via the hardened reference LP."""
    S = np.asarray(S, dtype=float)
    if S.size == 0 or not np.all(np.isfinite(S)):
        return float('inf')
    return dp._framability_lp(dp._kron_power(S, 2), gate)


def process_point(f: Path, name: str, args, manifest: list) -> tuple[int, int]:
    """Confirm one npz.  Returns (n_confirmed_de, n_manifest_de)."""
    z = np.load(f, allow_pickle=True)
    if (not args.force and 'conf_version' in z.files
            and str(z['conf_version']) == CONF_VERSION):
        z.close()
        return 0, 0
    data = {k: z[k] for k in z.files}
    z.close()

    model = MODELS[name]
    p1, p2 = float(data['p1']), float(data['p2'])
    dt, dim = float(data['dt']), int(data['dim'])
    H1, H2, j1, j2 = model.build(p1, p2)
    gate = bond_trotter_gate(H1, H2, j1, j2, dim, dt)
    fl = dp.spectral_floor(gate)
    if abs(fl - float(data['floor'])) > FLOOR_TOL:
        print(f'[WARN] {name}/{f.name}: rebuilt floor {fl:.8f} != stored '
              f'{float(data["floor"]):.8f} -- gate mismatch?', flush=True)

    ix, iy = int(data['ix']), int(data['iy'])
    n_conf = n_man = 0
    for de in D_EXTS:
        opt_S = data[f'opt_S_{de}']
        conf_opt = _conf(opt_S, gate)
        best_val, best_S = conf_opt, np.asarray(opt_S)
        # Only re-check the stored Powell frame when the scheme's winner is NOT
        # (confirmed) framable -- for an already-framable winner the floor is 1
        # and a lower sub-1 value is immaterial.  Roughly halves the cost in
        # framable regions.
        winner_framable = np.isfinite(conf_opt) and conf_opt <= 1.0 + FRA_ONE_TOL
        if not winner_framable and f'prev_S_{de}' in data:
            conf_prev = _conf(data[f'prev_S_{de}'], gate)
            if np.isfinite(conf_prev) and conf_prev < best_val:
                best_val, best_S = conf_prev, np.asarray(data[f'prev_S_{de}'])

        if f'unconf_fra_{de}' not in data:                # capture once
            data[f'unconf_fra_{de}'] = np.array(float(data[f'opt_fra_{de}']))
        data[f'conf_fra_{de}'] = np.array(best_val)
        data[f'opt_fra_{de}'] = np.array(best_val)         # corrected
        data[f'opt_S_{de}'] = best_S
        n_conf += 1

        # Only *false positives* -- points the scan claimed framable that are
        # not, after confirming the best stored frame -- need cluster
        # re-optimisation.  Genuinely non-framable points are corrected in place
        # and left out of the manifest (re-optimising them is the whole scan).
        claimed = float(data[f'unconf_fra_{de}']) <= 1.0 + FRA_ONE_TOL
        confirmed_framable = np.isfinite(best_val) and best_val <= 1.0 + FRA_ONE_TOL
        if claimed and not confirmed_framable:
            manifest.append((name, ix, iy, de, p1, p2,
                             float(data[f'unconf_fra_{de}']),
                             best_val if np.isfinite(best_val) else np.inf,
                             int(claimed)))
            n_man += 1

    data['conf_version'] = np.array(CONF_VERSION)
    if not args.dry_run:
        tmp = f.with_name(f'{f.stem}.tmp{os.getpid()}.npz')
        np.savez(tmp, **data)
        os.replace(tmp, f)
    return n_conf, n_man


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', type=str, default='results_trotter_v3')
    p.add_argument('--models', type=str,
                   default='model1,model2,model3,model4,model5')
    p.add_argument('--out_manifest', type=str, default='recompute_manifest',
                   help='basename for the <name>.npz and <name>.txt manifest')
    p.add_argument('--force', action='store_true',
                   help='re-confirm points already stamped with CONF_VERSION')
    p.add_argument('--dry_run', action='store_true',
                   help='report only; do not modify any npz')
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    names = [s.strip() for s in args.models.split(',') if s.strip()]
    manifest: list = []
    t0 = time.perf_counter()

    for name in names:
        model = MODELS[name]
        n_files = n_conf = n_man0 = 0
        n_man0 = len(manifest)
        for ix in range(model.N_X):
            for iy in range(model.N_Y):
                f = in_dir / name / f'pt_{ix:03d}_{iy:03d}.npz'
                if not f.exists():
                    continue
                c, _ = process_point(f, name, args, manifest)
                n_files += 1
                n_conf += c
        print(f'{name}: {n_files} points confirmed, '
              f'{len(manifest) - n_man0} (point,d_ext) need recompute  '
              f'[{time.perf_counter() - t0:.0f}s elapsed]', flush=True)

    # ---- manifest --------------------------------------------------------
    if manifest:
        arr = np.array([(MODELS_IDX[m], ix, iy, de, p1, p2, un, cf, cl)
                        for (m, ix, iy, de, p1, p2, un, cf, cl) in manifest],
                       dtype=float)
        np.savez(f'{args.out_manifest}.npz', rows=arr,
                 columns=np.array(['model_idx', 'ix', 'iy', 'de', 'p1', 'p2',
                                   'unconf_fra', 'conf_fra', 'was_claimed']),
                 model_names=np.array(list(MODELS)))
        with open(f'{args.out_manifest}.txt', 'w') as fh:
            fh.write('# model ix iy d_ext p1 p2 unconf_fra conf_fra was_claimed\n')
            for (m, ix, iy, de, p1, p2, un, cf, cl) in manifest:
                cf_s = 'inf' if not np.isfinite(cf) else f'{cf:.6f}'
                fh.write(f'{m} {ix} {iy} {de} {p1:.4f} {p2:.4f} '
                         f'{un:.6f} {cf_s} {cl}\n')
        print(f'\nRecompute manifest ({len(manifest)} entries) -> '
              f'{args.out_manifest}.npz / .txt', flush=True)
    else:
        print('\nNo points need recomputation: every stored frame confirms '
              '<= 1 + FRA_ONE_TOL.', flush=True)

    print(f'\n{"DRY RUN -- no files modified" if args.dry_run else "npz updated in place"}. '
          f'Total {time.perf_counter() - t0:.0f}s.', flush=True)


MODELS_IDX = {name: i for i, name in enumerate(MODELS)}

if __name__ == '__main__':
    main()
