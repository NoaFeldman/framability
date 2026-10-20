"""
Collect the KW-asymmetry scan (scripts/kw_asym_worker.py) of one run tag into
one file, and report missing points and the consistency checks.

Reads   <in_dir>/<tag>/pt_<PPP>.npz      (missing points stay NaN)
Writes  <in_dir>/<tag>/summary.npz with
    t                         (n_t,)
    diag_h, diag_gamma        the (h, D1 = D2 = gamma) panel axes
    off_d1, off_d2            the (D1, D2) panel axes (h = J = 1)
    diag_<key>                (n_h, n_gamma, n_t)   key in a b c tr tr_hat dS_gen dS_even
    off_<key>                 (n_d1, n_d2, n_t)
    diag_wall, off_wall       per-point wall time [s]
    tag, n_missing

Printed checks: max |dS_gen - dS_even| and max |b - a| (zero on the even
sector), max |Tr rho - 1| and |Tr rho_hat - 1|, and the off panel's D1 = D2
points against the diag panel at h = 1 (the same runs twice).

Usage:
    python scripts/kw_asym_collect.py --tag ring12_local_ghz_s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kw_asymmetry import (scan_points, DIAG_H, DIAG_GAMMA, OFF_D1, OFF_D2,  # noqa: E402
                          OBS_KEYS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--in_dir', default='results_kw_asym')
    args = ap.parse_args(argv)

    run = Path(args.in_dir) / args.tag
    pts = scan_points()
    t = None
    shapes = {'diag': (len(DIAG_H), len(DIAG_GAMMA)), 'off': (len(OFF_D1), len(OFF_D2))}
    data, missing = {}, []
    for p, pt in enumerate(pts):
        f = run / f'pt_{p:03d}.npz'
        if not f.exists():
            missing.append(p)
            continue
        z = np.load(f, allow_pickle=False)
        if t is None:
            t = z['t']
            for pan, sh in shapes.items():
                for k in OBS_KEYS:
                    data[f'{pan}_{k}'] = np.full(sh + (len(t),), np.nan)
                data[f'{pan}_wall'] = np.full(sh, np.nan)
        elif len(z['t']) != len(t) or np.abs(z['t'] - t).max() > 1e-12:
            raise SystemExit(f'{f}: time grid differs from the first file')
        pan, i, j = pt['panel'], pt['i'], pt['j']
        if abs(float(z['h']) - pt['h']) > 1e-12 or abs(float(z['d1']) - pt['d1']) > 1e-12 \
                or abs(float(z['d2']) - pt['d2']) > 1e-12:
            raise SystemExit(f'{f}: parameters do not match scan_points()[{p}]')
        for k in OBS_KEYS:
            data[f'{pan}_{k}'][i, j] = z[k]
        data[f'{pan}_wall'][i, j] = float(z['wall'])

    print(f'[kw collect] {args.tag}: {len(pts) - len(missing)}/{len(pts)} points')
    if missing:
        print(f'[kw collect] missing points: {missing}')
    if t is None:
        raise SystemExit('[kw collect] no point files found')

    def mx(x):
        return float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else float('nan')

    for pan in shapes:
        print(f'[kw collect] {pan}: max|dS_gen - dS_even| = '
              f"{mx(data[f'{pan}_dS_gen'] - data[f'{pan}_dS_even']):.2e}, "
              f"max|b - a| = {mx(data[f'{pan}_b'] - data[f'{pan}_a']):.2e}, "
              f"max|Tr rho - 1| = {mx(data[f'{pan}_tr'] - 1):.2e}, "
              f"max|Tr rho_hat - 1| = {mx(data[f'{pan}_tr_hat'] - 1):.2e}, "
              f"wall median {np.nanmedian(data[f'{pan}_wall']):.0f} s / "
              f"max {np.nanmax(data[f'{pan}_wall']):.0f} s")

    ih = int(np.argmin(np.abs(DIAG_H - 1.0)))
    diffs = []
    for i1, a in enumerate(OFF_D1):
        for j2, b in enumerate(OFF_D2):
            jg = np.flatnonzero(np.abs(DIAG_GAMMA - a) < 1e-12)
            if abs(a - b) < 1e-12 and len(jg):
                diffs.append(mx(data['off_dS_even'][i1, j2] - data['diag_dS_even'][ih, jg[0]]))
    if diffs:
        print(f'[kw collect] off panel D1 = D2 vs diag panel at h = 1: '
              f'max|dS difference| = {np.nanmax(diffs):.2e} over {len(diffs)} point(s)')

    out = run / 'summary.npz'
    np.savez(out, t=t, diag_h=DIAG_H, diag_gamma=DIAG_GAMMA, off_d1=OFF_D1,
             off_d2=OFF_D2, tag=args.tag, n_missing=len(missing), **data)
    print(f'[kw collect] wrote {out}')


if __name__ == '__main__':
    main()
