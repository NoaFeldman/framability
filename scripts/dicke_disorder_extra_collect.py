"""
Collect scripts/dicke_disorder_extra_worker.py into one panel file per
(family, N, M):

    <root>/<family>_N<N>_M<M>_extra_panels.npz
        widths, lam_ratio_vals, y_vals, y_name,
        F_stab3, F_prod_<c>, F_schro_<d>      (n_widths, n_lam, n_y), NaN = missing
        F_schro_best                          point-wise min over the d of F_schro_<d>

(same layout as <family>_N<N>_M<M>_panels.npz of dicke_disorder_rate_collect.py,
which holds Pauli and the optimised observable frames).  Prints the number of
missing points per width and measure.

Usage:
    python scripts/dicke_disorder_extra_collect.py
    python scripts/dicke_disorder_extra_collect.py --families dicke_dec --widths 0 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from trotter_lindbladian_scan import MODELS                           # noqa: E402

PROD_CHIS = (10, 40)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', nargs='+', default=['dicke_dec', 'dicke_dec_k025'])
    ap.add_argument('--ns', nargs='+', type=int, default=[100])
    ap.add_argument('--widths', nargs='+', type=float, default=[0.0, 0.2, 0.5, 0.8])
    ap.add_argument('--M', type=int, default=4)
    ap.add_argument('--schro_dexts', nargs='+', type=int, default=[6, 8])
    ap.add_argument('--root', default='results_dicke_disorder')
    args = ap.parse_args()

    root = REPO / args.root
    keys = ['F_stab3'] + [f'F_prod_{c}' for c in PROD_CHIS] + \
           [f'F_schro_{d}' for d in args.schro_dexts]
    for fam in args.families:
        for N in args.ns:
            spec = MODELS[f'{fam}_N{N}']
            x, y = np.asarray(spec.p1_vals, float), np.asarray(spec.p2_vals, float)
            nw = len(args.widths)
            out = {k: np.full((nw, len(x), len(y)), np.nan) for k in keys}
            for iw, w in enumerate(args.widths):
                d = root / f'{fam}_N{N}_M{args.M}_w{w:.2f}_extra'
                for f in sorted(d.glob('pt_[0-9][0-9][0-9]_[0-9][0-9][0-9].npz')):
                    try:
                        with np.load(f, allow_pickle=True) as z:
                            ix, iy = int(z['ix']), int(z['iy'])
                            for k in keys:
                                if k in z.files:
                                    out[k][iw, ix, iy] = float(z[k])
                    except Exception as e:                    # noqa: BLE001
                        print(f'  unreadable {f}: {type(e).__name__}: {e}', flush=True)
                miss = '  '.join(f'{k[2:]}:{int(np.isnan(out[k][iw]).sum())}' for k in keys)
                print(f'[{fam}_N{N} M={args.M} w={w:.2f}] missing points  {miss}', flush=True)
            schro = [out[f'F_schro_{d}'] for d in args.schro_dexts]
            with np.errstate(invalid='ignore'):
                out['F_schro_best'] = np.fmin.reduce(schro) if len(schro) > 1 else schro[0]
            path = root / f'{fam}_N{N}_M{args.M}_extra_panels.npz'
            np.savez(path, widths=np.asarray(args.widths, float), lam_ratio_vals=x,
                     y_vals=y, y_name=spec.p2_name, schro_dexts=np.asarray(args.schro_dexts),
                     **out)
            print(f'  wrote {path}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
