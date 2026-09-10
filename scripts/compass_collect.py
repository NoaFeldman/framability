"""
Collect one compass-chain case and draw its eight-panel (gamma, h) figure.

Reads the per-unit npz files written by
  * scripts/compass_rates_worker.py    -> <in_dir>/<case>/rates/pt_<ig>_<ih>_<measure>.npz
  * scripts/compass_manybody_worker.py -> <in_dir>/<case>/manybody/pt_<ig>_<ih>.npz

and draws

    row 1 |  Pauli rate           |  stabilizer-3 rate  |  opt Heisenberg d=4  |  d=6
    row 2 |  opt Schrodinger d=4  |  d=6                |  8q osc rate         |  8q gap

Panels 1-6 (viridis) show the framability rate of the more expensive bond gate
at the balanced dephasing split.  They are drawn by
model4_rate_panels_collect._panel itself: each panel on its own finite colour
range, log scale only past 4 decades, and a white contour around the rate
floor mu* = 0.  A white x marks points where the split search did not reach
the relative-error target.  gamma = 0 is never marked: without dephasing there
is nothing to split.  Panels 7-8 (inferno) are the full-chain oscillation rate
and Lindbladian gap.

The merged grids, including each measure's chosen split alpha and both gates'
rates, go to <out_dir>/<case>/compass_<case>_panels.npz next to the png.

Usage:
    python scripts/compass_collect.py --case jx1.0_jy1.0_hx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compass_chain import CASES, GAMMA_VALS, H_VALS                      # noqa: E402
from compass_rates_worker import RATE_KEYS, unit_path                    # noqa: E402
from compass_manybody_worker import (TAG as MB_TAG, N_QUBITS,            # noqa: E402
                                     point_path)
from model4_rate_panels_collect import _panel                            # noqa: E402

FRA_CMAP = 'viridis'
MB_CMAP = 'inferno'

RATE_FIELDS = ('rate', 'alpha', 'mu_xx', 'mu_yy', 'mismatch')
MB_FIELDS = ('osc_rate', 'gap')


def mb_labels(n_qubits: int):
    return [('osc_rate', rf'{n_qubits}q osc rate  $\max_k|{{\rm Im}}\lambda_k/'
                         rf'{{\rm Re}}\lambda_k|$'),
            ('gap',      f'{n_qubits}q Lindbladian gap')]


def load_rates(in_dir, case_name: str):
    ng, nh = len(GAMMA_VALS), len(H_VALS)
    out, rel_tol = {}, None
    for key, _ in RATE_KEYS:
        grids = {f: np.full((ng, nh), np.nan) for f in RATE_FIELDS}
        unbalanced = np.zeros((ng, nh), dtype=bool)
        found = 0
        for ig in range(ng):
            for ih in range(nh):
                f = unit_path(in_dir, case_name, ig, ih, key)
                if not f.exists():
                    continue
                try:
                    d = np.load(f, allow_pickle=True)
                    for name in RATE_FIELDS:
                        grids[name][ig, ih] = float(d[name])
                    # n_evals == 1 means no search took place (gamma = 0, or
                    # balanced at the even share): never flag those.
                    unbalanced[ig, ih] = (not bool(d['balanced'])
                                          and int(d['n_evals']) > 1)
                    rel_tol = float(d['rel_tol'])
                except Exception as e:
                    print(f'  warning: {f.name}: {e}', flush=True)
                    continue
                found += 1
        print(f'[{case_name}] {key}: {found}/{ng * nh} points, '
              f'{int(unbalanced.sum())} not balanced', flush=True)
        out[key] = dict(grids, unbalanced=unbalanced)
    return out, rel_tol


def load_manybody(in_dir, case_name: str):
    ng, nh = len(GAMMA_VALS), len(H_VALS)
    grids = {f: np.full((ng, nh), np.nan) for f in MB_FIELDS}
    meta = dict(n_qubits=N_QUBITS, topology='chain')
    found = 0
    for ig in range(ng):
        for ih in range(nh):
            f = point_path(in_dir, case_name, ig, ih)
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
                for name in MB_FIELDS:
                    grids[name][ig, ih] = float(d[name])
                meta = dict(n_qubits=int(d['N']), topology=str(d['topology']))
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
                continue
            found += 1
    print(f'[{case_name}] {MB_TAG}: {found}/{ng * nh} points', flush=True)
    return grids, meta


def _axis_labels(ax) -> None:
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$h$')


def plot(case, rates: dict, mb: dict, meta: dict, rel_tol, png: Path, *,
         floor: float = 0.0) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    tol_txt = 'target' if rel_tol is None else f'{rel_tol:g} relative error'
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    fig.suptitle(
        f'{case.title}\n'
        rf'panels 1-6: rate $\mu^*$ of the more expensive bond gate at the '
        rf'balanced dephasing split (XX gate $\alpha\gamma$, YY gate '
        rf'$(1-\alpha)\gamma$; white x: {tol_txt} not reached)'
        f"  |  panels 7-8: full {meta['n_qubits']}-qubit {meta['topology']}",
        fontsize=13)

    for ax, (key, label) in zip(axes.flat[:6], RATE_KEYS):
        r = rates[key]
        _panel(fig, ax, GAMMA_VALS, H_VALS, r['rate'], label, FRA_CMAP,
               floor_contour=floor)
        ig, ih = np.nonzero(r['unbalanced'])
        if ig.size:
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            ax.plot(GAMMA_VALS[ig], H_VALS[ih], ls='none', marker='x',
                    color='white', ms=5, mew=1.2)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        _axis_labels(ax)

    for ax, (key, label) in zip(axes.flat[6:], mb_labels(meta['n_qubits'])):
        _panel(fig, ax, GAMMA_VALS, H_VALS, mb[key], label, MB_CMAP)
        _axis_labels(ax)

    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[{case.name}] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--case',    type=str, required=True, choices=list(CASES))
    ap.add_argument('--in_dir',  type=str, default='results_compass')
    ap.add_argument('--out_dir', type=str, default='results_compass')
    ap.add_argument('--floor',   type=float, default=0.0,
                    help='value the white contour outlines on the rate panels '
                         '(default 0.0 = the framability-rate floor)')
    args = ap.parse_args()

    case = CASES[args.case]
    out_dir = Path(args.out_dir) / case.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rates, rel_tol = load_rates(args.in_dir, case.name)
    mb, meta = load_manybody(args.in_dir, case.name)

    stem = f'compass_{case.name}_panels'
    arrays = dict(case=case.name, Jx=case.Jx, Jy=case.Jy, field=case.field,
                  gamma_vals=GAMMA_VALS, h_vals=H_VALS,
                  N_manybody=meta['n_qubits'], topology=meta['topology'],
                  rel_tol=(np.nan if rel_tol is None else rel_tol))
    for key, _ in RATE_KEYS:
        arrays[key] = rates[key]['rate']
        for name in RATE_FIELDS[1:] + ('unbalanced',):
            arrays[f'{key}_{name}'] = rates[key][name]
    arrays.update(mb)
    np.savez(out_dir / f'{stem}.npz', **arrays)
    print(f'[{case.name}] wrote {out_dir / (stem + ".npz")}', flush=True)

    plot(case, rates, mb, meta, rel_tol, out_dir / f'{stem}.png',
         floor=args.floor)


if __name__ == '__main__':
    main()
