"""
Optimised Heisenberg framability rate mu*(Delta) on the diagonal cut
Delta_1 = Delta_2 = Delta of the model10 rate figure, one line per d_ext.

Sources:
    d_ext = 4 / 8 / 12: panel_rate_heis_* of --seeded_npz (written by
        scripts/model10_seeded_panels_collect.py; the values drawn in
        model10_rate_panels.png, i.e. the per-point minima over every
        certified upper bound)
    d_ext = 6: rate_heis_6 of --base_npz (the earlier, unseeded pipeline:
        base run + nrefine / qrefine rounds; not in the seeded pipeline),
        floored by the d_ext = 4 value (padding a d4 frame gives a valid d6
        frame)
    Pauli reference: rate_pauli of --base_npz (equals stabilizer-3 on the cut)

Each legend entry states the grid onset of mu* = 0: the smallest Delta from
which the rate stays within CONTOUR_TOL of 0 to the end of the cut.

Usage:
    python scripts/model10_opt_rate_diagonal_plot.py [--dmax 2.5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CONTOUR_TOL = 1e-6          # as scripts/model10_panels_common.py

# (d_ext, colour, linestyle): one blue ramp, light -> dark with d_ext
DEXT_STYLE = [
    (4, '#6da7ec', '-'),
    (6, '#2a78d6', (0, (5, 2.5))),
    (8, '#1c5cab', '-'),
    (12, '#0d366b', '-'),
]
PAULI_COLOR = '#8a8984'


def diag(Z) -> np.ndarray:
    Z = np.asarray(Z, float)
    n = min(Z.shape)
    return Z[np.arange(n), np.arange(n)]


def zero_onset(delta, rate, tol: float = CONTOUR_TOL):
    """Smallest Delta with rate <= tol from there to the end of the cut
    (None if the last point is already above tol)."""
    on = np.isfinite(rate) & (rate <= tol)
    if not on[-1]:
        return None
    i = len(on) - 1
    while i > 0 and on[i - 1]:
        i -= 1
    return float(delta[i])


def load_cuts(seeded_npz, base_npz):
    """(delta, {d_ext: mu* on the diagonal}, Pauli mu* on the diagonal, title)."""
    s = np.load(seeded_npz, allow_pickle=True)
    b = np.load(base_npz, allow_pickle=True)
    delta = np.asarray(s['delta1_vals'], float)
    for d in (s['delta2_vals'], b['delta1_vals'], b['delta2_vals']):
        if not np.allclose(delta, d):
            raise SystemExit('Delta grids differ: no common diagonal cut')
    cuts = {m: diag(s[f'panel_rate_heis_{m}']) for m in (4, 8, 12)}
    cuts[6] = np.fmin(diag(b['rate_heis_6']), cuts[4])
    return delta, cuts, diag(b['rate_pauli']), str(b['title'])


def add_io_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument('--seeded_npz', type=str,
                    default='results_model10_rate/model10_seeded_panels.npz')
    ap.add_argument('--base_npz', type=str,
                    default='results_model10_rate/model10_rate_panels.npz')


def main() -> None:
    ap = argparse.ArgumentParser()
    add_io_args(ap)
    ap.add_argument('--out_png', type=str,
                    default='results_model10_rate/model10_opt_rate_diagonal.png')
    ap.add_argument('--dmax', type=float, default=1.3,
                    help='right end of the Delta axis (the grid ends at 2.5)')
    args = ap.parse_args()

    delta, cuts, pauli, title = load_cuts(args.seeded_npz, args.base_npz)

    print(f'{"Delta":>6s} ' + ' '.join(f'{"d" + str(m):>8s}' for m, *_ in DEXT_STYLE)
          + f' {"Pauli":>8s}')
    for i, dv in enumerate(delta):
        print(f'{dv:6.2f} ' + ' '.join(f'{cuts[m][i]:8.4f}' for m, *_ in DEXT_STYLE)
              + f' {pauli[i]:8.4f}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    clip = lambda r: np.where(r <= CONTOUR_TOL, 0.0, r)
    dc = zero_onset(delta, pauli)
    ax.plot(delta, clip(pauli), color=PAULI_COLOR, lw=1.5, ls=(0, (2, 2)),
            label=rf'Pauli (= stabilizer-3)  ($\mu^*=0$ for $\Delta\geq{dc:.2f}$)')
    for m, color, ls in DEXT_STYLE:
        r = cuts[m]
        dc = zero_onset(delta, r)
        on = (rf'$\mu^*=0$ for $\Delta\geq{dc:.2f}$' if dc is not None
              else r'$\mu^*>0$ on the cut')
        note = ', earlier pipeline' if m == 6 else ''
        ax.plot(delta, clip(r), color=color, ls=ls, lw=2,
                label=rf'$d_{{\rm ext}}={m}${note}  ({on})')
        print(f'd_ext={m:<2d} mu*(0) = {r[0]:.4f}   mu* = 0 from '
              f'{"-" if dc is None else f"{dc:.2f}"}')

    # past the last onset every line sits on 0: crop and say so
    last = max(zero_onset(delta, r) or delta[-1] for r in (*cuts.values(), pauli))
    if args.dmax < delta[-1] and last <= args.dmax:
        ax.text(0.98, 0.06, rf'$\mu^*=0$ for every $d_{{\rm ext}}$' '\n'
                rf'on ${last:.2f}\leq\Delta\leq{delta[-1]:.2f}$' '\n(end of the grid)',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                color='0.35')

    ax.axhline(0, color='0.6', lw=0.8, zorder=0)
    ax.set_xlim(delta[0], min(args.dmax, delta[-1]))
    ax.set_ylim(-0.08, 1.05 * max(np.nanmax(pauli), *(np.nanmax(r) for r in cuts.values())))
    ax.set_xlabel(r'$\Delta$   ($\Delta_1=\Delta_2=\Delta$)')
    ax.set_ylabel(r'optimised framability rate $\mu^*$')
    ax.set_title(f'{title}\n'
                 r'opt Heisenberg rate on the diagonal cut, by $d_{\rm ext}$',
                 fontsize=10)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.25)

    png = Path(args.out_png)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'wrote {png}', flush=True)


if __name__ == '__main__':
    main()
