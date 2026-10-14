"""
Optimised Heisenberg framability rate mu* as a function of d_ext at fixed
points Delta_1 = Delta_2 = Delta of the model10 diagonal cut (one line per
Delta).

Data, first available:
  --scan_npz (results_model10_rate/model10_dext_scan/summary.npz, written by
      scripts/model10_dext_scan_collect.py): every d_ext of the scan and the
      pipelines, nested in d_ext.  Markers:
        filled circle  scanned, converged (>= min_hits restarts agree, at
                       least min_cold_hits of them cold, last hop round
                       gained nothing)
        open circle    scanned, not converged
        square         pipeline value, not re-tested by the scan
      faint dots: every independent search restart (cold and seeded)
  else the stored figure data (model10_opt_rate_diagonal_plot.load_cuts):
      d_ext 4 / 8 / 12 of the seeded pipeline (squares) and d_ext 6 of the
      earlier, unseeded pipeline (open circle)

Usage:
    python scripts/model10_opt_rate_vs_dext_plot.py
    python scripts/model10_opt_rate_vs_dext_plot.py --no_scan --deltas 0.5 0.6 0.7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model10_opt_rate_diagonal_plot import add_io_args, load_cuts      # noqa: E402

# blue ramp steps 300..700; n lines take evenly spaced steps (n = 3: the
# validated ordinal ramp #6da7ec / #256abf / #0d366b)
RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
        '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
MAX_LINES = 5
EARLIER = (6,)              # no-scan mode: d_ext of the earlier pipeline
# status -> (marker, open?, legend label)
STATUS = {
    'conv':  ('o', False, 'scan: converged'),
    'open':  ('o', True, 'scan: not converged'),
    'pipe':  ('s', False, 'pipeline value, not re-tested'),
    'early': ('o', True, 'earlier (unseeded) pipeline'),
}


def ramp_colors(n: int) -> list:
    if n == 1:
        return [RAMP[5]]
    return [RAMP[int(round(i))] for i in np.linspace(1, len(RAMP) - 1, n)]


def rows_from_scan(f: Path, deltas) -> tuple:
    """[(Delta, d_ext, rate, status, {d_ext: restart values})], title."""
    z = np.load(f, allow_pickle=True)
    dext = np.asarray(z['d_ext'], int)
    rows = []
    for ip, dv in enumerate(np.asarray(z['deltas'], float)):
        if deltas is not None and not np.any(np.isclose(dv, deltas)):
            continue
        scanned, conv = z['scanned'][ip], z['converged'][ip]
        src = np.asarray(z['source'][ip], str)
        status = np.where(scanned & conv & (src == 'scan'), 'conv',
                          np.where(scanned, 'open', 'pipe'))
        restarts = {int(m): z['search_vals'][ip, im][np.isfinite(z['search_vals'][ip, im])]
                    for im, m in enumerate(dext) if scanned[im]}
        ok = np.isfinite(z['rate'][ip])
        rows.append((float(dv), dext[ok], np.asarray(z['rate'][ip], float)[ok],
                     status[ok], restarts))
    return rows, str(z['title'])


def rows_from_panels(args, deltas) -> tuple:
    delta, cuts, _, title = load_cuts(args.seeded_npz, args.base_npz)
    dext = np.array(sorted(cuts))
    status = np.where(np.isin(dext, EARLIER), 'early', 'pipe')
    rows = []
    for dv in sorted(deltas):
        i = int(np.argmin(np.abs(delta - dv)))
        if abs(delta[i] - dv) > 1e-9:
            raise SystemExit(f'Delta = {dv} is not on the grid (nearest {delta[i]:g})')
        rows.append((float(delta[i]), dext, np.array([cuts[m][i] for m in dext]),
                     status, {}))
    return rows, title


def main() -> None:
    ap = argparse.ArgumentParser()
    add_io_args(ap)
    ap.add_argument('--scan_npz', type=str,
                    default='results_model10_rate/model10_dext_scan/summary.npz')
    ap.add_argument('--no_scan', action='store_true',
                    help='ignore --scan_npz, draw the stored figure data')
    ap.add_argument('--deltas', type=float, nargs='+', default=None,
                    help='Delta values to draw (default: all of the scan, or '
                         '0.5 0.6 0.7 without one)')
    ap.add_argument('--out_png', type=str,
                    default='results_model10_rate/model10_opt_rate_vs_dext.png')
    args = ap.parse_args()

    scan = Path(args.scan_npz)
    if scan.exists() and not args.no_scan:
        rows, title = rows_from_scan(scan, args.deltas)
        print(f'data: {scan}')
    else:
        rows, title = rows_from_panels(args, args.deltas or [0.5, 0.6, 0.7])
        print(f'data: {args.seeded_npz} + {args.base_npz}')
    if not rows:
        raise SystemExit('no Delta to draw')
    if len(rows) > MAX_LINES:
        raise SystemExit(f'at most {MAX_LINES} Delta values')

    for dv, dext, r, st, rs in rows:
        print(f'Delta={dv:.2f}: ' + '  '.join(
            f'd{m}={v:.6f}{"" if s in ("conv", "pipe") else "*"}'
            + (f' ({len(rs[m])} restarts)' if m in rs else '')
            for m, v, s in zip(dext, r, st)))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.0), constrained_layout=True)
    colors = ramp_colors(len(rows))
    top = 1.08 * max(r.max() for _, _, r, _, _ in rows)
    n_above = 0
    for k, ((dv, dext, r, st, rs), color) in enumerate(zip(rows, colors)):
        ax.plot(dext, r, color=color, lw=2, zorder=2, label=rf'$\Delta={dv:.2f}$')
        for key, (mk, hollow, _) in STATUS.items():
            sel = st == key
            if sel.any():
                ax.plot(dext[sel], r[sel], ls='none', marker=mk, ms=8, zorder=3,
                        color=color, mfc='white' if hollow else color,
                        mec=color if hollow else 'white', mew=2 if hollow else 1.5)
        off = (k - (len(rows) - 1) / 2) * 0.12
        for m, vals in rs.items():
            n_above += int((vals > top).sum())
            ax.plot(np.full(len(vals), m + off), vals, ls='none', marker='.',
                    ms=4, color=color, alpha=0.35, zorder=1)
        ax.text(dext[-1] + 0.3, r[-1], rf'$\Delta={dv:.1f}$', color='0.15',
                fontsize=9, va='center')
    used = {s for _, _, _, st, _ in rows for s in st}
    for key, (mk, hollow, lab) in STATUS.items():
        if key in used:
            ax.plot([], [], ls='none', marker=mk, ms=8, color='0.4',
                    mfc='white' if hollow else '0.4',
                    mec='0.4' if hollow else 'white', mew=2 if hollow else 1.5,
                    label=lab)
    if any(rs for *_, rs in rows):
        ax.plot([], [], ls='none', marker='.', ms=6, color='0.4', alpha=0.5,
                label='independent search restarts'
                      + (f' ({n_above} above the axis)' if n_above else ''))

    xt = sorted({int(m) for _, dext, _, _, _ in rows for m in dext})
    ax.set_xticks(xt)
    ax.set_xlim(xt[0] - 0.5, xt[-1] + 1.6)
    ax.set_ylim(0, top)
    ax.set_xlabel(r'$d_{\rm ext}$')
    ax.set_ylabel(r'optimised framability rate $\mu^*$')
    ax.set_title(f'{title}\n'
                 r'opt Heisenberg rate vs $d_{\rm ext}$ at $\Delta_1=\Delta_2=\Delta$',
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
