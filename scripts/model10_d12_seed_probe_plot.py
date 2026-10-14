"""
Figure of the d_ext = 12 seed probe (scripts/model10_d12_seed_probe_worker.py):
where the targets sit on the d_ext = 12 panel, which of them the new families
improved and by how much, which family did it, and what the winning frames
look like.  Read-only; works with targets.npz alone (targets over the panel)
and fills in the rest once probe point files are on disk.

Row 1:  the figure's d_ext = 12 panel (min over d_ext 4/8/12, viridis, white
        rate-0 contour) with every target outlined -- grey: not processed,
        white: processed, no gain; orange: improved (> --gain); red: new
        rate-0 point  |  the drop  rate_ref - rate_new  at the processed
        targets  |  per family: points where its optimum alone beats the
        reference and points where it is the origin of the final gain
Row 2:  the --n_frames largest improvements: the 11 free columns of the new
        frame projected on the YZ plane, colour = X component, marker size =
        1 + |c_I| (the identity share).  A column off the YZ plane (lat*)
        shows as a coloured point; a warped polygon as uneven spacing.

Usage:
    python scripts/model10_d12_seed_probe_plot.py --out_dir results_model10_rate_dim2
    python scripts/model10_d12_seed_probe_plot.py                      # 1D run
Output: <probe_dir>/model10_d12_seed_probe.png  (or --out_png)
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from model10_seeded_worker import MODEL                                   # noqa: E402
from model10_panels_common import draw_panel, FRA_CMAP                    # noqa: E402
from model10_d12_seed_probe_worker import probe_dir, S_KEY                # noqa: E402
from model10_d12_seed_probe_collect import load_probe, CONTROL_NAMES      # noqa: E402

STATUS_COLORS = {'pending': '0.55', 'no gain': 'white', 'improved': 'tab:orange',
                 'new rate-0': 'tab:red'}


def status_of(row, gain: float) -> str:
    if row is None:
        return 'pending'
    if row['new_zero']:
        return 'new rate-0'
    return 'improved' if row['v_ref'] - row['v_new'] > gain else 'no gain'


def draw_frame(ax, S, title: str):
    """Free columns of S (4 x 12) on the YZ plane; colour = X, size = c_I."""
    free = np.asarray(S, float)[:, 1:]
    cI, cx, cy, cz = free
    sc = ax.scatter(cy, cz, c=cx, cmap='coolwarm', vmin=-1, vmax=1,
                    s=40 * (1 + 3 * np.abs(cI)), edgecolors='k', linewidths=0.5,
                    zorder=3)
    ax.scatter(-cy, -cz, c=-cx, cmap='coolwarm', vmin=-1, vmax=1,
               s=40 * (1 + 3 * np.abs(cI)), edgecolors='k', linewidths=0.5,
               alpha=0.35, zorder=2)                       # the +- partners
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color='0.7', lw=0.8, zorder=1)
    ax.axhline(0, color='0.85', lw=0.6)
    ax.axvline(0, color='0.85', lw=0.6)
    ax.set_aspect('equal')
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title(title, fontsize=8)
    return sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default=MODEL, choices=sorted(MODELS))
    ap.add_argument('--out_dir', type=str, default='results_model10_rate')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out_png', type=str, default=None)
    ap.add_argument('--gain', type=float, default=1e-6)
    ap.add_argument('--n_frames', type=int, default=4)
    args = ap.parse_args()
    spec = MODELS[args.model]
    pd = probe_dir(args.out_dir, args.stride, args.model)
    tf = pd / 'targets.npz'
    if not tf.exists():
        sys.exit(f'{tf} missing (run the prepare job first)')
    t = np.load(tf, allow_pickle=True)
    p1, p2 = np.asarray(t['p1'], float), np.asarray(t['p2'], float)
    panel = np.asarray(t['panel'], float)
    tix, tiy = np.asarray(t['ix'], int), np.asarray(t['iy'], int)
    rows = load_probe(pd)
    by_pt = {(r['ix'], r['iy']): r for r in rows}
    out_png = Path(args.out_png) if args.out_png else pd / 'model10_d12_seed_probe.png'

    improved = [r for r in rows if r['v_ref'] - r['v_new'] > args.gain]
    improved.sort(key=lambda r: r['v_new'] - r['v_ref'])
    n_fr = min(args.n_frames, len(improved))
    n_rows = 2 if n_fr else 1
    fig = plt.figure(figsize=(16, 4.6 * n_rows))
    gs = fig.add_gridspec(n_rows, 3 if n_rows == 1 else max(3, n_fr),
                          height_ratios=[1] * n_rows)
    xl, yl = spec.p1_label, spec.p2_label
    ncol = gs.ncols
    span = max(1, ncol // 3)

    # ---- (a) panel + targets -------------------------------------------------
    ax = fig.add_subplot(gs[0, 0:span])
    draw_panel(fig, ax, p1, p2, panel,
               rf'$d_{{\rm ext}}=12$ panel, {len(tix)} targets ({str(t["targets"])})',
               FRA_CMAP, floor_contour=0.0, xlabel=xl, ylabel=yl)
    counts = Counter()
    for ix, iy in zip(tix, tiy):
        st = status_of(by_pt.get((ix, iy)), args.gain)
        counts[st] += 1
        ax.scatter(p1[ix], p2[iy], marker='s', s=26, facecolors='none',
                   edgecolors=STATUS_COLORS[st], linewidths=1.1, zorder=4)
    for st, c in STATUS_COLORS.items():
        if counts[st]:
            ax.scatter([], [], marker='s', s=26, facecolors='none', edgecolors=c,
                       label=f'{st} ({counts[st]})')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.7)

    # ---- (b) drop at the processed targets -----------------------------------
    ax = fig.add_subplot(gs[0, span:2 * span])
    if rows:
        drop = np.full(panel.shape, np.nan)
        for r in rows:
            drop[r['ix'], r['iy']] = max(r['v_ref'] - r['v_new'], 0.0)
        pos = drop[np.isfinite(drop)]
        ax.pcolormesh(p1, p2, np.where(np.isfinite(panel), 0.0, np.nan).T,
                      cmap='Greys', vmin=0, vmax=1, shading='nearest', alpha=0.15)
        cm = ax.scatter(p1[[r['ix'] for r in rows]], p2[[r['iy'] for r in rows]],
                        c=[max(r['v_ref'] - r['v_new'], 0.0) for r in rows],
                        cmap='magma_r', vmin=0, vmax=max(float(pos.max()), 1e-9),
                        marker='s', s=34, edgecolors='k', linewidths=0.4)
        fig.colorbar(cm, ax=ax, label=r'drop  $\mu^*_{\rm ref}-\mu^*_{\rm new}$')
        ax.set_xlim(p1[0], p1[-1])
        ax.set_ylim(p2[0], p2[-1])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f'{len(rows)} processed: {len(improved)} improved, '
                     f'{sum(r["new_zero"] for r in rows)} new rate-0, '
                     f'largest drop {max(pos.max(), 0):.2e}', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'no probe point files yet\n(targets.npz only)',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()

    # ---- (c) per family ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2 * span:])
    if rows:
        fams = sorted({n for r in rows for n in r['names']})
        alone, origin = Counter(), Counter()
        for r in rows:
            for n, v in zip(r['names'], r['vals']):
                if np.isfinite(v) and r['v_ref'] - v > args.gain:
                    alone[n] += 1
            if r in improved:
                m = re.match(r'probe (\S+) \+polish', r['label'])
                if m:
                    origin[m.group(1)] += 1
        y = np.arange(len(fams))
        ax.barh(y + 0.2, [alone[f] for f in fams], height=0.4, label='optimum alone < ref')
        ax.barh(y - 0.2, [origin[f] for f in fams], height=0.4, label='origin of final gain')
        ax.set_yticks(y)
        ax.set_yticklabels([f + (' (ctrl)' if f in CONTROL_NAMES else '') for f in fams],
                           fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('points')
        ax.set_title('per family', fontsize=10)
        ax.legend(fontsize=7)
    else:
        ax.set_axis_off()

    # ---- row 2: winning frames -----------------------------------------------
    sc = None
    for k in range(n_fr):
        r = improved[k]
        S = np.asarray(np.load(r['file'], allow_pickle=True)[S_KEY], float)
        ax = fig.add_subplot(gs[1, k])
        sc = draw_frame(ax, S, f'({p1[r["ix"]]:.2f}, {p2[r["iy"]]:.2f})  '
                               f'{r["v_ref"]:.2e} -> {r["v_new"]:.2e}\n{r["label"]}')
    if sc is not None:
        fig.colorbar(sc, ax=fig.axes[-n_fr:], label='X component', shrink=0.8)

    fig.suptitle(f'{args.model} {args.out_dir}: d_ext = 12 seed probe', fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f'[probe plot] {len(tix)} targets, {len(rows)} processed, {len(improved)} '
          f'improved -> {out_png}', flush=True)


if __name__ == '__main__':
    main()
