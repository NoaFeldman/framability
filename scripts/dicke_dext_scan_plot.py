"""
Best framability rate against d_ext of the atom-only open Dicke model at
zero local dephasing (gamma_phi = 0), one line per lam / lam_c, one panel
per N -- from the d_ext scan of scripts/submit_dicke_dext_scan.sh:

    results_dicke_N<N>_seeded/dicke_N<N>_dext_scan/summary.npz
        (scripts/model10_dext_scan_collect.py --model dicke_N<N> --p1 ... --p2 0)

y = the lowest certified rate any search / hop restart or the seeded
pipeline (d_ext 4, 8, 12) found at that d_ext, nested in d_ext (a frame
padded to more columns keeps its rate) -- an UPPER bound on mu*(d_ext).
  filled marker   converged (>= MIN_HITS restarts agree, >= 1 of them cold,
                  last hop round gained <= tol)
  open marker     not converged, or pipeline value only (d_ext not scanned)
  dotted line     floor: spectral abscissa of the bond generator, a lower
                  bound for EVERY frame at every d_ext (--no_floor skips it)

    python scripts/dicke_dext_scan_plot.py                   # every N on disk
    python scripts/dicke_dext_scan_plot.py --ns 10 100 --logy

Writes <out>.png (default results_model4_rate/dicke_dext_scan.png) and
prints the table.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.lines import Line2D                                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (MODELS, MODEL_DICKE_N,              # noqa: E402
                                      MODEL_DICKE_OMEGA, MODEL_DICKE_OMEGA0,
                                      MODEL_DICKE_KAPPA, build_bond_lindbladian)

# validated categorical slots 1-4 (dataviz reference palette), fixed order
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
MARKERS = ['o', 's', '^', 'D']
TEXT = '#0b0b0b'
MUTED = '#52514e'


def summary_path(root: Path, model: str) -> Path:
    return root / f'results_{model}_seeded' / f'{model}_dext_scan' / 'summary.npz'


def floor(model: str, p1: float, p2: float) -> float:
    """Spectral abscissa of the bond generator (frame-free lower bound)."""
    spec = MODELS[model]
    M = build_bond_lindbladian(*spec.build(float(p1), float(p2)), spec.dim)
    return float(np.max(np.linalg.eigvals(M).real))


def load(root: Path, n: int, with_floor: bool) -> dict | None:
    model = f'dicke_N{n}'
    f = summary_path(root, model)
    if not f.exists():
        print(f'[dicke dext plot] N={n}: {f} missing -- skipped', flush=True)
        return None
    z = np.load(f, allow_pickle=True)
    p1, p2 = np.asarray(z['p1'], float), np.asarray(z['p2'], float)
    d = dict(model=model, n=n, p1=p1, p2=p2, d_ext=np.asarray(z['d_ext'], int),
             rate=np.asarray(z['rate'], float), scanned=np.asarray(z['scanned'], bool),
             converged=np.asarray(z['converged'], bool),
             source=np.asarray(z['source'], str))
    d['floor'] = (np.array([floor(model, a, b) for a, b in zip(p1, p2)])
                  if with_floor else None)
    return d


def spread(ys: list, gap: float) -> list:
    """Shift sorted label positions apart by at least `gap` (keeps order)."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for k in range(1, len(order)):
        i, j = order[k], order[k - 1]
        out[i] = max(out[i], out[j] + gap)
    return list(out)


def draw(ax, d: dict, logy: bool) -> None:
    x_all = d['d_ext']
    ends = []
    for ip, lam in enumerate(d['p1']):
        c, mk = COLORS[ip % len(COLORS)], MARKERS[ip % len(MARKERS)]
        y = d['rate'][ip]
        ok = np.isfinite(y) & ((y > 0) if logy else True)
        if not ok.any():
            continue
        ax.plot(x_all[ok], y[ok], '-', color=c, lw=2, zorder=3)
        good = ok & d['scanned'][ip] & d['converged'][ip]
        bad = ok & ~good
        ax.plot(x_all[good], y[good], mk, ms=8, color=c, mec='white', mew=1.5,
                ls='none', zorder=4)
        ax.plot(x_all[bad], y[bad], mk, ms=8, mfc='white', mec=c, mew=2,
                ls='none', zorder=4)
        if d['floor'] is not None:
            fl = d['floor'][ip]
            if not logy or fl > 0:
                ax.axhline(max(fl, 0.0), color=c, ls=':', lw=1.2, zorder=2)
        ends.append((float(y[ok][-1]), float(x_all[ok][-1]), lam))
    # selective direct labels at the right end, in text ink
    if not logy:
        ax.set_ylim(bottom=0.0)
    if ends:
        lo, hi = ax.get_ylim()
        if logy:
            ys = spread([np.log10(e[0]) for e in ends],
                        0.06 * (np.log10(hi) - np.log10(lo)))
            ys = [10 ** v for v in ys]
        else:
            ys = spread([e[0] for e in ends], 0.06 * (hi - lo))
        for (_, x, lam), yl in zip(ends, ys):
            ax.annotate(rf'${lam:g}\,\lambda_c$', (x, yl), xytext=(8, 0),
                        textcoords='offset points', va='center', fontsize=9,
                        color=TEXT)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ns', type=int, nargs='+', default=sorted(MODEL_DICKE_N.values()),
                   help='N values (default: every registered dicke_N<N>; missing skipped)')
    p.add_argument('--root', default='.', help='repo root holding results_dicke_N*_seeded')
    p.add_argument('--out', default='results_model4_rate/dicke_dext_scan')
    p.add_argument('--logy', action='store_true')
    p.add_argument('--no_floor', action='store_true')
    args = p.parse_args()

    root = Path(args.root)
    data = [d for d in (load(root, n, not args.no_floor) for n in args.ns) if d is not None]
    if not data:
        sys.exit('[dicke dext plot] no summary file')

    nc = min(len(data), 3)
    nr = -(-len(data) // nc)
    fig, axes = plt.subplots(nr, nc, figsize=(max(5.2 * nc, 7.0), 4.0 * nr + 1.2),
                             squeeze=False, constrained_layout=True)
    for k, d in enumerate(data):
        ax = axes[k // nc, k % nc]
        if args.logy:
            ax.set_yscale('log')
        draw(ax, d, args.logy)
        dx = d['d_ext']
        ax.set_xticks(dx)
        ax.set_xlim(dx.min() - 0.6, dx.max() + 2.4)       # room for the labels
        ax.set_title(rf'$N={d["n"]}$,  $\gamma_\phi={d["p2"][0]:g}$', fontsize=11,
                     color=TEXT)
        ax.set_xlabel(r'$d_{\rm ext}$', color=TEXT)
        ax.set_ylabel(r'best rate found (upper bound on $\mu^*$)', color=TEXT)
        ax.grid(True, color='#e4e3df', lw=0.8, zorder=0)
        ax.tick_params(colors=MUTED)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    for k in range(len(data), nr * nc):
        axes[k // nc, k % nc].set_visible(False)

    # one legend: series identity + marker / line meaning
    lams = data[0]['p1']
    handles = [Line2D([], [], color=COLORS[i % 4], marker=MARKERS[i % 4], lw=2, ms=7,
                      mec='white', label=rf'$\lambda/\lambda_c={lam:g}$')
               for i, lam in enumerate(lams)]
    handles += [Line2D([], [], color=MUTED, marker='o', ls='none', ms=7, mec='white',
                       label='converged'),
                Line2D([], [], color=MUTED, marker='o', ls='none', ms=7, mfc='white',
                       mew=2, label='not converged / pipeline only')]
    if not args.no_floor:
        handles.append(Line2D([], [], color=MUTED, ls=':', lw=1.2,
                              label='floor (spectral abscissa, any frame)'))
    fig.legend(handles=handles, loc='outside lower center',
               ncol=2 if nc == 1 else 4, fontsize=9, frameon=False)
    fig.suptitle('Open Dicke model (cavity eliminated), zero local dephasing:\n'
                 r'best framability rate vs $d_{\rm ext}$;  '
                 rf'$\omega={MODEL_DICKE_OMEGA:g}$, $\kappa={MODEL_DICKE_KAPPA:g}$, '
                 rf'$\omega_0={MODEL_DICKE_OMEGA0:g}$', fontsize=11, color=TEXT)
    out = Path(args.out).with_suffix('.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'[dicke dext plot] wrote {out}', flush=True)

    # ---- table ----------------------------------------------------------------
    for d in data:
        print(f'[dicke dext plot] N={d["n"]}  (* = converged, p = padded / pipeline)')
        print(f'  {"lam/lam_c":>9s} ' + ' '.join(f'{m:>11d}' for m in d['d_ext'])
              + (f' {"floor":>11s}' if d['floor'] is not None else ''))
        for ip, lam in enumerate(d['p1']):
            cells = []
            for im in range(len(d['d_ext'])):
                v = d['rate'][ip, im]
                tag = ('*' if d['scanned'][ip, im] and d['converged'][ip, im]
                       else ('p' if str(d['source'][ip, im]) != 'scan' else ' '))
                cells.append(f'{v:10.6f}{tag}' if np.isfinite(v) else f'{"-":>11s}')
            fl = (f' {d["floor"][ip]:11.6f}' if d['floor'] is not None else '')
            print(f'  {lam:9.2f} ' + ' '.join(cells) + fl)


if __name__ == '__main__':
    main()
