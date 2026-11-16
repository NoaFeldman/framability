"""
Paper figure 2: best optimised Heisenberg framability rate against d_ext
(4, 6, ..., 16), one panel per model, one line per parameter point.

Reads the d_ext-scan summaries written by scripts/model10_dext_scan_collect.py
(<out_dir>/<model>_dext_scan/summary.npz; rate is already nested in d_ext, i.e.
the value at d_ext m is the best certified rate of any frame with <= m
columns).  Filled markers: the scan converged at that d_ext (independent
restarts agree and the last hop round gained nothing); open markers: not
converged, or a pipeline-only value -- an upper bound that may still drop.
Dicke panels also show the disordered model (unequal omega0, pair-averaged
rate, per-species frames) at the same points, one line style per width
(--disorder_root; those scans have no convergence test, so open markers).

--scans takes LABEL=PATH pairs; default: model4, model10 and the Dicke models
of --dicke_models at their standard locations (missing files are skipped).

Output: <out>.png and <out>.pdf (default results_plots/paper/rate_vs_dext).

Usage:
    python scripts/paper_dext_plot.py
    python scripts/paper_dext_plot.py --dicke_models dicke_dec_N100 dicke_dec_k025_N100 --logy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS, MODEL_DICKE_PARAMS          # noqa: E402

DEFAULT_SCANS = {
    'model4': 'results_model4_seeded/model4_dext_scan/summary.npz',
    'model10': 'results_model10_rate/model10_dext_scan/summary.npz',
}
# axis names -> LaTeX for the legend
TEX = {'gamma': r'\gamma', 'gamma_p': r"\gamma'", 'delta1': r'\Delta_1',
       'delta2': r'\Delta_2', 'lam_ratio': r'\lambda/\lambda_c',
       'gamma_phi': r'\gamma_\phi', 'gamma_down': r'\gamma_\downarrow'}


def panel_title(model: str) -> str:
    """Dicke models: 'Dicke, kappa = .., N = ..'; others: the model name."""
    p = MODEL_DICKE_PARAMS.get(model)
    if p is None:
        return model
    return rf'Dicke $\kappa={p["kappa"]:g}$, $N={p["N"]}$'


def point_labels(z) -> list:
    """Legend label of every scanned point."""
    if 'p1' in z.files and 'p1_name' in z.files:
        n1, n2 = str(z['p1_name']), str(z['p2_name'])
        return [rf'${TEX.get(n1, n1)}={a:g},\ {TEX.get(n2, n2)}={b:g}$'
                for a, b in zip(z['p1'], z['p2'])]
    return [rf'$\Delta_1=\Delta_2={d:g}$' for d in z['deltas']]   # model10 diagonal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scans', nargs='*', default=None,
                    help='LABEL=PATH to a summary.npz (default: model4, model10, '
                         '--dicke_models)')
    ap.add_argument('--dicke_models', nargs='*',
                    default=['dicke_dec_N100', 'dicke_dec_k025_N100'])
    ap.add_argument('--disorder_root', default='results_dicke_disorder_dext',
                    help='overlay the disordered-Dicke d_ext summaries of a panel '
                         'model found here (dashed / dotted per width; '
                         'scripts/dicke_disorder_dext_collect.py); "" = none')
    ap.add_argument('--out', default='results_plots/paper/rate_vs_dext')
    ap.add_argument('--logy', action='store_true')
    args = ap.parse_args()

    if args.scans:
        scans = dict(s.split('=', 1) for s in args.scans)
    else:
        scans = dict(DEFAULT_SCANS)
        for m in args.dicke_models:
            scans[m] = f'results_{m}_seeded/{m}_dext_scan/summary.npz'
    # disordered-Dicke overlays: <disorder_root>/<model>_M<M>_w<w>_dext_summary.npz
    overlays = {k: sorted(Path(args.disorder_root).glob(f'{k}_M*_w*_dext_summary.npz'))
                for k in scans} if args.disorder_root else {k: [] for k in scans}
    panels = {}
    for k, v in scans.items():
        p = Path(v)
        ok = p.exists()
        print(f'  {k:>16s}: {v}' + ('' if ok else '   (missing)')
              + ''.join(f'\n  {"":>16s}  + {o}' for o in overlays[k]))
        if ok or overlays[k]:
            panels[k] = (p if ok else None, overlays[k])
    if not panels:
        print('ERROR: no summary found', file=sys.stderr)
        sys.exit(1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), constrained_layout=True,
                             squeeze=False)
    styles = ['--', ':', '-.']
    for ax, (lab, (path, ovl)) in zip(axes[0], panels.items()):
        colours: dict = {}                    # (p1, p2) -> colour, shared by overlays
        d_all: set = set()
        model = lab
        sets = ([(path, None)] if path is not None else []) + [(o, 'w') for o in ovl]
        handles = []
        for iset, (p, kind) in enumerate(sets):
            with np.load(p, allow_pickle=True) as z:
                d_ext = np.asarray(z['d_ext'], int)
                rate = np.asarray(z['rate'], float)
                conv = (np.asarray(z['converged'], bool) if 'converged' in z.files
                        else np.zeros_like(rate, bool))
                labels = point_labels(z)
                keys = (list(zip(np.round(z['p1'], 9), np.round(z['p2'], 9)))
                        if 'p1' in z.files else [(float(d), float(d)) for d in z['deltas']])
                width = float(z['width']) if 'width' in z.files else 0.0
                if 'model' in z.files:
                    model = str(z['model'])
            d_all |= set(d_ext.tolist())
            ls = '-' if kind is None else styles[(iset - (path is not None)) % len(styles)]
            for ip, pl in enumerate(labels):
                y = rate[ip]
                c = colours.get(keys[ip])
                line, = ax.plot(d_ext, y, ls, lw=1.3, color=c,
                                label=pl if keys[ip] not in colours else None)
                c = colours.setdefault(keys[ip], line.get_color())
                ax.plot(d_ext[conv[ip]], y[conv[ip]], 'o', color=c, ms=5.5)
                ax.plot(d_ext[~conv[ip]], y[~conv[ip]], 'o', mfc='white', mec=c, ms=5.5)
                nc = int((~conv[ip]).sum())
                wl = '' if kind is None else f' [w={width:g}]'
                print(f'  {lab}{wl} {pl}: ' + '  '.join(f'{m}:{v:.4f}' for m, v in zip(d_ext, y))
                      + (f'   ({nc} not converged)' if nc else ''), flush=True)
            if kind is not None or ovl:
                handles.append(Line2D([], [], color='0.3', ls=ls,
                                      label=rf'$\omega_0$ width $w={width:g}$'
                                      if kind is not None else 'uniform'))
        ax.set_title(panel_title(model if lab in MODELS else lab))
        ax.set_xlabel(r'$d_{\rm ext}$')
        ax.set_ylabel(r'$\mu^*$ (best found)')
        ax.set_xticks(sorted(d_all))
        if args.logy:
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
        leg = ax.legend(fontsize=8, frameon=False)
        if handles:
            ax.add_artist(leg)
            ax.legend(handles=handles, fontsize=8, frameon=False, loc='lower left')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        f = out.with_suffix(f'.{ext}') if not args.logy else \
            out.with_name(out.name + '_logy').with_suffix(f'.{ext}')
        fig.savefig(f, dpi=200)
        print(f'wrote {f}', flush=True)


if __name__ == '__main__':
    main()
