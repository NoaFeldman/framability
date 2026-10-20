"""
Quick figure of a collected KW-asymmetry scan (scripts/kw_asym_collect.py).

Panels:
  1. Delta S_KW at J t = --t_show on the (h, gamma) plane (D1 = D2 = gamma),
     the self-dual line h = J dashed (the paper's Fig. 2 analogue)
  2. the same on the (D1, D2) plane at h = J, the self-dual diagonal dashed
  3. Delta S_KW(t) along the h = J column of panel 1 (one curve per gamma)
  4. Delta S_KW(t) at gamma = --gamma_show for several h

Usage:
    python scripts/kw_asym_plot.py --summary results_kw_asym/ring12_local_ghz_s/summary.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--t_show', type=float, default=0.25, help='J t of panels 1-2')
    ap.add_argument('--gamma_show', type=float, default=0.6)
    ap.add_argument('--key', default='dS_even', choices=('dS_even', 'dS_gen'))
    ap.add_argument('--out_png', default=None)
    args = ap.parse_args(argv)

    z = np.load(args.summary, allow_pickle=False)
    t, H, G, D1, D2 = z['t'], z['diag_h'], z['diag_gamma'], z['off_d1'], z['off_d2']
    it = int(np.argmin(np.abs(t - args.t_show)))
    dg, do = z[f'diag_{args.key}'], z[f'off_{args.key}']
    tag = str(z['tag'])

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    a = ax[0, 0]
    m = a.pcolormesh(G, H, dg[:, :, it], shading='nearest', cmap='viridis')
    a.axhline(1.0, color='w', ls='--', lw=1)
    a.set_xlabel('gamma = D1 = D2'); a.set_ylabel('h / J')
    a.set_title(f'Delta S_KW at J t = {t[it]:.2f}')
    fig.colorbar(m, ax=a)

    a = ax[0, 1]
    m = a.pcolormesh(D2, D1, do[:, :, it], shading='nearest', cmap='viridis')
    lo, hi = min(D1.min(), D2.min()), max(D1.max(), D2.max())
    a.plot([lo, hi], [lo, hi], 'w--', lw=1)
    a.set_xlabel('D2'); a.set_ylabel('D1')
    a.set_title(f'h = J: Delta S_KW at J t = {t[it]:.2f}')
    fig.colorbar(m, ax=a)

    a = ax[1, 0]
    ih = int(np.argmin(np.abs(H - 1.0)))
    for j, g in enumerate(G):
        a.plot(t, dg[ih, j], color=plt.cm.plasma(j / max(len(G) - 1, 1)), label=f'{g:g}')
    a.set_xlabel('J t'); a.set_ylabel('Delta S_KW')
    a.set_title(f'self-dual h = J (D1 = D2 = gamma)')
    a.legend(title='gamma', fontsize=7, ncol=2)

    a = ax[1, 1]
    jg = int(np.argmin(np.abs(G - args.gamma_show)))
    for i, h in enumerate(H[::2]):
        a.plot(t, dg[2 * i, jg], color=plt.cm.coolwarm(2 * i / max(len(H) - 1, 1)),
               label=f'{h:g}')
    a.set_xlabel('J t'); a.set_ylabel('Delta S_KW')
    a.set_title(f'gamma = {G[jg]:g}')
    a.legend(title='h / J', fontsize=7, ncol=2)

    fig.suptitle(f'KW asymmetry, {tag}')
    fig.tight_layout()
    out = Path(args.out_png) if args.out_png else Path(args.summary).with_name(f'kw_asym_{tag}.png')
    fig.savefig(out, dpi=150)
    print(f'[kw plot] wrote {out}')


if __name__ == '__main__':
    main()
