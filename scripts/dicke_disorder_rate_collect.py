"""
Collect and draw the disordered-Dicke average-rate run
(scripts/dicke_disorder_rate_worker.py, framability_rate_disorder).

Reads  <root>/<family>_N<N>_M<M>_w<width:.2f>/pt_<ix>_<iy>.npz
writes <root>/<family>_N<N>_M<M>_panels.npz       arrays (n_widths, n_x, n_y)
       <fig_dir>/<family>_N<N>_disorder_panels.png
           rows: widths;  cols: Pauli | shared d_1 | ... | shared d_max | per-species d_max
           white mu = 0 (F <= 1e-6), red dashed every pair floor = 0 (outside it
           no frame assignment reaches 0), black dashed mean-field transition of
           the species mixture (N -> inf, mixture_lam_c)
       <fig_dir>/<family>_N<N>_disorder_changes.png
           rows: widths > 0;  cols: disorder cost shared(w) - shared(0) at d_max |
           species gain shared - species at d_max | the same at d_min
       <fig_dir>/<family>_N<N>_disorder_cuts.png
           F against lam/lam_c at the --cuts of the second axis, one line per
           width, shared (solid) and per-species (dashed), d_max
and prints, per width: mu = 0 counts (shared / species), mean F, the largest
species gain and disorder cost, missing points, batched-vs-reference LP
mismatches; for w = 0 also the deviation from the stored uniform panels
(results_<family>_N<N>_seeded/..._rate_panels.npz) and the check of the
mixture transition against dicke_n_compare_plot.mf_lam_c.

Missing points are NaN (drawn blank), so it can run while the array is going:
    python scripts/dicke_disorder_rate_collect.py
    python scripts/dicke_disorder_rate_collect.py --families dicke_dec --ns 20 --widths 0 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from trotter_lindbladian_scan import MODELS                              # noqa: E402
from framability_rate_disorder import (family_params, family_rates, lam_c0,  # noqa: E402
                                       species_omegas, mixture_lam_c)
from dicke_n_compare_plot import mf_lam_c                                # noqa: E402

ZERO_TOL = 1e-6
FLOOR_TOL = 1e-9
SCALARS = ['F_pauli', 'floor_max', 'floor_avg']
PER_D = ['F_seed', 'F_shared', 'F_shared_ref', 'F_shared_cert', 'F_species',
         'F_species_ref', 'F_species_cert', 'worst_shared', 'worst_species', 't']


def folder(root, fam, N, M, w):
    return root / f'{fam}_N{N}_M{M}_w{w:.2f}'


def load_width(root, fam, N, M, w, d_exts, nx, ny):
    keys = SCALARS + [f'{k}_{m}' for k in PER_D for m in d_exts]
    out = {k: np.full((nx, ny), np.nan) for k in keys}
    d = folder(root, fam, N, M, w)
    n = 0
    for f in sorted(d.glob('pt_*_*.npz')) if d.is_dir() else []:
        if f.name.endswith('.tmp.npz'):
            continue
        try:
            z = np.load(f, allow_pickle=True)
            ix, iy = int(z['ix']), int(z['iy'])
        except Exception as e:                              # noqa: BLE001
            print(f'  unreadable {f}: {e}')
            continue
        n += 1
        for k in keys:
            if k in z.files:
                out[k][ix, iy] = float(z[k])
    return out, n


def transition_curve(fam, w, M, y):
    """lam_c(y) / lam_c0 of the species mixture (nan where none below 6 lam_c0)."""
    p = family_params(fam)
    om = species_omegas(p['omega0'], w, M)
    ab = np.full(len(om), 1.0 / len(om))
    lc = lam_c0(fam)
    xs = []
    for yy in y:
        gd, gp = family_rates(fam, yy)
        xs.append(mixture_lam_c(gd + 4.0 * gp, om, ab, p['omega'], p['kappa'],
                                lam_max=6.0 * lc) / lc)
    return np.array(xs)


def check_transition(fam, y):
    """mixture_lam_c at one species vs the closed-form uniform mf_lam_c."""
    p = family_params(fam)
    worst = 0.0
    for yy in y:
        gd, gp = family_rates(fam, yy)
        a = mixture_lam_c(gd + 4.0 * gp, [p['omega0']], [1.0], p['omega'], p['kappa'],
                          lam_max=6.0 * lam_c0(fam))
        b = mf_lam_c(gd + 4.0 * gp, p['omega'], p['omega0'], p['kappa'])
        if np.isfinite(a) and np.isfinite(b):
            worst = max(worst, abs(a - b) / b)
        elif np.isfinite(a) != np.isfinite(b):
            worst = np.inf
    return worst


def draw(ax, x, y, Z, F, xt, vmin, vmax, cmap='viridis'):
    im = ax.pcolormesh(x, y, Z.T, shading='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    ok = np.isfinite(Z)
    if ok.any() and np.nanmin(Z) <= ZERO_TOL:
        ax.contour(x, y, np.where(ok, (Z <= ZERO_TOL).astype(float), np.nan).T,
                   levels=[0.5], colors='w', linewidths=1.2)
    if F is not None and np.isfinite(F).any() and np.nanmin(F) <= FLOOR_TOL < np.nanmax(F):
        ax.contour(x, y, (F <= FLOOR_TOL).astype(float).T, levels=[0.5],
                   colors='r', linewidths=1.0, linestyles='--')
    if xt is not None:
        sel = np.isfinite(xt) & (xt <= x.max())
        ax.plot(xt[sel], y[sel], 'k--', lw=1.0)
    ax.set_xlim(x.min() - 0.05, x.max() + 0.05)
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--families', nargs='+', default=['dicke_dec', 'dicke_dec_k025'])
    ap.add_argument('--ns', nargs='+', type=int, default=[100])
    ap.add_argument('--widths', nargs='+', type=float, default=[0.0, 0.2, 0.5, 0.8])
    ap.add_argument('--M', type=int, default=4)
    ap.add_argument('--d_exts', nargs='+', type=int, default=[4, 8])
    ap.add_argument('--root', default='results_dicke_disorder')
    ap.add_argument('--fig_dir', default='results_model4_rate')
    ap.add_argument('--cuts', nargs='+', type=float, default=[0.04, 0.1])
    args = ap.parse_args()

    root = REPO / args.root
    fig_dir = REPO / args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    widths = sorted(args.widths)
    d_exts = sorted(args.d_exts)
    dmin, dmax = d_exts[0], d_exts[-1]

    for fam in args.families:
        p = family_params(fam)
        for N in args.ns:
            spec = MODELS[f'{fam}_N{N}']
            x, y = np.asarray(spec.p1_vals, float), np.asarray(spec.p2_vals, float)
            ylab = r'$\gamma_\downarrow$' if p['axis'] == 'gamma_down' else r'$\gamma_\phi$'
            print(f'===== {fam} N={N}: kappa={p["kappa"]:g} omega0={p["omega0"]:g} M={args.M} =====')
            dev = check_transition(fam, y)
            print(f'  mixture transition at one species vs mf_lam_c: max rel dev {dev:.2e}'
                  + ('' if dev < 1e-6 else '   <-- CHECK mixture_mf_matrix'))
            data, trans = {}, {}
            for w in widths:
                data[w], n = load_width(root, fam, N, args.M, w, d_exts, len(x), len(y))
                trans[w] = transition_curve(fam, w, args.M, y)
                miss = len(x) * len(y) - n
                print(f'  w={w:.2f}: {n} points' + (f', {miss} MISSING' if miss else ''))
            have = [w for w in widths if np.isfinite(data[w][f'F_shared_cert_{dmax}']).any()]
            if not have:
                print('  no data -- skipped')
                continue

            # ---- summary -----------------------------------------------------
            ref0 = data.get(0.0)
            for w in have:
                D = data[w]
                for m in d_exts:
                    sh, sp = D[f'F_shared_cert_{m}'], D[f'F_species_cert_{m}']
                    ok = np.isfinite(sh)
                    if not ok.any():
                        continue
                    gain = sh - sp
                    mism = np.nansum(np.abs(D[f'F_shared_{m}'] - D[f'F_shared_ref_{m}']) > 1e-6)
                    line = (f'  w={w:.2f} d={m:2d}: mu=0 shared {int(np.nansum(sh <= ZERO_TOL))}'
                            f' species {int(np.nansum(sp <= ZERO_TOL))} / {int(ok.sum())};'
                            f' mean F shared {np.nanmean(sh):.4f} species {np.nanmean(sp):.4f};'
                            f' species gain max {np.nanmax(gain):.4f}'
                            f' (> 1e-3 at {int(np.nansum(gain > 1e-3))} pts)')
                    if ref0 is not None and w > 0:
                        cost = sh - ref0[f'F_shared_cert_{m}']
                        line += (f'; disorder cost max {np.nanmax(cost):.4f}'
                                 f' min {np.nanmin(cost):.4f}')
                    if mism:
                        line += f'; {int(mism)} batched/reference LP mismatches'
                    print(line)
            stored = (REPO / f'results_{fam}_N{N}_seeded' /
                      f'{fam}_N{N}_seeded_rate_panels.npz')
            if ref0 is not None and stored.exists():
                z = np.load(stored, allow_pickle=True)
                for m in d_exts:
                    k = f'panel_rate_heis_{m}'
                    if k in z.files and z[k].shape == ref0[f'F_shared_cert_{m}'].shape:
                        dd = ref0[f'F_shared_cert_{m}'] - np.asarray(z[k], float)
                        print(f'  w=0 vs stored uniform d={m}: max {np.nanmax(dd):+.2e} '
                              f'min {np.nanmin(dd):+.2e} (lower at {int(np.nansum(dd < -1e-6))} pts)')

            # ---- npz ---------------------------------------------------------
            keys = list(data[have[0]].keys())
            np.savez(root / f'{fam}_N{N}_M{args.M}_panels.npz',
                     widths=np.array(have), lam_ratio_vals=x, y_vals=y, y_name=p['axis'],
                     transition=np.array([trans[w] for w in have]),
                     **{k: np.array([data[w][k] for w in have]) for k in keys})

            # ---- panels ------------------------------------------------------
            cols = [('F_pauli', 'Pauli')] + \
                   [(f'F_shared_cert_{m}', rf'shared $S$, $d_{{\rm ext}}={m}$') for m in d_exts] + \
                   [(f'F_species_cert_{dmax}', rf'$S$ per species, $d_{{\rm ext}}={dmax}$')]
            nr, nc = len(have), len(cols)
            fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.3 * nr), squeeze=False)
            for c, (k, lab) in enumerate(cols):
                vals = np.concatenate([data[w][k].ravel() for w in have])
                vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
                for r, w in enumerate(have):
                    ax = axes[r, c]
                    im = draw(ax, x, y, data[w][k], data[w]['floor_max'], trans[w], 0.0, vmax)
                    ax.set_title(f'{lab},  $w={w:g}$', fontsize=9)
                    ax.set_xlabel(r'$\lambda/\lambda_c$')
                    ax.set_ylabel(ylab)
                fig.colorbar(im, ax=axes[:, c].tolist(), shrink=0.6, label=r'$\bar\mu^*$')
            fig.suptitle(
                f'Open Dicke model, $N={N}$, unequal $\\omega_0$: {args.M} species at '
                f'$\\omega_0(1 + w(2s+1-{args.M})/{args.M})$, $\\bar\\omega_0={p["omega0"]:g}$, '
                f'$\\omega=1$, $\\kappa={p["kappa"]:g}$; pair-averaged rate $\\bar\\mu^*$\n'
                r'white: $\bar\mu^*=0$;  red dashed: every pair floor $=0$;  '
                r'black dashed: mean-field transition of the mixture;  '
                r'$\lambda_c$ of the mean $\omega_0$', fontsize=10)
            f1 = fig_dir / f'{fam}_N{N}_disorder_panels.png'
            fig.savefig(f1, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f'  wrote {f1}')

            # ---- changes -----------------------------------------------------
            wpos = [w for w in have if w > 0]
            if wpos:
                chg = []
                if ref0 is not None:
                    chg.append((lambda D, m=dmax: D[f'F_shared_cert_{m}'] - ref0[f'F_shared_cert_{m}'],
                                rf'disorder cost: shared($w$) $-$ shared(0), $d={dmax}$'))
                chg.append((lambda D, m=dmax: D[f'F_shared_cert_{m}'] - D[f'F_species_cert_{m}'],
                            rf'species gain: shared $-$ per species, $d={dmax}$'))
                if dmin != dmax:
                    chg.append((lambda D, m=dmin: D[f'F_shared_cert_{m}'] - D[f'F_species_cert_{m}'],
                                rf'species gain, $d={dmin}$'))
                fig, axes = plt.subplots(len(wpos), len(chg),
                                         figsize=(4.2 * len(chg), 3.3 * len(wpos)), squeeze=False)
                for c, (fn, lab) in enumerate(chg):
                    Zs = [fn(data[w]) for w in wpos]
                    v = max(np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 0 for Z in Zs) or 1e-3
                    for r, (w, Z) in enumerate(zip(wpos, Zs)):
                        ax = axes[r, c]
                        im = ax.pcolormesh(x, y, Z.T, shading='nearest', cmap='RdBu_r',
                                           vmin=-v, vmax=v)
                        sel = np.isfinite(trans[w]) & (trans[w] <= x.max())
                        ax.plot(trans[w][sel], y[sel], 'k--', lw=1.0)
                        ax.set_title(f'{lab},  $w={w:g}$', fontsize=9)
                        ax.set_xlabel(r'$\lambda/\lambda_c$')
                        ax.set_ylabel(ylab)
                    fig.colorbar(im, ax=axes[:, c].tolist(), shrink=0.6)
                fig.suptitle(f'{fam}, $N={N}$: what unequal $\\omega_0$ costs, and what a frame '
                             f'per species buys (red = positive)', fontsize=10)
                f2 = fig_dir / f'{fam}_N{N}_disorder_changes.png'
                fig.savefig(f2, dpi=110, bbox_inches='tight')
                plt.close(fig)
                print(f'  wrote {f2}')

            # ---- cuts --------------------------------------------------------
            fig, axes = plt.subplots(1, len(args.cuts), figsize=(5.2 * len(args.cuts), 3.8),
                                     squeeze=False)
            colors = plt.cm.viridis(np.linspace(0, 0.85, len(have)))
            for c, g in enumerate(args.cuts):
                ax = axes[0, c]
                iy = int(np.argmin(np.abs(y - g)))
                for col, w in zip(colors, have):
                    ax.plot(x, data[w][f'F_shared_cert_{dmax}'][:, iy], 'o-', ms=3, color=col,
                            label=f'$w={w:g}$ shared')
                    if w > 0:
                        ax.plot(x, data[w][f'F_species_cert_{dmax}'][:, iy], 's--', ms=3,
                                color=col, mfc='none', label=f'$w={w:g}$ per species')
                    xt = trans[w][iy]
                    if np.isfinite(xt) and xt <= x.max():
                        ax.axvline(xt, color=col, ls=':', lw=1)
                ax.set_title(f'{ylab} $= {y[iy]:g}$,  $d_{{\\rm ext}}={dmax}$', fontsize=10)
                ax.set_xlabel(r'$\lambda/\lambda_c$')
                ax.set_ylabel(r'$\bar\mu^*$')
                ax.legend(fontsize=7)
            fig.suptitle(f'{fam}, $N={N}$: pair-averaged rate vs coupling for each disorder '
                         f'width (dotted: mixture transition)', fontsize=10)
            f3 = fig_dir / f'{fam}_N{N}_disorder_cuts.png'
            fig.savefig(f3, dpi=110, bbox_inches='tight')
            plt.close(fig)
            print(f'  wrote {f3}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
