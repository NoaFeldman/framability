"""
Collect + figures of the reweighted-Pauli w_min scan of the open Dicke model
(scripts/dicke_pauli_weight_worker.py).

Figures (FIG_DIR, default results_model4_rate):
  <family>_N<N>_pauli_weight_curves.png
      F*(w_min) against 1/w_min (log-log), one panel per case (uniform /
      disordered).  Solid = shared weights, dashed = per-species weights.
      A curve that reaches F = 0 ends with a downward triangle at 1/w_star;
      dotted horizontal = pair-floor average (lower bound in ANY frame).
      Reading: with w_min = N^-c the bottom axis is N^c.
  <family>_N<N>_pauli_weight_cost.png
      log10 of the optimal sample count against N at total times --T:
          ln(samples) = min_w [ 2 k' ln(1/w) + N F*(w) T ],
      k' = 1 (one-site observable), N F T = 2 x total rate (N/2) F x T
      (pair-averaged Dicke bound).  The pair patch is the N of the run;
      for Dicke it depends on N only through (N-1)/N, (N-2)/N (~1% between
      N = 100 and 1000), so the curve is a fixed-patch extrapolation.
      Flat = zero rate at finite w_star; ~log N growth = power law
      F ~ w^p (polynomial, degree 2k'/p); linear in N = plateau.
  <family>_N<N>_pauli_weight_wstar.png   (only if grid files exist)
      log10 w_star over the (lam/lam_c, gamma_down) grid per width and mode,
      with the plain Pauli zero region outlined; grey = not reached.

Also writes <root>/<family>_N<N>_M<M>_pauli_weight_summary.npz and prints a
table (label, expectation, Pauli F, F at the smallest w_min, tail slope,
w_star, floor, generic-LP deviation).

Usage:
    python scripts/dicke_pauli_weight_collect.py
    python scripts/dicke_pauli_weight_collect.py --T 10 100 --fig_dir results_model4_rate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                          # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from framability_rate_pauli_weights import local_slopes                  # noqa: E402
from trotter_lindbladian_scan import MODELS                              # noqa: E402

_ZERO = 1e-12
_COLORS = {'zero': 'tab:green', 'edge': 'tab:orange', 'plateau': 'tab:purple',
           'certified': 'tab:red'}


def load_points(root: Path, family: str, N: int, M: int) -> list:
    d = root / f'{family}_N{N}_M{M}' / 'points'
    out = []
    for f in sorted(d.glob('*.npz')):
        if f.name.endswith('.tmp.npz'):
            continue
        with np.load(f, allow_pickle=True) as z:
            out.append({k: z[k] for k in z.files})
    return out


def opt_log_samples(w, F, N, T, k_obs=1):
    """min over the sweep levels of 2 k ln(1/w) + N F(w) T (natural log)."""
    w, F = np.asarray(w, float), np.asarray(F, float)
    vals = 2 * k_obs * np.log(1.0 / w)[None, :] + np.outer(N, F) * T
    return vals.min(axis=1)


def summary_row(r, mode):
    w, F = r['w_grid'], r[f'F_{mode}']
    sl = local_slopes(w, F)
    tail = sl[np.isfinite(sl)][-3:]
    return dict(label=str(r['label']), expect=str(r['expect']), mode=mode,
                width=float(r['width']), lam=float(r['lam_ratio']), y=float(r['y']),
                F_pauli=float(r['F_pauli']), F_end=float(F[-1]), s_end=float(r[f's_{mode}'][-1]),
                slope=float(np.mean(tail)) if tail.size else np.nan,
                wstar=float(r[f'wstar_{mode}']), floor=float(r['floor_avg']),
                ref_dev=float(r[f'ref_dev_{mode}']), nonmono=int(r[f'nonmono_{mode}']))


def plot_curves(pts, path: Path, title: str) -> None:
    cases = [('uniform $\\omega_0$', [r for r in pts if float(r['width']) == 0]),
             ('disordered $\\omega_0$', [r for r in pts if float(r['width']) > 0])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, (name, rs) in zip(axes, cases):
        for r in rs:
            c = _COLORS.get(str(r['expect']), 'k')
            inv = 1.0 / r['w_grid']
            lbl = (f"{r['label']}: $\\lambda/\\lambda_c$={float(r['lam_ratio']):g}, "
                   f"$\\gamma_\\downarrow$={float(r['y']):g}")
            for mode, ls in (('shared', '-'), ('species', '--')):
                if f'F_{mode}' not in r:
                    continue
                F = np.asarray(r[f'F_{mode}'], float)
                pos = F > _ZERO
                ax.plot(inv[pos], F[pos], ls, color=c, marker='o', ms=2.5,
                        label=lbl if mode == 'shared' else None)
                ws = float(r[f'wstar_{mode}'])
                if np.isfinite(ws):
                    ax.plot([1.0 / ws], [ax_floor_val(F)], 'v', color=c, ms=8,
                            mfc='none' if mode == 'species' else c)
            if float(r['floor_avg']) > 0:
                ax.axhline(float(r['floor_avg']), color=c, ls=':', lw=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$1/w_{\min}$  ($=N^{c}$ for $w_{\min}=N^{-c}$)')
        ax.set_title(name)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=7, loc='lower left')
    axes[0].set_ylabel(r'reweighted Pauli rate $F^*(w_{\min})$')
    fig.suptitle(title + '\nsolid: shared weights, dashed: per-species weights; '
                 r'$\blacktriangledown$ at $1/w_\star$ (rate reaches 0); dotted: floor',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f'[fig] {path}')


def ax_floor_val(F):
    pos = np.asarray(F)[np.asarray(F) > _ZERO]
    return float(pos.min()) * 0.5 if pos.size else 1e-3


def plot_cost(pts, path: Path, title: str, Ts, k_obs: int) -> None:
    Ns = np.logspace(1, 8, 141)
    fig, axes = plt.subplots(1, len(Ts), figsize=(5.2 * len(Ts), 4.3), squeeze=False)
    for ax, T in zip(axes[0], Ts):
        for r in pts:
            c = _COLORS.get(str(r['expect']), 'k')
            for mode, ls in (('shared', '-'), ('species', '--')):
                if f'F_{mode}' not in r:
                    continue
                ls_ = opt_log_samples(r['w_grid'], r[f'F_{mode}'], Ns, T, k_obs) / np.log(10)
                ax.plot(Ns, ls_, ls, color=c,
                        label=(f"{r['label']} ({r['expect']})" if mode == 'shared' else None))
            Fp = float(r['F_pauli'])
            ax.plot(Ns, Ns * Fp * T / np.log(10), ':', color=c, lw=0.8)
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=1.0)
        ax.set_xlabel('N')
        ax.set_title(f'T = {T:g}')
        ax.grid(alpha=0.3, which='both')
    axes[0, 0].set_ylabel(r'$\log_{10}$ samples (optimal $w_{\min}$)')
    axes[0, -1].legend(fontsize=7, loc='upper left')
    fig.suptitle(title + f'\noptimal over w: 2k\'ln(1/w) + N F*(w) T, k\'={k_obs}; '
                 'dotted: unit-weight Pauli', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f'[fig] {path}')


def plot_wstar_maps(root: Path, family: str, N: int, M: int, path: Path) -> bool:
    dirs = sorted(root.glob(f'{family}_N{N}_M{M}_w*'))
    panels = []
    for d in dirs:
        files = [f for f in d.glob('pt_*.npz') if not f.name.endswith('.tmp.npz')]
        if not files:
            continue
        recs = []
        for f in files:
            with np.load(f, allow_pickle=True) as z:
                recs.append({k: z[k] for k in z.files})
        spec = MODELS[f'{family}_N{N}']            # full grid axes (missing points stay nan)
        lam = np.asarray(spec.p1_vals, float)
        yv = np.asarray(spec.p2_vals, float)
        nx, ny = len(lam), len(yv)
        width = float(recs[0]['width'])
        for mode in ('shared', 'species'):
            if f'wstar_{mode}' not in recs[0]:
                continue
            W = np.full((nx, ny), np.nan)
            P0 = np.full((nx, ny), np.nan)
            for r in recs:
                W[int(r['ix']), int(r['iy'])] = float(r[f'wstar_{mode}'])
                P0[int(r['ix']), int(r['iy'])] = float(r['F_pauli'])
            panels.append((width, mode, lam, yv, W, P0, len(recs)))
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
    for ax, (width, mode, lam, yv, W, P0, n) in zip(axes[0], panels):
        Z = np.log10(W).T
        ax.pcolormesh(lam, yv, np.isnan(Z).astype(float), cmap='Greys', vmin=0, vmax=3,
                      shading='nearest')
        im = ax.pcolormesh(lam, yv, np.ma.masked_invalid(Z), cmap='viridis_r',
                           vmin=-8, vmax=0, shading='nearest')
        try:
            cs = ax.contour(lam, yv, Z, levels=[-6, -4, -3, -2, -1], colors='w', linewidths=0.8)
            ax.clabel(cs, fmt=lambda v: f'N={10 ** (-v):.0e}', fontsize=6)
            ax.contour(lam, yv, (P0 <= _ZERO).astype(float).T, levels=[0.5], colors='r',
                       linewidths=1.0)
        except (ValueError, TypeError):
            pass
        ax.set_title(f'w = {width:g}, {mode} ({n} pts)', fontsize=9)
        ax.set_xlabel(r'$\lambda/\lambda_c$')
        fig.colorbar(im, ax=ax, label=r'$\log_{10} w_\star$')
    axes[0, 0].set_ylabel(r'$\gamma_\downarrow$')
    fig.suptitle(f'{family} N={N}: smallest weight needed for rate 0 (reweighted Pauli); '
                 'red: plain Pauli zero; grey: not reached; white labels: N = 1/w* (c = 1)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'[fig] {path}')
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--family', default='dicke_dec')
    ap.add_argument('--N', type=int, default=100)
    ap.add_argument('--M', type=int, default=4)
    ap.add_argument('--root', default='results_dicke_pauli_weight')
    ap.add_argument('--fig_dir', default='results_model4_rate')
    ap.add_argument('--T', nargs='+', type=float, default=[10.0, 100.0])
    ap.add_argument('--k_obs', type=int, default=1, help='support of the observable terms')
    args = ap.parse_args()

    root = REPO / args.root
    fig_dir = REPO / args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    pts = load_points(root, args.family, args.N, args.M)
    print(f'[collect] {len(pts)} point files')
    rows = []
    for r in pts:
        for mode in ('shared', 'species'):
            if f'F_{mode}' in r:
                rows.append(summary_row(r, mode))
    hdr = (f'{"pt":4s} {"expect":9s} {"mode":7s} {"Pauli":>7s} {"F(wend)":>9s} {"s(wend)":>9s} '
           f'{"slope":>6s} {"w*":>9s} {"floor":>7s} {"LPdev":>8s} nonmono')
    print(hdr)
    for q in rows:
        print(f'{q["label"]:4s} {q["expect"]:9s} {q["mode"]:7s} {q["F_pauli"]:7.4f} '
              f'{q["F_end"]:9.3e} {q["s_end"]:9.2e} {q["slope"]:6.2f} {q["wstar"]:9.2e} '
              f'{q["floor"]:7.4f} {q["ref_dev"]:8.1e} {q["nonmono"]}')
        if q['F_end'] < q['floor'] - 1e-7:
            print(f'   [BUG] {q["label"]} {q["mode"]}: below the floor')
    if rows:
        np.savez(root / f'{args.family}_N{args.N}_M{args.M}_pauli_weight_summary.npz',
                 **{k: np.array([q[k] for q in rows]) for k in rows[0]})
    title = f'{args.family}, N = {args.N}: reweighted Pauli frame, weights >= w_min'
    if pts:
        plot_curves(pts, fig_dir / f'{args.family}_N{args.N}_pauli_weight_curves.png', title)
        plot_cost(pts, fig_dir / f'{args.family}_N{args.N}_pauli_weight_cost.png', title,
                  args.T, args.k_obs)
    if not plot_wstar_maps(root, args.family, args.N, args.M,
                           fig_dir / f'{args.family}_N{args.N}_pauli_weight_wstar.png'):
        print('[collect] no grid files (run the worker with --grid for the w* maps)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
