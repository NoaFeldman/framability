"""
Cross-N comparison of the framability-rate panels of the atom-only open Dicke
model (trotter_lindbladian_scan._make_model_dicke / _make_model_dicke_dec,
dicke_lindbladian.md), from the per-N figure data of the seeded rate pipeline
(scripts/submit_dicke_rate.sh -> scripts/submit_rate_seeded.sh):

    results_<family>_N<N>_seeded/<family>_N<N>_seeded_rate_panels.npz

families (MODEL_DICKE_PARAMS): dicke (second axis gamma_phi, kappa = 1),
dicke_dec (gamma_down, gamma_phi = 0, kappa = 1), dicke_dec_k025 (the same,
kappa = 0.25).

Figure 1 (<out>_panels.png):
    rows | N = N_1 | N_2 | ... | N_1 - N_last (difference)
    cols | Pauli | opt Heisenberg d_ext = 4 | 8 | 12
  one colour scale per column across the N rows;
  white contour  = mu* = 0 (within CONTOUR_TOL);
  red dashed     = floor = 0: the spectral abscissa of the bond generator is a
                   lower bound on mu* for EVERY frame, so right of / below it
                   no frame of this patch reaches 0 (--no_floor skips it);
  black dashed   = mean-field transition: lam/lam_c = 1 on the gamma_phi axis
                   (a phase boundary only at gamma_phi = 0, on-site dephasing
                   destroys the transition), the curve lam_c(gamma_down)/lam_c
                   on the gamma_down axis (mf_lam_c, N -> inf).
Figure 2 (<out>_cuts.png): mu* (solid) and the floor (dashed) against
  lam/lam_c at the second-axis values --cuts, one colour per N, one panel per
  (column, cut).

Missing N files are skipped (with a message), so the script also draws a
partial comparison while some runs are still going.

    python scripts/dicke_n_compare_plot.py                       # dicke, every registered N
    python scripts/dicke_n_compare_plot.py --ns 5 20 100
    python scripts/dicke_n_compare_plot.py --family dicke_dec_k025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (MODELS, MODEL_DICKE_PARAMS,         # noqa: E402
                                      dicke_coefficients, build_bond_lindbladian)

CONTOUR_TOL = 1e-6          # as scripts/model10_panels_common.py
FLOOR_TOL = 1e-9            # spectral abscissa: a Lindbladian's own 0 is ~1e-15
COLS = [('rate_pauli', 'Pauli'),
        ('panel_rate_heis_4', r'opt $d_{\rm ext}=4$'),
        ('panel_rate_heis_8', r'opt $d_{\rm ext}=8$'),
        ('panel_rate_heis_12', r'opt $d_{\rm ext}=12$')]
DEFAULT_CUTS = {'gamma_phi': [0.0, 0.5], 'gamma_down': [0.04, 0.1]}


def family_models(family: str) -> dict:
    """{N: model name} of a Dicke family."""
    return {p['N']: name for name, p in MODEL_DICKE_PARAMS.items() if p['family'] == family}


def mf_lam_c(gamma_perp: float, omega: float, omega0: float, kappa: float) -> float:
    """Mean-field (N -> inf) critical coupling of the atom-only model with
    on-site coherence decay gamma_perp (note convention: gamma_down, or
    4 gamma_phi): Holstein-Primakoff J^- = sqrt N b around the all-down state,
        d<b>/dt = -(i(omega0 + u) + d) b - (i u + v) b*,
        u = lam^2 Im A,  v = lam^2 Im B,
        d = gamma_perp + kappa lam^2 (|c_-|^2 - |c_+|^2)   (net collective damping),
    eigenvalues -d +- sqrt(u^2 + v^2 - (omega0 + u)^2); unstable from the
    smallest s = lam^2 > 0 with -2 omega0 u - omega0^2 + v^2 = d^2.  Reduces to
    lam_c^2 = (omega^2 + kappa^2)(omega0^2 + gamma_perp^2)/(4 omega omega0)
    for omega0, gamma_perp << omega, and to dicke_coefficients' lam_c at 0."""
    co = dicke_coefficients(omega, omega0, kappa)
    a, b = co['A'].imag, co['B'].imag
    dl = kappa * (abs(co['c_m']) ** 2 - abs(co['c_p']) ** 2)
    # (b^2 - dl^2) s^2 + (-2 omega0 a - 2 gamma_perp dl) s - (omega0^2 + gamma_perp^2) = 0
    roots = np.roots([b ** 2 - dl ** 2, -2 * omega0 * a - 2 * gamma_perp * dl,
                      -(omega0 ** 2 + gamma_perp ** 2)])
    s = [r.real for r in roots if abs(r.imag) < 1e-12 and r.real > 0]
    return float(np.sqrt(min(s))) if s else np.nan


def transition(model: str, yy: np.ndarray):
    """(x, y) of the mean-field transition line in the (lam/lam_c, axis) plane."""
    p = MODEL_DICKE_PARAMS[model]
    if p['axis'] != 'gamma_down':
        return np.ones_like(yy), yy
    lam_c0 = dicke_coefficients(p['omega'], p['omega0'], p['kappa'])['lam_c']
    xs = np.array([mf_lam_c(g, p['omega'], p['omega0'], p['kappa']) for g in yy]) / lam_c0
    return xs, yy


def floor_grid(model: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spectral abscissa of the bond generator on the grid (frame-free lower bound)."""
    spec = MODELS[model]
    F = np.empty((len(x), len(y)))
    for i, a in enumerate(x):
        for j, b in enumerate(y):
            M = build_bond_lindbladian(*spec.build(float(a), float(b)), spec.dim)
            F[i, j] = np.max(np.linalg.eigvals(M).real)
    return F


def load(model: str, root: Path, with_floor: bool) -> dict | None:
    f = root / f'results_{model}_seeded' / f'{model}_seeded_rate_panels.npz'
    if not f.exists():
        print(f'[dicke compare] {model}: {f} missing -- skipped', flush=True)
        return None
    d = np.load(f, allow_pickle=True)
    spec = MODELS[model]
    out = {'x': np.asarray(d[f'{spec.p1_name}_vals'], float),
           'y': np.asarray(d[f'{spec.p2_name}_vals'], float)}
    for k, _ in COLS:
        out[k] = np.asarray(d[k], float) if k in d.files else None
    out['floor'] = floor_grid(model, out['x'], out['y']) if with_floor else None
    yy = np.linspace(out['y'].min(), out['y'].max(), 200)
    out['trans'] = transition(model, yy)
    return out


def draw_panel(ax, d, Z, vmin, vmax, cmap, overlays=True):
    x, y = d['x'], d['y']
    im = ax.pcolormesh(x, y, Z.T, shading='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    if overlays:
        if np.isfinite(Z).any() and np.nanmin(Z) <= CONTOUR_TOL:
            ax.contour(x, y, (Z <= CONTOUR_TOL).astype(float).T, levels=[0.5],
                       colors='w', linewidths=1.2)
        F = d['floor']
        if F is not None and F.min() <= FLOOR_TOL < F.max():
            ax.contour(x, y, (F <= FLOOR_TOL).astype(float).T, levels=[0.5],
                       colors='r', linestyles='--', linewidths=1.0)
    tx, ty = d['trans']
    keep = tx <= x.max() + 0.5 * (x[1] - x[0])
    ax.plot(np.where(keep, tx, np.nan), ty, color='k', ls='--', lw=0.8)
    ax.set_xlim(x.min() - 0.5 * (x[1] - x[0]), x.max() + 0.5 * (x[1] - x[0]))
    ax.set_ylim(y.min() - 0.5 * (y[1] - y[0]), y.max() + 0.5 * (y[1] - y[0]))
    return im


def main() -> None:
    families = sorted({p['family'] for p in MODEL_DICKE_PARAMS.values()})
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--family', default='dicke', choices=families)
    p.add_argument('--ns', type=int, nargs='+', default=None,
                   help='N values (default: every registered N of the family; '
                        'missing files skipped)')
    p.add_argument('--root', default='.', help='repo root holding results_<family>_N*_seeded')
    p.add_argument('--out', default=None,
                   help='output stem (default results_model4_rate/<family>_n_compare)')
    p.add_argument('--cuts', type=float, nargs='+', default=None,
                   help='second-axis values of the line-cut figure '
                        '(default 0, 0.5 for gamma_phi; 0.04, 0.1 for gamma_down)')
    p.add_argument('--no_floor', action='store_true',
                   help='skip the floor (spectral abscissa) contour and cut curves')
    args = p.parse_args()

    models = family_models(args.family)
    ns_req = args.ns if args.ns is not None else sorted(models)
    unknown = [n for n in ns_req if n not in models]
    if unknown:
        sys.exit(f'[dicke compare] {args.family}: N = {unknown} not registered '
                 f'(have {sorted(models)})')
    root = Path(args.root)
    data = {n: load(models[n], root, not args.no_floor) for n in ns_req}
    data = {n: d for n, d in data.items() if d is not None}
    if not data:
        sys.exit('[dicke compare] no data')
    ns = sorted(data)
    m0 = models[ns[0]]
    spec, par = MODELS[m0], MODEL_DICKE_PARAMS[m0]
    xlab, ylab = spec.p1_label, spec.p2_label
    cuts = args.cuts if args.cuts is not None else DEFAULT_CUTS[par['axis']]
    out = Path(args.out or f'results_model4_rate/{args.family}_n_compare')
    out.parent.mkdir(parents=True, exist_ok=True)
    noise = (r'$\gamma_\phi D[Z]$' if par['axis'] == 'gamma_phi'
             else r'$\gamma_\downarrow D[\sigma^-]$, $\gamma_\phi=0$')
    title = ('Open Dicke model (cavity eliminated) + ' + noise + ', framability rate of '
             r'the dimer patch $(N-1)\,\mathcal{L}_{ij}$;  '
             rf'$\omega={par["omega"]:g}$, $\kappa={par["kappa"]:g}$, '
             rf'$\omega_0={par["omega0"]:g}$' + '\n'
             r'white: $\mu^*=0$;  red dashed: floor $=0$ (no frame of this patch '
             r'reaches 0 beyond it);  black dashed: mean-field $\lambda_c$')

    # ---- figure 1: panels ----------------------------------------------------
    with_diff = len(ns) >= 2
    nrow = len(ns) + int(with_diff)
    fig, axes = plt.subplots(nrow, len(COLS), figsize=(4.0 * len(COLS), 3.2 * nrow),
                             squeeze=False, constrained_layout=True)
    for c, (key, label) in enumerate(COLS):
        grids = [data[n][key] for n in ns if data[n][key] is not None]
        if grids:
            vmax = max(float(np.nanmax(g)) for g in grids)
            vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        else:
            vmax = 1.0
        im = None
        for r, n in enumerate(ns):
            ax = axes[r, c]
            Z = data[n][key]
            if Z is None:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
            else:
                im = draw_panel(ax, data[n], Z, 0.0, vmax, 'viridis')
            ax.set_title(f'{label},  $N={n}$', fontsize=10)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
        if im is not None:
            fig.colorbar(im, ax=axes[:len(ns), c], shrink=0.8, label=r'$\mu^*$')
        if with_diff:
            ax = axes[-1, c]
            a, b = data[ns[0]], data[ns[-1]]
            if a[key] is not None and b[key] is not None and a[key].shape == b[key].shape:
                D = a[key] - b[key]
                lim = float(np.nanmax(np.abs(D))) if np.isfinite(D).any() else 0.0
                lim = lim if lim > 0 else 1e-12
                imd = draw_panel(ax, a, D, -lim, lim, 'RdBu_r', overlays=False)
                fig.colorbar(imd, ax=ax, shrink=0.8,
                             label=rf'$\mu^*_{{N={ns[0]}}}-\mu^*_{{N={ns[-1]}}}$')
            else:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
            ax.set_title(f'{label}:  $N={ns[0]}$ minus $N={ns[-1]}$', fontsize=10)
            ax.set_xlabel(xlab)
            ax.set_ylabel(ylab)
    fig.suptitle(title, fontsize=11)
    f1 = out.with_name(out.name + '_panels.png')
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print(f'[dicke compare] wrote {f1}', flush=True)

    # ---- figure 2: line cuts at fixed second-axis value ----------------------
    fig, axes = plt.subplots(len(cuts), len(COLS),
                             figsize=(4.0 * len(COLS), 3.0 * len(cuts)),
                             squeeze=False, constrained_layout=True)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for r, g in enumerate(cuts):
        for c, (key, label) in enumerate(COLS):
            ax = axes[r, c]
            for k, n in enumerate(ns):
                d = data[n]
                col = colors[k % len(colors)]
                iy = int(np.argmin(np.abs(d['y'] - g)))
                if d[key] is not None:
                    ax.plot(d['x'], d[key][:, iy], 'o-', ms=3, color=col, label=f'$N={n}$')
                if d['floor'] is not None:
                    ax.plot(d['x'], np.maximum(d['floor'][:, iy], 0.0), '--', lw=1,
                            color=col, label='floor' if k == 0 else None)
            tx, ty = data[ns[0]]['trans']
            ax.axvline(float(tx[int(np.argmin(np.abs(ty - g)))]), color='k', ls='--', lw=0.8)
            ax.set_title(rf'{label},  {ylab}$={g:g}$', fontsize=10)
            ax.set_xlabel(xlab)
            ax.set_ylabel(r'$\mu^*$')
            ax.legend(fontsize=8)
    f2 = out.with_name(out.name + '_cuts.png')
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f'[dicke compare] wrote {f2}', flush=True)

    # ---- numbers for the log ---------------------------------------------------
    for n in ns:
        F = data[n]['floor']
        if F is not None:
            print(f'[dicke compare] {"floor":>22s} N={n:<5d} min {F.min():.4g} '
                  f'max {F.max():.4g}  floor-0 points {int(np.sum(F <= FLOOR_TOL))}/{F.size}',
                  flush=True)
    for key, label in COLS:
        for n in ns:
            Z = data[n][key]
            if Z is None:
                continue
            nz = int(np.sum(Z <= CONTOUR_TOL))
            print(f'[dicke compare] {label:>22s} N={n:<5d} min {np.nanmin(Z):.4g} '
                  f'max {np.nanmax(Z):.4g}  rate-0 points {nz}/{Z.size}', flush=True)


if __name__ == '__main__':
    main()
