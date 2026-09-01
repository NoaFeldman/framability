"""dt=0 extrapolation of the seven DT_BASE-line framabilities -> N colormaps/model.

For every (gamma, gamma') grid point of a model (the same 51x51 grid
trotter_lindbladian_scan scans), trotter_dtbase_line swept the Trotter control
DT_BASE over its bottom 10 values (see trotter_dtbase_line_worker.base_grid) and
stored the seven framabilities

    stab_fra   pauli_fra   opt_fra_4   opt_fra_6   gamma_ch1   sch_fra_6   prod_fra_10

together with the adaptive step  dt = DT_BASE / max(||H||_1, {gamma_k})  at each
DT_BASE.  Raw framability tends trivially to 1 as dt -> 0 (the gate -> identity);
the quantity with a non-trivial continuous-time limit is the per-unit-time power

    q(dt) = framability ** (1/dt)                 (the y-axis of the collect plot)

whose dt -> 0 limit is exp(rate0) with rate0 = lim ln(framability)/dt.  This
script extrapolates each measure to dt = 0 and draws a 5-panel colormap over
(gamma, gamma') for each model.

The DT_BASE lines are not always smooth over the full 0.01..0.99 range, so the
fit uses only the FIT_N points nearest dt = 0 (default 15) and a low-order
polynomial in dt (default degree 1: rate0 = the dt=0 intercept of ln(fra)/dt).

Usage:
    python scripts/trotter_dtbase_line_extrap.py                 # model3 + model4
    python scripts/trotter_dtbase_line_extrap.py --models model3
    python scripts/trotter_dtbase_line_extrap.py --fit_n 12 --deg 2
    python scripts/trotter_dtbase_line_extrap.py --raw           # extrapolate raw fra
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS
from trotter_dtbase_line_worker import (  # noqa: E402  (scripts/ is on the path)
    MEASURES, base_grid, point_tag, N_BASE,
)


# ---------------------------------------------------------------------------
#  Load one point's DT_BASE line (dt and the five measures, ordered by base)
# ---------------------------------------------------------------------------
def _load_point_raw(model: str, p1: float, p2: float, in_dir: Path):
    """load_point()'s original source: the per-base npz directory written by
    trotter_dtbase_line_worker.py, or None if that directory is absent/empty."""
    pt_dir = in_dir / point_tag(model, p1, p2)
    if not pt_dir.is_dir():
        return None
    arrs = {k: np.full(N_BASE, np.nan) for k, _ in MEASURES}
    dt_vals = np.full(N_BASE, np.nan)
    for f in pt_dir.glob('base_*.npz'):
        if '_qrefine_' in f.stem:          # item 5 refine files: not a source
            continue                       # of dt (merged in separately below)
        try:
            d = np.load(f, allow_pickle=True)
            idx = int(d['base_idx'])
        except Exception:
            continue
        if not (0 <= idx < N_BASE):
            continue
        dt_vals[idx] = float(d['dt'])
        for k, _ in MEASURES:
            if k in d.files:
                arrs[k][idx] = float(d[k])
    if not np.isfinite(dt_vals).any():
        return None

    # item 5: elementwise-minimum merge of every quick-refine round for
    # opt_fra_4/opt_fra_6 (same convention as trotter_dtbase_line_collect.py).
    for f in pt_dir.glob('base_*_qrefine_r*.npz'):
        try:
            d = np.load(f, allow_pickle=True)
            idx = int(d['base_idx']) if 'base_idx' in d.files else int(f.stem.split('_')[1])
        except Exception:
            continue
        if not (0 <= idx < N_BASE):
            continue
        for k in ('opt_fra_4', 'opt_fra_6'):
            if k in d.files:
                v = float(d[k])
                if np.isfinite(v) and (not np.isfinite(arrs[k][idx]) or v < arrs[k][idx]):
                    arrs[k][idx] = v
    return dt_vals, arrs


def _load_point_collected(model: str, p1: float, p2: float, in_dir: Path):
    """Fallback source: the aggregate <tag>_dtbase_line.npz written by
    trotter_dtbase_line_collect.py, used when the raw per-base directory has
    been cleaned up but the collected summary survives. Same (dt_vals, arrs)
    shape as _load_point_raw since collect.py stores dt_vals/measures directly."""
    npz = in_dir / f'{point_tag(model, p1, p2)}_dtbase_line.npz'
    if not npz.is_file():
        return None
    try:
        d = np.load(npz, allow_pickle=True)
        dt_vals = np.asarray(d['dt_vals'], dtype=float)
        arrs = {k: (np.asarray(d[k], dtype=float) if k in d.files
                   else np.full(dt_vals.shape, np.nan)) for k, _ in MEASURES}
    except Exception:
        return None
    if not np.isfinite(dt_vals).any():
        return None
    return dt_vals, arrs


def load_point(model: str, p1: float, p2: float, in_dir: Path):
    """Return (dt_vals, {measure: array}) over the base grid, or None if the
    point has no data. Tries the raw per-base directory first, falling back to
    the collected per-point summary npz (results_dtbase_line/<tag>_dtbase_line.npz)
    when the raw directory has been removed/consolidated."""
    return (_load_point_raw(model, p1, p2, in_dir)
            or _load_point_collected(model, p1, p2, in_dir))


# ---------------------------------------------------------------------------
#  dt -> 0 extrapolation
# ---------------------------------------------------------------------------
def extrapolate(dt_vals: np.ndarray, fra_vals: np.ndarray, *,
                fit_n: int, deg: int, raw: bool) -> float:
    """dt=0 value of framability**(1/dt) (or of raw framability if raw=True),
    fit on the fit_n points nearest dt=0 with a degree-`deg` polynomial in dt.

    framability >= 1 always, so the power form is fit through the log-rate
    r(dt) = ln(framability)/dt, whose dt=0 intercept is the exponent rate0;
    the returned value is exp(rate0).  raw=True instead fits framability(dt)
    directly and returns its dt=0 intercept (should approach 1)."""
    m = np.isfinite(dt_vals) & np.isfinite(fra_vals) & (dt_vals > 0)
    dt = dt_vals[m]
    fra = fra_vals[m]
    if dt.size < 2:
        return np.nan

    order = np.argsort(dt)                 # ascending dt: nearest dt=0 first
    dt = dt[order]
    fra = fra[order]
    k = int(min(fit_n, dt.size))
    dtf = dt[:k]
    fraf = np.maximum(fra[:k], 1.0)        # framability lower bound is 1

    d = int(min(deg, k - 1))
    if raw:
        c = np.polyfit(dtf, fraf, d)
        return float(max(c[-1], 1.0))      # intercept (dt^0), clipped to >= 1

    rate = np.log(fraf) / dtf              # -> rate0 (const) as dt -> 0
    c = np.polyfit(dtf, rate, d)
    rate0 = c[-1]                          # dt=0 intercept of the rate
    return float(np.exp(rate0))


# ---------------------------------------------------------------------------
#  Per-model grid extrapolation
# ---------------------------------------------------------------------------
def extrapolate_model(model: str, in_dir: Path, *, fit_n: int, deg: int,
                      raw: bool, stride: int = 1,
                      max_dt_base: float = 0.10) -> dict:
    m = MODELS[model]
    p1_vals = np.asarray(m.p1_vals[::stride], float)     # gamma  (x-axis)
    p2_vals = np.asarray(m.p2_vals[::stride], float)     # gamma' (y-axis)
    nx, ny = len(p1_vals), len(p2_vals)

    # base_grid() indexes align 1:1 with dt_vals/arrs (both loaders preserve
    # base_idx position), so this mask restricts the fit pool to DT_BASE <=
    # max_dt_base regardless of fit_n.
    base_mask = base_grid() <= max_dt_base

    grids = {k: np.full((nx, ny), np.nan) for k, _ in MEASURES}
    found = 0
    for ix, p1 in enumerate(p1_vals):
        for iy, p2 in enumerate(p2_vals):
            pt = load_point(model, float(p1), float(p2), in_dir)
            if pt is None:
                continue
            found += 1
            dt_vals, arrs = pt
            for key, _ in MEASURES:
                grids[key][ix, iy] = extrapolate(
                    dt_vals[base_mask], arrs[key][base_mask],
                    fit_n=fit_n, deg=deg, raw=raw)
    print(f'[extrap] {model}: {found}/{nx * ny} grid points had data', flush=True)
    return dict(p1_vals=p1_vals, p2_vals=p2_vals, found=found, **grids)


# ---------------------------------------------------------------------------
#  Figure: 5 colormaps over (gamma, gamma')
# ---------------------------------------------------------------------------
def osc_rate_panels(model: str, osc_dir: Path, stride: int = 1) -> list:
    """The oscillation-rate panel spec for `model`, or [] when no data exists
    yet (so the figure simply keeps its framability panels).  Shared by this
    script's main() and scripts/collect_and_plot_all.py so both build the panel
    the same way."""
    try:
        import osc_rate_collect
    except ImportError:
        return []
    d = osc_rate_collect.load(model, osc_dir, stride)
    if d is None:
        return []
    n_q = 8
    return [dict(p1_vals=d['p1_vals'], p2_vals=d['p2_vals'], Z=d['osc_rate'],
                 label=f'{osc_rate_collect.OSC_LABEL}\n({n_q}-qubit ring '
                       f'Lindbladian)',
                 cbar_label=r'$\max_k |\mathrm{Im}\lambda_k/\mathrm{Re}\lambda_k|$',
                 cmap='magma', vmin=0.0)]


def plot_model(model: str, data: dict, png: Path, *, raw: bool,
               fra_tol: float = 1e-3, extra: list | None = None) -> None:
    """`extra`: optional list of additional panels appended after the MEASURES
    panels, each a dict with keys p1_vals, p2_vals, Z, label, cbar_label and
    optionally cmap/vmin.  Used for quantities that are NOT dt-extrapolated
    framabilities -- currently the 8-qubit-ring oscillation rate, which is a
    property of the Lindbladian itself and so carries no dt dependence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    m = MODELS[model]
    p1_vals = data['p1_vals']
    p2_vals = data['p2_vals']

    def edges(v):
        v = np.asarray(v, float)
        mid = (v[:-1] + v[1:]) / 2
        return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])
    ex, ey = edges(p1_vals), edges(p2_vals)

    extra = list(extra or [])
    n = len(MEASURES) + len(extra)
    ncol = min(4, n) if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 5 * nrow),
                             constrained_layout=True, squeeze=False)
    qty = 'framability' if raw else r'framability$^{1/dt}$'
    fig.suptitle(f'{m.title}\n' + r'$dt\to0$ extrapolation of ' + qty
                 + r'   ($dt=\mathrm{DT\_BASE}/\max(\|H\|_1,\{\gamma_k\})$)',
                 fontsize=13)

    axflat = list(axes.flat)
    for ax, (key, label) in zip(axflat, MEASURES):
        Z = data[key].T                        # rows = gamma' (y), cols = gamma (x)
        vmin = 1.0
        vmax = float(np.nanmax(Z)) if np.isfinite(Z).any() else vmin + 1e-6
        if vmax <= vmin:
            vmax = vmin + 1e-6
        pcm = ax.pcolormesh(ex, ey, Z, cmap='viridis', vmin=vmin, vmax=vmax,
                            shading='flat')
        if np.isfinite(Z).any() and np.nanmin(Z) <= 1.0 + fra_tol <= np.nanmax(Z):
            cs = ax.contour(p1_vals, p2_vals, Z, levels=[1.0 + fra_tol],
                            colors='white', linewidths=1.3)
            ax.clabel(cs, fmt=f'framable (=1+{fra_tol:g})', fontsize=7)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(m.p1_label)              # gamma
        ax.set_ylabel(m.p2_label)              # gamma'
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label(qty + r'  at $dt=0$')
    # Extra (non-framability) panels: own colour scale, own grid, vmin=0 rather
    # than the framability floor of 1.
    for ax, spec in zip(axflat[len(MEASURES):], extra):
        ex2, ey2 = edges(spec['p1_vals']), edges(spec['p2_vals'])
        Z = np.asarray(spec['Z'], float).T
        # inf is a physically meaningful outcome here (an undamped oscillating
        # mode), but it would collapse the colour scale, so render it as a gap
        # and report how many points were affected.
        n_inf = int(np.sum(np.isinf(Z)))
        if n_inf:
            print(f'[extrap] {spec["label"]}: {n_inf} point(s) with infinite '
                  f'rate shown as blank', flush=True)
        Z = np.where(np.isfinite(Z), Z, np.nan)
        finite = np.isfinite(Z)
        vmin = spec.get('vmin', 0.0)
        vmax = float(np.nanmax(Z)) if finite.any() else vmin + 1e-6
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6
        pcm = ax.pcolormesh(ex2, ey2, Z, cmap=spec.get('cmap', 'magma'),
                            vmin=vmin, vmax=vmax, shading='flat')
        ax.set_title(spec['label'], fontsize=11)
        ax.set_xlabel(m.p1_label)
        ax.set_ylabel(m.p2_label)
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label(spec.get('cbar_label', ''))

    for ax in axflat[len(MEASURES) + len(extra):]:   # hide unused panels
        ax.axis('off')

    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[extrap] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['model3', 'model4'],
                    choices=list(MODELS))
    ap.add_argument('--in_dir', type=str, default='results_dtbase_line')
    ap.add_argument('--out_dir', type=str, default='results_dtbase_line')
    ap.add_argument('--stride', type=int, default=1,
                    help='use every stride-th p1/p2 grid value, matching the '
                         'sweep resolution actually computed (e.g. STRIDE used '
                         'at submit time)')
    ap.add_argument('--fit_n', type=int, default=15,
                    help='number of points nearest dt=0 used in the fit')
    ap.add_argument('--deg', type=int, default=1,
                    help='polynomial degree of the dt-extrapolation fit')
    ap.add_argument('--max_dt_base', type=float, default=0.10,
                    help='only DT_BASE values <= this are eligible for the fit '
                         '(default 0.10 = the bottom N_BASE_KEEP=10 values)')
    ap.add_argument('--raw', action='store_true',
                    help='extrapolate raw framability instead of framability**(1/dt)')
    ap.add_argument('--fra_tol', type=float, default=1e-3)
    ap.add_argument('--osc_dir', type=str, default='results_osc_rate',
                    help='directory of scripts/osc_rate_worker.py output; the '
                         'oscillation-rate panel is appended when data exists '
                         'there (silently omitted otherwise)')
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = 'extrap_raw' if args.raw else 'extrap'

    for model in args.models:
        data = extrapolate_model(model, in_dir, fit_n=args.fit_n, deg=args.deg,
                                 raw=args.raw, stride=args.stride,
                                 max_dt_base=args.max_dt_base)
        npz = out_dir / f'{model}_dtbase_{suffix}.npz'
        png = out_dir / f'{model}_dtbase_{suffix}.png'
        np.savez(npz, model=model, fit_n=args.fit_n, deg=args.deg,
                 raw=args.raw, measures=[k for k, _ in MEASURES], **data)
        print(f'[extrap] saved {npz}', flush=True)
        plot_model(model, data, png, raw=args.raw, fra_tol=args.fra_tol,
                   extra=osc_rate_panels(model, Path(args.osc_dir), args.stride))


if __name__ == '__main__':
    main()
