"""Aggregate the XXZ magic / framability scan and draw the seven colormaps.

Reads every <in_dir>/<model>/pt_<ix>_<iy>.npz written by
scripts/xxz_magic_worker.py, assembles one (n_p2, n_p1) array per quantity
(rows = p2 = h, columns = p1 = Delta, the pcolormesh orientation) and writes

    <in_dir>/xxz_magic_summary_<model>.npz          all maps + axes + metadata
    results_plots/xxz_magic_<model>_fra_D1.png      (1) D = 1
    results_plots/xxz_magic_<model>_fra_D2.png      (1) D = 2
    results_plots/xxz_magic_<model>_nc_2a.png       (2a) .. (2e)
    ...
    results_plots/xxz_magic_<model>_overview.png    all seven panels

The dt -> 0 framability limit is recomputed here from the stored raw ladder
(trotter_rom_dtbase.extrapolate_to_zero), so --fit_n / --deg can be changed
without re-running the workers; points that only carry the worker's value are
used as they are.

Missing points stay NaN and are drawn as blank cells, so the script is safe to
run on a partially finished scan.

Usage:
    python scripts/xxz_magic_collect.py
    python scripts/xxz_magic_collect.py --deg 2 --log_base 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LogNorm, Normalize                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from xxz_magic_scan import (                                       # noqa: E402
    MODELS, TASK_KEYS, TASK_LABELS, FRA_KEYS, NC_KEYS, XXZ_MAGIC_VERSION,
    framability_limit, framability_rate,
)
from trotter_rom_dtbase import FIT_N_DEFAULT, DEG_DEFAULT          # noqa: E402

DEFAULT_IN = 'results_xxz_magic'
DEFAULT_PLOTS = 'results_plots'

FRA_CMAP = 'viridis'
NC_CMAP = 'magma'
# log of the framability limit: finite even where exp(rate) overflows float64.
RATE_KEYS = ('fra_rate_D1', 'fra_rate_D2')
# Extra scalar maps kept in the summary npz (diagnostics, not plotted).
DIAG_KEYS = ('dt_pt', 'gap', 'gap_next', 't_2e')
EXTRA_KEYS = RATE_KEYS + DIAG_KEYS


def _edges(centers: np.ndarray) -> np.ndarray:
    """Cell edges of an evenly or unevenly spaced axis of cell centres."""
    c = np.asarray(centers, float)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])


def load_maps(in_dir: Path, model, *, fit_n: int, deg: int,
              log_base: float | None) -> dict:
    """(n_p2, n_p1) arrays for every task key, plus diagnostics and counts."""
    n1, n2 = model.shape
    maps = {k: np.full((n2, n1), np.nan) for k in TASK_KEYS}
    maps.update({k: np.full((n2, n1), np.nan) for k in EXTRA_KEYS})

    pdir = in_dir / model.name
    files = sorted(pdir.glob('pt_*.npz')) if pdir.is_dir() else []
    n_ok = n_bad = n_refit = 0

    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
        except Exception:
            n_bad += 1
            continue
        if str(d.get('code_version', '')) != XXZ_MAGIC_VERSION:
            n_bad += 1
            continue
        ix, iy = int(d['ix']), int(d['iy'])
        if not (0 <= ix < n1 and 0 <= iy < n2):
            n_bad += 1
            continue

        for k in FRA_KEYS:
            dim = k[-1]
            raw_key, dts_key, rate_key = f'fra_raw_D{dim}', 'fra_dts', f'fra_rate_D{dim}'
            if raw_key in d.files and dts_key in d.files:
                # Re-fit from the ladder so --fit_n / --deg take effect.
                dts = np.asarray(d[dts_key], float)
                raw = np.asarray(d[raw_key], float)
                maps[k][iy, ix] = framability_limit(dts, raw, fit_n=fit_n, deg=deg)
                maps[rate_key][iy, ix] = framability_rate(dts, raw,
                                                          fit_n=fit_n, deg=deg)
                n_refit += 1
            else:
                if k in d.files:
                    maps[k][iy, ix] = float(d[k])
                if rate_key in d.files:
                    maps[rate_key][iy, ix] = float(d[rate_key])

        for k in NC_KEYS:
            src = f'{k}_bits' if (log_base == 2 and f'{k}_bits' in d.files) else k
            if src in d.files:
                v = float(d[src])
                if log_base == 2 and src == k:
                    v = v / np.log(2.0)
                maps[k][iy, ix] = v

        for k in DIAG_KEYS:               # RATE_KEYS were set with the ladders
            if k in d.files:
                maps[k][iy, ix] = float(d[k])
        n_ok += 1

    maps['_counts'] = {'files': len(files), 'ok': n_ok, 'bad': n_bad,
                       'refit': n_refit}
    return maps


def _norm(vals: np.ndarray, mode: str):
    """Colour normalization; 'auto' picks log for a wide strictly-positive range."""
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return Normalize()
    lo, hi = float(np.min(v)), float(np.max(v))
    if mode == 'log' or (mode == 'auto' and lo > 0 and hi / max(lo, 1e-300) > 50.0):
        return LogNorm(vmin=max(lo, 1e-300), vmax=max(hi, lo * 1.000001 + 1e-300))
    if hi <= lo:
        hi = lo + 1e-12
    return Normalize(vmin=lo, vmax=hi)


def _panel(ax, model, key, Z, *, cmap, norm, unit_note=''):
    xe, ye = _edges(model.p1_vals), _edges(model.p2_vals)
    Zm = np.ma.masked_invalid(Z)
    pcm = ax.pcolormesh(xe, ye, Zm, cmap=cmap, norm=norm, shading='flat')
    ax.set_xlabel(rf'$\{model.p1_name}$' if model.p1_name == 'Delta'
                  else model.p1_name)
    ax.set_ylabel(model.p2_name)
    ax.set_title(TASK_LABELS[key] + unit_note, fontsize=9)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    return pcm


def _fra_field(maps, key: str, mode: str) -> tuple[np.ndarray, str]:
    """The array to draw for a framability key, plus a title note.

    'limit' is the dt -> 0 value of framability**(1/dt); it is +inf wherever the
    rate exceeds ~709 (float64 exp overflow).  'rate' is its logarithm, always
    finite.  'auto' falls back to the rate as soon as any cell overflowed.
    """
    lim, rate = maps[key], maps[f'fra_rate_D{key[-1]}']
    finite_lim = np.isfinite(lim) | np.isnan(lim)        # NaN = missing point
    if mode == 'rate' or (mode == 'auto' and not finite_lim.all()):
        n_of = int((~finite_lim).sum())
        note = '\n' + (r'shown as $\ln$ of the limit'
                       + (f' ({n_of} cells overflowed)' if n_of else ''))
        return rate, note
    return lim, ''


def make_plots(model, maps, out_dir: Path, *, fra_color: str, fra_plot: str,
               log_base, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    unit = ' [bits]' if log_base == 2 else ' [nats]'
    written: list[Path] = []

    for key in TASK_KEYS:
        is_fra = key in FRA_KEYS
        Z, note = _fra_field(maps, key, fra_plot) if is_fra else (maps[key], unit)
        cmap = FRA_CMAP if is_fra else NC_CMAP
        norm = _norm(Z, fra_color if (is_fra and not note) else 'linear')

        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        pcm = _panel(ax, model, key, Z, cmap=cmap, norm=norm, unit_note=note)
        fig.colorbar(pcm, ax=ax)
        fig.tight_layout()
        p = out_dir / f'xxz_magic_{model.name}_{key}.png'
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        written.append(p)

    fig, axes = plt.subplots(2, 4, figsize=(21, 8.6))
    for ax, key in zip(axes.ravel(), TASK_KEYS):
        is_fra = key in FRA_KEYS
        Z, note = _fra_field(maps, key, fra_plot) if is_fra else (maps[key], unit)
        pcm = _panel(ax, model, key, Z,
                     cmap=FRA_CMAP if is_fra else NC_CMAP,
                     norm=_norm(Z, fra_color if (is_fra and not note) else 'linear'),
                     unit_note=note)
        fig.colorbar(pcm, ax=ax)
    axes.ravel()[-1].axis('off')
    fig.suptitle(f'{model.name}: stabilizer-3 framability ($dt\\to0$) and '
                 f'Choi-state non-cliffordness   '
                 f'({", ".join(f"{k}={v}" for k, v in model.consts.items())})')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out_dir / f'xxz_magic_{model.name}_overview.png'
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    written.append(p)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model',    default='xxz', choices=list(MODELS))
    ap.add_argument('--in_dir',   default=DEFAULT_IN)
    ap.add_argument('--plot_dir', default=DEFAULT_PLOTS)
    ap.add_argument('--fit_n',    type=int, default=FIT_N_DEFAULT)
    ap.add_argument('--deg',      type=int, default=DEG_DEFAULT)
    ap.add_argument('--log_base', choices=('e', '2'), default='e',
                    help='units of the non-cliffordness maps (default nats)')
    ap.add_argument('--fra_color', choices=('auto', 'linear', 'log'),
                    default='auto',
                    help='colour scale of the framability maps; the dt->0 '
                         'limit of fra**(1/dt) can span decades')
    ap.add_argument('--fra_plot', choices=('auto', 'limit', 'rate'),
                    default='auto',
                    help="'limit' draws the dt->0 value of fra**(1/dt), 'rate' "
                         'its logarithm (finite even when the limit overflows '
                         "float64); 'auto' switches to the rate if any cell did")
    ap.add_argument('--dpi',      type=int, default=160)
    ap.add_argument('--no_plots', action='store_true',
                    help='write only the summary npz')
    a = ap.parse_args()

    model = MODELS[a.model]
    in_dir = Path(a.in_dir)
    log_base = 2 if a.log_base == '2' else None

    maps = load_maps(in_dir, model, fit_n=a.fit_n, deg=a.deg, log_base=log_base)
    counts = maps.pop('_counts')
    n_cells = model.n_points
    print(f'[collect] {a.model}: {counts["ok"]}/{n_cells} points loaded '
          f'({counts["bad"]} unreadable/stale, {counts["refit"]} ladders re-fit)')
    for k in list(TASK_KEYS) + list(RATE_KEYS):
        M = maps[k]
        v = M[np.isfinite(M)]
        miss = int(np.isnan(M).sum())
        over = int(np.isposinf(M).sum())
        rng = f'{v.min():.6g} .. {v.max():.6g}' if v.size else 'no data'
        tail = f'   missing {miss}' + (f', overflowed {over}' if over else '')
        print(f'    {k:12s} {rng:>28s}{tail}')

    in_dir.mkdir(parents=True, exist_ok=True)
    summary = in_dir / f'xxz_magic_summary_{a.model}.npz'
    np.savez(
        summary,
        p1_vals=model.p1_vals, p2_vals=model.p2_vals,
        p1_name=np.array(model.p1_name), p2_name=np.array(model.p2_name),
        consts=np.array(str(model.consts)),
        fit_n=np.array(a.fit_n), deg=np.array(a.deg),
        log_base=np.array(a.log_base),
        code_version=np.array(XXZ_MAGIC_VERSION),
        **{k: maps[k] for k in list(TASK_KEYS) + list(EXTRA_KEYS)},
    )
    print(f'[collect] wrote {summary}')

    if a.no_plots:
        return
    for p in make_plots(model, maps, Path(a.plot_dir),
                        fra_color=a.fra_color, fra_plot=a.fra_plot,
                        log_base=log_base, dpi=a.dpi):
        print(f'[plot] wrote {p}')


if __name__ == '__main__':
    main()
