"""
Collect the complex-frame rate pipeline (scripts/rate_complex_worker.py) into
panel files and figures, one per model; plus the small queries the driver
(scripts/submit_rate_complex.sh) needs.

Per model (coarse grid, --stride):
    <out_root>/<model>_complex_rate_panels.npz
        model, dim, stride, <p1>_vals, <p2>_vals, floor, rate_pauli,
        rate_stab3, panel_rate_heis_{4,8,12}  complex frames (best of every
            file of the point; monotone envelope over d_ext -- a padded
            smaller frame is a valid larger frame -- and never above the
            Hermitian value, a real frame being a complex frame),
        real_rate_heis_{4,8,12}  earlier Hermitian run at the same points
            (same monotone envelope),
        gain_{4,8,12} = real - complex,  imag_{4,8,12}  ||Im S|| / ||S|| of
            the best complex frame, label_{4,8,12}
    <out_root>/<model>_complex_rate_panels.png
        row 1: complex d_ext 4 | 8 | 12 | Pauli
        row 2: Hermitian d_ext 4 | 8 | 12 | stabilizer   (same colour scale
               per column as row 1)
        row 3: gain d_ext 4 | 8 | 12 | imaginary fraction of the d_ext=12 frame
        white contour: mu* = 0 (rate <= --zero_tol)

Queries (print one line, for the driver):
    --count seed|d12          "<points done> <points total>"
    --last_round D_EXT        highest refine round on disk (0 if none)
    --round_gain D_EXT ROUND  "<largest drop> <new rate-0 points> <files>"

Usage:
    python scripts/rate_complex_collect.py                       # all models
    python scripts/rate_complex_collect.py --models model4 --zero_tol 1e-5
    python scripts/rate_complex_collect.py --round_gain 8 3
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                               # noqa: E402
from rate_complex_worker import (DEFAULT_MODELS, D_EXTS, RATE_TOL,        # noqa: E402
                                 grid, point_list, model_dir, point_files,
                                 imag_fraction, PIPELINE_VERSION)


# ---------------------------------------------------------------------------
#  Driver queries
# ---------------------------------------------------------------------------
def count_done(args, stage: str) -> tuple[int, int]:
    pts = point_list(args.models, args.stride)
    sfx = '' if stage == 'seed' else '_d12'
    done = sum((model_dir(args.out_root, m, args.stride)
                / f'pt_{ix:03d}_{iy:03d}{sfx}.npz').exists() for m, ix, iy in pts)
    return done, len(pts)


def refine_files(args, m: int, r: int | None = None) -> list:
    tag = f'_refine_d{m}_r' + ('[0-9][0-9]' if r is None else f'{r:02d}')
    out = []
    for model in args.models:
        out += sorted(model_dir(args.out_root, model, args.stride).glob(f'pt_*{tag}.npz'))
    return [f for f in out if '.tmp' not in f.name]


def last_round(args, m: int) -> int:
    rounds = [int(re.search(r'_r(\d\d)\.npz$', f.name).group(1))
              for f in refine_files(args, m)]
    return max(rounds, default=0)


def round_gain(args, m: int, r: int) -> tuple[float, int, int]:
    mx, zeros, n = 0.0, 0, 0
    for f in refine_files(args, m, r):
        try:
            d = np.load(f, allow_pickle=True)
            new, prev = float(d[f'rate_{m}']), float(d[f'rate_{m}_prev'])
        except Exception:                                   # noqa: BLE001
            continue
        n += 1
        if np.isfinite(new) and np.isfinite(prev):
            mx = max(mx, prev - new)
            zeros += prev > RATE_TOL >= new
    return mx, zeros, n


# ---------------------------------------------------------------------------
#  Panels
# ---------------------------------------------------------------------------
def collect_model(model: str, args) -> dict:
    spec = MODELS[model]
    p1, p2 = grid(model, args.stride)
    nx, ny = len(p1), len(p2)
    folder = model_dir(args.out_root, model, args.stride)
    nan = lambda: np.full((nx, ny), np.nan)                   # noqa: E731
    g = {k: nan() for k in ('floor', 'rate_pauli', 'rate_stab3')}
    for m in D_EXTS:
        g[f'c_{m}'], g[f'real_{m}'], g[f'imag_{m}'] = nan(), nan(), nan()
        g[f'label_{m}'] = np.full((nx, ny), '', dtype=object)
    n_files = 0
    for ix in range(nx):
        for iy in range(ny):
            best = {m: (np.inf, None, '') for m in D_EXTS}
            for f in point_files(folder, ix, iy):
                try:
                    d = np.load(f, allow_pickle=True)
                except Exception as e:                      # noqa: BLE001
                    print(f'  warning: {f.name}: {e}', flush=True)
                    continue
                n_files += 1
                for k in ('floor', 'rate_pauli', 'rate_stab3'):
                    if k in d.files and not np.isfinite(g[k][ix, iy]):
                        g[k][ix, iy] = float(d[k])
                for m in D_EXTS:
                    if f'real_{m}' in d.files and np.isfinite(float(d[f'real_{m}'])):
                        g[f'real_{m}'][ix, iy] = float(d[f'real_{m}'])
                    if f'rate_{m}' in d.files and f'S_{m}' in d.files:
                        v = float(d[f'rate_{m}'])
                        if np.isfinite(v) and v < best[m][0]:
                            lab = str(d[f'label_{m}']) if f'label_{m}' in d.files else f.stem
                            best[m] = (v, np.asarray(d[f'S_{m}']), lab)
            for m in D_EXTS:
                v, S, lab = best[m]
                if S is not None:
                    g[f'c_{m}'][ix, iy] = v
                    g[f'imag_{m}'][ix, iy] = imag_fraction(S)
                    g[f'label_{m}'][ix, iy] = lab

    out = {'model': model, 'dim': spec.dim, 'stride': args.stride,
           f'{spec.p1_name}_vals': p1, f'{spec.p2_name}_vals': p2,
           'pipeline_version': PIPELINE_VERSION,
           'floor': g['floor'], 'rate_pauli': g['rate_pauli'],
           'rate_stab3': g['rate_stab3']}
    env_c, env_r = None, None
    for m in D_EXTS:
        c, r = g[f'c_{m}'], g[f'real_{m}']
        env_r = r if env_r is None else np.fmin(r, env_r)
        env_c = c if env_c is None else np.fmin(c, env_c)
        n_real_wins = int(np.sum(np.isfinite(env_r) & ~(env_c <= env_r + 1e-9)))
        env_c = np.fmin(env_c, env_r)              # a real frame is a complex frame
        out[f'panel_rate_heis_{m}'] = env_c.copy()
        out[f'real_rate_heis_{m}'] = env_r.copy()
        out[f'gain_{m}'] = env_r - env_c
        out[f'imag_{m}'] = g[f'imag_{m}']
        out[f'label_{m}'] = g[f'label_{m}']
        out[f'raw_complex_{m}'] = c
        out[f'n_hermitian_floor_{m}'] = n_real_wins
    print(f'[{model}] {nx}x{ny} points, {n_files} files in {folder}', flush=True)
    for m in D_EXTS:
        c, r = out[f'panel_rate_heis_{m}'], out[f'real_rate_heis_{m}']
        fin = np.isfinite(c) & np.isfinite(r)
        gain = (r - c)[fin]
        print(f'  d_ext {m:2d}: {int(np.isfinite(g[f"c_{m}"]).sum()):4d} complex points | '
              f'rate 0: complex {int((c <= args.zero_tol).sum()):4d}, Hermitian '
              f'{int((r <= args.zero_tol).sum()):4d} | gain median '
              f'{np.median(gain) if gain.size else np.nan:.4f} max '
              f'{gain.max() if gain.size else np.nan:.4f} | complex frame best at '
              f'{int((g[f"imag_{m}"] > 1e-6).sum())} pts | Hermitian value used at '
              f'{out[f"n_hermitian_floor_{m}"]} pts', flush=True)
    return out


def draw(ax, fig, x, y, Z, title, zero_tol, xlabel, ylabel, *, cmap='viridis',
         vmin=0.0, vmax=None, contour=True) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if Z is None or not np.isfinite(Z).any():
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center',
                va='center', color='0.5')
        return
    if contour:
        Z = np.where(np.isfinite(Z) & (Z <= zero_tol), 0.0, Z)
    pc = ax.pcolormesh(x, y, Z.T, shading='nearest', cmap=cmap, vmin=vmin,
                       vmax=vmax, rasterized=True)
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    if contour:
        zero = np.isfinite(Z) & (Z <= zero_tol)
        if zero.any() and not zero.all():
            ax.contour(x, y, zero.T.astype(float), levels=[0.5], colors='white',
                       linewidths=1.3)


def plot_model(out: dict, args, png: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = MODELS[out['model']]
    x, y = out[f'{spec.p1_name}_vals'], out[f'{spec.p2_name}_vals']
    xl, yl = spec.p1_label, spec.p2_label
    DL = r'$d_{\rm ext}'
    fig, axes = plt.subplots(3, 4, figsize=(17, 11.5), constrained_layout=True)
    for col, m in enumerate(D_EXTS):
        c, r = out[f'panel_rate_heis_{m}'], out[f'real_rate_heis_{m}']
        both = np.concatenate([a[np.isfinite(a)] for a in (c, r)]) if \
            np.isfinite(c).any() or np.isfinite(r).any() else np.array([1.0])
        vmax = float(both.max()) if both.size else None
        draw(axes[0, col], fig, x, y, c, rf'complex frames, {DL}={m}$', args.zero_tol,
             xl, yl, vmax=vmax)
        draw(axes[1, col], fig, x, y, r, rf'Hermitian frames, {DL}={m}$', args.zero_tol,
             xl, yl, vmax=vmax)
        draw(axes[2, col], fig, x, y, out[f'gain_{m}'],
             rf'gain (Hermitian $-$ complex), {DL}={m}$', args.zero_tol, xl, yl,
             cmap='magma', contour=False)
    draw(axes[0, 3], fig, x, y, out['rate_pauli'], 'Pauli', args.zero_tol, xl, yl)
    draw(axes[1, 3], fig, x, y, out['rate_stab3'], 'Stabilizer', args.zero_tol, xl, yl)
    draw(axes[2, 3], fig, x, y, out['imag_12'],
         rf'$\|{{\rm Im}}\,S\|/\|S\|$, {DL}=12$', args.zero_tol, xl, yl,
         cmap='cividis', vmax=1.0, contour=False)
    fig.suptitle(f'{out["model"]}: {spec.title}  (stride {out["stride"]})', fontsize=11)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=170)
    plt.close(fig)
    print(f'  wrote {png}', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=list(DEFAULT_MODELS))
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--out_root', default='results_rate_complex')
    p.add_argument('--zero_tol', type=float, default=1e-5)
    p.add_argument('--count', choices=('seed', 'd12'))
    p.add_argument('--last_round', type=int, metavar='D_EXT')
    p.add_argument('--round_gain', type=int, nargs=2, metavar=('D_EXT', 'ROUND'))
    p.add_argument('--no_plot', action='store_true')
    args = p.parse_args()

    if args.count:
        print(*count_done(args, args.count))
        return
    if args.last_round is not None:
        print(last_round(args, args.last_round))
        return
    if args.round_gain:
        mx, nz, nf = round_gain(args, *args.round_gain)
        print(f'{mx:.6e} {nz} {nf}')
        return
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        out = collect_model(model, args)
        f = root / f'{model}_complex_rate_panels.npz'
        np.savez(f, **out)
        print(f'  wrote {f}', flush=True)
        if not args.no_plot:
            plot_model(out, args, root / f'{model}_complex_rate_panels.png')


if __name__ == '__main__':
    main()
