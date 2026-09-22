"""
Collect scripts/polygon_seed_worker.py and draw one figure per (model, d_ext).

Each figure has up to four viridis panels on the (gamma, gamma') grid, white
contour at the framable floor mu* = 0:
    optimised (polygon-seeded)  |  identity-free polygon frame (fixed)
    base scan at the same d_ext (if --base_dir has it)  |  optimised - base

Usage:
    python scripts/polygon_seed_collect.py
    python scripts/polygon_seed_collect.py --base_dir_model4 results_model4_rate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from polygon_seed_worker import grid_vals, D_EXTS_DEFAULT, STRIDE_DEFAULT  # noqa: E402

CONTOUR_TOL = 1e-6


def load(pt_dir: Path, model: str, stride: int, d_exts):
    p1, p2 = grid_vals(model, stride)
    nx, ny = len(p1), len(p2)
    keys = [f'{k}_{m}' for m in d_exts for k in ('rate_opt', 'rate_seed', 'rate_free')]
    g = {k: np.full((nx, ny), np.nan) for k in keys}
    found = 0
    for ix in range(nx):
        for iy in range(ny):
            f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            d = np.load(f, allow_pickle=True)
            for k in keys:
                if k in d:
                    g[k][ix, iy] = float(d[k])
            found += 1
    print(f'[{model}] {found}/{nx * ny} points from {pt_dir}', flush=True)
    return p1, p2, g


def load_base(base_dir: Path, model: str, stride: int, d_exts):
    """Base full-resolution scan (results_model4_rate layout) at stride-2 indices."""
    p1, p2 = grid_vals(model, stride)
    out = {}
    for m in d_exts:
        Z = np.full((len(p1), len(p2)), np.nan)
        ok = False
        for ix in range(len(p1)):
            for iy in range(len(p2)):
                f = base_dir / model / f'pt_{ix * stride:03d}_{iy * stride:03d}.npz'
                if f.exists():
                    d = np.load(f, allow_pickle=True)
                    if f'rate_heis_{m}' in d:
                        Z[ix, iy] = float(d[f'rate_heis_{m}'])
                        ok = True
        if ok:
            out[m] = Z
    return out


def _edges(v):
    v = np.asarray(v, float)
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])


def _panel(fig, ax, xv, yv, Z, title, *, floor=0.0, cmap='viridis'):
    Zt = np.asarray(Z, float).T
    fin = np.isfinite(Zt)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r"$\gamma'$")
    if not fin.any():
        ax.set_title(f'{title}\n(no data)', fontsize=10)
        return
    lo, hi = float(Zt[fin].min()), float(Zt[fin].max())
    if hi <= lo:
        hi = lo + 1e-12
    pcm = ax.pcolormesh(_edges(xv), _edges(yv), Zt, cmap=cmap, vmin=lo, vmax=hi,
                        shading='flat')
    if floor is not None:
        mask = np.where(fin, np.abs(Zt - floor) <= CONTOUR_TOL, False).astype(float)
        if 0.0 < mask.sum() < mask.size:
            ax.contour(xv, yv, mask, levels=[0.5], colors='white', linewidths=1.3)
    fig.colorbar(pcm, ax=ax)


def plot_model_dext(model, m, p1, p2, g, base, png: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_base = base is not None
    n_pan = 4 if has_base else 2
    fig, axes = plt.subplots(1, n_pan, figsize=(5.5 * n_pan, 4.8),
                             constrained_layout=True)
    fig.suptitle(f'{model}: Heisenberg framability rate, d_ext = {m} '
                 f'(polygon-seeded scan, 0.4 steps)', fontsize=12)
    _panel(fig, axes[0], p1, p2, g[f'rate_opt_{m}'],
           'optimised (seeded with projector-polygon frame)')
    _panel(fig, axes[1], p1, p2, g[f'rate_free_{m}'],
           'identity-free projector-polygon frame (fixed)')
    if has_base:
        _panel(fig, axes[2], p1, p2, base, f'base scan rate_heis_{m}')
        _panel(fig, axes[3], p1, p2, g[f'rate_opt_{m}'] - base,
               'optimised - base  (negative = seed helped)', floor=None,
               cmap='coolwarm')
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'  wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', type=str, default='results_polygon_seed')
    ap.add_argument('--out_dir', type=str, default='results_polygon_seed')
    ap.add_argument('--stride', type=int, default=STRIDE_DEFAULT)
    ap.add_argument('--d_exts', type=int, nargs='+', default=list(D_EXTS_DEFAULT))
    ap.add_argument('--base_dir_model4', type=str, default='results_model4_rate')
    ap.add_argument('--base_dir_model3', type=str, default='')
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in ('model3', 'model4'):
        pt_dir = in_dir / model
        if not pt_dir.exists():
            print(f'[{model}] no results directory, skipping', flush=True)
            continue
        p1, p2, g = load(pt_dir, model, args.stride, args.d_exts)
        base_dir = getattr(args, f'base_dir_{model}')
        base = (load_base(Path(base_dir), model, args.stride, args.d_exts)
                if base_dir else {})
        np.savez(out_dir / f'polygon_seed_{model}.npz', model=model,
                 gamma_vals=p1, gamma_p_vals=p2, d_exts=np.array(args.d_exts),
                 **g, **{f'base_heis_{m}': Z for m, Z in base.items()})
        for m in args.d_exts:
            plot_model_dext(model, m, p1, p2, g, base.get(m),
                            out_dir / f'polygon_seed_{model}_d{m}.png')


if __name__ == '__main__':
    main()
