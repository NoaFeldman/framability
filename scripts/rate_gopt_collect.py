"""
Collect scripts/rate_gopt_worker.py and draw the globally optimised rate maps.

Always writes, for the model,
    results_<model>_rate/<model>_rate_gopt.npz
    results_<model>_rate/<model>_rate_gopt.png
        row 1: rate_heis_<m> for every d_ext (viridis, white contour at the
               floor mu* = 0)
        row 2: previous best (base scan + refine rounds) and the improvement,
               where a previous scan exists

With --replot it then redraws the model's standard figure with the new
values folded in:
    model4 : scripts/model4_rate_panels_collect.py  (its loader takes the min
             over base, refine rounds and gopt files) -> model4_rate_panels.png
    others : scripts/trotter_dtbase_line_extrap.py --from_npz --rate_dir ...
             -> results_dtbase_line/<model>_dtbase_extrap.png with the three
             exp(mu*) panels appended; --from_npz reuses the cached
             dt-extrapolation instead of re-reading ~26k per-base files.

Usage:
    python scripts/rate_gopt_collect.py --model model4 --replot
    python scripts/rate_gopt_collect.py --model model3 --replot
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                              # noqa: E402
from rate_gopt_worker import D_EXTS_DEFAULT, SUFFIX, grid_vals           # noqa: E402

CONTOUR_TOL = 1e-6


def load_gopt_grids(model: str, in_dir: Path, stride: int = 1,
                    d_exts=D_EXTS_DEFAULT) -> dict:
    """Grids rate_heis_<m> (gopt files) and prev_<m> (min over base scan and
    refine rounds, nan where absent) on the model's strided grid."""
    p1, p2 = grid_vals(model, stride)
    nx, ny = len(p1), len(p2)
    pt_dir = in_dir / model
    g = {f'rate_heis_{m}': np.full((nx, ny), np.nan) for m in d_exts}
    g.update({f'prev_{m}': np.full((nx, ny), np.nan) for m in d_exts})
    found = 0
    for ix in range(nx):
        for iy in range(ny):
            f = pt_dir / f'pt_{ix:03d}_{iy:03d}{SUFFIX}.npz'
            if f.exists():
                try:
                    d = np.load(f, allow_pickle=True)
                    for m in d_exts:
                        if f'rate_heis_{m}' in d.files:
                            g[f'rate_heis_{m}'][ix, iy] = float(d[f'rate_heis_{m}'])
                    found += 1
                except Exception as e:                      # noqa: BLE001
                    print(f'  warning: {f.name}: {e}', flush=True)
            others = [pt_dir / f'pt_{ix:03d}_{iy:03d}.npz']
            others += sorted(pt_dir.glob(f'pt_{ix:03d}_{iy:03d}_*refine_r*.npz'))
            for f in others:
                if not f.exists():
                    continue
                try:
                    d = np.load(f, allow_pickle=True)
                except Exception:                           # noqa: BLE001
                    continue
                for m in d_exts:
                    if f'rate_heis_{m}' in d.files:
                        v = float(d[f'rate_heis_{m}'])
                        cur = g[f'prev_{m}'][ix, iy]
                        if not np.isfinite(cur) or v < cur:
                            g[f'prev_{m}'][ix, iy] = v
    print(f'[{model} gopt] {found}/{nx * ny} points loaded from {pt_dir}', flush=True)
    return dict(p1_vals=p1, p2_vals=p2, n_points=found, d_exts=list(d_exts), **g)


def _edges(v):
    v = np.asarray(v, float)
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])


def _panel(fig, ax, spec, xv, yv, Z, title, *, floor=0.0, cmap='viridis',
           symmetric=False):
    Zt = np.asarray(Z, float).T
    fin = np.isfinite(Zt)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(spec.p1_label)
    ax.set_ylabel(spec.p2_label)
    if not fin.any():
        ax.set_title(f'{title}\n(no data)', fontsize=10)
        return
    lo, hi = float(Zt[fin].min()), float(Zt[fin].max())
    if symmetric:
        a = max(abs(lo), abs(hi), 1e-12)
        lo, hi = -a, a
    if hi <= lo:
        hi = lo + 1e-12
    pcm = ax.pcolormesh(_edges(xv), _edges(yv), Zt, cmap=cmap, vmin=lo, vmax=hi,
                        shading='flat')
    if floor is not None:
        mask = np.where(fin, np.abs(Zt - floor) <= CONTOUR_TOL, False).astype(float)
        if 0.0 < mask.sum() < mask.size:
            ax.contour(xv, yv, mask, levels=[0.5], colors='white', linewidths=1.3)
    fig.colorbar(pcm, ax=ax)


def plot(model: str, g: dict, png: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spec = MODELS[model]
    d_exts = g['d_exts']
    has_prev = any(np.isfinite(g[f'prev_{m}']).any() for m in d_exts)
    nrow = 3 if has_prev else 1
    fig, axes = plt.subplots(nrow, len(d_exts), figsize=(5.6 * len(d_exts), 4.8 * nrow),
                             constrained_layout=True, squeeze=False)
    fig.suptitle(f'{spec.title}\nHeisenberg framability rate $\\mu^*$, global '
                 f'seeded optimiser (white: $\\mu^*=0$)', fontsize=12)
    for c, m in enumerate(d_exts):
        Z = g[f'rate_heis_{m}']
        n0 = int((np.abs(Z) < CONTOUR_TOL).sum())
        _panel(fig, axes[0, c], spec, g['p1_vals'], g['p2_vals'], Z,
               f'opt Heisenberg rate, $d_{{\\rm ext}}={m}$  ({n0} pts at floor)')
        if has_prev:
            P = g[f'prev_{m}']
            _panel(fig, axes[1, c], spec, g['p1_vals'], g['p2_vals'], P,
                   f'previous best (base + refine), $d_{{\\rm ext}}={m}$')
            _panel(fig, axes[2, c], spec, g['p1_vals'], g['p2_vals'], P - Z,
                   f'improvement (previous $-$ new), $d_{{\\rm ext}}={m}$',
                   floor=None, cmap='coolwarm', symmetric=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[{model} gopt] wrote {png}', flush=True)


def replot_standard(model: str, in_dir: Path, py: str) -> None:
    root = Path(__file__).resolve().parent.parent
    if model == 'model4':
        cmd = [py, str(root / 'scripts' / 'model4_rate_panels_collect.py'),
               '--model', model, '--in_dir', str(in_dir), '--out_dir', str(in_dir)]
    else:
        cached = root / 'results_dtbase_line' / f'{model}_dtbase_extrap.npz'
        if not cached.exists():
            print(f'[{model} gopt] no cached {cached.name}; standard figure not '
                  f'replotted (the separate gopt figure is the result)', flush=True)
            return
        cmd = [py, str(root / 'scripts' / 'trotter_dtbase_line_extrap.py'),
               '--models', model, '--from_npz', '--rate_dir', str(in_dir)]
    print(f'[{model} gopt] replot: {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd, cwd=root)
    if r.returncode != 0:
        print(f'[{model} gopt] replot failed (exit {r.returncode}); the separate '
              f'gopt figure stands on its own', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True, choices=list(MODELS))
    ap.add_argument('--in_dir', type=str, default=None)
    ap.add_argument('--out_dir', type=str, default=None)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--d_exts', type=int, nargs='+', default=list(D_EXTS_DEFAULT))
    ap.add_argument('--replot', action='store_true',
                    help="also redraw the model's standard figure with the new "
                         "rates folded in")
    args = ap.parse_args()
    in_dir = Path(args.in_dir or f'results_{args.model}_rate')
    out_dir = Path(args.out_dir or f'results_{args.model}_rate')
    out_dir.mkdir(parents=True, exist_ok=True)

    g = load_gopt_grids(args.model, in_dir, args.stride, args.d_exts)
    spec = MODELS[args.model]
    np.savez(out_dir / f'{args.model}_rate_gopt.npz', model=args.model,
             stride=args.stride, d_exts=np.array(args.d_exts),
             **{f'{spec.p1_name}_vals': g['p1_vals'],
                f'{spec.p2_name}_vals': g['p2_vals']},
             **{k: g[k] for k in g if k.startswith(('rate_heis_', 'prev_'))})
    for m in args.d_exts:
        Z, P = g[f'rate_heis_{m}'], g[f'prev_{m}']
        both = np.isfinite(Z) & np.isfinite(P)
        msg = (f'  d_ext={m}: {int((np.abs(Z) < CONTOUR_TOL).sum())} points at the '
               f'floor')
        if both.any():
            msg += (f'; improved at {int((P[both] - Z[both] > 1e-6).sum())} of '
                    f'{int(both.sum())} previously scanned points, max gain '
                    f'{float(np.max(P[both] - Z[both])):.4f}')
        print(msg, flush=True)
    plot(args.model, g, out_dir / f'{args.model}_rate_gopt.png')
    if args.replot:
        replot_standard(args.model, in_dir, sys.executable)


if __name__ == '__main__':
    main()
