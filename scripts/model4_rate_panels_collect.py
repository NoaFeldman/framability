"""
Collect the model4 rate pipeline and draw its eight-panel figure.

Reads the per-point npz files written by
  * scripts/model4_rate_panels_worker.py  -> <in_dir>/model4/pt_<ix>_<iy>.npz
       panels 1-6: framability rates of the two-qubit bond generator
  * scripts/model4_manybody_worker.py     -> <in_dir>/model4_8q/pt_<ix>_<iy>.npz
       panels 7-8: oscillation rate and Lindbladian gap of the full 8-qubit
       2x4 lattice

assembles each quantity on its (gamma, gamma') grid, stores the merged arrays
and draws

    row 1 |  stabilizer-3 rate  |  Pauli rate  |  opt Heisenberg d=4  |  d=6
    row 2 |  opt Schrodinger d=4 |  d=6        |  8q osc rate         |  8q gap

The two groups may live on different strides (the many-body panels default to
--mb_stride 5, an 11x11 grid, against the framability panels' full 51x51); the
panels are drawn with pcolormesh on their own axis values, so they still share
the same (gamma, gamma') data coordinates and line up.

Usage:
    python scripts/model4_rate_panels_collect.py
    python scripts/model4_rate_panels_collect.py --stride 1 --mb_stride 5
    python scripts/model4_rate_panels_collect.py --shared_rate_scale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS, MODEL4_H                    # noqa: E402
from model4_rate_panels_worker import RATE_KEYS, MODEL_NAME              # noqa: E402
from model4_manybody_worker import (TAG as MB_TAG, N_QUBITS,             # noqa: E402
                                    LATTICE_LX, LATTICE_LY)

# (npz key, panel label, colormap) in figure order.
MB_KEYS = [
    ('osc_rate', rf'{N_QUBITS}q osc rate  $\max_k|{{\rm Im}}\lambda_k/'
                 rf'{{\rm Re}}\lambda_k|$', 'magma'),
    ('gap',      f'{N_QUBITS}q Lindbladian gap', 'viridis'),
]
RATE_CMAP = 'coolwarm'


def grid_vals(stride: int):
    m = MODELS[MODEL_NAME]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def load_group(pt_dir: Path, keys, stride: int, label: str) -> dict:
    """Assemble every key in `keys` on the strided grid from per-point files."""
    p1_vals, p2_vals = grid_vals(stride)
    nx, ny = len(p1_vals), len(p2_vals)
    grids = {k: np.full((nx, ny), np.nan) for k in keys}
    found = 0
    for ix in range(nx):
        for iy in range(ny):
            f = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not f.exists():
                continue
            try:
                d = np.load(f, allow_pickle=True)
            except Exception as e:
                print(f'  warning: {f.name}: {e}', flush=True)
                continue
            for k in keys:
                if k in d:
                    grids[k][ix, iy] = float(d[k])
            found += 1
    print(f'[{label}] {found}/{nx * ny} grid points loaded from {pt_dir}',
          flush=True)
    for k in keys:
        n_ok = int(np.isfinite(grids[k]).sum())
        if n_ok < found:
            print(f'  note: {k} finite at {n_ok}/{found} loaded points',
                  flush=True)
    return dict(p1_vals=p1_vals, p2_vals=p2_vals, n_points=found, **grids)


def _edges(v):
    """Cell edges of a (possibly non-uniform) axis, for pcolormesh shading."""
    v = np.asarray(v, float)
    if v.size == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])


def _panel(fig, ax, xv, yv, Z, title, cmap, *, vmin=None, vmax=None,
           center_zero=False):
    """One pcolormesh panel; Z is indexed [ix, iy] so it is drawn transposed."""
    Zt = np.asarray(Z, float).T
    finite = np.isfinite(Zt)
    if not finite.any():
        ax.set_title(f'{title}\n(no data)', fontsize=10)
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r"$\gamma'$")
        return
    lo = float(np.nanmin(Zt)) if vmin is None else vmin
    hi = float(np.nanmax(Zt)) if vmax is None else vmax
    if center_zero:
        # Rates straddle 0 (mu <= 0 means the frame contracts): a diverging
        # colormap is only honest if 0 sits at its midpoint.
        m = max(abs(lo), abs(hi), 1e-12)
        lo, hi = -m, m
    if hi <= lo:
        hi = lo + 1e-12
    pcm = ax.pcolormesh(_edges(xv), _edges(yv), Zt, cmap=cmap,
                        vmin=lo, vmax=hi, shading='flat')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r"$\gamma'$")
    fig.colorbar(pcm, ax=ax)


def plot(rates: dict, mb: dict, png: Path, *, shared_rate_scale: bool) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)
    fig.suptitle(
        rf"model4:  $H = J\sum_{{\langle ij\rangle}} Z_iZ_j + {MODEL4_H}\sum_i X_i$,  "
        rf"jumps $\sqrt{{\gamma}}\,|{{-}}\rangle\langle{{+}}|_i,\ \sqrt{{\gamma'}}Z_i$  "
        rf"($J=1$)"
        "\n"
        rf"framability rates $\mu^*=\lim_{{dt\to0}}({{\rm fra}}-1)/dt$ of the bond "
        rf"generator  |  panels 7-8: full {N_QUBITS}-qubit "
        rf"{LATTICE_LY}x{LATTICE_LX} lattice Lindbladian",
        fontsize=13)

    rate_grids = [rates[k] for k, _ in RATE_KEYS]
    v = np.concatenate([g[np.isfinite(g)].ravel() for g in rate_grids]) \
        if any(np.isfinite(g).any() for g in rate_grids) else np.array([0.0])
    shared = (float(v.min()), float(v.max())) if shared_rate_scale else (None, None)

    for ax, (key, label) in zip(axes.flat[:6], RATE_KEYS):
        _panel(fig, ax, rates['p1_vals'], rates['p2_vals'], rates[key],
               label, RATE_CMAP, vmin=shared[0], vmax=shared[1],
               center_zero=not shared_rate_scale)

    for ax, (key, label, cmap) in zip(axes.flat[6:], MB_KEYS):
        _panel(fig, ax, mb['p1_vals'], mb['p2_vals'], mb[key], label, cmap,
               vmin=0.0 if key == 'gap' else None)

    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[model4-rate] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir',  type=str, default='results_model4_rate')
    ap.add_argument('--out_dir', type=str, default='results_model4_rate')
    ap.add_argument('--stride',    type=int, default=1,
                    help='stride used by model4_rate_panels_worker (panels 1-6)')
    ap.add_argument('--mb_stride', type=int, default=5,
                    help='stride used by model4_manybody_worker (panels 7-8)')
    ap.add_argument('--shared_rate_scale', action='store_true',
                    help='put panels 1-6 on one common colour scale instead of '
                         'a per-panel scale centred on 0')
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rates = load_group(in_dir / MODEL_NAME, [k for k, _ in RATE_KEYS],
                       args.stride, 'model4-rates')
    mb = load_group(in_dir / MB_TAG, [k for k, _, _ in MB_KEYS],
                    args.mb_stride, f'model4-{N_QUBITS}q')

    np.savez(out_dir / 'model4_rate_panels.npz',
             model=MODEL_NAME, h=MODEL4_H, N_manybody=N_QUBITS,
             lattice=f'{LATTICE_LY}x{LATTICE_LX}',
             stride=args.stride, mb_stride=args.mb_stride,
             gamma_vals=rates['p1_vals'], gamma_p_vals=rates['p2_vals'],
             mb_gamma_vals=mb['p1_vals'], mb_gamma_p_vals=mb['p2_vals'],
             **{k: rates[k] for k, _ in RATE_KEYS},
             **{k: mb[k] for k, _, _ in MB_KEYS})
    print(f'[model4-rate] wrote {out_dir / "model4_rate_panels.npz"}', flush=True)

    plot(rates, mb, out_dir / 'model4_rate_panels.png',
         shared_rate_scale=args.shared_rate_scale)


if __name__ == '__main__':
    main()
