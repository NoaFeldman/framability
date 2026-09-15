"""
Collect scripts/liouvillian_q_worker.py's per-point files into (gamma, gamma')
grids, and draw the Q panels / Q contours shared by

  * scripts/model4_rate_panels_collect.py   -> results_model4_rate/model4_rate_panels.png
  * scripts/trotter_dtbase_line_extrap.py   -> results_dtbase_line/<model>_dtbase_extrap.png
    (also via scripts/collect_and_plot_all.py)

Grids hold Q_max over the damped modes, set to +inf where the spectrum has an
undamped oscillating mode (closed-dynamics edges); the panels show those at the
colour cap and say how many there are.

Usage:
    python scripts/liouvillian_q_collect.py --model model3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'scripts'))

from liouvillian_q_worker import grid_vals          # noqa: E402

# (grid key, per-point prefix)
Q_GROUPS = [('bond_Q_max', 'bond'), ('lat_Q_max', 'lat')]

Q_CAP = 4.0                  # colour cap: the physics of interest is Q ~ 1
Q_LEVELS_DEFAULT = (1.0,)    # Q = 1: the product-frame square in a correlated plane
Q_CONTOUR_COLOR = 'cyan'


def load(model: str, in_dir: Path, stride: int = 1) -> dict | None:
    """Q grids of `model`, or None when no worker output exists yet."""
    pt_dir = Path(in_dir) / model
    if not pt_dir.is_dir():
        return None
    p1_vals, p2_vals = grid_vals(model, stride)
    nx, ny = len(p1_vals), len(p2_vals)
    grids = {k: np.full((nx, ny), np.nan) for k, _ in Q_GROUPS}
    n_undamped = {k: 0 for k, _ in Q_GROUPS}
    found, lattice = 0, None
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
            found += 1
            for key, pre in Q_GROUPS:
                if pre == 'lat' and not bool(d['lat_done']):
                    continue
                if int(d[f'{pre}_n_undamped']) > 0:
                    grids[key][ix, iy] = np.inf
                    n_undamped[key] += 1
                else:
                    grids[key][ix, iy] = float(d[f'{pre}_Q_max'])
            if lattice is None and bool(d['lat_done']):
                # lat_topology is 'ring' for the 1D ring models; files written
                # before the key existed are open-boundary lattices
                topo = (str(d['lat_topology']) if 'lat_topology' in d.files
                        else 'lattice')
                lattice = (int(d['lat_Ly']), int(d['lat_Lx']), topo)
    if found == 0:
        return None
    print(f'[liouvillian_q] {model}: {found}/{nx * ny} grid points loaded; '
          + ', '.join(f'{k} finite/undamped '
                      f'{int(np.isfinite(grids[k]).sum())}/{n_undamped[k]}'
                      for k, _ in Q_GROUPS), flush=True)
    ly, lx, topo = lattice if lattice else (np.nan, np.nan, 'lattice')
    return dict(p1_vals=p1_vals, p2_vals=p2_vals, found=found,
                lat_Ly=ly, lat_Lx=lx, lat_topology=topo, **grids)


def labels(d: dict) -> dict:
    """Panel titles for the two Q grids."""
    q = r'$Q_{\max}=\max_k|\mathrm{Im}\,\lambda_k/\mathrm{Re}\,\lambda_k|$'
    if not np.isfinite(d['lat_Ly']):
        lat = 'lattice'
    elif d.get('lat_topology') == 'ring':
        lat = f"{d['lat_Lx']}-site ring"
    else:
        lat = f"{d['lat_Ly']}x{d['lat_Lx']} lattice"
    return {'bond_Q_max': f'{q}\nbond generator (exact)',
            'lat_Q_max': f'{q}\n{lat} Lindbladian (exact)'}


def _edges(v):
    v = np.asarray(v, float)
    if v.size == 1:
        return np.array([v[0] - 0.5, v[0] + 0.5])
    mid = (v[:-1] + v[1:]) / 2
    return np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])


def draw_q_panel(fig, ax, xv, yv, Z, title, *, xlabel, ylabel, cmap='magma',
                 cap: float = Q_CAP, levels=Q_LEVELS_DEFAULT) -> None:
    """Q grid Z[ix, iy] as a colormap clipped at `cap`, with white dashed
    contours at `levels`.  +inf (undamped oscillation) is drawn at the cap."""
    Zt = np.asarray(Z, float).T
    n_inf = int(np.isinf(Zt).sum())
    Zs = np.where(np.isinf(Zt), cap, Zt)
    finite = np.isfinite(Zs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not finite.any():
        ax.set_title(f'{title}\n(no data)', fontsize=10)
        return
    hi = float(Zs[finite].max())
    clipped = hi > cap
    vmax = cap if clipped else max(hi, 1e-12)
    pcm = ax.pcolormesh(_edges(xv), _edges(yv), np.minimum(Zs, cap), cmap=cmap,
                        vmin=0.0, vmax=vmax, shading='flat')
    for lev in levels:
        if float(Zs[finite].min()) < lev < hi:
            ax.contour(np.asarray(xv, float), np.asarray(yv, float), Zs,
                       levels=[lev], colors='white', linestyles='--',
                       linewidths=1.2)
    note = []
    if clipped:
        note.append(f'clipped at {cap:g}, max {hi:.3g}')
    if n_inf:
        note.append(f'{n_inf} undamped pt(s) at cap')
    if levels:
        note.append('dashed: Q=' + ','.join(f'{lev:g}' for lev in levels))
    ax.set_title(title + (f"\n({'; '.join(note)})" if note else ''), fontsize=10)
    fig.colorbar(pcm, ax=ax, extend='max' if clipped or n_inf else 'neither')


def draw_q_contour(ax, xv, yv, Z, *, levels=Q_LEVELS_DEFAULT,
                   color: str = Q_CONTOUR_COLOR, linestyle: str = '--') -> None:
    """Overlay Q = level contours of Z[ix, iy] on an existing panel."""
    Zt = np.asarray(Z, float).T
    Zs = np.where(np.isinf(Zt), 1e6, Zt)
    finite = np.isfinite(Zs)
    if not finite.any():
        return
    lo, hi = float(Zs[finite].min()), float(Zs[finite].max())
    for lev in levels:
        if lo < lev < hi:
            ax.contour(np.asarray(xv, float), np.asarray(yv, float), Zs,
                       levels=[lev], colors=color, linestyles=linestyle,
                       linewidths=1.1)


# ---------------------------------------------------------------------------
#  Observable quality factor Q_obs (scripts/observable_q_worker.py)
# ---------------------------------------------------------------------------
# (grid key, label key)
OBS_GROUPS = [('obs_Q', 'obs_label'), ('obs_opt_Q', 'obs_opt_label')]
# Q_obs = 1 contour style on the rate / framability panels
OBS_STYLE = {'obs_Q': ('magenta', '-.'), 'obs_opt_Q': ('orange', ':')}


def canonical_label(s: str) -> str:
    """Merge a two-letter string with its mirror image ('IZ' -> 'ZI/IZ')."""
    r = s[::-1]
    if r == s:
        return s
    a, b = (s, r) if s >= r else (r, s)
    return f'{a}/{b}'


def load_obs(model: str, in_dir: Path, stride: int = 1) -> dict | None:
    """Q_obs grids and binding-string grids of `model`, or None without data.
    Missing optimised-basis keys (a --no_opt run) leave NaN / ''."""
    pt_dir = Path(in_dir) / model
    if not pt_dir.is_dir():
        return None
    p1_vals, p2_vals = grid_vals(model, stride)
    nx, ny = len(p1_vals), len(p2_vals)
    out = {k: np.full((nx, ny), np.nan) for k, _ in OBS_GROUPS}
    out.update({lk: np.full((nx, ny), '', dtype=object) for _, lk in OBS_GROUPS})
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
            found += 1
            for k, lk in OBS_GROUPS:
                if k in d.files:
                    out[k][ix, iy] = float(d[k])
                    out[lk][ix, iy] = str(d[lk])
    if found == 0:
        return None
    print(f'[observable_q] {model}: {found}/{nx * ny} grid points loaded; '
          + ', '.join(f'{k} finite {int(np.isfinite(out[k]).sum())}'
                      for k, _ in OBS_GROUPS), flush=True)
    return dict(p1_vals=p1_vals, p2_vals=p2_vals, found=found, **out)


def obs_titles(d: dict) -> dict:
    q = r'$Q_{\rm obs}=\max_P\sum_{P^\prime\neq P}|A_{P^\prime P}|/(-A_{PP})$'
    return {'obs_Q': f'{q}\nbond generator, Pauli basis',
            'obs_opt_Q': f'{q}\nbond generator, optimised local basis',
            'obs_label': r'string attaining $Q_{\rm obs}$ (Pauli basis)',
            'obs_opt_label': r'axes attaining $Q_{\rm obs}$ (optimised basis,'
                             '\nnamed by nearest Pauli axis)'}


def draw_label_panel(fig, ax, xv, yv, labels, title, *, xlabel, ylabel) -> None:
    """Categorical map of the string attaining Q_obs (mirror pairs merged)."""
    from matplotlib import colormaps
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    Lt = np.asarray(labels, dtype=object).T
    canon = np.empty(Lt.shape, dtype=object)
    for idx, s in np.ndenumerate(Lt):
        canon[idx] = canonical_label(s) if s else ''
    names = sorted({s for s in canon.ravel() if s})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not names:
        ax.set_title(f'{title}\n(no data)', fontsize=10)
        return
    code = {s: i for i, s in enumerate(names)}
    Z = np.full(canon.shape, np.nan)
    for idx, s in np.ndenumerate(canon):
        if s:
            Z[idx] = code[s]
    colors = [colormaps['tab20'](i % 20) for i in range(len(names))]
    ax.pcolormesh(_edges(xv), _edges(yv), Z, cmap=ListedColormap(colors),
                  vmin=-0.5, vmax=len(names) - 0.5, shading='flat')
    ax.legend(handles=[Patch(color=colors[i], label=s)
                       for i, s in enumerate(names)],
              fontsize=7, loc='upper right', framealpha=0.85)
    ax.set_title(title, fontsize=10)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='model3')
    ap.add_argument('--in_dir', type=str, default='results_liouvillian_q')
    ap.add_argument('--out_dir', type=str, default='results_liouvillian_q')
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    d = load(args.model, Path(args.in_dir), args.stride)
    if d is None:
        print(f'[liouvillian_q] no data under {args.in_dir}/{args.model}')
        return
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f'{args.model}_liouvillian_q.npz'
    np.savez(npz, model=args.model, **d)
    print(f'[liouvillian_q] saved {npz}', flush=True)


if __name__ == '__main__':
    main()
