"""
Collect the model4 rate pipeline and draw its eight-panel figure.

Reads the per-point npz files written by
  * scripts/model4_rate_panels_worker.py  -> <in_dir>/model4/pt_<ix>_<iy>.npz
       panels 1-6: framability rates of the two-qubit bond generator
  * scripts/model4_manybody_worker.py     -> <in_dir>/model4_8q/pt_<ix>_<iy>.npz
       panels 7-8: oscillation rate and Lindbladian gap of the full 8-qubit
       2x4 lattice
  * scripts/liouvillian_q_worker.py       -> <q_dir>/model4/pt_<ix>_<iy>.npz
       panels 9-10: quality factor Q_max of the bond generator and of the
       2x3-lattice Lindbladian (exact spectra); bond Q_max = 1 is also drawn as
       a dashed cyan contour on the six rate panels

assembles each quantity on its (gamma, gamma') grid, stores the merged arrays
and draws

    row 1 |  stabilizer-3 rate  |  Pauli rate  |  opt Heisenberg d=4  |  d=6
    row 2 |  opt Schrodinger d=4 |  d=6        |  8q osc rate         |  8q gap
    row 3 |  bond Q_max          |  lattice Q_max          (when Q data exists)

The two groups may live on different strides (the many-body panels default to
--mb_stride 5, an 11x11 grid, against the framability panels' full 51x51); the
panels are drawn with pcolormesh on their own axis values, so they still share
the same (gamma, gamma') data coordinates and line up.

Colour conventions
------------------
Colours match results_dtbase_line (scripts/trotter_dtbase_line_extrap.py):
viridis for the six framability-rate panels, magma for the two panels that are
not framabilities, and a white contour on the framable floor.

Each panel spans exactly its own finite data range: no padding to 0, no
symmetrisation, no scale shared between panels, so every colour on a colourbar
is realised somewhere in its image.  (This is the one deliberate departure from
the reference, which pins vmin to the framability floor of 1 -- here the
stabilizer-3 and Schrodinger rates never approach their floor, so pinning it
would waste most of the colour range.)  A quantity spanning more than
_LOG_DECADES decades of strictly positive values is drawn logarithmically and
says so in its title (only osc_rate qualifies).

The six framability-rate panels carry a white contour around the region where
the rate sits at its floor, mu* = 0 -- the rate-picture image of framability =
1, since mu* = max(0, coherence rate) exactly as framability = max(1, margin).
Inside that contour the frame does not inflate at all.

Usage:
    python scripts/model4_rate_panels_collect.py
    python scripts/model4_rate_panels_collect.py --stride 1 --mb_stride 5
    python scripts/model4_rate_panels_collect.py --floor 0.0
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
import liouvillian_q_collect as qcollect                                 # noqa: E402

# (npz key, panel label) in figure order.
MB_KEYS = [
    ('osc_rate', rf'{N_QUBITS}q osc rate  $\max_k|{{\rm Im}}\lambda_k/'
                 rf'{{\rm Re}}\lambda_k|$'),
    ('gap',      f'{N_QUBITS}q Lindbladian gap'),
]

# Quantities the quick neighbour-refine pipeline
# (scripts/model4_rate_quick_refine_worker.py) can improve: the collect takes
# the min over the base scan and every refine round for these.
RATE_REFINE_KEYS = frozenset({'rate_heis_4', 'rate_heis_6'})

# Colours follow results_dtbase_line (scripts/trotter_dtbase_line_extrap.py):
# viridis for the framability panels with a white contour on the framable
# floor, magma for the panels that are not framabilities.
FRA_CMAP = 'viridis'
MB_CMAP = 'magma'
CONTOUR_LW = 1.3

# A quantity whose finite values are strictly positive and span more than this
# many decades is drawn on a logarithmic colour scale.  Only osc_rate does:
# max|Im/Re| diverges as Re(lambda) -> 0, so it runs from ~1e-15 (a purely real
# spectrum: no ringing at all) to ~1e18 at a near-undamped mode.  On a linear
# scale that single point flattens the whole panel to one colour.
_LOG_DECADES = 4.0

# Values within this of a panel's floor count as sitting ON the floor when the
# white contour is drawn.  The rate LPs are accurate to ~1e-9, so 1e-6 is a
# safe "numerically at the floor" band; it picks up the thin fringe of points
# the optimiser drove to 1e-7-ish rather than exactly 0.
CONTOUR_TOL = 1e-6


def grid_vals(stride: int):
    m = MODELS[MODEL_NAME]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def load_group(pt_dir: Path, keys, stride: int, label: str, *,
               refine_keys=frozenset()) -> dict:
    """Assemble every key in `keys` on the strided grid from per-point files.

    Keys in `refine_keys` take the MINIMUM over the base scan file and every
    quick-refine round written next to it (pt_<ix>_<iy>_qrefine_r*.npz, from
    scripts/model4_rate_quick_refine_worker.py).  That is sound because every
    stored rate is a certified upper bound on the point's true minimum, so the
    smallest one is the best bound known.  Every other key is read from the
    base file alone.
    """
    p1_vals, p2_vals = grid_vals(stride)
    nx, ny = len(p1_vals), len(p2_vals)
    grids = {k: np.full((nx, ny), np.nan) for k in keys}
    found = n_refined = 0
    for ix in range(nx):
        for iy in range(ny):
            base = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not base.exists():
                continue
            rounds = sorted(pt_dir.glob(f'pt_{ix:03d}_{iy:03d}_qrefine_r*.npz'))
            improved = False
            for f in [base, *rounds]:
                try:
                    d = np.load(f, allow_pickle=True)
                except Exception as e:
                    print(f'  warning: {f.name}: {e}', flush=True)
                    continue
                is_base = f == base
                for k in keys:
                    if k not in d or not (is_base or k in refine_keys):
                        continue
                    v = float(d[k])
                    cur = grids[k][ix, iy]
                    if not np.isfinite(cur) or v < cur:
                        if not is_base and np.isfinite(cur):
                            improved = True
                        grids[k][ix, iy] = v
            found += 1
            n_refined += bool(improved)
    print(f'[{label}] {found}/{nx * ny} grid points loaded from {pt_dir}',
          flush=True)
    if refine_keys:
        print(f'  quick-refine: {n_refined} point(s) improved over the base scan',
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


def _panel(fig, ax, xv, yv, Z, title, cmap, *, floor_contour=None):
    """One pcolormesh panel; Z is indexed [ix, iy] so it is drawn transposed.

    The colour scale always spans exactly the panel's own finite data -- no
    padding to 0, no symmetrisation about 0, no scale shared with another
    panel -- so every colour on the bar occurs somewhere in the image.

    floor_contour : float | None
        Draw a white outline around the region where Z sits at this value (to
        within CONTOUR_TOL), as results_dtbase_line does around its framable
        region.  The region is outlined by contouring its 0/1 indicator mask
        at 0.5 rather than contouring Z at the level itself: the floor is a
        flat plateau, so Z has no crossing there and a direct contour would
        draw nothing.
    """
    Zt = np.asarray(Z, float).T
    finite = np.isfinite(Zt)
    if not finite.any():
        ax.set_title(f'{title}\n(no data)', fontsize=10)
        ax.set_xlabel(r'$\gamma$')
        ax.set_ylabel(r"$\gamma'$")
        return

    vals = Zt[finite]
    lo, hi = float(vals.min()), float(vals.max())
    norm = None
    if lo > 0 and hi / lo > 10.0 ** _LOG_DECADES:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=lo, vmax=hi)
    elif hi <= lo:
        hi = lo + 1e-12

    if norm is not None:
        pcm = ax.pcolormesh(_edges(xv), _edges(yv), Zt, cmap=cmap, norm=norm,
                            shading='flat')
        title = f'{title}  [log scale]'
    else:
        pcm = ax.pcolormesh(_edges(xv), _edges(yv), Zt, cmap=cmap,
                            vmin=lo, vmax=hi, shading='flat')

    if floor_contour is not None:
        mask = np.where(finite, np.abs(Zt - floor_contour) <= CONTOUR_TOL,
                        False).astype(float)
        if 0.0 < mask.sum() < mask.size:
            ax.contour(np.asarray(xv, float), np.asarray(yv, float), mask,
                       levels=[0.5], colors='white', linewidths=CONTOUR_LW)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r"$\gamma'$")
    fig.colorbar(pcm, ax=ax)


def plot(rates: dict, mb: dict, png: Path, *, floor: float = 0.0,
         q: dict | None = None,
         q_levels=qcollect.Q_LEVELS_DEFAULT) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nrow = 2 if q is None else 3
    fig, axes = plt.subplots(nrow, 4, figsize=(22, 5 * nrow),
                             constrained_layout=True)
    levels_txt = ','.join(f'{lev:g}' for lev in q_levels)
    q_line = ('' if q is None else
              "\n" + rf"row 3: quality factor $Q_{{\max}}$ of the most coherent "
              rf"damped mode (exact spectra)  |  dashed cyan on the rate "
              rf"panels: bond $Q_{{\max}}={levels_txt}$")
    fig.suptitle(
        rf"model4:  $H = J\sum_{{\langle ij\rangle}} Z_iZ_j + {MODEL4_H}\sum_i X_i$,  "
        rf"jumps $\sqrt{{\gamma}}\,|{{-}}\rangle\langle{{+}}|_i,\ \sqrt{{\gamma'}}Z_i$  "
        rf"($J=1$)"
        "\n"
        rf"framability rates $\mu^*=\lim_{{dt\to0}}({{\rm fra}}-1)/dt$ of the bond "
        rf"generator  |  panels 7-8: full {N_QUBITS}-qubit "
        rf"{LATTICE_LY}x{LATTICE_LX} lattice Lindbladian" + q_line,
        fontsize=13)

    # Panels 1-6 are framability rates, so they get the white floor contour:
    # mu* = 0 is the rate-picture image of framability = 1 (mu* = max(0,
    # coherence rate) exactly as framability = max(1, margin)), i.e. the
    # boundary of the region where the frame does not inflate at all.
    for ax, (key, label) in zip(axes.flat[:6], RATE_KEYS):
        _panel(fig, ax, rates['p1_vals'], rates['p2_vals'], rates[key],
               label, FRA_CMAP, floor_contour=floor)
        if q is not None:
            qcollect.draw_q_contour(ax, q['p1_vals'], q['p2_vals'],
                                    q['bond_Q_max'], levels=q_levels)

    # Panels 7-8 are not framabilities and have no such floor.
    for ax, (key, label) in zip(axes.flat[6:8], MB_KEYS):
        _panel(fig, ax, mb['p1_vals'], mb['p2_vals'], mb[key], label, MB_CMAP)

    # Panels 9-10: Lindbladian quality factor (bond generator, exact lattice).
    if q is not None:
        titles = qcollect.labels(q)
        for ax, (key, _) in zip(axes.flat[8:10], qcollect.Q_GROUPS):
            qcollect.draw_q_panel(fig, ax, q['p1_vals'], q['p2_vals'], q[key],
                                  titles[key], xlabel=r'$\gamma$',
                                  ylabel=r"$\gamma'$", cmap=MB_CMAP,
                                  levels=q_levels)
        for ax in axes.flat[10:]:
            ax.axis('off')

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
    ap.add_argument('--floor', type=float, default=0.0,
                    help='value the white contour outlines on the rate panels '
                         '(default 0.0 = the framability rate floor, the '
                         'rate-picture image of framability = 1)')
    ap.add_argument('--q_dir', type=str, default='results_liouvillian_q',
                    help='scripts/liouvillian_q_worker.py output; the Q row and '
                         'the bond-Q contour are added when data exists there')
    ap.add_argument('--q_stride', type=int, default=1,
                    help='stride used by liouvillian_q_worker')
    ap.add_argument('--q_levels', type=float, nargs='+',
                    default=list(qcollect.Q_LEVELS_DEFAULT),
                    help='Q values contoured (dashed) on the rate and Q panels')
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rates = load_group(in_dir / MODEL_NAME, [k for k, _ in RATE_KEYS],
                       args.stride, 'model4-rates',
                       refine_keys=RATE_REFINE_KEYS)
    mb = load_group(in_dir / MB_TAG, [k for k, _ in MB_KEYS],
                    args.mb_stride, f'model4-{N_QUBITS}q')
    q = qcollect.load(MODEL_NAME, Path(args.q_dir), args.q_stride)
    if q is None:
        print(f'[model4-rate] no Q data under {args.q_dir}/{MODEL_NAME}; '
              f'Q row and contour omitted', flush=True)
    q_arrays = {} if q is None else dict(
        q_gamma_vals=q['p1_vals'], q_gamma_p_vals=q['p2_vals'],
        q_stride=args.q_stride, **{k: q[k] for k, _ in qcollect.Q_GROUPS})

    np.savez(out_dir / 'model4_rate_panels.npz',
             model=MODEL_NAME, h=MODEL4_H, N_manybody=N_QUBITS,
             lattice=f'{LATTICE_LY}x{LATTICE_LX}',
             stride=args.stride, mb_stride=args.mb_stride,
             gamma_vals=rates['p1_vals'], gamma_p_vals=rates['p2_vals'],
             mb_gamma_vals=mb['p1_vals'], mb_gamma_p_vals=mb['p2_vals'],
             **{k: rates[k] for k, _ in RATE_KEYS},
             **{k: mb[k] for k, _ in MB_KEYS},
             **q_arrays)
    print(f'[model4-rate] wrote {out_dir / "model4_rate_panels.npz"}', flush=True)

    plot(rates, mb, out_dir / 'model4_rate_panels.png', floor=args.floor,
         q=q, q_levels=tuple(args.q_levels))


if __name__ == '__main__':
    main()
