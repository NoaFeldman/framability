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
  * scripts/observable_q_worker.py        -> <obs_dir>/model4/pt_<ix>_<iy>.npz
       panels 11-14: observable quality factor Q_obs of the bond generator in
       the Pauli basis and in the optimised local basis, and the string
       attaining each; Q_obs = 1 is drawn on the rate panels (magenta
       dash-dot = Pauli basis, orange dotted = optimised basis)

assembles each quantity on its (gamma, gamma') grid, stores the merged arrays
and draws

    row 1 |  stabilizer-3 rate  |  Pauli rate  |  opt Heisenberg d=4  |  d=6
    row 2 |  opt Schrodinger d=4 |  d=6        |  8q osc rate         |  8q gap
    row 3 |  bond Q_max          |  lattice Q_max  |  Q_obs Pauli  |  Q_obs opt
    row 4 |  binding string (Pauli) | binding axes (opt)
          (rows 3-4 hold whichever of the Q / Q_obs groups have data)

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

--model model8 collects the same pipeline run for model8 (default in/out dir
results_<model>_rate, figure <model>_rate_panels.png, axes = the model's own
scan parameters), --model model10 the run for the Shibata-Katsura dissipative
quantum Ising chain (https://arxiv.org/abs/1904.12505), whose panels 7-8 come
from an 8-site periodic ring instead of the 2x4 lattice
(scripts/submit_model10_rate.sh chains every stage).

Usage:
    python scripts/model4_rate_panels_collect.py
    python scripts/model4_rate_panels_collect.py --stride 1 --mb_stride 5
    python scripts/model4_rate_panels_collect.py --model model8 --q_levels 1 2.414 3.732
    python scripts/model4_rate_panels_collect.py --floor 0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS                              # noqa: E402
from model4_rate_panels_worker import (RATE_KEYS, MODEL_NAME,            # noqa: E402
                                       SUPPORTED_MODELS)
from model4_manybody_worker import (mb_tag, N_QUBITS,                    # noqa: E402
                                    mb_geometry)
import liouvillian_q_collect as qcollect                                 # noqa: E402

# (npz key, panel label) in figure order.
MB_KEYS = [
    ('osc_rate', rf'{N_QUBITS}q osc rate  $\max_k|{{\rm Im}}\lambda_k/'
                 rf'{{\rm Re}}\lambda_k|$'),
    ('gap',      f'{N_QUBITS}q Lindbladian gap'),
]

# Quantities the neighbour-refine pipelines can improve -- quick
# (scripts/model4_rate_quick_refine_worker.py, boundary points only) and full
# (scripts/model4_rate_nb_refine_worker.py, every point): the collect takes the
# min over the base scan and every refine round of both for these.
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


def grid_vals(stride: int, model: str = MODEL_NAME):
    m = MODELS[model]
    return (np.asarray(m.p1_vals[::stride], float),
            np.asarray(m.p2_vals[::stride], float))


def load_group(pt_dir: Path, keys, stride: int, label: str, *,
               refine_keys=frozenset(), model: str = MODEL_NAME) -> dict:
    """Assemble every key in `keys` on the strided grid from per-point files.

    Keys in `refine_keys` take the MINIMUM over the base scan file and every
    neighbour-refine round written next to it: quick rounds
    (pt_<ix>_<iy>_qrefine_r*.npz, scripts/model4_rate_quick_refine_worker.py)
    and full rounds (pt_<ix>_<iy>_nrefine_r*.npz,
    scripts/model4_rate_nb_refine_worker.py).  That is sound because every
    stored rate is a certified upper bound on the point's true minimum, so the
    smallest one is the best bound known.  Every other key is read from the
    base file alone.
    """
    p1_vals, p2_vals = grid_vals(stride, model)
    nx, ny = len(p1_vals), len(p2_vals)
    grids = {k: np.full((nx, ny), np.nan) for k in keys}
    found = n_refined = 0
    for ix in range(nx):
        for iy in range(ny):
            base = pt_dir / f'pt_{ix:03d}_{iy:03d}.npz'
            if not base.exists():
                continue
            rounds = sorted(pt_dir.glob(f'pt_{ix:03d}_{iy:03d}_*refine_r*.npz'))
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
        print(f'  neighbour refine: {n_refined} point(s) improved over the base scan',
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


def _panel(fig, ax, xv, yv, Z, title, cmap, *, floor_contour=None,
           xlabel=r'$\gamma$', ylabel=r"$\gamma'$"):
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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(pcm, ax=ax)


def plot(rates: dict, mb: dict, png: Path, *, floor: float = 0.0,
         model: str = MODEL_NAME, q: dict | None = None,
         q_levels=qcollect.Q_LEVELS_DEFAULT, obs: dict | None = None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    m = MODELS[model]
    lab = dict(xlabel=m.p1_label, ylabel=m.p2_label)

    # Panels after the eight rate / many-body panels, in figure order:
    # (kind, data, key, title, contour levels)
    extra = []
    if q is not None:
        titles = qcollect.labels(q)
        extra += [('q', q, key, titles[key], q_levels)
                  for key, _ in qcollect.Q_GROUPS]
    if obs is not None:
        titles = qcollect.obs_titles(obs)
        extra += [('q', obs, key, titles[key], (1.0,))
                  for key, _ in qcollect.OBS_GROUPS]
        extra += [('label', obs, lkey, titles[lkey], None)
                  for _, lkey in qcollect.OBS_GROUPS]

    nrow = 2 + int(np.ceil(len(extra) / 4))
    fig, axes = plt.subplots(nrow, 4, figsize=(22, 5 * nrow),
                             constrained_layout=True)
    levels_txt = ','.join(f'{lev:g}' for lev in q_levels)
    notes = []
    if q is not None:
        notes.append(rf"$Q_{{\max}}$: most coherent damped mode (exact spectra), "
                     rf"dashed cyan = bond $Q_{{\max}}={levels_txt}$")
    if obs is not None:
        notes.append(r"$Q_{\rm obs}$: observable quality factor, $=1$ as "
                     r"magenta dash-dot (Pauli basis) / orange dotted "
                     r"(optimised basis)")
    fig.suptitle(
        f'{model}:  {m.title}'
        "\n"
        rf"framability rates $\mu^*=\lim_{{dt\to0}}({{\rm fra}}-1)/dt$ of the bond "
        rf"generator  |  panels 7-8: full {N_QUBITS}-qubit "
        rf"{mb_geometry(model)['label']} Lindbladian"
        + ('\n' + '  |  '.join(notes) if notes else ''),
        fontsize=13)

    # Panels 1-6 are framability rates, so they get the white floor contour:
    # mu* = 0 is the rate-picture image of framability = 1 (mu* = max(0,
    # coherence rate) exactly as framability = max(1, margin)), i.e. the
    # boundary of the region where the frame does not inflate at all.
    for ax, (key, label) in zip(axes.flat[:6], RATE_KEYS):
        _panel(fig, ax, rates['p1_vals'], rates['p2_vals'], rates[key],
               label, FRA_CMAP, floor_contour=floor, **lab)
        if q is not None:
            qcollect.draw_q_contour(ax, q['p1_vals'], q['p2_vals'],
                                    q['bond_Q_max'], levels=q_levels)
        if obs is not None:
            for key, (color, ls) in qcollect.OBS_STYLE.items():
                qcollect.draw_q_contour(ax, obs['p1_vals'], obs['p2_vals'],
                                        obs[key], levels=(1.0,), color=color,
                                        linestyle=ls)

    # Panels 7-8 are not framabilities and have no such floor.
    for ax, (key, label) in zip(axes.flat[6:8], MB_KEYS):
        _panel(fig, ax, mb['p1_vals'], mb['p2_vals'], mb[key], label, MB_CMAP,
               **lab)

    # Panels 9+: mode quality factor Q_max (bond, lattice), observable quality
    # factor Q_obs (Pauli, optimised basis) and the strings attaining Q_obs.
    for ax, (kind, d, key, title, levels) in zip(axes.flat[8:], extra):
        if kind == 'q':
            qcollect.draw_q_panel(fig, ax, d['p1_vals'], d['p2_vals'], d[key],
                                  title, cmap=MB_CMAP, levels=levels, **lab)
        else:
            qcollect.draw_label_panel(fig, ax, d['p1_vals'], d['p2_vals'],
                                      d[key], title, **lab)
    for ax in axes.flat[8 + len(extra):]:
        ax.axis('off')

    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f'[{model}-rate] wrote {png}', flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',   type=str, default=MODEL_NAME,
                    choices=SUPPORTED_MODELS)
    ap.add_argument('--in_dir',  type=str, default=None,
                    help='default results_<model>_rate')
    ap.add_argument('--out_dir', type=str, default=None,
                    help='default results_<model>_rate')
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
    ap.add_argument('--obs_dir', type=str, default='results_observable_q',
                    help='scripts/observable_q_worker.py output; the Q_obs '
                         'panels and Q_obs = 1 contours are added when data '
                         'exists there')
    ap.add_argument('--obs_stride', type=int, default=1,
                    help='stride used by observable_q_worker')
    args = ap.parse_args()

    model = args.model
    m = MODELS[model]
    in_dir = Path(args.in_dir or f'results_{model}_rate')
    out_dir = Path(args.out_dir or f'results_{model}_rate')
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'[{model}-rate]'

    rates = load_group(in_dir / model, [k for k, _ in RATE_KEYS],
                       args.stride, f'{model}-rates',
                       refine_keys=RATE_REFINE_KEYS, model=model)
    mb = load_group(in_dir / mb_tag(model), [k for k, _ in MB_KEYS],
                    args.mb_stride, f'{model}-{N_QUBITS}q', model=model)
    q = qcollect.load(model, Path(args.q_dir), args.q_stride)
    if q is None:
        print(f'{tag} no Q data under {args.q_dir}/{model}; '
              f'Q row and contour omitted', flush=True)
    obs = qcollect.load_obs(model, Path(args.obs_dir), args.obs_stride)
    if obs is None:
        print(f'{tag} no Q_obs data under {args.obs_dir}/{model}; '
              f'Q_obs panels and contours omitted', flush=True)

    # axis arrays are named by the model's scan parameters, so model4 keeps
    # its gamma_vals / gamma_p_vals / mb_... / q_... keys
    x, y = m.p1_name, m.p2_name
    q_arrays = {} if q is None else {
        f'q_{x}_vals': q['p1_vals'], f'q_{y}_vals': q['p2_vals'],
        'q_stride': args.q_stride, **{k: q[k] for k, _ in qcollect.Q_GROUPS}}
    obs_arrays = {} if obs is None else {
        f'obs_{x}_vals': obs['p1_vals'], f'obs_{y}_vals': obs['p2_vals'],
        'obs_stride': args.obs_stride,
        **{k: obs[k] for k, _ in qcollect.OBS_GROUPS},
        **{lk: np.asarray(obs[lk], dtype='U8') for _, lk in qcollect.OBS_GROUPS}}

    npz = out_dir / f'{model}_rate_panels.npz'
    geo = mb_geometry(model)
    np.savez(npz, model=model, title=m.title, N_manybody=N_QUBITS,
             lattice=(f"{geo['Ly']}x{geo['Lx']}" if geo['topology'] == 'lattice'
                      else geo['topology']),
             stride=args.stride, mb_stride=args.mb_stride,
             **{f'{x}_vals': rates['p1_vals'], f'{y}_vals': rates['p2_vals'],
                f'mb_{x}_vals': mb['p1_vals'], f'mb_{y}_vals': mb['p2_vals']},
             **{k: rates[k] for k, _ in RATE_KEYS},
             **{k: mb[k] for k, _ in MB_KEYS},
             **q_arrays, **obs_arrays)
    print(f'{tag} wrote {npz}', flush=True)

    plot(rates, mb, out_dir / f'{model}_rate_panels.png', floor=args.floor,
         model=model, q=q, q_levels=tuple(args.q_levels), obs=obs)


if __name__ == '__main__':
    main()
