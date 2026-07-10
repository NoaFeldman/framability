"""
Collect the dephasing-anneal chains (trotter_deph_anneal_worker.py) and plot
the optimised framability (Heisenberg, d_ext_single = 6) as a function of the
added dephasing rate kappa.

Per model, all seed chains of the selected scan point are combined into a
per-kappa minimum envelope, separately for the ramp-up stage and the anneal-
down (frame-continuation) stage.  The figure shows, for each model:

  * the anneal-down envelope (the main curve, kappa_star -> 0),
  * the ramp-up envelope (0 -> kappa_star),
  * a dashed horizontal line at the *original* scan opt_fra_6 (the value at
    kappa = 0 before the whole procedure),
  * dashed vertical lines at the model's coefficients (J, h, gamma, ...) so
    the kappa below which dephasing becomes negligible can be read off,
  * a dotted line at framability = 1 (the framable threshold).

The x-axis is symlog so the exact kappa = 0 endpoint is on the plot.

Outputs: <out_dir>/deph_anneal_<model>.png   (one per model)
         <out_dir>/deph_anneal_all.png       (one panel per model)
         <in_dir>/<model>/deph_anneal_summary.npz

Usage:
    python scripts/trotter_deph_anneal_collect.py \
        --models model1 model2 model5 \
        --in_dir results_deph_anneal --out_dir results_plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS

# Series colors: first two steps of the validated default categorical palette
# (dataviz reference palette); reference lines stay in recessive grays.
C_DOWN = '#4269d0'      # anneal-down envelope (the main curve)
C_UP   = '#efb118'      # ramp-up envelope
C_REF  = '#555555'      # dashed reference lines (orig fra, coefficients)
C_ONE  = '#999999'      # framability = 1 threshold


# ---------------------------------------------------------------------------
def load_model(in_dir: Path, model_name: str) -> dict | None:
    """Load all chains of one model, keep the point group with minimal scan
    opt_fra_6 (chains of a stale point selection are dropped with a warning),
    and reduce the seeds to per-kappa min envelopes on the union kappa grid."""
    mdir = in_dir / model_name
    chains = []
    for f in sorted(mdir.glob('chain_*_seed*.npz')):
        try:
            chains.append(dict(np.load(f, allow_pickle=True)))
        except Exception as e:
            print(f'  warning: {f.name}: {e}', flush=True)
    if not chains:
        print(f'{model_name}: no chains found in {mdir}', flush=True)
        return None

    groups: dict[tuple[int, int], list[dict]] = {}
    for c in chains:
        groups.setdefault((int(c['ix']), int(c['iy'])), []).append(c)
    if len(groups) > 1:
        keep = min(groups, key=lambda k: float(groups[k][0]['fra_orig']))
        print(f'{model_name}: WARNING -- chains for {len(groups)} different '
              f'points found {sorted(groups)}; keeping {keep} '
              f'(minimal scan opt_fra_6)', flush=True)
        chains = groups[keep]

    kappas = np.unique(np.concatenate([c['kappas'] for c in chains]))
    n, m = len(kappas), len(chains)
    up = np.full((m, n), np.nan)
    down = np.full((m, n), np.nan)
    for r, c in enumerate(chains):
        idx = np.searchsorted(kappas, c['kappas'])
        up[r, idx] = c['fra_up']
        down[r, idx] = c['fra_down']

    c0 = chains[0]
    n_reached = sum(bool(c['reached']) for c in chains)
    if n_reached < len(chains):
        print(f'{model_name}: WARNING -- only {n_reached}/{len(chains)} chains '
              f'reached framability 1 during the ramp-up', flush=True)
    return dict(
        model=model_name, n_chains=m, kappas=kappas,
        env_up=np.fmin.reduce(up), env_down=np.fmin.reduce(down),
        per_seed_down=down,
        fra_orig=float(c0['fra_orig']),
        ix=int(c0['ix']), iy=int(c0['iy']),
        p1=float(c0['p1']), p2=float(c0['p2']),
        p1_name=str(c0['p1_name']), p2_name=str(c0['p2_name']),
        dt=float(c0['dt']), d_ext=int(c0['d_ext']),
        coef_names=[str(s) for s in c0['coef_names']],
        coef_values=np.asarray(c0['coef_values'], dtype=float),
        kappa_star=float(np.max([c['kappas'][int(c['i_star'])] for c in chains])),
    )


# ---------------------------------------------------------------------------
def plot_panel(ax, d: dict, show_legend: bool = True) -> None:
    kappas, env_down, env_up = d['kappas'], d['env_down'], d['env_up']
    pos = kappas[kappas > 0]
    linthresh = float(pos.min()) if len(pos) else 1e-3
    ax.set_xscale('symlog', linthresh=linthresh, linscale=0.5)

    # reference: framable threshold and the original optimised framability
    ax.axhline(1.0, color=C_ONE, lw=0.8, ls=':', zorder=1)
    if np.isfinite(d['fra_orig']):
        ax.axhline(d['fra_orig'], color=C_REF, lw=1.2, ls='--', zorder=2,
                   label=r'original opt fra ($\kappa=0$)')

    # reference: the model coefficients (vertical dashed lines, direct-labeled)
    for name, val in zip(d['coef_names'], d['coef_values']):
        if not np.isfinite(val) or val <= 0:
            continue          # a zero coefficient has no scale to mark
        ax.axvline(val, color=C_REF, lw=1.0, ls='--', alpha=0.7, zorder=2)
        ax.annotate(name, xy=(val, 0.985), xycoords=('data', 'axes fraction'),
                    ha='right', va='top', rotation=90, fontsize=8, color=C_REF)

    m_up = np.isfinite(env_up)
    m_dn = np.isfinite(env_down)
    ax.plot(kappas[m_up], env_up[m_up], color=C_UP, lw=1.5, marker='.', ms=4,
            zorder=3, label=r'ramp up (to fra $=1$)')
    ax.plot(kappas[m_dn], env_down[m_dn], color=C_DOWN, lw=2.0, marker='o',
            ms=3.5, zorder=4, label='anneal down (frame continuation)')

    # direct label: the headline kappa = 0 continuation value
    if m_dn[0]:
        ax.annotate(f'{env_down[0]:.4f}', xy=(kappas[0], env_down[0]),
                    xytext=(6, 8), textcoords='offset points', fontsize=8,
                    color=C_DOWN)

    ax.set_xlim(left=-0.5 * linthresh)
    ax.set_xlabel(r'dephasing rate $\kappa$  (added jump $\sqrt{\kappa}\,Z$)')
    ax.set_title(f"{d['model']}:  {d['p1_name']}={d['p1']:.2f}, "
                 f"{d['p2_name']}={d['p2']:.2f}  "
                 f"(pt {d['ix']:03d}_{d['iy']:03d}, {d['n_chains']} seeds)",
                 fontsize=10)
    ax.grid(True, which='major', alpha=0.25, lw=0.5)
    if show_legend:
        ax.legend(fontsize=8, frameon=False, loc='best')


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=str, nargs='+',
                   default=['model1', 'model2', 'model5'],
                   choices=list(MODELS))
    p.add_argument('--in_dir',  type=str, default='results_deph_anneal')
    p.add_argument('--out_dir', type=str, default='results_plots')
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for name in args.models:
        d = load_model(in_dir, name)
        if d is None:
            continue
        loaded.append(d)
        print(f"{name}: {d['n_chains']} chains, point "
              f"({d['p1_name']}={d['p1']:.2f}, {d['p2_name']}={d['p2']:.2f}), "
              f"orig fra={d['fra_orig']:.6f}, kappa_star={d['kappa_star']:.4g}, "
              f"anneal-down fra(0)={d['env_down'][0]:.6f}", flush=True)

        np.savez(in_dir / name / 'deph_anneal_summary.npz',
                 kappas=d['kappas'], env_up=d['env_up'],
                 env_down=d['env_down'], per_seed_down=d['per_seed_down'],
                 fra_orig=np.array(d['fra_orig']),
                 ix=np.array(d['ix']), iy=np.array(d['iy']),
                 p1=np.array(d['p1']), p2=np.array(d['p2']),
                 coef_names=np.array(d['coef_names']),
                 coef_values=d['coef_values'],
                 kappa_star=np.array(d['kappa_star']),
                 n_chains=np.array(d['n_chains']))

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        plot_panel(ax, d)
        ax.set_ylabel(r'optimised framability  (Heisenberg, $d_{\rm ext}=6$)')
        fig.tight_layout()
        png = out_dir / f'deph_anneal_{name}.png'
        fig.savefig(png, dpi=200)
        plt.close(fig)
        print(f'  wrote {png}', flush=True)

    if len(loaded) > 1:
        fig, axes = plt.subplots(1, len(loaded),
                                 figsize=(5.4 * len(loaded), 4.4))
        for k, (ax, d) in enumerate(zip(np.atleast_1d(axes), loaded)):
            plot_panel(ax, d, show_legend=(k == 0))
            if k == 0:
                ax.set_ylabel(r'optimised framability  '
                              r'(Heisenberg, $d_{\rm ext}=6$)')
        fig.tight_layout()
        png = out_dir / 'deph_anneal_all.png'
        fig.savefig(png, dpi=200)
        plt.close(fig)
        print(f'wrote {png}', flush=True)


if __name__ == '__main__':
    main()
