"""
Paper figure 1: framability-rate panels, one figure per model.

Eight panels in a 2 x 4 grid, in this order:
    Pauli | Stabilizer | Heisenberg d_ext=4 | Heisenberg d_ext=8
    Heisenberg d_ext=12 | Schrodinger (optimised) | Product chi=10 | Product chi=40

Linear viridis colour scale per panel (own colour bar).  The ONLY line drawn
is the white mu* = 0 contour (the boundary of the cells with rate <= --zero_tol).
Values below --zero_tol (LP noise of order 1e-14 on the stabilizer panel, tiny
negative LP residuals) are set to exactly 0 before plotting.  A panel whose
data does not exist yet is drawn empty with "no data", so the script can run
on partial results.

Cases (--cases): model4, model10 (1D chain), model10_dim2 (square lattice),
and any registered Dicke model (dicke_N10, dicke_dec_N10, ...).  Data sources
per case, first existing array wins (see SOURCES below):
  seeded panels  results_<model>_seeded/<model>_seeded_rate_panels.npz
                 (model10: results_model10_rate/model10_seeded_panels.npz,
                  model10_dim2: results_model10_rate_dim2/model10_dim2_rate_panels.npz)
                 -> Pauli, stabilizer, Heisenberg 4 / 8 / 12, product
  rate panels    results_<model>_rate/<model>_rate_panels.npz (original six-rate
                 scan) -> fallback for Pauli, stabilizer, Heisenberg 4, product,
                 and min(rate_schro_4, rate_schro_6) for the Schrodinger panel
  schro panels   <dir>/<model>_schro_panels.npz (scripts/paper_schro_collect.py)

Disordered Dicke cases (unequal omega0, framability_rate_disorder):
<family>_N<N>_M<M>_w<width>, e.g. dicke_dec_N100_M4_w0.50.  All eight panels are
PAIR-AVERAGED rates F (= the bond rate at width 0) of the width's slice of
  results_dicke_disorder/<family>_N<N>_M<M>_panels.npz        (dicke_disorder_rate_collect.py)
      -> Pauli, Heisenberg 4 / 8 / 12 (--disorder_frames species: one frame per
         species, the best found; shared: one frame on every atom)
  results_dicke_disorder/<family>_N<N>_M<M>_extra_panels.npz  (dicke_disorder_extra_collect.py)
      -> stabilizer, Schrodinger (shared state frame, best d_ext), product 10 / 40

Output: <out_dir>/<case>_rate_panels.{png,pdf}  (default results_plots/paper).

Complex frames (--complex_root DIR, e.g. results_rate_complex): the three
Heisenberg panels come from DIR/<model>_complex_rate_panels.npz
(scripts/rate_complex_collect.py: non-Hermitian frame elements, complex
coefficients) and the whole figure is drawn on that file's coarse grid
(every stride-th point); Pauli / stabilizer also come from that file, the
Schrodinger and product panels (state frames, Hermitian by definition, so
unchanged by complex coefficients) from the usual full-grid sources,
subsampled.  Output <case>_complex_rate_panels.{png,pdf}.

Usage:
    python scripts/paper_rate_panels_plot.py --cases model4 model10 dicke_dec_N100
    python scripts/paper_rate_panels_plot.py --cases dicke_dec_N100_M4_w0.50
    python scripts/paper_rate_panels_plot.py --cases model4 model14 \\
        --complex_root results_rate_complex
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import MODELS                              # noqa: E402

DL = r'$d_{\rm ext}'
PANELS = [
    ('pauli',  'Pauli'),
    ('stab',   'Stabilizer'),
    ('heis4',  rf'Heisenberg, {DL}=4$'),
    ('heis8',  rf'Heisenberg, {DL}=8$'),
    ('heis12', rf'Heisenberg, {DL}=12$'),
    ('schro',  'Schrödinger (optimized)'),
    ('prod10', r'Product, $\chi=10$'),
    ('prod40', r'Product, $\chi=40$'),
]


DIS_RE = re.compile(r'^(dicke\w*?_N\d+)_M(\d+)_w(\d+(?:\.\d+)?)$')


def case_files(case: str) -> tuple[str, dict]:
    """(registry model, {'seeded', 'rate', 'schro': npz path or None}); for a
    disordered Dicke case {'dis', 'disx', 'width'}."""
    m = DIS_RE.match(case)
    if m:
        model, M, w = m.group(1), int(m.group(2)), float(m.group(3))
        root = 'results_dicke_disorder'
        return model, dict(dis=f'{root}/{model}_M{M}_panels.npz',
                           disx=f'{root}/{model}_M{M}_extra_panels.npz', width=w)
    if case == 'model4':
        return 'model4', dict(seeded='results_model4_seeded/model4_seeded_rate_panels.npz',
                              rate='results_model4_rate/model4_rate_panels.npz',
                              schro='results_model4_rate/model4_schro_panels.npz')
    if case == 'model10':
        return 'model10', dict(seeded='results_model10_rate/model10_seeded_panels.npz',
                               rate='results_model10_rate/model10_rate_panels.npz',
                               schro='results_model10_rate/model10_schro_panels.npz')
    if case == 'model10_dim2':
        return 'model10', dict(seeded='results_model10_rate_dim2/model10_dim2_rate_panels.npz',
                               rate=None,
                               schro='results_model10_rate_dim2/model10_schro_panels.npz')
    if case in MODELS:
        root = f'results_{case}_seeded'
        return case, dict(seeded=f'{root}/{case}_seeded_rate_panels.npz',
                          rate=f'results_{case}_rate/{case}_rate_panels.npz',
                          schro=f'{root}/{case}_schro_panels.npz')
    raise ValueError(f'unknown case {case!r}')


# panel -> candidates (file role, key or tuple of keys = point-wise min), in order
SOURCES = {
    'pauli':  [('seeded', 'rate_pauli'), ('rate', 'rate_pauli')],
    'stab':   [('seeded', 'rate_stab3'), ('rate', 'rate_stab3')],
    'heis4':  [('seeded', 'panel_rate_heis_4'), ('rate', ('rate_heis_4', 'rate_heis_6'))],
    'heis8':  [('seeded', 'panel_rate_heis_8')],
    'heis12': [('seeded', 'panel_rate_heis_12')],
    'schro':  [('schro', 'rate_schro_best'), ('rate', ('rate_schro_4', 'rate_schro_6'))],
    'prod10': [('seeded', 'rate_prod_10'), ('rate', 'rate_prod_10')],
    'prod40': [('seeded', 'rate_prod_40'), ('rate', 'rate_prod_40')],
}


def dis_sources(frames: str) -> dict:
    """Panel sources of a disordered Dicke case (arrays (n_widths, nx, ny))."""
    return {
        'pauli':  [('dis', 'F_pauli')],
        'stab':   [('disx', 'F_stab3')],
        'heis4':  [('dis', f'F_{frames}_cert_4')],
        'heis8':  [('dis', f'F_{frames}_cert_8')],
        'heis12': [('dis', f'F_{frames}_cert_12')],
        'schro':  [('disx', 'F_schro_best')],
        'prod10': [('disx', 'F_prod_10')],
        'prod40': [('disx', 'F_prod_40')],
    }


def complex_sources(sources: dict) -> dict:
    """SOURCES with the Heisenberg panels (and Pauli / stabilizer first) taken
    from the complex-frame panel file."""
    out = {k: list(v) for k, v in sources.items()}
    for m in (4, 8, 12):
        out[f'heis{m}'] = [('complex', f'panel_rate_heis_{m}')]
    out['pauli'] = [('complex', 'rate_pauli')] + out['pauli']
    out['stab'] = [('complex', 'rate_stab3')] + out['stab']
    return out


def width_slice(z: dict, k: str, shape: tuple, width, full=None):
    """z[k] if it has the panel shape; for a (n_widths,) + shape array the
    slice of `width` (None when that width is not in the file); with
    full = (full_shape, stride), a full-grid array is subsampled to the
    coarse panel grid."""
    a = np.asarray(z[k], float)
    if a.shape == shape:
        return a
    if full is not None and a.shape == full[0]:
        return a[::full[1], ::full[1]]
    if width is not None and 'widths' in z and a.shape == (len(z['widths']),) + shape:
        i = np.flatnonzero(np.abs(np.asarray(z['widths'], float) - width) < 1e-9)
        return a[i[0]] if i.size else None
    return None


def load_panel(files: dict, cands: list, shape: tuple, cache: dict):
    """First candidate array of the right shape with any finite entry, and its
    'file:key' label; (None, '') when nothing exists."""
    width = files.get('width')
    full = files.get('full')
    for role, key in cands:
        path = files.get(role)
        if not path or not Path(path).exists():
            continue
        if path not in cache:
            cache[path] = dict(np.load(path, allow_pickle=True))
        z = cache[path]
        keys = key if isinstance(key, tuple) else (key,)
        arrs = [a for a in (width_slice(z, k, shape, width, full) for k in keys if k in z)
                if a is not None]
        if not arrs:
            continue
        a = arrs[0] if len(arrs) == 1 else np.fmin.reduce(arrs)
        if np.isfinite(a).any():
            return a, f'{path}:{"|".join(keys)}'
    return None, ''


def draw(ax, fig, x, y, Z, title, zero_tol, xlabel, ylabel) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if Z is None:
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center',
                va='center', color='0.5')
        return
    Z = np.where(np.isfinite(Z) & (Z <= zero_tol), 0.0, Z)
    pc = ax.pcolormesh(x, y, Z.T, shading='nearest', cmap='viridis',
                       vmin=0.0, rasterized=True)
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    zero = np.isfinite(Z) & (Z <= zero_tol)
    if zero.any() and not zero.all():
        ax.contour(x, y, zero.T.astype(float), levels=[0.5], colors='white',
                   linewidths=1.4)


def plot_case(case: str, out_dir: Path, zero_tol: float, suptitle: bool,
              disorder_frames: str = 'species', complex_root=None) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model, files = case_files(case)
    sources = dis_sources(disorder_frames) if 'dis' in files else SOURCES
    spec = MODELS[model]
    x, y = np.asarray(spec.p1_vals, float), np.asarray(spec.p2_vals, float)
    panels, stem = PANELS, 'rate_panels'
    if complex_root is not None and 'dis' not in files:
        cf = Path(complex_root) / f'{model}_complex_rate_panels.npz'
        stride = int(np.load(cf, allow_pickle=True)['stride']) if cf.exists() else 2
        files = dict(files, complex=str(cf), full=((len(x), len(y)), stride))
        sources = complex_sources(sources)
        x, y = x[::stride], y[::stride]
        panels = [(k, t.replace('Heisenberg', 'Heisenberg (complex)')) for k, t in PANELS]
        stem = 'complex_rate_panels'
    shape = (len(x), len(y))
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), constrained_layout=True)
    cache: dict = {}
    print(f'[{case}]', flush=True)
    for ax, (key, title) in zip(axes.flat, panels):
        Z, src = load_panel(files, sources[key], shape, cache)
        n0 = int((np.isfinite(Z) & (Z <= zero_tol)).sum()) if Z is not None else 0
        print(f'  {key:>7s}: {src or "MISSING"}' + (f'  ({n0} zero cells)' if Z is not None else ''),
              flush=True)
        draw(ax, fig, x, y, Z, title, zero_tol, spec.p1_label, spec.p2_label)
    if suptitle:
        extra = (f', omega0 width {files["width"]:g}, {disorder_frames} frames'
                 if 'dis' in files else '')
        fig.suptitle(f'{case}: {spec.title}{extra}', fontsize=12)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        f = out_dir / f'{case}_{stem}.{ext}'
        fig.savefig(f, dpi=200)
        print(f'  wrote {f}', flush=True)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='+', default=['model4', 'model10'])
    ap.add_argument('--out_dir', default='results_plots/paper')
    ap.add_argument('--zero_tol', type=float, default=1e-5,
                    help='rates <= this count as mu* = 0 (set to 0, inside the contour)')
    ap.add_argument('--suptitle', action='store_true',
                    help='add the model title on top (off: caption goes in the paper)')
    ap.add_argument('--disorder_frames', choices=('species', 'shared'), default='species',
                    help='Heisenberg panels of disordered Dicke cases: one frame per '
                         'species (best found) or one shared frame')
    ap.add_argument('--complex_root', default=None,
                    help='root of the complex-frame panels (rate_complex_collect.py); '
                         'draws the Heisenberg panels with complex frames on its '
                         'coarse grid')
    args = ap.parse_args()
    for case in args.cases:
        plot_case(case, Path(args.out_dir), args.zero_tol, args.suptitle,
                  args.disorder_frames, args.complex_root)


if __name__ == '__main__':
    main()
