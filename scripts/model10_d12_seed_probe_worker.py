"""
Seed probe at d_ext = 12: do structured frames OUTSIDE the current seed set
beat the best known d_ext = 12 rate at the boundary of the rate-0 region?

Motivation (rate_optimization_methods_note.tex, Secs. 3-4): at d_ext = 12 the
pipeline fills the 12 columns with the padded d_ext = 8 optimum, its greedy
augmentation, the closed forms C10 / P9 and three families (YZ 10-gon,
YZ 8-gon + XY pair, YZ hexagon + XY square); the seed library tops out at 7
columns and differential evolution is off.  Every candidate is a single
affine-regular polygon in a coordinate plane plus axis columns, and the
d_ext 8 -> 12 gain of the figure is ~1 pp.  This worker evaluates, at the
positive points next to the rate-0 region, three kinds of frames none of the
stages can reach:

  1. warped YZ polygon  yzw10     {I, xX, YZ 20-gon with a non-uniform vertex
     density t -> t + beta sin 2t}: dense near the Y axis (ZZ-leak cost
     2Jc/x - 2 D2 c^2 peaks there) and sparse near Z, where only the field
     rotation costs.  The affine ellipse ties density to radius the wrong
     way round for B-like radii (Z major => dense near Z); beta decouples them.
  2. latitude rings about X (prism / antiprism)
       lat10         {I, xX, ring  x0 X + r(cos t Y + sin t Z), t = full circle}
       yz5lat5, yz6lat4, yz4lat6   equatorial YZ polygon + one latitude ring
       lat5lat5      two latitude rings at x0, x1
     The field still rotates every ring into itself, X dephasing is still a
     credit (x0 <= x), and the product columns carry their own X (x) Z
     components, so the ZZ leak need not go through the thin xX (x) Z route.
     No current seed or family has a column off the coordinate planes.
  3. other plane splits and P + C combined
       yz4xy6, yz5xy5, yz7xy3, yz8xy2    (only yz6xy4 is in the pipeline)
       zpyz4xy4, zpyz5xy3, zpyz6xy2      {I, aI +- bZ, xX, YZ n-gon, XY k-gon}
  control: yz6xy4 re-run with the probe's own budget -- improvements that
     also show up here are the budget, not the new seeds.

Per target point: every family is optimised by multistart Nelder-Mead
(framability_rate_families.optimize_family) from starts built from the point's
stored d_ext = 8 / 12 family optima and the analytic B / C weights; the best
--n_polish_cands NEW candidates get the pipeline's polish (bundle polish, then
margin polish, which at positive rate is a rate polish over all near-binding
columns) and the per-column reference LP.  Nothing is compared against a
polished incumbent unless --control is set (same polish applied to the best
known frame; costs about as much again).

Reads   <out_dir>/model10_seeded[_s]/, <out_dir>/model10_seeded_d12[_s]/
        (best_known over worker / xeval / qrefine / margin / rrefine files)
Writes  <out_dir>/model10_seeded_d12_probe[_s]/targets.npz        (--prepare)
        <out_dir>/model10_seeded_d12_probe[_s]/pt_<ix>_<iy>.npz   (per target)
          rate_12_prev = best known d_ext <= 12 value, rate_12 / S_12 / label_12
          = best NEW certified value (written whether or not it improves),
          fam_names / fam_vals / fam_kinds, famp_<name>, polished_* , improved
Nothing is written into the pipeline directories; use
scripts/model10_d12_seed_probe_collect.py (--adopt) to promote improvements.

Usage (the whole chain: bash scripts/submit_model10_d12_seed_probe.sh):
    python scripts/model10_d12_seed_probe_worker.py --prepare --n_proc 8      # targets.npz
    python scripts/model10_d12_seed_probe_worker.py --task_id 0 --n_chunks 200
    python scripts/model10_d12_seed_probe_worker.py --task_id 0 --n_chunks 200 \
        --out_dir results_model10_rate_dim2 --dim 2
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trotter_lindbladian_scan import MODELS, MODEL10_J, MODEL10_H          # noqa: E402
from framability_rate import RATE_VERSION                                  # noqa: E402
from framability_rate_global import (frame_rate_value, bundle_rate_polish,  # noqa: E402
                                     product_weight_optimum, WEIGHT_FLOOR,
                                     RATE_GLOBAL_VERSION)
from framability_rate_margin import margin_polish, RATE_MARGIN_VERSION     # noqa: E402
from framability_rate_families import (Family, affine_polygon, optimize_family,  # noqa: E402
                                       yz_xy_poly, model10_weights,
                                       _frame, _plane_cols, _w,
                                       RATE_FAMILIES_VERSION)
from model10_seeded_worker import MODEL, grid_vals, pt_dir_name, generator  # noqa: E402
from model10_seeded_qrefine_worker import best_known, pt_paths             # noqa: E402
from model10_d12_worker import d12_dir, d8_dir                             # noqa: E402

PROBE_VERSION = '1.0-d12-seed-probe'
M = 12
TOL = 1e-9
KEY, S_KEY, LAB_KEY = f'rate_{M}', f'S_{M}', f'label_{M}'
K4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)


def probe_dir(out_dir: str, stride: int, model: str = MODEL) -> Path:
    return Path(out_dir) / (pt_dir_name(stride, model)
                            .replace('_seeded', '_seeded_d12_probe', 1))


# ---------------------------------------------------------------------------
#  New building blocks
# ---------------------------------------------------------------------------
def warped_polygon(n: int, a1: float, a2: float, phi: float, theta0: float,
                   beta: float) -> np.ndarray:
    """affine_polygon with a non-uniform vertex density: the regular phases
    t_k are warped to t_k + beta sin(2 t_k) before the ellipse map, so
    beta < 0 crowds the vertices towards the a1 axis (Y in the YZ plane) and
    beta > 0 towards the a2 axis.  |beta| < 1/2 keeps the map monotone."""
    t = theta0 + np.pi * np.arange(n) / n
    t = t + float(np.clip(beta, -0.45, 0.45)) * np.sin(2.0 * t)
    u = np.vstack([a1 * np.cos(t), a2 * np.sin(t)])
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]]) @ u


def lat_ring_cols(n: int, x0: float, r1: float, r2: float, psi: float,
                  theta1: float) -> list:
    """n columns  x0 X + R(psi) (r1 cos t, r2 sin t) in the YZ plane, t over
    the FULL circle (the antipodes sit at -x0, so no +- pair is duplicated).
    Columns outside the ball are projected by _frame."""
    t = theta1 + 2.0 * np.pi * np.arange(n) / n
    u = np.vstack([r1 * np.cos(t), r2 * np.sin(t)])
    c, s = np.cos(psi), np.sin(psi)
    u = np.array([[c, -s], [s, c]]) @ u
    x0 = float(np.clip(x0, -1.0 + WEIGHT_FLOOR, 1.0 - WEIGHT_FLOOR))
    return [[0.0, x0, u[0, k], u[1, k]] for k in range(n)]


# ---------------------------------------------------------------------------
#  New families (all with m = 12 at the sizes used below)
# ---------------------------------------------------------------------------
def yzw_poly(n: int) -> Family:
    """{I, xX, warped YZ n-gon}; p = (x, a1, a2, phi, theta0, beta)."""
    def build(p):
        x, a1, a2, phi, th, beta = p
        V = warped_polygon(n, _w(a1), _w(a2), phi, th, beta)
        return _frame([[0.0, _w(x), 0.0, 0.0]] + _plane_cols(V, 'yz'))
    h = np.pi / (2 * n)
    return Family(f'yzw{n}', 'yzw', n, n + 2, build,
                  (WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h, -0.4),
                  (1.0, 1.0, 1.0, np.pi / 2, h, 0.4))


def lat_poly(n: int) -> Family:
    """{I, xX, latitude ring of n}; p = (x, x0, r1, r2, psi, theta1)."""
    def build(p):
        x, x0, r1, r2, psi, th = p
        return _frame([[0.0, _w(x), 0.0, 0.0]]
                      + lat_ring_cols(n, x0, _w(r1), _w(r2), psi, th))
    h = np.pi / n
    return Family(f'lat{n}', 'lat', n, n + 2, build,
                  (WEIGHT_FLOOR, 0.0, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h),
                  (1.0, 0.8, 1.0, 1.0, np.pi / 2, h))


def yz_lat_poly(n_yz: int, n_lat: int) -> Family:
    """{I, xX, affine YZ n_yz-gon, latitude ring of n_lat};
    p = (x, a1, a2, phi, theta0, x0, r1, r2, psi, theta1)."""
    def build(p):
        x, a1, a2, phi, th, x0, r1, r2, psi, th1 = p
        V = affine_polygon(n_yz, _w(a1), _w(a2), phi, th)
        return _frame([[0.0, _w(x), 0.0, 0.0]] + _plane_cols(V, 'yz')
                      + lat_ring_cols(n_lat, x0, _w(r1), _w(r2), psi, th1))
    h0, h1 = np.pi / (2 * n_yz), np.pi / n_lat
    return Family(f'yz{n_yz}lat{n_lat}', 'yz+lat', n_yz, n_yz + n_lat + 2, build,
                  (WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h0,
                   0.0, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h1),
                  (1.0, 1.0, 1.0, np.pi / 2, h0,
                   0.8, 1.0, 1.0, np.pi / 2, h1))


def lat_lat_poly(n0: int, n1: int) -> Family:
    """{I, xX, latitude ring n0 at x0, latitude ring n1 at x1};
    p = (x, x0, r1, r2, psi, theta1, x1, s1, s2, chi, theta2)."""
    def build(p):
        x, x0, r1, r2, psi, th1, x1, s1, s2, chi, th2 = p
        return _frame([[0.0, _w(x), 0.0, 0.0]]
                      + lat_ring_cols(n0, x0, _w(r1), _w(r2), psi, th1)
                      + lat_ring_cols(n1, x1, _w(s1), _w(s2), chi, th2))
    h0, h1 = np.pi / n0, np.pi / n1
    return Family(f'lat{n0}lat{n1}', 'lat+lat', n0, n0 + n1 + 2, build,
                  (WEIGHT_FLOOR, 0.0, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h0,
                   0.0, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h1),
                  (1.0, 0.8, 1.0, 1.0, np.pi / 2, h0,
                   0.9, 1.0, 1.0, np.pi / 2, h1))


def zpair_yz_xy_poly(n_yz: int, n_xy: int) -> Family:
    """P + C at once: {I, aI +- bZ, xX, affine YZ n_yz-gon, affine XY n_xy-gon};
    p = (a, b, x, a1, a2, phi, theta0, b1, b2, psi, theta1)."""
    def build(p):
        a, b, x, a1, a2, phi, th, b1, b2, psi, th1 = p
        a = float(np.clip(a, 0.0, 1.0 - WEIGHT_FLOOR))
        V = affine_polygon(n_yz, _w(a1), _w(a2), phi, th)
        W = affine_polygon(n_xy, _w(b1), _w(b2), psi, th1)
        return _frame([[a, 0.0, 0.0, _w(b)], [a, 0.0, 0.0, -_w(b)],
                       [0.0, _w(x), 0.0, 0.0]]
                      + _plane_cols(V, 'yz') + _plane_cols(W, 'xy'))
    h0, h1 = np.pi / (2 * n_yz), np.pi / (2 * n_xy)
    return Family(f'zpyz{n_yz}xy{n_xy}', 'zp+yz+xy', n_yz, n_yz + n_xy + 4, build,
                  (0.0, WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR,
                   -np.pi / 2, -h0, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h1),
                  (0.5, 1.0, 1.0, 1.0, 1.0, np.pi / 2, h0, 1.0, 1.0, np.pi / 2, h1))


GROUPS = {
    'warp':    [yzw_poly(10)],
    'lat':     [lat_poly(10), yz_lat_poly(5, 5), yz_lat_poly(6, 4),
                yz_lat_poly(4, 6), lat_lat_poly(5, 5)],
    'split':   [yz_xy_poly(4, 6), yz_xy_poly(5, 5), yz_xy_poly(7, 3),
                yz_xy_poly(8, 2)],
    'pc':      [zpair_yz_xy_poly(4, 4), zpair_yz_xy_poly(5, 3),
                zpair_yz_xy_poly(6, 2)],
    'control': [yz_xy_poly(6, 4)],
}
CONTROL_NAMES = {f.name for f in GROUPS['control']}


def probe_families(groups) -> list:
    fams = [f for g in groups for f in GROUPS[g]]
    assert all(f.m == M for f in fams), [(f.name, f.m) for f in fams]
    return fams


# ---------------------------------------------------------------------------
#  Starts
# ---------------------------------------------------------------------------
def _famp(pt: Path, ix: int, iy: int, m: int) -> dict:
    """{family name: params} stored for the point at d_ext = m (seeded
    worker file for m = 8, d12 worker file for m = 12)."""
    f = pt / f'pt_{ix:03d}_{iy:03d}.npz'
    if not f.exists():
        return {}
    d = np.load(f, allow_pickle=True)
    sfx = f'_{m}'
    return {k[len('famp_'):-len(sfx)]: np.asarray(d[k], float)
            for k in d.files if k.startswith('famp_') and k.endswith(sfx)}


def _dedupe(rows, tol: float = 1e-6) -> list:
    out = []
    for r in rows:
        r = np.asarray(r, float)
        if not any(len(q) == len(r) and np.allclose(q, r, atol=tol) for q in out):
            out.append(r)
    return out


def yz_part_starts(model: str, d1: float, d2: float, A, famp8: dict,
                   famp12: dict) -> list:
    """5-vectors (x, a1, a2, phi, theta0) of a YZ polygon: the point's stored
    optima first (d12 yz10, d8 yz6, the YZ parts of yzxy / yz+xy), then the
    analytic B / C starts (model10) or the rescaled-Pauli weights."""
    rows = []
    for famp in (famp12, famp8):
        for name, p in famp.items():
            if name.startswith('yz') and len(p) >= 5:
                rows.append(p[:5])
    if model == 'model10':
        x_b, y_b, x_c = model10_weights(d1, d2, MODEL10_J, MODEL10_H)
        rows += [(x_b, y_b, 1.0, 0.0, 0.0), (x_c, 1.0, 1.0, 0.0, 0.0),
                 (x_c, 1.0, 1.0, 0.0, np.pi / 20)]
    else:
        sx, sy, sz = (_w(v) for v in np.diag(product_weight_optimum(A)[1])[1:])
        rows += [(sx, sy, sz, 0.0, 0.0), (sx, 1.0, 1.0, 0.0, 0.0),
                 (sx, 1.0, 1.0, 0.0, np.pi / 20)]
    return _dedupe(rows)


def xy_part_starts(famp12: dict, yz5) -> list:
    """4-vectors (b1, b2, psi, theta1) of an XY polygon: the stored yz6xy4 XY
    part and the transfer rule of the pipeline (X / Y extents of the YZ part)."""
    rows = [p[5:9] for name, p in famp12.items()
            if name.startswith('yz') and 'xy' in name and len(p) == 9]
    x, a1 = float(yz5[0]), float(yz5[1])
    rows += [(x, a1, 0.0, 0.0), (x, 1.0, 0.0, np.pi / 8)]
    return _dedupe(rows)


def starts_for(fam: Family, yz_starts: list, famp12: dict, max_starts: int) -> list:
    """Analytic / transferred starts of a probe family; stored-optimum-derived
    rows come first so a cap keeps them."""
    out = []
    for yz5 in yz_starts:
        x, a1, a2, phi, th = (float(v) for v in yz5)
        if fam.kind == 'yzw':
            out += [(x, a1, a2, phi, th, b) for b in (0.0, -0.2, 0.2)]
        elif fam.kind == 'lat':
            for x0 in (0.2, 0.4):
                sh = np.sqrt(1.0 - x0 * x0)
                out.append((x, x0, a1 * sh, a2 * sh, phi, th))
        elif fam.kind == 'yz+lat':
            k = fam.m - fam.n - 2
            for x0 in (0.25, 0.5):
                sh = np.sqrt(1.0 - x0 * x0)
                for off in (0.0, np.pi / k):            # prism / antiprism
                    out.append((x, a1, a2, phi, th, x0, a1 * sh, a2 * sh, phi, th + off))
        elif fam.kind == 'lat+lat':
            k = fam.m - fam.n - 2
            for x0, x1 in ((0.2, 0.55), (0.3, 0.7)):
                s0, s1 = np.sqrt(1.0 - x0 * x0), np.sqrt(1.0 - x1 * x1)
                out.append((x, x0, a1 * s0, a2 * s0, phi, th,
                            x1, a1 * s1, a2 * s1, phi, th + np.pi / k))
        elif fam.kind == 'yz+xy':
            for xy4 in xy_part_starts(famp12, yz5):
                out.append((x, a1, a2, phi, th, *xy4))
        elif fam.kind == 'zp+yz+xy':
            for a, b in ((0.5, 0.5), (0.0, 1.0)):
                for xy4 in xy_part_starts(famp12, yz5)[:2]:
                    out.append((a, b, x, a1, a2, phi, th, *xy4))
    out = _dedupe(out)
    return [np.asarray(p, float) for p in out[:max_starts]]


# ---------------------------------------------------------------------------
#  Targets (--prepare)
# ---------------------------------------------------------------------------
def _point_rates(task):
    """(ix, iy, v4, v8, v12): the best known values of one point, with every
    file of the point (model10_seeded_qrefine_worker.pt_paths) opened ONCE and
    only the rate keys read -- the same minimum as three best_known calls, at
    a third of the file opens and without touching the frames."""
    pt8, pt12, ix, iy = task
    best = {4: np.inf, 8: np.inf, 12: np.inf}
    for pt, ms in ((pt8, (4, 8)), (pt12, (12,))):
        for f in pt_paths(pt, ix, iy):
            try:
                d = np.load(f, allow_pickle=True)
            except Exception:                               # noqa: BLE001
                continue
            for m in ms:
                k = f'rate_{m}'
                if k in d.files and f'S_{m}' in d.files:
                    v = float(d[k])
                    if np.isfinite(v) and v < best[m]:
                        best[m] = v
    return ix, iy, best[4], best[8], best[12]


def panel_grids(out_dir: str, stride: int, model: str, nx: int, ny: int,
                n_proc: int = 1):
    """best known rate_4 / rate_8 / rate_12 per point (nan where missing),
    read with n_proc processes (the file opens dominate on a network FS)."""
    pt8, pt12 = d8_dir(out_dir, stride, model), d12_dir(out_dir, stride, model)
    tasks = [(pt8, pt12, ix, iy) for ix in range(nx) for iy in range(ny)]
    g = {m: np.full((nx, ny), np.nan) for m in (4, 8, 12)}
    if n_proc > 1:
        from multiprocessing import Pool
        with Pool(n_proc) as pool:
            rows = list(pool.imap_unordered(_point_rates, tasks, chunksize=16))
    else:
        rows = [_point_rates(t) for t in tasks]
    for ix, iy, v4, v8, v12 in rows:
        for m, v in ((4, v4), (8, v8), (12, v12)):
            if np.isfinite(v):
                g[m][ix, iy] = v
    return g


def prepare(args) -> None:
    p1, p2 = grid_vals(args.stride, args.model)
    nx, ny = len(p1), len(p2)
    t0 = time.perf_counter()
    g = panel_grids(args.out_dir, args.stride, args.model, nx, ny, args.n_proc)
    panel = np.fmin(np.fmin(g[12], g[8]), g[4])          # the figure's d12 panel
    z = panel <= args.tol
    pos = np.isfinite(panel) & ~z
    boundary = pos & binary_dilation(z, structure=K4)
    near = pos & binary_dilation(z, structure=np.ones((2 * args.radius + 1,) * 2, bool))
    sel = {'boundary': boundary, 'near': near, 'all': pos}[args.targets]
    ixs, iys = np.nonzero(sel)
    order = np.argsort(panel[ixs, iys])                   # smallest rates first
    ixs, iys = ixs[order], iys[order]
    pd = probe_dir(args.out_dir, args.stride, args.model)
    pd.mkdir(parents=True, exist_ok=True)
    f = pd / 'targets.npz'
    np.savez(f, ix=ixs, iy=iys, v_ref=panel[ixs, iys], targets=args.targets,
             radius=args.radius, tol=args.tol, panel=panel, rate_4=g[4],
             rate_8=g[8], rate_12=g[12], p1=p1, p2=p2, model=args.model)
    print(f'[probe prepare] {args.model} {args.out_dir}: {int(np.isfinite(panel).sum())}'
          f'/{nx * ny} points, {int(z.sum())} at rate 0, {int(boundary.sum())} boundary, '
          f'{int(near.sum())} within radius {args.radius}, {int(pos.sum())} positive; '
          f'targets={args.targets}: {len(ixs)} -> {f}  ({time.perf_counter() - t0:.0f}s)',
          flush=True)


# ---------------------------------------------------------------------------
#  One point
# ---------------------------------------------------------------------------
def _certify(S, A) -> float:
    v = frame_rate_value(S, A, reference=True)
    return max(float(v), 0.0) if np.isfinite(v) and v > -TOL else float(v)


def polish(S, A, args):
    """The pipeline's polish: bundle (rate) then margin (all near-binding
    columns), certified; returns (value, frame, n_lp)."""
    n_lp = 0
    v = frame_rate_value(S, A)
    if args.bundle_iters > 0 and v > TOL:
        vb, Sb = bundle_rate_polish(S, A, n_iter=args.bundle_iters)
        if np.isfinite(vb) and vb < v:
            v, S = float(vb), Sb
    if args.margin_iters > 0 and v > TOL:
        F, Sm, n = margin_polish(S, A, n_iter=args.margin_iters)
        n_lp += n
        if np.isfinite(F) and F < v:                      # F = rate at positive rate
            v, S = float(F), Sm
    return _certify(S, A), S, n_lp


def compute_point(ix: int, iy: int, args) -> dict:
    spec = MODELS[args.model]
    p1, p2 = grid_vals(args.stride, args.model)
    d1, d2 = float(p1[ix]), float(p2[iy])
    A = generator(d1, d2, args.dim, args.model)
    pt8 = d8_dir(args.out_dir, args.stride, args.model)
    pt12 = d12_dir(args.out_dir, args.stride, args.model)
    t0 = time.perf_counter()

    v12, S12, lab12 = best_known(pt12, ix, iy, KEY, S_KEY)
    v8, _, _ = best_known(pt8, ix, iy, 'rate_8', 'S_8')
    v4, _, _ = best_known(pt8, ix, iy, 'rate_4', 'S_4')
    v_ref = float(np.nanmin([v12, v8, v4]))             # the figure's d12 value
    out = {spec.p1_name: d1, spec.p2_name: d2, 'dim': args.dim,
           'rate_12_known': v12, 'rate_8_known': v8, 'rate_4_known': v4,
           'label_12_known': lab12, f'{KEY}_prev': v_ref}

    famp8, famp12 = _famp(pt8, ix, iy, 8), _famp(pt12, ix, iy, 12)
    yz_starts = yz_part_starts(args.model, d1, d2, A, famp8, famp12)
    rng = np.random.default_rng([args.seed, ix, iy, M, 7])
    fams = probe_families(args.groups)
    names, vals, kinds, times, frames = [], [], [], [], []
    for fam in fams:
        t1 = time.perf_counter()
        starts = starts_for(fam, yz_starts, famp12, args.max_starts)
        v, S, p, n = optimize_family(fam, A, starts, n_random=args.fam_random,
                                     n_polish=args.fam_polish,
                                     maxfev=args.fam_maxfev, rng=rng)
        names.append(fam.name)
        vals.append(float(v))
        kinds.append(fam.kind)
        times.append(time.perf_counter() - t1)
        frames.append(S)
        out[f'famp_{fam.name}'] = np.asarray(p, float)
        print(f'    ({ix},{iy}) {fam.name:>10s}: {v:.6e}  [{len(starts)} starts, '
              f'{n} evals, {times[-1]:.0f}s]  ref {v_ref:.6e}', flush=True)
    out.update(fam_names=np.array(names), fam_vals=np.array(vals),
               fam_kinds=np.array(kinds), fam_t=np.array(times))

    # polish the best NEW candidates (the control family is not eligible)
    order = [i for i in np.argsort(vals) if names[i] not in CONTROL_NAMES
             and np.isfinite(vals[i])]
    pol_names, pol_vals = [], []
    best_v, best_S, best_lab = np.inf, None, ''
    for i in order[:args.n_polish_cands]:
        v, S, _ = polish(frames[i], A, args)
        pol_names.append(names[i])
        pol_vals.append(v)
        print(f'    ({ix},{iy}) polish {names[i]}: {vals[i]:.6e} -> {v:.6e}', flush=True)
        if np.isfinite(v) and v < best_v:
            best_v, best_S, best_lab = v, S, f'probe {names[i]} +polish'
    # the control family polished the same way (the budget's own effect)
    ctrl_pol = np.nan
    ic = [i for i, nm in enumerate(names) if nm in CONTROL_NAMES]
    if ic and np.isfinite(vals[ic[0]]):
        ctrl_pol, _, _ = polish(frames[ic[0]], A, args)
    ctrl_inc = np.nan
    if args.control and S12 is not None:
        ctrl_inc, _, _ = polish(S12, A, args)
        print(f'    ({ix},{iy}) control (incumbent polished): {v12:.6e} -> '
              f'{ctrl_inc:.6e}', flush=True)
    if best_S is None:
        raise RuntimeError('no finite probe frame')
    out.update({KEY: best_v, S_KEY: best_S, LAB_KEY: best_lab,
                'polished_names': np.array(pol_names),
                'polished_vals': np.array(pol_vals),
                'control_fam_polished': ctrl_pol,
                'control_incumbent_polished': ctrl_inc,
                'improved': bool(best_v < v_ref - args.gain_tol),
                'new_zero': bool(best_v <= args.tol < v_ref),
                't': time.perf_counter() - t0})
    return out


def run_point(ix: int, iy: int, args) -> None:
    pd = probe_dir(args.out_dir, args.stride, args.model)
    out_f = pd / f'pt_{ix:03d}_{iy:03d}.npz'
    if out_f.exists() and not args.force:
        print(f'[skip] {out_f.name}', flush=True)
        return
    try:
        res = compute_point(ix, iy, args)
    except Exception as e:                                  # noqa: BLE001
        print(f'  ERROR at ({ix},{iy}): {type(e).__name__}: {e}', flush=True)
        return
    pd.mkdir(parents=True, exist_ok=True)
    tmp = out_f.with_suffix('.tmp.npz')
    np.savez(tmp, model=args.model, ix=ix, iy=iy, stride=args.stride, d_ext=M,
             groups=np.array(args.groups), probe_version=PROBE_VERSION,
             rate_version=RATE_VERSION, rate_global_version=RATE_GLOBAL_VERSION,
             rate_families_version=RATE_FAMILIES_VERSION,
             rate_margin_version=RATE_MARGIN_VERSION, **res)
    os.replace(tmp, out_f)
    tag = 'IMPROVED' if res['improved'] else 'no gain'
    print(f'  saved {out_f.name}  ref {res[f"{KEY}_prev"]:.6e} -> probe {res[KEY]:.6e} '
          f'[{res[LAB_KEY]}] {tag}{"  NEW RATE-0" if res["new_zero"] else ""}  '
          f'({res["t"]:.0f}s)', flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--prepare', action='store_true',
                   help='write targets.npz (the target list) and exit')
    p.add_argument('--n_proc', type=int,
                   default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')),
                   help='--prepare: processes reading the point files')
    p.add_argument('--task_id', type=int, default=0)
    p.add_argument('--n_chunks', type=int, default=1,
                   help='<=1: task_id indexes the target list; otherwise the '
                        'list is strided over n_chunks array tasks')
    p.add_argument('--model', type=str, default=MODEL, choices=sorted(MODELS))
    p.add_argument('--out_dir', type=str, default='results_model10_rate')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--dim', type=int, default=None)
    p.add_argument('--targets', choices=('boundary', 'near', 'all'), default='boundary',
                   help='positive points with a rate-0 4-neighbour | within '
                        '--radius of the rate-0 region | every positive point')
    p.add_argument('--radius', type=int, default=2)
    p.add_argument('--tol', type=float, default=1e-6, help='rate counted as 0')
    p.add_argument('--gain_tol', type=float, default=1e-9)
    p.add_argument('--groups', nargs='+', default=list(GROUPS),
                   choices=sorted(GROUPS), help='family groups to probe')
    p.add_argument('--max_starts', type=int, default=12,
                   help='analytic / transferred starts per family')
    p.add_argument('--fam_random', type=int, default=4)
    p.add_argument('--fam_polish', type=int, default=2)
    p.add_argument('--fam_maxfev', type=int, default=80)
    p.add_argument('--n_polish_cands', type=int, default=2,
                   help='best new family optima that get the full polish')
    p.add_argument('--bundle_iters', type=int, default=25)
    p.add_argument('--margin_iters', type=int, default=40)
    p.add_argument('--control', action='store_true',
                   help='also polish the best known frame with the same budget')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    if args.dim is None:
        args.dim = MODELS[args.model].dim
    if args.prepare:
        prepare(args)
        return

    f = probe_dir(args.out_dir, args.stride, args.model) / 'targets.npz'
    if not f.exists():
        sys.exit(f'{f} missing: run with --prepare first (login node)')
    t = np.load(f, allow_pickle=True)
    ixs, iys = np.asarray(t['ix'], int), np.asarray(t['iy'], int)
    n_total = len(ixs)
    if args.n_chunks <= 1:
        if not (0 <= args.task_id < n_total):
            sys.exit(f'task_id must be in [0, {n_total})')
        ids = [args.task_id]
    else:
        if not (0 <= args.task_id < args.n_chunks):
            sys.exit(f'chunk id must be in [0, {args.n_chunks})')
        ids = list(range(args.task_id, n_total, args.n_chunks))
        print(f'[chunk {args.task_id}/{args.n_chunks}] d12 seed probe '
              f'({str(t["targets"])}): {len(ids)} of {n_total} targets, '
              f'groups {args.groups}', flush=True)
    for i in ids:
        run_point(int(ixs[i]), int(iys[i]), args)


if __name__ == '__main__':
    main()
