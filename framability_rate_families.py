r"""
framability_rate_families.py -- structured, low-dimensional observable frames
for the dt-free Heisenberg framability rate mu*(S (x) S), the closed-form
frames of the model10 rate-zero analysis, and a small structured optimiser
whose results seed framability_rate_global.minimize_rate_global.

Why
---
minimize_rate / minimize_rate_global search the full 4(m-1)-dimensional frame
space.  The hand analysis of the Shibata-Katsura chain (model10,
H = -J ZZ - h X, jumps sqrt(D1) X, sqrt(D2) ZZ, bond generator with dim = 1)
singles out frames described by a handful of numbers that are exact at their
own thresholds:

  B  rescaled Pauli {I, xX, yY, Z} with y = h/D1, x = D2 y/J:
     rate 0 iff D1 >= h and D2 (2 D2 + D1 - h^2/D1) >= 2 J^2.
  C  {I, xX, unit circle in the YZ plane}: the field hX rotates the circle
     into itself (tangent moves are free), X noise shrinks it uniformly.
     Sufficient for rate 0, for every h: x = min(1, D2/J) and
     D1 >= max(J/x, max_c (2Jc/x - 2 D2 c^2)).
     A finite version is an n-gon in the YZ plane (d_ext = n + 2); rotating a
     regular 2n-gon at angular speed w costs w tan(pi/2n).
  P  a Z pair + XY polygon (the conditional-rotation frame of
     scripts/polygon_frame_gate.py): ZZ costs 2J tan(pi/2n) per bond.

Every family below is an AFFINE-REGULAR polygon in one Pauli plane (the image
of a regular 2n-gon under a linear map, so the vertices lie on an ellipse
with semi-axes a1, a2 rotated by phi, at phase theta0) plus a few axis
columns.  An affine-regular square is a general parallelogram, so at n = 2 the
YZ family contains every 4-frame made of a pure X column and two YZ columns:
B (phi = theta0 = 0) as well as the skewed bases that absorb the field
when the YZ block is overdamped (D2 >~ h).

Frame conventions are those of optimize_framability / framability_rate:
S is 4 x m with rows (I, X, Y, Z), column 0 the identity, and every other
column in the ball |c_I| + ||(c_X, c_Y, c_Z)||_2 <= 1.  A = L^T acts on
observable coefficients.

Public API
----------
affine_polygon(n, a1, a2, phi, theta0)   2 x n polygon vertices
Family                                   parametrised frame family
families_for(m)                          the families with m columns
yz_xy_poly(n_yz, n_xy)                   two-plane family (YZ + XY polygons)
winning_families_for(m)                  families that won at d_ext = 8 (model10)
transfer_params(fam, famp)               starts from smaller-d_ext optima
candidate_columns(S), greedy_augment(S, A, m)   grow a frame column by column
model10_weights(d1, d2, J, h)            closed-form (x_B, y_B, x_C)
model10_family_starts(fam, d1, d2, J, h) analytic parameter starts
model10_frames(d1, d2, J, h, m)          closed-form frames B / C / P
optimize_family(fam, A, starts)          multistart Nelder-Mead in the family
b_boundary(d1, J, h) / c_boundary(d2, J) analytic rate-zero curves

    python framability_rate_families.py      # quick self-test (model10)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from optimize_framability import _FIXED_COLS, _project_columns_bloch
from framability_rate_global import (frame_rate_value, fit_columns,
                                     WEIGHT_FLOOR)

RATE_FAMILIES_VERSION = '1.0-affine-polygons'

_BIG = 1e3                    # objective value of a singular frame
_TOL = 1e-9


# ---------------------------------------------------------------------------
#  Building blocks
# ---------------------------------------------------------------------------
def _w(v) -> float:
    """A column weight, clipped to [WEIGHT_FLOOR, 1]."""
    return float(np.clip(v, WEIGHT_FLOOR, 1.0))


def affine_polygon(n: int, a1: float, a2: float, phi: float,
                   theta0: float) -> np.ndarray:
    """2 x n vertices (one per +- pair) of the affine-regular 2n-gon with
    semi-axes (a1, a2), axes rotated by phi, first vertex at phase theta0."""
    t = theta0 + np.pi * np.arange(n) / n
    u = np.vstack([a1 * np.cos(t), a2 * np.sin(t)])
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]]) @ u


def _frame(cols) -> np.ndarray:
    """Identity column + the given (I, X, Y, Z) columns, projected onto the
    operator-norm ball."""
    free = np.array(cols, float).T
    return np.hstack([_FIXED_COLS, _project_columns_bloch(free)])


def _plane_cols(V, plane: str):
    """Columns (I, X, Y, Z) of plane vertices V (2 x n) in 'xy' or 'yz'."""
    rows = {'xy': (1, 2), 'yz': (2, 3)}[plane]
    out = []
    for k in range(V.shape[1]):
        c = [0.0, 0.0, 0.0, 0.0]
        c[rows[0]], c[rows[1]] = V[0, k], V[1, k]
        out.append(c)
    return out


# ---------------------------------------------------------------------------
#  Families
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Family:
    """A parametrised frame family: build(p) -> S (4 x m).  lo / hi bound
    the random starts only (weights are clipped inside build)."""
    name: str
    kind: str
    n: int
    m: int
    build: Callable[[np.ndarray], np.ndarray]
    lo: tuple
    hi: tuple


def yz_poly(n: int) -> Family:
    """{I, xX, affine-regular YZ n-gon}; p = (x, a1, a2, phi, theta0).
    n = 2, phi = theta0 = 0 is the rescaled Pauli frame diag(1, x, a1, a2)."""
    def build(p):
        x, a1, a2, phi, th = p
        V = affine_polygon(n, _w(a1), _w(a2), phi, th)
        return _frame([[0.0, _w(x), 0.0, 0.0]] + _plane_cols(V, 'yz'))
    h = np.pi / (2 * n)
    return Family(f'yz{n}', 'yz', n, n + 2, build,
                  (WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h),
                  (1.0, 1.0, 1.0, np.pi / 2, h))


def xy_zpair(n: int) -> Family:
    """{I, (a I +- b Z), affine-regular XY n-gon}; p = (a, b, a1, a2, phi,
    theta0).  a = b = 1/2 is the projector pair of the conditional-rotation
    frame, a = 0 a single Z column."""
    def build(p):
        a, b, a1, a2, phi, th = p
        a = float(np.clip(a, 0.0, 1.0 - WEIGHT_FLOOR))
        b = _w(b)
        V = affine_polygon(n, _w(a1), _w(a2), phi, th)
        return _frame([[a, 0.0, 0.0, b], [a, 0.0, 0.0, -b]]
                      + _plane_cols(V, 'xy'))
    h = np.pi / (2 * n)
    return Family(f'xy{n}', 'xy', n, n + 3, build,
                  (0.0, WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h),
                  (0.5, 1.0, 1.0, 1.0, np.pi / 2, h))


def xpair_yz(n: int) -> Family:
    """{I, (a I +- b X), affine-regular YZ n-gon}; p = (a, b, a1, a2, phi,
    theta0).  a = b = 1/2 are the X-eigenstate projectors (I +- X)/2."""
    def build(p):
        a, b, a1, a2, phi, th = p
        a = float(np.clip(a, 0.0, 1.0 - WEIGHT_FLOOR))
        b = _w(b)
        V = affine_polygon(n, _w(a1), _w(a2), phi, th)
        return _frame([[a, b, 0.0, 0.0], [a, -b, 0.0, 0.0]]
                      + _plane_cols(V, 'yz'))
    h = np.pi / (2 * n)
    return Family(f'xp{n}', 'xp', n, n + 3, build,
                  (0.0, WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h),
                  (0.5, 1.0, 1.0, 1.0, np.pi / 2, h))


def yz_poly_xy_pair(n: int) -> Family:
    """{I, xX, affine-regular YZ n-gon, c(cos psi X +- sin psi Y)};
    p = (x, a1, a2, phi, theta0, c, psi): both planes at once."""
    def build(p):
        x, a1, a2, phi, th, c, psi = p
        c = _w(c)
        V = affine_polygon(n, _w(a1), _w(a2), phi, th)
        pair = [[0.0, c * np.cos(psi), s * c * np.sin(psi), 0.0]
                for s in (1.0, -1.0)]
        return _frame([[0.0, _w(x), 0.0, 0.0]] + _plane_cols(V, 'yz') + pair)
    h = np.pi / (2 * n)
    return Family(f'yzxy{n}', 'yzxy', n, n + 4, build,
                  (WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h,
                   WEIGHT_FLOOR, 0.0),
                  (1.0, 1.0, 1.0, np.pi / 2, h, 1.0, np.pi / 2))


def yz_xy_poly(n_yz: int, n_xy: int) -> Family:
    """Two planes at once: {I, xX, affine-regular YZ n_yz-gon, affine-regular
    XY n_xy-gon}; p = (x, a1, a2, phi, theta0, b1, b2, psi, theta1).  The YZ
    polygon makes the field rotation cheap, the XY polygon the ZZ conditional
    rotation (the perpendicular-plane conflict of the finite frames)."""
    def build(p):
        x, a1, a2, phi, th, b1, b2, psi, th1 = p
        V = affine_polygon(n_yz, _w(a1), _w(a2), phi, th)
        W = affine_polygon(n_xy, _w(b1), _w(b2), psi, th1)
        return _frame([[0.0, _w(x), 0.0, 0.0]] + _plane_cols(V, 'yz')
                      + _plane_cols(W, 'xy'))
    h1, h2 = np.pi / (2 * n_yz), np.pi / (2 * n_xy)
    return Family(f'yz{n_yz}xy{n_xy}', 'yz+xy', n_yz, n_yz + n_xy + 2, build,
                  (WEIGHT_FLOOR, WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h1,
                   WEIGHT_FLOOR, WEIGHT_FLOOR, -np.pi / 2, -h2),
                  (1.0, 1.0, 1.0, np.pi / 2, h1, 1.0, 1.0, np.pi / 2, h2))


def families_for(m: int) -> list:
    """The structured families with exactly m columns (identity included)."""
    fams = [yz_poly(m - 2)] if m >= 4 else []
    if m - 3 >= 2:
        fams += [xy_zpair(m - 3), xpair_yz(m - 3)]
    if m - 4 >= 2:
        fams.append(yz_poly_xy_pair(m - 4))
    return fams


def winning_families_for(m: int) -> list:
    """The families worth optimising at large m, chosen from the d_ext = 8
    model10 run: YZ affine polygons won most of the positive-rate region and
    'YZ polygon + XY pair' the rate-0 islands, while the X-pair and Z-pair
    families never won.  So: the finer YZ polygon, YZ polygon + XY pair, and
    the two-plane frame (YZ hexagon + XY (m-8)-gon)."""
    fams = [yz_poly(m - 2)]
    if m - 4 >= 2:
        fams.append(yz_poly_xy_pair(m - 4))
    if m - 8 >= 2:
        fams.append(yz_xy_poly(6, m - 8))
    return fams


def transfer_params(fam: Family, famp: dict) -> list:
    """Starts for `fam` from optimised parameters of the same kind at a
    smaller size (famp: {family name: params} of the d_ext = 8 run).  The
    parameters of an affine-regular polygon do not depend on its vertex
    count, so they carry over unchanged; the two-plane family takes the YZ
    part of a 'yz' optimum and an XY polygon matching its X / Y extents."""
    out = []
    for name, p in famp.items():
        p = np.asarray(p, float)
        if fam.kind in ('yz', 'yzxy') and name.startswith(fam.kind) and \
                name[len(fam.kind):].isdigit() and len(p) == len(fam.lo):
            out.append(p.copy())
        elif fam.kind == 'yz+xy' and name.startswith('yz') and \
                name[2:].isdigit() and len(p) == 5:
            x, a1 = float(p[0]), float(p[1])
            out.append(np.concatenate([p, [x, a1, 0.0, 0.0]]))
    return out


# ---------------------------------------------------------------------------
#  model10 closed forms
# ---------------------------------------------------------------------------
def model10_weights(d1: float, d2: float, J: float, h: float):
    """(x_B, y_B, x_C): the rescaled-Pauli weights of frame B
    (y = h/D1, x = D2 y/J) and the X weight of frame C (x = D2/J), clipped
    to [WEIGHT_FLOOR, 1]."""
    y_b = _w(h / d1) if d1 > 0 else 1.0
    x_b = _w(d2 * y_b / J)
    x_c = _w(d2 / J)
    return x_b, y_b, x_c


def model10_family_starts(fam: Family, d1: float, d2: float, J: float,
                          h: float) -> list:
    """Analytic parameter starts for `fam` at the model10 point (d1, d2)."""
    x_b, y_b, x_c = model10_weights(d1, d2, J, h)
    off = np.pi / (2 * fam.n)          # polygon rotated by half a step
    if fam.kind == 'yz':
        return [np.array(p, float) for p in (
            (x_b, y_b, 1.0, 0.0, 0.0),          # B (exact at n = 2)
            (x_c, 1.0, 1.0, 0.0, 0.0),          # C, vertices on Y and Z
            (x_c, 1.0, 1.0, 0.0, off),          # C, rotated polygon
            (x_b, y_b, 1.0, 0.0, off))]
    if fam.kind == 'xy':
        return [np.array(p, float) for p in (
            (0.5, 0.5, 1.0, 1.0, 0.0, 0.0),     # projectors + regular polygon
            (0.0, 1.0, x_b, y_b, 0.0, 0.0),     # Z column + B ellipse
            (0.5, 0.5, x_b, y_b, 0.0, 0.0))]
    if fam.kind == 'xp':
        return [np.array(p, float) for p in (
            (0.0, x_b, y_b, 1.0, 0.0, 0.0),     # X column + B ellipse
            (0.0, x_c, 1.0, 1.0, 0.0, 0.0),     # X column + C circle
            (0.5, 0.5, y_b, 1.0, 0.0, 0.0))]    # (I +- X)/2 + B ellipse
    if fam.kind == 'yzxy':
        return [np.array(p, float) for p in (
            (x_b, y_b, 1.0, 0.0, 0.0, 0.3, np.pi / 4),
            (x_c, 1.0, 1.0, 0.0, 0.0, 0.3, np.pi / 4))]
    if fam.kind == 'yz+xy':
        # XY polygon with the same X / Y extents as the YZ part (ellipsoid)
        return [np.array(p, float) for p in (
            (x_b, y_b, 1.0, 0.0, 0.0, x_b, y_b, 0.0, 0.0),
            (x_c, 1.0, 1.0, 0.0, 0.0, x_c, 1.0, 0.0, 0.0),
            (x_c, 1.0, 1.0, 0.0, off, x_c, 1.0, 0.0, np.pi / 8))]
    return []


# ---------------------------------------------------------------------------
#  Greedy column augmentation
# ---------------------------------------------------------------------------
def candidate_columns(S) -> list:
    """New-column candidates for a frame S (identity first): the normalised
    sums and differences of every pair of free columns (a polygon's edge
    midpoints, the bisectors between planes), rescaled to the pair's mean
    size, and the three Pauli axes at the frame's largest extent along each."""
    free = np.asarray(S, float)[:, 1:]
    size = np.abs(free[0]) + np.linalg.norm(free[1:], axis=0)
    out = []
    k = free.shape[1]
    for i in range(k):
        for j in range(i + 1, k):
            r = 0.5 * (size[i] + size[j])
            for s in (1.0, -1.0):
                v = free[:, i] + s * free[:, j]
                nv = abs(v[0]) + np.linalg.norm(v[1:])
                if nv > 1e-9:
                    out.append(v * (r / nv))
    for a in (1, 2, 3):
        ext = float(np.max(np.abs(free[a]))) if k else 1.0
        if ext > 1e-9:
            e = np.zeros(4)
            e[a] = min(ext, 1.0)
            out.append(e)
    return out


def greedy_augment(S, A, m: int, *, tol: float = _TOL):
    """Grow S to m columns one column at a time, each time adding the
    candidate_columns entry with the lowest rate.  Adding a column also adds
    new product columns to S (x) S, so the rate is not monotone; duplicating
    the last column (rate unchanged) is always among the choices, so the
    result is never worse than padding.  Returns (value, S_m, n_evals)."""
    A = np.asarray(A, float)
    S = np.asarray(S, float).copy()
    v = float(frame_rate_value(S, A))
    n_ev = 1
    while S.shape[1] < m:
        if v <= tol:
            return v, fit_columns(S, m), n_ev
        best_v, best_S = v, fit_columns(S, S.shape[1] + 1)   # duplicate
        for c in candidate_columns(S):
            T = np.hstack([S, _project_columns_bloch(np.asarray(c)[:, None])])
            val = frame_rate_value(T, A)
            n_ev += 1
            if np.isfinite(val) and val < best_v - tol:
                best_v, best_S = float(val), T
        S, v = best_S, best_v
    return v, S, n_ev


def model10_frames(d1: float, d2: float, J: float, h: float, m: int) -> dict:
    """Closed-form frames with m columns: 'B' (padded to m), 'C<n>' (YZ n-gon
    at the C weights, n = m - 2) and, for m >= 5, 'P<n>' (projector pair +
    regular XY n-gon, n = m - 3)."""
    x_b, y_b, x_c = model10_weights(d1, d2, J, h)
    out = {'B': fit_columns(np.diag([1.0, x_b, y_b, 1.0]), m)}
    if m >= 4:
        out[f'C{m - 2}'] = yz_poly(m - 2).build(
            np.array([x_c, 1.0, 1.0, 0.0, 0.0]))
    if m >= 5:
        out[f'P{m - 3}'] = xy_zpair(m - 3).build(
            np.array([0.5, 0.5, 1.0, 1.0, 0.0, 0.0]))
    return out


def b_boundary(d1, J: float = 1.0, h: float = 1.0):
    """Frame-B rate-zero boundary: the D2 solving D2 (2 D2 + D1 - h^2/D1)
    = 2 J^2, defined for D1 >= h (nan below)."""
    d1 = np.asarray(d1, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        b = d1 - h * h / d1
        d2 = (-b + np.sqrt(b * b + 16.0 * J * J)) / 4.0
    return np.where(d1 >= h, d2, np.nan)


def c_boundary(d2, J: float = 1.0):
    """Continuous-frame-C sufficient boundary: the smallest D1 with rate 0
    at D2 (any h):  J (D2 >= J),  J^2/D2 (J/sqrt2 <= D2 <= J),
    2 (J^2/D2 - D2) (D2 < J/sqrt2)."""
    d2 = np.asarray(d2, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mid = J * J / d2
        low = 2.0 * (J * J / d2 - d2)
    return np.where(d2 >= J, J, np.where(d2 >= J / np.sqrt(2.0), mid, low))


# ---------------------------------------------------------------------------
#  Structured optimiser
# ---------------------------------------------------------------------------
def optimize_family(fam: Family, A, starts=(), *, n_random: int = 12,
                    n_polish: int = 3, maxfev: int = 200, rng=None,
                    tol: float = _TOL):
    """Multistart Nelder-Mead over fam's parameters on the batched-LP rate
    (closed form for 4 columns).  The starts and n_random box samples are
    evaluated, the n_polish best are polished.  Stops early at the floor.
    Returns (value, S, p, n_evals)."""
    A = np.asarray(A, float)
    rng = np.random.default_rng(rng)
    lo, hi = np.asarray(fam.lo, float), np.asarray(fam.hi, float)
    n_ev = 0

    def obj(p):
        nonlocal n_ev
        n_ev += 1
        v = frame_rate_value(fam.build(p), A)
        return float(v) if np.isfinite(v) else _BIG

    pts = [np.asarray(p, float) for p in starts]
    pts += [rng.uniform(lo, hi) for _ in range(n_random)]
    vals = []
    for p in pts:
        vals.append(obj(p))
        if vals[-1] <= tol:
            return vals[-1], fam.build(p), p, n_ev
    order = np.argsort(vals)
    best_v, best_p = float(vals[order[0]]), pts[order[0]]
    step = 0.15 * (hi - lo)
    for i in order[:n_polish]:
        p0 = pts[i]
        simplex = np.vstack([p0] + [p0 + step[k] * np.eye(len(p0))[k]
                                    for k in range(len(p0))])
        r = minimize(obj, p0, method='Nelder-Mead',
                     options=dict(maxfev=maxfev, initial_simplex=simplex,
                                  xatol=1e-7, fatol=1e-11))
        if r.fun < best_v:
            best_v, best_p = float(r.fun), np.asarray(r.x, float)
        if best_v <= tol:
            break
    return best_v, fam.build(best_p), best_p, n_ev


# ---------------------------------------------------------------------------
#  Self-test (model10)
# ---------------------------------------------------------------------------
def self_check(verbose: bool = True) -> bool:
    """Closed-form checks on the model10 bond generator; returns True if all
    pass.  Cheap (closed-form 4-frames plus a few 8-column LPs)."""
    from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,
                                          MODEL10_J, MODEL10_H)
    from framability_rate_frames import pauli_rate
    spec = MODELS['model10']
    J, h = MODEL10_J, MODEL10_H

    def gen(d1, d2):
        L = build_bond_lindbladian(*spec.build(d1, d2), spec.dim).real
        return L, L.T

    ok = True

    def check(name, cond, val):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f'  [{"ok" if cond else "FAIL"}] {name}: {val:.3e}', flush=True)

    # B is exactly at the floor on its side of the boundary, positive outside
    for d1, d2, inside in ((2.5, 0.65, True), (2.5, 0.55, False),
                           (1.5, 0.95, True), (1.2, 0.8, False)):
        L, A = gen(d1, d2)
        S = model10_frames(d1, d2, J, h, 4)['B']
        v = frame_rate_value(S, A)
        pred = d2 >= b_boundary(d1, J, h)
        check(f'B at ({d1},{d2}) inside={pred}', (v <= 1e-9) == inside == pred, v)
    # yz2 at the B parameters is the B frame
    L, A = gen(2.5, 0.65)
    fam = yz_poly(2)
    p = model10_family_starts(fam, 2.5, 0.65, J, h)[0]
    v1 = frame_rate_value(fam.build(p), A)
    v2 = frame_rate_value(model10_frames(2.5, 0.65, J, h, 4)['B'], A)
    check('yz2(B params) == B', abs(v1 - v2) < 1e-10, abs(v1 - v2))
    # Pauli frame: zero exactly on D1 >= h, D2 >= J
    L, A = gen(1.5, 1.2)
    check('Pauli at (1.5,1.2)', pauli_rate(L) <= 1e-9, pauli_rate(L))
    # informational: finite C / P frames at d_ext = 8 inside region C
    L, A = gen(2.0, 0.8)
    for name, S in model10_frames(2.0, 0.8, J, h, 8).items():
        v = frame_rate_value(S, A)
        if verbose:
            print(f'  [info] {name} (d_ext=8) at (2.0,0.8): {v:.4e}', flush=True)
    return ok


if __name__ == '__main__':
    print('framability_rate_families self-check:', flush=True)
    print('passed' if self_check() else 'FAILED')
