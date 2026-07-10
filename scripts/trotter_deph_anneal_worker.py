"""
Dephasing-annealed optimised framability chain for one Trotter-scan point.

For the scan point of a model with the *minimal* stored optimised Heisenberg
framability at d_ext_single = 6 (opt_fra_6, results_trotter_v3), this worker
runs a two-stage continuation in the strength kappa of an added single-qubit
pure-dephasing channel sqrt(kappa) Z (one-qubit jump, so it carries the same
1/(2d) bond share as the model's own jumps -- kappa is directly comparable
with the model's rates):

  1. ramp up:   walk kappa upward along a geometric grid (kappa = 0, then
                kappa_min .. kappa_max, extended by the grid ratio up to
                kappa_ext_max if needed), warm-starting every optimisation
                from the previous step's frame (the kappa = 0 step starts
                from the scan's stored opt_S_6 frame), until the optimised
                framability reaches 1 (<= 1 + tol).
  2. anneal down: take the frame found at that kappa_star and walk kappa back
                down the same grid to exactly 0, warm-starting each step from
                the previous (higher-kappa) optimum (plus the ramp-up frame
                at the same kappa, when available).

The bond Trotter gate keeps the *scan point's dt fixed* for every kappa, so
the gate family is continuous in kappa and the kappa = 0 endpoint is exactly
the scan gate.

Chains for different --seed values are fully independent (they differ in the
jitter around the warm starts); the collect script takes the per-kappa minimum
across seeds.  The kappa grid depends only on the CLI arguments, so all seeds
of one submission share the same grid points.

Output: <out_dir>/<model>/chain_<ix:03d>_<iy:03d>_seed<seed:03d>.npz

Usage (one chain):
    python scripts/trotter_deph_anneal_worker.py --model model1 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dissipative_PT import (
    _SZ, optimise_framability, frame_from_params, params_from_frame,
)
from trotter_lindbladian_scan import MODELS, MODEL4_H, bond_trotter_gate, choose_dt

DEPH_ANNEAL_VERSION = '1.0'
D_EXT = 6                      # d_ext_single of the optimised Heisenberg frame


# ---------------------------------------------------------------------------
#  Model coefficients (the rates/couplings kappa is compared against)
# ---------------------------------------------------------------------------
def model_coefficients(model_name: str, p1: float, p2: float) -> list[tuple[str, float]]:
    """(name, value) pairs of the model's coefficients at the point (p1, p2).
    These become the vertical reference lines of the framability-vs-kappa plot
    (zero-valued coefficients are kept here and skipped at plot time)."""
    if model_name == 'model1':
        return [('J', 1.0), ('|h|', abs(p1)), (r'$\gamma$', p2)]
    if model_name == 'model2':
        return [('J', 1.0), ('|h|', abs(p1)), (r'$\gamma$', p2)]
    if model_name == 'model3':
        return [('J', 1.0), (r'$\gamma$', p1), (r"$\gamma'$", p2)]
    if model_name == 'model4':
        return [('J', 1.0), ('h', MODEL4_H), (r'$\gamma$', p1), (r"$\gamma'$", p2)]
    if model_name == 'model5':
        return [(r'$J_x$', 0.9), (r'$J_y$', abs(p1)), (r'$J_z$', 1.0),
                (r'$\gamma$', p2)]
    raise ValueError(f'unknown model {model_name}')


# ---------------------------------------------------------------------------
#  Point selection: minimal stored opt_fra_6 over the scan grid
# ---------------------------------------------------------------------------
def select_point(scan_dir: Path, model, ix: int | None, iy: int | None) -> dict:
    """The scan point with minimal opt_fra_6 (deterministic: sorted file order,
    strict '<' tie-break), or the explicit (--ix, --iy) point if given.

    Returns dict(ix, iy, p1, p2, dt, dim, fra_orig, S6) where S6 is the stored
    optimal frame (or None) used to warm-start the kappa = 0 optimisation.
    """
    mdir = scan_dir / model.name

    def _unpack(f: Path) -> dict:
        d = np.load(f)
        S6 = np.asarray(d['opt_S_6'], dtype=float) if 'opt_S_6' in d else None
        if S6 is not None and not np.all(np.isfinite(S6)):
            S6 = None
        return dict(ix=int(d['ix']), iy=int(d['iy']),
                    p1=float(d['p1']), p2=float(d['p2']),
                    dt=float(d['dt']), dim=int(d['dim']),
                    fra_orig=float(d['opt_fra_6']), S6=S6)

    if ix is not None and iy is not None:
        f = mdir / f'pt_{ix:03d}_{iy:03d}.npz'
        if f.exists():
            return _unpack(f)
        # point not on disk: rebuild its parameters from the model grid
        p1 = float(model.p1_vals[ix])
        p2 = float(model.p2_vals[iy])
        H1, H2, j1, j2 = model.build(p1, p2)
        dt = model.dt if model.dt is not None else choose_dt(H1, H2, j1, j2)
        return dict(ix=ix, iy=iy, p1=p1, p2=p2, dt=dt, dim=model.dim,
                    fra_orig=float('nan'), S6=None)

    best_val, best_file = float('inf'), None
    for f in sorted(mdir.glob('pt_*.npz')):
        try:
            d = np.load(f)
            v = float(d['opt_fra_6'])
        except Exception:
            continue
        if np.isfinite(v) and v < best_val:
            best_val, best_file = v, f
    if best_file is None:
        raise FileNotFoundError(f'no usable pt_*.npz with opt_fra_6 in {mdir}')
    return _unpack(best_file)


# ---------------------------------------------------------------------------
#  kappa grid and gate family
# ---------------------------------------------------------------------------
def base_kappas(kmin: float, kmax: float, per_decade: int) -> np.ndarray:
    """[0, kmin, ..., kmax] -- geometric grid with `per_decade` points per
    decade, prepended with the exact kappa = 0 endpoint."""
    n = int(round(per_decade * np.log10(kmax / kmin))) + 1
    return np.concatenate([[0.0], np.geomspace(kmin, kmax, n)])


def make_gate_factory(model, p1: float, p2: float, dim: int, dt: float):
    """gate(kappa): bond Trotter gate with an added sqrt(kappa) Z one-qubit
    jump, at the *fixed* dt of the base point (continuous gate family)."""
    H1, H2, jumps1, jumps2 = model.build(p1, p2)

    def gate(kappa: float) -> np.ndarray:
        j1 = list(jumps1 or [])
        if kappa > 0.0:
            j1 = j1 + [np.sqrt(kappa) * _SZ]
        return bond_trotter_gate(H1, H2, j1, jumps2, dim, dt)

    return gate


# ---------------------------------------------------------------------------
#  One continuation step
# ---------------------------------------------------------------------------
def opt_step(gate: np.ndarray, warm_xs, *, maxfev: int, n_jitter: int,
             jitter: float, rng: np.random.Generator, seed: int):
    """One optimise_framability call seeded from the warm starts (each with
    n_jitter jittered copies) plus the standard ixyz seed (n_restarts=1)."""
    extras = []
    for xw in warm_xs:
        if xw is None:
            continue
        xw = np.asarray(xw, dtype=float).ravel()
        if xw.size != 4 * (D_EXT - 1) or not np.all(np.isfinite(xw)):
            continue
        extras.append(xw)
        for _ in range(n_jitter):
            extras.append(xw + jitter * rng.standard_normal(xw.size))
    f, x = optimise_framability(gate, D_EXT, n_restarts=1, maxfev=maxfev,
                                seed=seed, extra_init_xs=extras, return_x=True)
    return float(f), np.asarray(x, dtype=float)


# ---------------------------------------------------------------------------
#  Chain
# ---------------------------------------------------------------------------
def run_chain(model, sel: dict, args) -> dict:
    gate_at = make_gate_factory(model, sel['p1'], sel['p2'], sel['dim'], sel['dt'])
    rng = np.random.default_rng(1000 + args.seed)

    kappas = list(base_kappas(args.kappa_min, args.kappa_max, args.per_decade))
    ratio = (args.kappa_max / args.kappa_min) ** (1.0 / (len(kappas) - 2))

    fra_up, x_up = [], []
    warm0 = params_from_frame(sel['S6']) if sel['S6'] is not None else None

    # ── stage 1: ramp kappa up until framability reaches 1 ──────────────────
    reached, i_star, i = False, None, 0
    while True:
        t0 = time.perf_counter()
        warms = [x_up[i - 1]] if i > 0 else [warm0]
        f, x = opt_step(gate_at(kappas[i]), warms, maxfev=args.maxfev,
                        n_jitter=args.n_jitter, jitter=args.jitter, rng=rng,
                        seed=args.seed * 10007 + i)
        fra_up.append(f)
        x_up.append(x)
        print(f'[up   {i:3d}] kappa={kappas[i]:.6g}  fra={f:.6f}  '
              f'({time.perf_counter() - t0:.0f}s)', flush=True)
        if f <= 1.0 + args.tol:
            reached, i_star = True, i
            break
        if i == len(kappas) - 1:
            # deterministic extension: kmax * ratio, kmax * ratio^2, ...
            nxt = kappas[-1] * ratio
            if nxt > args.kappa_ext_max:
                i_star = int(np.argmin(fra_up))
                print(f'WARNING: framability never reached 1 + {args.tol} up to '
                      f'kappa={kappas[-1]:.3g}; annealing down from the best '
                      f'point (i={i_star}, fra={fra_up[i_star]:.6f})', flush=True)
                break
            kappas.append(nxt)
        i += 1

    n = len(fra_up)
    kappas = np.array(kappas[:n])
    fra_up = np.array(fra_up)

    # ── stage 2: anneal kappa down to 0 with frame continuation ─────────────
    fra_down = np.full(n, np.nan)
    x_down = [None] * n
    fra_down[i_star] = fra_up[i_star]
    x_down[i_star] = x_up[i_star]
    for i in range(i_star - 1, -1, -1):
        t0 = time.perf_counter()
        f, x = opt_step(gate_at(kappas[i]), [x_down[i + 1], x_up[i]],
                        maxfev=args.maxfev, n_jitter=args.n_jitter,
                        jitter=args.jitter, rng=rng,
                        seed=args.seed * 10007 + 5000 + i)
        fra_down[i] = f
        x_down[i] = x
        print(f'[down {i:3d}] kappa={kappas[i]:.6g}  fra={f:.6f}  '
              f'({time.perf_counter() - t0:.0f}s)', flush=True)

    def _stack_x(xs):
        out = np.full((n, 4 * (D_EXT - 1)), np.nan)
        for j, xj in enumerate(xs):
            if xj is not None:
                out[j] = np.asarray(xj, dtype=float).ravel()
        return out

    S_down = np.full((n, 4, D_EXT), np.nan)
    for j, xj in enumerate(x_down):
        if xj is not None:
            S_down[j] = frame_from_params(xj, D_EXT)

    names, values = zip(*model_coefficients(model.name, sel['p1'], sel['p2']))
    return dict(kappas=kappas, fra_up=fra_up, fra_down=fra_down,
                x_up=_stack_x(x_up), x_down=_stack_x(x_down), S_down=S_down,
                i_star=i_star, reached=reached,
                coef_names=np.array(names), coef_values=np.array(values, float))


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model',    type=str, required=True, choices=list(MODELS))
    p.add_argument('--seed',     type=int, default=0,
                   help='chain id: independent jitter around the warm starts')
    p.add_argument('--scan_dir', type=str, default='results_trotter_v3',
                   help='trotter-scan results to select the point from')
    p.add_argument('--out_dir',  type=str, default='results_deph_anneal')
    p.add_argument('--ix', type=int, default=None,
                   help='override the point (with --iy) instead of the argmin')
    p.add_argument('--iy', type=int, default=None)
    p.add_argument('--kappa_min',     type=float, default=1e-3)
    p.add_argument('--kappa_max',     type=float, default=1e4)
    p.add_argument('--kappa_ext_max', type=float, default=1e7,
                   help='hard cap when extending the grid past kappa_max')
    p.add_argument('--per_decade', type=int, default=5,
                   help='geometric grid density (points per decade of kappa)')
    p.add_argument('--tol',      type=float, default=1e-3,
                   help='stop the ramp-up once fra <= 1 + tol')
    p.add_argument('--maxfev',   type=int, default=400,
                   help='Powell budget of every optimise_framability start')
    p.add_argument('--n_jitter', type=int, default=1,
                   help='jittered copies of each warm start per step')
    p.add_argument('--jitter',   type=float, default=0.05)
    args = p.parse_args()
    if (args.ix is None) != (args.iy is None):
        p.error('--ix and --iy must be given together')

    model = MODELS[args.model]
    sel = select_point(Path(args.scan_dir), model, args.ix, args.iy)

    out_dir = Path(args.out_dir) / model.name
    out = out_dir / f"chain_{sel['ix']:03d}_{sel['iy']:03d}_seed{args.seed:03d}.npz"
    if out.exists():
        try:
            d = np.load(out, allow_pickle=True)
            if str(d['code_version']) == DEPH_ANNEAL_VERSION:
                print(f'[skip] {out} already at version {DEPH_ANNEAL_VERSION}',
                      flush=True)
                return
        except Exception:
            pass

    print(f"[{model.name} seed {args.seed}] point ix={sel['ix']} iy={sel['iy']} "
          f"({model.p1_name}={sel['p1']:.3f}, {model.p2_name}={sel['p2']:.3f})  "
          f"dt={sel['dt']:.6g}  scan opt_fra_6={sel['fra_orig']:.6f}", flush=True)

    t0 = time.perf_counter()
    res = run_chain(model, sel, args)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        model=np.array(model.name), seed=np.array(args.seed),
        ix=np.array(sel['ix']), iy=np.array(sel['iy']),
        p1=np.array(sel['p1']), p2=np.array(sel['p2']),
        p1_name=np.array(model.p1_name), p2_name=np.array(model.p2_name),
        dt=np.array(sel['dt']), dim=np.array(sel['dim']),
        d_ext=np.array(D_EXT), fra_orig=np.array(sel['fra_orig']),
        kappas=res['kappas'], fra_up=res['fra_up'], fra_down=res['fra_down'],
        x_up=res['x_up'], x_down=res['x_down'], S_down=res['S_down'],
        i_star=np.array(res['i_star']), reached=np.array(res['reached']),
        coef_names=res['coef_names'], coef_values=res['coef_values'],
        code_version=np.array(DEPH_ANNEAL_VERSION),
    )
    print(f'saved {out}  ({time.perf_counter() - t0:.0f}s total, '
          f"kappa_star={res['kappas'][res['i_star']]:.6g}, "
          f"fra(kappa=0) down={res['fra_down'][0]:.6f} vs "
          f"orig={sel['fra_orig']:.6f})", flush=True)


if __name__ == '__main__':
    main()
