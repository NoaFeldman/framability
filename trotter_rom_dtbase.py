"""
DT_BASE sweep and dt -> 0 extrapolation of the two trotter_rom_state quantities:
the stabilizer-3 framability of the two-qubit bond gate, and the RoM of the
2x2-lattice state after one application of expm(L_full dt).

This is to trotter_rom_state what results_dtbase_line is to the framability part
of trotter_lindbladian_scan.  For every (p1, p2) point of a model the Trotter
control DT_BASE is swept over base_grid(), each base giving the adaptive step

    dt = DT_BASE / max(||H||_1, {gamma_k})          (choose_dt(..., base=DT_BASE))

and both quantities are evaluated on the gates built at that dt.

Why the RATE is the quantity that survives the limit
----------------------------------------------------
Both raw numbers tend trivially to 1 as dt -> 0: the bond gate tends to the
identity (framable) and the evolved state tends back to the stabilizer start
state (RoM 1).  What has a non-trivial continuous-time limit is the
per-unit-time power

    q(dt) = value ** (1/dt),

whose dt -> 0 limit is exp(rate0) with rate0 = lim_{dt->0} ln(value)/dt.  Both
values are >= 1, so ln >= 0 and the rate is fit through

    r(dt) = ln(value)/dt   ->   rate0   as dt -> 0,

a degree-`deg` polynomial in dt through the `fit_n` points nearest dt = 0.  This
is the same procedure, with the same defaults, as
scripts/trotter_dtbase_line_extrap.extrapolate -- so the framability limits of
this pipeline and of results_dtbase_line are directly comparable.

Base grids
----------
The fit only ever consumes the points nearest dt = 0 (fit_n = 15 by default), so
the default sweep is the 20 smallest bases rather than the full 0.01..0.99 line
of results_dtbase_line -- a 5x saving with no effect on the extrapolant.  Pass
mode='full' to reproduce the results_dtbase_line grid exactly.

    'fit'   0.01 .. 0.20  step 0.01   (20 bases, default)
    'full'  0.01 .. 0.99  step 0.01   (99 bases, the results_dtbase_line grid)

Both grids start at 0.01 = trotter_lindbladian_scan.DT_BASE, so the first base of
either sweep reproduces the main scan's operating point.

Usage:
    python trotter_rom_dtbase.py --self_test
    python trotter_rom_dtbase.py --model model1 --p1 0.0 --p2 1.0

Cluster pipeline:
    scripts/trotter_rom_dtbase_worker.py   per-point array worker (whole base line)
    scripts/trotter_rom_dtbase.slurm.sh    200-task array (one model per submit)
    scripts/submit_trotter_rom_dtbase.sh   submits all six models
    scripts/trotter_rom_dtbase_extrap.py   dt->0 extrapolation + colormaps
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import expm

from trotter_lindbladian_scan import (
    MODELS, DT_BASE, choose_dt, build_bond_lindbladian,
    build_full_lindbladian_model, lpdo_init_vector,
)
from trotter_rom_state import (
    N_LATTICE, STATE_ROM_MODELS, coeffs_to_pauli_vec, rom_of_pauli_vec,
)
from framability import stabilizer_3_framability

# Version stamp for cached results; bump when any stored quantity changes.
ROM_DTBASE_VERSION = '1.0'

# DT_BASE sweeps (see the module docstring).  Both start at DT_BASE = 0.01.
BASE_MODES = ('fit', 'full')
DEFAULT_BASE_MODE = 'fit'

# Extrapolation defaults, identical to scripts/trotter_dtbase_line_extrap.
FIT_N_DEFAULT = 15
DEG_DEFAULT = 1


def base_grid(mode: str = DEFAULT_BASE_MODE) -> np.ndarray:
    """The swept DT_BASE values for a base-grid mode."""
    assert mode in BASE_MODES, f'unknown base mode {mode!r}'
    n = 20 if mode == 'fit' else 99
    return np.array([1e-2 * i for i in range(1, n + 1)], dtype=float)


def n_base(mode: str = DEFAULT_BASE_MODE) -> int:
    return len(base_grid(mode))


# ---------------------------------------------------------------------------
#  The DT_BASE line at one grid point
# ---------------------------------------------------------------------------
def compute_base_line(model, p1: float, p2: float, *,
                      bases: np.ndarray | None = None,
                      mode: str = DEFAULT_BASE_MODE, dim: int | None = None,
                      verbose: bool = False) -> dict:
    """stab_fra and RoM at every DT_BASE of the sweep, for one (p1, p2) point.

    Neither Lindbladian depends on DT_BASE, so L_bond and L_full are built once
    and only the two matrix exponentials are redone per base -- building L_full
    (a 256x256 generator assembled Pauli column by Pauli column) is otherwise
    comparable in cost to the LPs themselves.

    Returns arrays over the base grid: base, dt, stab_fra, rom (plus log2_rom).
    """
    if bases is None:
        bases = base_grid(mode)
    bases = np.asarray(bases, dtype=float)

    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    if dim is None:
        dim = model.dim

    L_bond = build_bond_lindbladian(H1, H2, jumps1, jumps2, dim)   # 16 x 16
    L_full = build_full_lindbladian_model(H1, H2, jumps1, jumps2)  # 256 x 256
    c0 = lpdo_init_vector(model, N_LATTICE)

    n = len(bases)
    out = dict(base=bases,
               dt=np.full(n, np.nan), stab_fra=np.full(n, np.nan),
               rom=np.full(n, np.nan), log2_rom=np.full(n, np.nan))

    for i, base in enumerate(bases):
        t0 = time.perf_counter()
        dt = choose_dt(H1, H2, jumps1, jumps2, base=float(base))
        out['dt'][i] = dt
        out['stab_fra'][i] = float(
            stabilizer_3_framability(expm(L_bond * dt).real))
        c1 = expm(L_full * dt) @ c0
        r = rom_of_pauli_vec(coeffs_to_pauli_vec(c1), N_LATTICE)
        out['rom'][i] = r['rom']
        out['log2_rom'][i] = r['log2_rom']
        if verbose:
            print(f'    base={base:.3f} dt={dt:.4e}  '
                  f'stab_fra={out["stab_fra"][i]:.6f}  rom={r["rom"]:.6f}  '
                  f'({time.perf_counter() - t0:.1f}s)', flush=True)

    out['dim'] = dim
    out['lpdo_init'] = model.lpdo_init
    return out


# ---------------------------------------------------------------------------
#  dt -> 0 extrapolation
# ---------------------------------------------------------------------------
def extrapolate_to_zero(dt_vals: np.ndarray, vals: np.ndarray, *,
                        fit_n: int = FIT_N_DEFAULT, deg: int = DEG_DEFAULT,
                        raw: bool = False) -> float:
    """dt = 0 value of vals**(1/dt), or of raw `vals` when raw=True.

    Same procedure as scripts/trotter_dtbase_line_extrap.extrapolate: the fit
    uses the `fit_n` points nearest dt = 0 and a degree-`deg` polynomial in dt.
    Both quantities are bounded below by 1, so the values are clipped there
    before the log is taken; the returned power-form limit is exp(rate0) with
    rate0 the dt = 0 intercept of r(dt) = ln(value)/dt.
    """
    dt_vals = np.asarray(dt_vals, float)
    vals = np.asarray(vals, float)
    m = np.isfinite(dt_vals) & np.isfinite(vals) & (dt_vals > 0)
    dt, val = dt_vals[m], vals[m]
    if dt.size < 2:
        return float('nan')

    order = np.argsort(dt)                 # ascending dt: nearest dt=0 first
    dt, val = dt[order], val[order]
    k = int(min(fit_n, dt.size))
    dtf = dt[:k]
    valf = np.maximum(val[:k], 1.0)        # both quantities are >= 1

    d = int(min(deg, k - 1))
    if raw:
        c = np.polyfit(dtf, valf, d)
        return float(max(c[-1], 1.0))      # dt^0 intercept, clipped to >= 1

    rate = np.log(valf) / dtf              # -> rate0 (const) as dt -> 0
    c = np.polyfit(dtf, rate, d)
    return float(np.exp(c[-1]))            # exp(dt=0 intercept of the rate)


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------
def self_test() -> None:
    # 1. Base grids start at the main scan's DT_BASE and are strictly ordered.
    for mode in BASE_MODES:
        g = base_grid(mode)
        print(f'[1] base_grid({mode!r}): {len(g)} values, '
              f'{g[0]:.2f} .. {g[-1]:.2f}')
        assert abs(g[0] - DT_BASE) < 1e-12
        assert np.all(np.diff(g) > 0)

    # 2. Exact recovery: a pure exponential value(dt) = exp(rate0 dt) has
    #    ln(value)/dt == rate0 for every dt, so the fit must return exp(rate0).
    dt = base_grid('fit') / 7.0
    for rate0 in (0.0, 3.5, 42.0):
        vals = np.exp(rate0 * dt)
        got = extrapolate_to_zero(dt, vals)
        print(f'[2] rate0 = {rate0:6.2f}: limit = {got:.10f}  '
              f'(expect {np.exp(rate0):.10f})')
        assert abs(got - np.exp(rate0)) < 1e-8 * max(1.0, np.exp(rate0))
        # the same line, extrapolated raw, must tend to 1
        assert abs(extrapolate_to_zero(dt, vals, raw=True) - 1.0) < 1e-6

    # 3. A first-order correction, value = exp((rate0 + c dt) dt), is removed
    #    exactly by the degree-1 fit but not by a degree-0 one.
    rate0, c = 12.0, 250.0
    vals = np.exp((rate0 + c * dt) * dt)
    lin = extrapolate_to_zero(dt, vals, deg=1)
    const = extrapolate_to_zero(dt, vals, deg=0)
    print(f'[3] with a linear-in-dt correction: deg=1 -> {lin:.8f}, '
          f'deg=0 -> {const:.8f}  (expect {np.exp(rate0):.8f})')
    assert abs(lin - np.exp(rate0)) < 1e-6 * np.exp(rate0)
    assert abs(const - np.exp(rate0)) > abs(lin - np.exp(rate0))

    # 4. One real base line: both quantities must decrease towards 1 with dt,
    #    and both limits must be finite and >= 1.
    model = MODELS['model1']
    res = compute_base_line(model, 0.0, 1.0, bases=base_grid('fit')[:6])
    print(f'[4] model1 (h=0, gamma=1): dt = {res["dt"][0]:.3e} .. '
          f'{res["dt"][-1]:.3e}')
    print(f'    stab_fra = {res["stab_fra"][0]:.6f} .. {res["stab_fra"][-1]:.6f}')
    print(f'    rom      = {res["rom"][0]:.6f} .. {res["rom"][-1]:.6f}')
    assert np.all(np.isfinite(res['stab_fra'])) and np.all(res['stab_fra'] >= 1 - 1e-9)
    assert np.all(np.isfinite(res['rom'])) and np.all(res['rom'] >= 1 - 1e-9)
    assert res['stab_fra'][0] <= res['stab_fra'][-1] + 1e-9   # grows with dt
    assert res['rom'][0] <= res['rom'][-1] + 1e-9

    for key in ('stab_fra', 'rom'):
        lim = extrapolate_to_zero(res['dt'], res[key])
        raw = extrapolate_to_zero(res['dt'], res[key], raw=True)
        print(f'[4] {key}: (^1/dt) limit = {lim:.6f},  raw limit = {raw:.8f}')
        assert np.isfinite(lim) and lim >= 1.0
        assert abs(raw - 1.0) < 1e-3

    print('self-test passed.')


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--self_test', action='store_true')
    p.add_argument('--model', type=str, choices=list(STATE_ROM_MODELS))
    p.add_argument('--p1',    type=float)
    p.add_argument('--p2',    type=float)
    p.add_argument('--mode',  type=str, default=DEFAULT_BASE_MODE,
                   choices=list(BASE_MODES))
    p.add_argument('--fit_n', type=int, default=FIT_N_DEFAULT)
    p.add_argument('--deg',   type=int, default=DEG_DEFAULT)
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if args.model is None or args.p1 is None or args.p2 is None:
        p.error('--model, --p1 and --p2 are required (or use --self_test)')

    model = MODELS[args.model]
    print(f'[{model.name}] {model.p1_name}={args.p1} {model.p2_name}={args.p2} '
          f'mode={args.mode}')
    res = compute_base_line(model, args.p1, args.p2, mode=args.mode, verbose=True)
    for key in ('stab_fra', 'rom'):
        lim = extrapolate_to_zero(res['dt'], res[key],
                                  fit_n=args.fit_n, deg=args.deg)
        raw = extrapolate_to_zero(res['dt'], res[key],
                                  fit_n=args.fit_n, deg=args.deg, raw=True)
        print(f'  {key:9s} dt->0:  ^(1/dt) limit = {lim:.6f}   raw = {raw:.8f}')


if __name__ == '__main__':
    main()
