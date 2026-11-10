"""
Sanity anchors and a first comparison for the dimer-frame rate
(framability_rate_dimer.py): a 2-qubit frame S on every dimer, the bond gate
applied INSIDE a dimer (frame S) and ACROSS two dimers (frame S (x) S).

Anchors (printed PASS / FAIL, tolerances in brackets):
  1. embed_bond == the 4-qubit Lindbladian built directly from the embedded
     H and jump, for every cut type ba / ab / aa / bb            [1e-12]
  2. Pauli frame: F_split = mu_in = mu_cut = pauli_rate(B)        [1e-10]
  3. product frame S1 (x) S1 (best stored / fresh d4 optimum):
     mu_in = the single-qubit closed-form rate, mu_cut <= it, F <= it  [1e-9]
  4. closed form == per-column LP for a random 2-qubit basis, split and
     joint                                                         [1e-7]
  5. envelope gradient == central finite difference of every term's binding
     column along a random direction (ties excluded)               [1e-5 rel]
  6. rotate+weight: _WeightProblem value == evaluate(rw_frame)    [1e-9]
then the full search per point, against the stored product-frame rates.

Points: model4 (gamma, gamma') = (10, 0), (4, 0)   [the 2026-09 intra-bond
check, square lattice dim = 2, brick tiling]; model10 (Delta1, Delta2) =
(0.5, 0.5), (1.0, 0.2) [chain]; model15 (gamma_1, h) = (1.0, 0.5) [chain].

    python scripts/dimer_rate_check.py                       # all points
    python scripts/dimer_rate_check.py --points model4,10,0 --quick
    python scripts/dimer_rate_check.py --anchors_only
"""
from __future__ import annotations

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import (MODELS, build_bond_lindbladian,      # noqa: E402
                                      _embed_two_site)
from dissipative_PT import _build_lindbladian                             # noqa: E402
from framability_rate_frames import pauli_rate                           # noqa: E402
from framability_rate_global import basis_rate, minimize_rate_global     # noqa: E402
from framability_rate_dimer import (                                      # noqa: E402
    DimerProblem, CUT_SITES, NQ, embed_bond, evaluate, term_parts,
    column_grad, frame_from_free, product_frame, rotate_terms, rw_frame,
    pauli_liouville_unitary, su4, _WeightProblem, minimize_dimer_rate,
    smoothed_value_grad, stored_product_frames, default_seed_dirs)

DEFAULT_POINTS = ['model4,10,0', 'model4,4,0', 'model10,0.5,0.5',
                  'model10,1.0,0.2', 'model15,1.0,0.5']


def _flag(ok: bool) -> str:
    return 'PASS' if ok else 'FAIL'


def grid_index(vals, v):
    vals = np.asarray(vals, float)
    i = int(np.argmin(np.abs(vals - v)))
    return i if abs(vals[i] - v) < 1e-9 else None


def bond_at(model: str, p1: float, p2: float, dim: int):
    H1, H2, j1, j2 = MODELS[model].build(p1, p2)
    return build_bond_lindbladian(H1, H2, j1, j2, dim).real


# ---------------------------------------------------------------------------
#  Anchors
# ---------------------------------------------------------------------------
def anchor_embedding(rng) -> bool:
    X = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H2 = X + X.conj().T
    L2 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    B = _build_lindbladian(H2, [(1.0, L2, L2.conj().T, L2.conj().T @ L2)], n=2)
    ok_all = True
    for c, (i, j) in CUT_SITES.items():
        H4 = _embed_two_site(H2, i, j, 4)
        L4 = _embed_two_site(L2, i, j, 4)
        ref = _build_lindbladian(H4, [(1.0, L4, L4.conj().T, L4.conj().T @ L4)],
                                 n=4)
        err = float(np.max(np.abs(embed_bond(B, (i, j)) - ref)))
        ok = err < 1e-12
        ok_all &= ok
        print(f'  [1] embed_bond {c} sites {(i, j)}: err {err:.1e}  {_flag(ok)}')
    return ok_all


def anchor_pauli(prob, B) -> bool:
    F, terms = evaluate(np.eye(NQ), prob, 'split')
    ref = pauli_rate(B)
    errs = [abs(F - ref)] + [abs(t['value'] - ref) for t in terms]
    ok = max(errs) < 1e-10
    print(f'  [2] Pauli frame: F={F:.10f} pauli_rate={ref:.10f} '
          f'parts {term_parts(terms)}  {_flag(ok)}')
    return ok


def anchor_product(prob, S1) -> bool:
    mu1 = basis_rate(S1, prob.A_in)
    F, terms = evaluate(product_frame(S1), prob, 'split')
    parts = term_parts(terms)
    ok = abs(parts['in'] - mu1) < 1e-9 and \
        all(v <= mu1 + 1e-9 for k, v in parts.items() if k.startswith('cut')) \
        and F <= mu1 + 1e-9
    print(f'  [3] product S1(x)S1: single-qubit rate {mu1:.8f}, F={F:.8f} '
          f'parts {", ".join(f"{k}={v:.8f}" for k, v in parts.items())}  {_flag(ok)}')
    return ok


def random_basis(rng, scale=0.6):
    while True:
        S = frame_from_free(scale * rng.normal(size=(NQ, NQ - 1)) / np.sqrt(NQ))
        if np.linalg.cond(S) < 1e3:
            return S


def anchor_closed_vs_lp(prob, rng) -> bool:
    S = random_basis(rng)
    ok_all = True
    for mode in ('split', 'joint'):
        t0 = time.perf_counter()
        Fc, _ = evaluate(S, prob, mode, method='closed')
        Fl, _ = evaluate(S, prob, mode, method='lp')
        ok = abs(Fc - Fl) < 1e-7 * max(1.0, abs(Fc))
        ok_all &= ok
        print(f'  [4] closed vs LP ({mode}): {Fc:.10f} vs {Fl:.10f} '
              f'({time.perf_counter() - t0:.1f}s)  {_flag(ok)}')
    return ok_all


def anchor_gradient(prob, rng, eps=1e-6) -> bool:
    S = random_basis(rng, scale=0.4)            # columns well inside the ball
    _, terms = evaluate(S, prob, 'split', keep=True)
    Delta = rng.normal(size=(NQ, NQ - 1))
    Delta /= np.linalg.norm(Delta)
    Sp = S.copy(); Sp[:, 1:] += eps * Delta
    Sm = S.copy(); Sm[:, 1:] -= eps * Delta
    _, tp = evaluate(Sp, prob, 'split', keep=True)
    _, tm = evaluate(Sm, prob, 'split', keep=True)
    ok_all = True
    for k, t in enumerate(terms):
        cols = t['cols']
        order = np.argsort(-cols)
        j = int(order[0])
        if cols[order[0]] - cols[order[1]] < 1e-4:
            print(f'  [5] gradient {t["name"]}: binding column tied, skipped')
            continue
        g = column_grad(S, t, j)[:, 1:]
        an = float(np.sum(g * Delta))
        fd = (tp[k]['cols'][j] - tm[k]['cols'][j]) / (2 * eps)
        ok = abs(an - fd) <= 1e-5 * max(1.0, abs(fd))
        ok_all &= ok
        print(f'  [5] gradient {t["name"]} col {j}: analytic {an:.8e} '
              f'FD {fd:.8e}  {_flag(ok)}')
    return ok_all


def anchor_smooth_grad(prob, rng, eps_fd=1e-6) -> bool:
    """[7] smoothed objective gradient (incl. the op-norm penalty: a scale
    that pushes some columns outside the ball) vs central differences."""
    S = random_basis(rng, scale=0.4)
    S[:, 1:4] *= 1.6                              # a few columns outside the ball
    Delta = rng.normal(size=(NQ, NQ - 1))
    Delta /= np.linalg.norm(Delta)
    ok_all = True
    for beta, eps in ((20.0, 1e-2), (500.0, 1e-4)):
        v, G = smoothed_value_grad(S, prob, 'split', beta, eps)
        Sp = S.copy(); Sp[:, 1:] += eps_fd * Delta
        Sm = S.copy(); Sm[:, 1:] -= eps_fd * Delta
        fd = (smoothed_value_grad(Sp, prob, 'split', beta, eps)[0]
              - smoothed_value_grad(Sm, prob, 'split', beta, eps)[0]) / (2 * eps_fd)
        an = float(np.sum(G[:, 1:] * Delta))
        ok = abs(an - fd) <= 1e-4 * max(1.0, abs(fd))
        ok_all &= ok
        print(f'  [7] smoothed gradient beta={beta:g} eps={eps:g}: analytic '
              f'{an:.8e} FD {fd:.8e}  {_flag(ok)}')
    return ok_all


def anchor_rotate_weight(prob, rng) -> bool:
    theta = rng.normal(0, 0.3, NQ - 1)
    u = rng.uniform(np.log(0.05), 0.0, NQ - 1)
    R = pauli_liouville_unitary(su4(theta))
    wp = _WeightProblem(rotate_terms(prob, 'split', R))
    v_w = wp.value(u)
    v_e, _ = evaluate(rw_frame(theta, u), prob, 'split')
    ok = abs(v_w - v_e) < 1e-9 * max(1.0, abs(v_e))
    print(f'  [6] rotate+weight: weight problem {v_w:.10f} vs evaluate '
          f'{v_e:.10f}  {_flag(ok)}')
    return ok


# ---------------------------------------------------------------------------
#  One point
# ---------------------------------------------------------------------------
def run_point(model, p1, p2, args, rng):
    spec = MODELS[model]
    dim = args.dim if args.dim is not None else spec.dim
    B = bond_at(model, p1, p2, dim)
    prob = DimerProblem(B, dim, args.tiling if dim == 2 else None)
    print(f'\n=== {model} {spec.p1_name}={p1:g} {spec.p2_name}={p2:g}  '
          f'dim={dim} tiling={prob.tiling} (swap-symmetric bond: {prob.symmetric})')

    ix, iy = grid_index(spec.p1_vals, p1), grid_index(spec.p2_vals, p2)
    frames, stored = ({}, {})
    if ix is not None and iy is not None:
        dirs = args.seed_dirs or default_seed_dirs(model, dim, spec.dim)
        frames, stored = stored_product_frames(dirs, ix, iy)
    d4 = frames.get(4, [])
    if d4:
        S1 = min(d4, key=lambda S: basis_rate(S, prob.A_in))
        src = f'stored ({len(d4)} d4 frames)'
    else:
        S1, _, _ = minimize_rate_global(prob.A_in, 4, seed=args.seed)
        src = 'fresh minimize_rate_global'
    print(f'  product d4 frame: {src}; stored product rates: '
          + (', '.join(f'd{m}={v:.5f}' for m, v in sorted(stored.items()))
             or 'none found'))

    oks = [anchor_pauli(prob, B), anchor_product(prob, S1)]
    if not args.skip_lp:
        oks.append(anchor_closed_vs_lp(prob, rng))
    oks += [anchor_gradient(prob, rng), anchor_rotate_weight(prob, rng),
            anchor_smooth_grad(prob, rng)]
    if args.anchors_only:
        return all(oks), None

    S, F, info = minimize_dimer_rate(
        prob, mode=args.mode, m_ext=args.m_ext, product_seeds=d4 or [S1],
        fresh_d4=False, rw_random=args.rw_random, rw_maxfev=args.rw_maxfev,
        n_polish=args.n_polish, smooth_iters=args.smooth_iters,
        n_kicks=args.n_kicks, bundle_iters=args.bundle_iters,
        ext_iters=args.ext_iters, certify=not args.skip_lp, seed=args.seed,
        verbose=args.verbose)
    mu1 = basis_rate(S1, prob.A_in)
    row = dict(model=model, p1=p1, p2=p2, dim=dim, pauli=pauli_rate(B),
               prod4=mu1, stored=stored, F=F, info=info)
    print(f'  seeds best   {info["mu_seed"]:.6f} ({info["seed_name"]})')
    print(f'  15 weights   {info["mu_weights16"]:.6f}   rotate+weight '
          f'{info["mu_rw"]:.6f}   smoothed {info["mu_smooth"]:.6f}   '
          f'polished {info["mu_polish"]:.6f} ({info["label"]})')
    print(f'  final F_{args.mode} = {F:.6f}   split {info["rate_split"]:.6f} '
          f'{info["parts_split"]}   joint {info["rate_joint"]:.6f}   '
          f'LP ref {info.get("mu_reference", np.nan):.6f}   '
          f'max op-norm {info["op_norm_max"]:.6f}   [{info["time"]:.0f}s]')
    if 'ext' in info:
        e = info['ext']
        print(f'  m={e["m"]}: {e["value"]:.6f} (padded {e["padded"]:.6f}, '
              f'split {e["rate_split"]:.6f}, joint {e["rate_joint"]:.6f})')
    ok = F <= mu1 + 1e-9 and info['rate_joint'] <= info['rate_split'] + 1e-9
    print(f'  monotone (F <= product d4) and joint <= split: {_flag(ok)}')
    oks.append(ok)
    return all(oks), row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--points', nargs='+', default=DEFAULT_POINTS,
                   help='model,p1,p2 (the model\'s own grid axes)')
    p.add_argument('--dim', type=int, default=None,
                   help="lattice dimension (default: each model's)")
    p.add_argument('--tiling', default='brick', choices=['brick', 'columnar'],
                   help='dim = 2 dimer tiling')
    p.add_argument('--mode', default='split', choices=['split', 'joint'])
    p.add_argument('--m_ext', type=int, default=None,
                   help='also polish an m_ext > 16 frame (per-column LPs: slow)')
    p.add_argument('--seed_dirs', nargs='*', default=None)
    p.add_argument('--rw_random', type=int, default=3)
    p.add_argument('--rw_maxfev', type=int, default=400)
    p.add_argument('--n_polish', type=int, default=3)
    p.add_argument('--smooth_iters', type=int, default=300)
    p.add_argument('--n_kicks', type=int, default=2)
    p.add_argument('--bundle_iters', type=int, default=40)
    p.add_argument('--ext_iters', type=int, default=20)
    p.add_argument('--quick', action='store_true',
                   help='rw_random 1, rw_maxfev 150, n_polish 2, smooth 150, '
                        'kicks 1, bundle 20')
    p.add_argument('--anchors_only', action='store_true')
    p.add_argument('--skip_lp', action='store_true',
                   help='skip the LP anchor and certification (the slow part)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.rw_random, args.rw_maxfev = 1, 150
        args.n_polish, args.bundle_iters = 2, 20
        args.smooth_iters, args.n_kicks = 150, 1

    rng = np.random.default_rng(args.seed)
    print('=== embedding')
    all_ok = anchor_embedding(rng)
    rows = []
    for spec in args.points:
        model, a, b = spec.split(',')
        ok, row = run_point(model, float(a), float(b), args, rng)
        all_ok &= ok
        if row is not None:
            rows.append(row)

    if rows:
        print('\nmodel     p1    p2  dim |  Pauli  prod d4  stored(best m) |'
              '  dimer split (in / cut)        joint')
        for r in rows:
            st = r['stored']
            best_m = min(st, key=st.get) if st else None
            sts = f'{st[best_m]:.4f}(d{best_m})' if best_m else '   n/a     '
            parts = r['info']['parts_split']
            cut = '/'.join(f'{v:.4f}' for k, v in parts.items() if k.startswith('cut'))
            print(f'{r["model"]:8s} {r["p1"]:5.2f} {r["p2"]:5.2f}  {r["dim"]}  | '
                  f'{r["pauli"]:6.4f}  {r["prod4"]:6.4f}  {sts:>14s} | '
                  f'{r["info"]["rate_split"]:.4f} ({parts["in"]:.4f} / {cut})  '
                  f'{r["info"]["rate_joint"]:.4f}')
    print(f'\nALL CHECKS: {_flag(all_ok)}')


if __name__ == '__main__':
    main()
