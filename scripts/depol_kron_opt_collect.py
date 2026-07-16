"""
Collect the heavy depol_kron optimisation (depol_kron_opt_worker.py) and, for
every (gate_label, d_ext_single) cell, select a ROBUSTLY REACHABLE optimum
rather than a fragile knife-edge minimum.

Robust-basin selection
-----------------------
Every restart's converged frame (across all batches) is pooled.  Then:
  1. cluster the restart frames into basins by a permutation/sign-invariant
     fingerprint (sorted column norms + sorted pairwise |cosines| of the full
     frame D = S (x) S), so restarts that landed on the *same* frame (up to the
     gauge symmetries) are grouped;
  2. keep the basins whose best framability is within eps_tol of the global
     best (the near-optimal basins);
  3. among those, choose the one populated by the MOST restarts -- the widest,
     easiest-to-reach basin -- and take its lowest-framability member as the
     representative optimal frame.

Reported per cell:
  robust framability + gap-to-floor  (the representative)
  strict_min framability             (the global argmin, for reference)
  reach_frac = fraction of ALL restarts within eps_tol of the global best
  basin_frac = fraction of ALL restarts landing in the chosen basin
  n_basins, chosen basin size, total restarts.

Outputs
-------
  <out_dir>/depol_kron_opt_summary.npz          -- arrays over (N_GATES, N_D)
  <out_dir>/best_frames/<label>_d<d>.npz         -- representative D, S, stats
  console table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

GATES = [
    ('g1_p0.00', (float(np.sqrt(0.5)), float(np.exp(-1.0)), float(np.pi)), 0.00),
    ('g1_p0.08', (float(np.sqrt(0.5)), float(np.exp(-1.0)), float(np.pi)), 0.08),
    ('g2_p0.00', (0.3, 0.3, 0.0), 0.00),
    ('g2_p0.08', (0.3, 0.3, 0.0), 0.08),
]
D_EXT_SINGLES = [4, 6, 8]
N_BATCHES     = 16

N_GATES = len(GATES)
N_D     = len(D_EXT_SINGLES)
N_QUBITS = 2


def _kron_power(S: np.ndarray, n: int) -> np.ndarray:
    out = S
    for _ in range(n - 1):
        out = np.kron(out, S)
    return out


def _fingerprint(S: np.ndarray) -> np.ndarray:
    """Permutation/sign-invariant fingerprint of the frame D = S (x) S.

    Each column d_j of D is stacked into a real vector [Re(d_j); Im(d_j)].
    The fingerprint concatenates the sorted column norms with the sorted
    pairwise |cosines|; both are invariant under column permutations and sign
    flips (the gauge symmetries of the frame polytope conv{+-d_j}).
    """
    D = _kron_power(S, N_QUBITS)                  # (16, d_ext)
    V = np.vstack([D.real, D.imag])               # (32, d_ext)
    norms = np.linalg.norm(V, axis=0)             # (d_ext,)
    safe = np.where(norms > 1e-12, norms, 1.0)
    U = V / safe
    G = np.abs(U.T @ U)                           # |cosines|, (d_ext, d_ext)
    iu = np.triu_indices(G.shape[0], k=1)
    return np.concatenate([np.sort(norms), np.sort(G[iu])])


def _cluster(fingerprints, fras, fp_tol):
    """Greedy fingerprint clustering, restarts visited best-framability first.

    Returns a list of clusters; each cluster is a dict with 'members' (indices),
    'best' (min framability) and 'rep' (index of the min-framability member).
    """
    order = np.argsort(fras)
    clusters = []
    for i in order:
        fp = fingerprints[i]
        placed = False
        for cl in clusters:
            if np.max(np.abs(fp - fingerprints[cl['rep']])) < fp_tol:
                cl['members'].append(i)
                placed = True
                break
        if not placed:
            clusters.append({'members': [i], 'best': float(fras[i]), 'rep': i})
    return clusters


def _process_cell(pool_fra, pool_S, eps_tol, fp_tol):
    """Robust-basin selection for one (gate, d) cell.  Returns a stats dict."""
    n = len(pool_fra)
    global_best = float(np.min(pool_fra))
    reach_frac  = float(np.mean(pool_fra <= global_best + eps_tol))

    fps = [_fingerprint(S) for S in pool_S]
    clusters = _cluster(fps, pool_fra, fp_tol)

    # near-optimal basins: cluster best within eps_tol of the global best.
    near = [cl for cl in clusters if cl['best'] <= global_best + eps_tol]
    if near:
        chosen = max(near, key=lambda cl: len(cl['members']))
    else:                                    # fallback (shouldn't happen)
        chosen = min(clusters, key=lambda cl: cl['best'])

    rep = chosen['rep']
    return {
        'robust_fra':  float(pool_fra[rep]),
        'strict_min':  global_best,
        'rep_S':       pool_S[rep],
        'reach_frac':  reach_frac,
        'basin_frac':  float(len(chosen['members']) / n),
        'basin_size':  int(len(chosen['members'])),
        'n_restarts':  int(n),
        'n_basins':    int(len(clusters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir',  default='results_depol_kron_opt')
    parser.add_argument('--out_dir', default='results_depol_kron_opt')
    parser.add_argument('--n_batches', type=int, default=N_BATCHES)
    parser.add_argument('--eps_tol', type=float, default=1e-4,
                        help='framability tolerance defining "near-optimal"')
    parser.add_argument('--fp_tol', type=float, default=5e-3,
                        help='fingerprint L-inf tolerance for basin clustering')
    args = parser.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    (out_dir / 'best_frames').mkdir(parents=True, exist_ok=True)

    robust_fra = np.full((N_GATES, N_D), np.nan)
    strict_min = np.full((N_GATES, N_D), np.nan)
    floor_arr  = np.full((N_GATES, N_D), np.nan)
    gap_arr    = np.full((N_GATES, N_D), np.nan)
    reach_frac = np.full((N_GATES, N_D), np.nan)
    basin_frac = np.full((N_GATES, N_D), np.nan)
    basin_size = np.zeros((N_GATES, N_D), dtype=int)
    n_restarts = np.zeros((N_GATES, N_D), dtype=int)
    n_basins   = np.zeros((N_GATES, N_D), dtype=int)
    use_complex = np.zeros((N_GATES, N_D), dtype=bool)

    for g_idx, (label, _, _) in enumerate(GATES):
        for d_idx, d in enumerate(D_EXT_SINGLES):
            pool_fra, pool_S, floor_val = [], [], None
            for b in range(args.n_batches):
                f = in_dir / f'{label}_d{d}_b{b:02d}.npz'
                if not f.exists():
                    continue
                data = np.load(f, allow_pickle=True)
                if 'pool_fra' not in data or data['pool_fra'].size == 0:
                    continue
                pf = np.asarray(data['pool_fra'], dtype=float)
                pS = np.asarray(data['pool_S'])
                good = np.isfinite(pf)
                pool_fra.extend(pf[good].tolist())
                pool_S.extend([pS[k] for k in np.nonzero(good)[0]])
                floor_val = float(data['floor'])

            if not pool_fra:
                print(f'  missing: no valid restart for {label} d={d}')
                continue

            pool_fra = np.asarray(pool_fra, dtype=float)
            st = _process_cell(pool_fra, pool_S, args.eps_tol, args.fp_tol)

            rep_S = st['rep_S']
            rep_D = _kron_power(rep_S, N_QUBITS)
            is_cplx = bool(np.max(np.abs(rep_S.imag)) > 1e-12)

            robust_fra[g_idx, d_idx] = st['robust_fra']
            strict_min[g_idx, d_idx] = st['strict_min']
            floor_arr[g_idx, d_idx]  = floor_val
            gap_arr[g_idx, d_idx]    = st['robust_fra'] - floor_val
            reach_frac[g_idx, d_idx] = st['reach_frac']
            basin_frac[g_idx, d_idx] = st['basin_frac']
            basin_size[g_idx, d_idx] = st['basin_size']
            n_restarts[g_idx, d_idx] = st['n_restarts']
            n_basins[g_idx, d_idx]   = st['n_basins']
            use_complex[g_idx, d_idx] = is_cplx

            a, b_, c = GATES[g_idx][1]
            p = GATES[g_idx][2]
            np.savez(
                out_dir / 'best_frames' / f'{label}_d{d}.npz',
                framability   = np.array(st['robust_fra']),
                strict_min    = np.array(st['strict_min']),
                floor         = np.array(floor_val),
                gap           = np.array(st['robust_fra'] - floor_val),
                reach_frac    = np.array(st['reach_frac']),
                basin_frac    = np.array(st['basin_frac']),
                basin_size    = np.array(st['basin_size']),
                n_restarts    = np.array(st['n_restarts']),
                n_basins      = np.array(st['n_basins']),
                D             = rep_D,
                S             = rep_S,
                use_complex   = np.array(is_cplx),
                alpha = np.array(a), beta = np.array(b_), gamma = np.array(c),
                p     = np.array(p),
                d_ext_single  = np.array(d),
                gate_label    = np.array(label),
            )

    np.savez(
        out_dir / 'depol_kron_opt_summary.npz',
        gate_labels   = np.array([g[0] for g in GATES]),
        alpha         = np.array([g[1][0] for g in GATES]),
        beta          = np.array([g[1][1] for g in GATES]),
        gamma         = np.array([g[1][2] for g in GATES]),
        p             = np.array([g[2] for g in GATES]),
        d_ext_singles = np.array(D_EXT_SINGLES),
        framability   = robust_fra,
        strict_min    = strict_min,
        floor         = floor_arr,
        gap           = gap_arr,
        reach_frac    = reach_frac,
        basin_frac    = basin_frac,
        basin_size    = basin_size,
        n_restarts    = n_restarts,
        n_basins      = n_basins,
        use_complex   = use_complex,
    )
    print(f'Saved {out_dir / "depol_kron_opt_summary.npz"}')

    # ── console table ─────────────────────────────────────────────────────────
    print('\nRobust-basin optimal framability  (reach = frac of restarts near '
          'global best;  basin = frac of restarts in chosen basin)')
    for g_idx, (label, _, p) in enumerate(GATES):
        print(f'\n{label}  (p={p:.2f})')
        print('   d   robust_fra    gap        strict_min   reach   basin   '
              'basins  restarts')
        for d_idx, d in enumerate(D_EXT_SINGLES):
            if np.isnan(robust_fra[g_idx, d_idx]):
                print(f'  {d:2d}   --')
                continue
            print(f'  {d:2d}   {robust_fra[g_idx, d_idx]:.6f}   '
                  f'{gap_arr[g_idx, d_idx]:.2e}   '
                  f'{strict_min[g_idx, d_idx]:.6f}    '
                  f'{reach_frac[g_idx, d_idx]:.2f}    '
                  f'{basin_frac[g_idx, d_idx]:.2f}    '
                  f'{n_basins[g_idx, d_idx]:4d}   '
                  f'{n_restarts[g_idx, d_idx]:5d}')

    n_missing = int(np.sum(np.isnan(robust_fra)))
    if n_missing:
        print(f'\nWarning: {n_missing}/{N_GATES * N_D} cells have no result yet')


if __name__ == '__main__':
    main()
