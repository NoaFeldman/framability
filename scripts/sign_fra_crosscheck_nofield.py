"""
Cross-validate the sign problem and framability optima for the no-field
two-qubit Lindbladian Trotter step.

Inputs (already on disk, both on the same 41 x 21 grid, J=1, h=0, dt=0.01):
  * results_sign_problem_nofield/sign_nofield_<ig>_<igp>.npz
        keys: sign_init, sign_opt, n_opt (3-vec: R(n) = exp(i pi n.sigma)),
              gamma, gamma_p, J, h, dt
  * results_trotter/trotter_4_<ig>_<igp>.npz
        keys: framability (= fra_opt), D (= 16x16 frame = S kron S),
              x, d_ext_single (=4), gamma, gamma_p, J, dt

Cross-checks (both at d_ext_single = 4, frames are 16x16):
    s_via_D   = s( D^{-1} U D )                with D from results_trotter
    fra_via_R = heisenberg_framability(D_R, U) with D_R = M(R) kron M(R)
                (M(R) = Pauli-basis 4x4 orthogonal superop of R(n_opt))

Improved values:
    s_final   = min(s_opt,   s_via_D)
    fra_final = min(fra_opt, fra_via_R)

Outputs:
    results_plots/sign_fra_crosscheck_nofield.npz
    results_plots/sign_fra_crosscheck_nofield.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sign_problem import sign_problem
from framability import heisenberg_framability
from scripts.sign_problem_lindbladian_worker import (
    build_lindbladian, _single_qubit_superop_in_pauli_basis,
    GAMMA_STEP, N_GAMMA, N_GP,
)


_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _R_from_n(n_vec: np.ndarray) -> np.ndarray:
    """Single-qubit unitary R(n) = exp(i pi (n_x X + n_y Y + n_z Z))."""
    H1 = n_vec[0] * _SX + n_vec[1] * _SY + n_vec[2] * _SZ
    return expm(1j * np.pi * H1)


# ── cross-check helpers ──────────────────────────────────────────────────────
def _sign_in_frame_D(gate: np.ndarray, D: np.ndarray) -> float:
    """Sign problem of D^{-1} U D (D is 16x16)."""
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        return float('nan')
    U_frame = D_inv @ gate @ D
    return float(sign_problem(U_frame))


def _fra_from_unitary_rotation(gate: np.ndarray, n_vec: np.ndarray) -> float:
    """Framability with D = kron(M(R), M(R)) where R = R(n_vec)."""
    R = _R_from_n(n_vec)
    M = _single_qubit_superop_in_pauli_basis(R)
    if np.max(np.abs(M.imag)) > 1e-10:
        return float('nan')
    M = M.real
    D = np.kron(M, M)
    try:
        return float(heisenberg_framability(D, gate))
    except Exception:
        return float('nan')


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sign_dir',   default='results_sign_problem_nofield')
    parser.add_argument('--trotter_dir', default='results_trotter')
    parser.add_argument('--sign_tag',   default='nofield')
    parser.add_argument('--J',          type=float, default=1.0)
    parser.add_argument('--h',          type=float, default=0.0)
    parser.add_argument('--dt',         type=float, default=0.01)
    parser.add_argument('--out_dir',    default='results_plots')
    args = parser.parse_args()

    sign_dir    = Path(args.sign_dir)
    trotter_dir = Path(args.trotter_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sign_init = np.full((N_GAMMA, N_GP), np.nan)
    sign_opt  = np.full((N_GAMMA, N_GP), np.nan)
    fra_opt   = np.full((N_GAMMA, N_GP), np.nan)
    s_via_D   = np.full((N_GAMMA, N_GP), np.nan)
    fra_via_R = np.full((N_GAMMA, N_GP), np.nan)

    t0 = time.perf_counter()
    n_done = n_total = 0

    for ig in range(N_GAMMA):
        for igp in range(N_GP):
            n_total += 1
            f_sign = sign_dir    / f'sign_{args.sign_tag}_{ig:03d}_{igp:03d}.npz'
            f_trot = trotter_dir / f'trotter_4_{ig:03d}_{igp:03d}.npz'
            if not f_sign.exists() or not f_trot.exists():
                continue

            d_s = np.load(f_sign)
            d_t = np.load(f_trot)

            sign_init[ig, igp] = float(d_s['sign_init'])
            sign_opt [ig, igp] = float(d_s['sign_opt'])
            fra_opt  [ig, igp] = float(d_t['framability'])

            gamma   = float(d_s['gamma'])
            gamma_p = float(d_s['gamma_p'])
            R_vec   = np.asarray(d_s['n_opt'], dtype=float)
            D_fra   = np.asarray(d_t['D'],     dtype=float)

            # Rebuild U (same recipe as the sign-problem worker).
            L    = build_lindbladian(args.J, gamma, gamma_p, args.h)
            gate = expm(L * args.dt).real

            s_via_D  [ig, igp] = _sign_in_frame_D(gate, D_fra)
            fra_via_R[ig, igp] = _fra_from_unitary_rotation(gate, R_vec)

            n_done += 1
            if n_done % 50 == 0 or n_done == 1:
                elapsed = time.perf_counter() - t0
                eta = elapsed / n_done * (N_GAMMA * N_GP - n_done)
                print(f'[{n_done}/{N_GAMMA * N_GP}]  (ig={ig:02d}, igp={igp:02d})  '
                      f's_opt={sign_opt[ig,igp]:.4f}  s_viaD={s_via_D[ig,igp]:.4f}  '
                      f'fra_opt={fra_opt[ig,igp]:.4f}  fra_viaR={fra_via_R[ig,igp]:.4f}  '
                      f'eta={eta/60:.1f} min', flush=True)

    # Improved (final) values: minimum of the two methods.
    s_final   = np.fmin(sign_opt, s_via_D)
    fra_final = np.fmin(fra_opt,  fra_via_R)

    # ── save raw data ───────────────────────────────────────────────────────
    gamma_vals = GAMMA_STEP * np.arange(N_GAMMA)
    gp_vals    = GAMMA_STEP * np.arange(N_GP)
    npz_path = out_dir / 'sign_fra_crosscheck_nofield.npz'
    np.savez(npz_path,
             gamma_values = gamma_vals,
             gp_values    = gp_vals,
             sign_init    = sign_init,
             sign_opt     = sign_opt,
             fra_opt      = fra_opt,
             s_via_D      = s_via_D,
             fra_via_R    = fra_via_R,
             s_final      = s_final,
             fra_final    = fra_final,
             J = args.J, h = args.h, dt = args.dt)
    print(f'Saved {npz_path}')

    # ── 2x2 plot: rows = sign / fra, cols = before / after ──────────────────
    def _panel(ax, data, title, *, vmin, vmax, mark_one=True):
        gp_edges = np.append(gp_vals - GAMMA_STEP / 2, gp_vals[-1] + GAMMA_STEP / 2)
        g_edges  = np.append(gamma_vals - GAMMA_STEP / 2, gamma_vals[-1] + GAMMA_STEP / 2)
        im = ax.pcolormesh(gp_edges, g_edges, data, cmap='viridis',
                           vmin=vmin, vmax=vmax, shading='flat')
        if mark_one and np.any(np.isfinite(data)):
            finite = data[np.isfinite(data)]
            if finite.min() < 1.0 < finite.max():
                try:
                    ax.contour(gp_vals, gamma_vals, data,
                               levels=[1.0], colors='red', linewidths=1.2)
                except Exception:
                    pass
        plt.colorbar(im, ax=ax)
        ax.set_xlabel(r"$\gamma'$")
        ax.set_ylabel(r'$\gamma$')
        ax.set_title(title, fontsize=10)
        n_pts = int(np.sum(np.isfinite(data)))
        ax.text(0.98, 0.98, f'{n_pts}/{N_GAMMA * N_GP}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=8, color='white')

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        r'No-field two-qubit Trotter step  $e^{L\,dt}$  '
        r'($J=1$, $h=0$, $dt=0.01$):  cross-validation  ($d_\mathrm{ext}=4$)',
        fontsize=12,
    )

    sign_all = np.concatenate([sign_opt.ravel(), s_via_D.ravel(), s_final.ravel()])
    fra_all  = np.concatenate([fra_opt.ravel(),  fra_via_R.ravel(), fra_final.ravel()])
    s_lo, s_hi = (float(np.nanmin(sign_all)), float(np.nanmax(sign_all)))
    f_lo, f_hi = (float(np.nanmin(fra_all)),  float(np.nanmax(fra_all)))

    _panel(axes[0, 0], sign_opt,
           r's_opt  (sign-problem optimiser)', vmin=s_lo, vmax=s_hi)
    _panel(axes[0, 1], s_final,
           r's_final = min(s_opt, s via D_fra)', vmin=s_lo, vmax=s_hi)
    _panel(axes[1, 0], fra_opt,
           r'fra_opt  (framability optimiser)', vmin=f_lo, vmax=f_hi)
    _panel(axes[1, 1], fra_final,
           r'fra_final = min(fra_opt, fra via R_sign)',
           vmin=f_lo, vmax=f_hi)

    fig.tight_layout()
    png_path = out_dir / 'sign_fra_crosscheck_nofield.png'
    fig.savefig(png_path, dpi=170)
    plt.close(fig)
    print(f'Saved {png_path}')

    # Summary stats.
    def _stats(name, arr):
        v = arr[np.isfinite(arr)]
        if len(v):
            print(f'  {name:12s}  min={v.min():.4f}  max={v.max():.4f}  '
                  f'median={np.median(v):.4f}  n={len(v)}')

    print('-- sign --')
    _stats('s_opt',    sign_opt)
    _stats('s_via_D',  s_via_D)
    _stats('s_final',  s_final)
    print('-- fra --')
    _stats('fra_opt',   fra_opt)
    _stats('fra_via_R', fra_via_R)
    _stats('fra_final', fra_final)

    s_better = int(np.nansum(s_via_D   < sign_opt - 1e-6))
    f_better = int(np.nansum(fra_via_R < fra_opt  - 1e-6))
    print(f'sign improved by fra frame at {s_better} points')
    print(f'fra  improved by sign frame at {f_better} points')


if __name__ == '__main__':
    main()
