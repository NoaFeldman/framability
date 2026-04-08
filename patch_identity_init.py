"""
Re-run gradient descent for data points where min_fra > pauli_fra,
seeding from the cycling-identity frame (which guarantees an initial
framability equal to pauli_framability, so the optimizer can only
improve from there).

Usage
-----
    python patch_identity_init.py [--n_pts 20] [--J 1.0]
                                  [--gamma_step 0.1] [--out_dir results]
                                  [--n_restarts 5] [--maxfev 1000]
"""

import argparse
import os

import numpy as np
from scipy.linalg import expm

from two_qubit_lindbladian import numeric_two_qubit_lindbladian
from framability import extended_pauli_D
from optimize_framability import minimize_framability, DEFAULT_METHOD

COL_PAULI_FRA = 3
COL_EXT_FRA   = 4
COL_MIN_FRA   = 5


def _make_gate(J, gamma, gamma_p, gamma_step):
    L = numeric_two_qubit_lindbladian(J=J, gamma=gamma, gamma_p=gamma_p)
    dt = 0.01 * gamma_step
    return expm(dt * L).real


def main():
    p = argparse.ArgumentParser(
        description='Patch data points where min_fra > pauli_fra.'
    )
    p.add_argument('--n_pts',      type=int,   default=20)
    p.add_argument('--J',          type=float, default=1.0)
    p.add_argument('--gamma_step', type=float, default=0.1)
    p.add_argument('--out_dir',    type=str,   default='results')
    p.add_argument('--n_restarts', type=int,   default=5)
    p.add_argument('--maxfev',     type=int,   default=1000)
    args = p.parse_args()

    n         = args.n_pts
    gs        = args.gamma_step
    out_dir   = args.out_dir
    d_ext     = extended_pauli_D().shape[1]   # 36

    # Load combined array
    combined = os.path.join(out_dir, 'scan_full.npy')
    data = np.load(combined)   # (n, n, 8)

    min_fra   = data[:, :, COL_MIN_FRA]
    pauli_fra = data[:, :, COL_PAULI_FRA]
    ext_fra   = data[:, :, COL_EXT_FRA]

    bad = [(ig, igp)
           for ig in range(n) for igp in range(n)
           if min_fra[ig, igp] > pauli_fra[ig, igp] + 1e-9]

    print(f'{len(bad)} data point(s) with min_fra > pauli_fra — re-optimising ...')

    n_improved = 0
    for k, (ig, igp) in enumerate(bad):
        gamma   = gs * ig
        gamma_p = gs * igp
        old_val = min_fra[ig, igp]

        gate = _make_gate(args.J, gamma, gamma_p, gs)

        # minimize_framability now always seeds with the cycling-identity
        # init (index 1, after ext-Pauli), so no extra_init_xs needed.
        _, f_new = minimize_framability(
            gate, d_ext=d_ext, mode='kronecker',
            n_restarts=args.n_restarts, method=DEFAULT_METHOD,
            maxfev=args.maxfev, verbose=False,
        )

        # Safety clamp
        f_new = min(f_new, ext_fra[ig, igp])

        tag = f'  [{k+1}/{len(bad)}] ({ig:2d},{igp:2d})  old={old_val:.6f}  new={f_new:.6f}'
        if f_new < old_val - 1e-9:
            data[ig, igp, COL_MIN_FRA] = f_new
            row_path = os.path.join(out_dir, f'row_{ig:04d}.npy')
            np.save(row_path, data[ig])
            n_improved += 1
            print(tag + f'  ✓ Δ={old_val - f_new:.6f}')
        else:
            print(tag + '  (no improvement)')

    print(f'\nImproved {n_improved}/{len(bad)} point(s).')

    # Always rewrite scan_full.npy with the current state
    np.save(combined, data)
    print(f'Updated {combined}')

    # Regenerate the diff plot
    import matplotlib.pyplot as plt

    diff = data[:, :, COL_MIN_FRA] - data[:, :, COL_PAULI_FRA]
    vmax = np.abs(diff).max()

    gammas   = np.arange(n) * gs
    gamma_ps = np.arange(n) * gs
    extent = [gamma_ps[0] - gs/2, gamma_ps[-1] + gs/2,
              gammas[0]   - gs/2, gammas[-1]   + gs/2]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(diff, origin='lower', aspect='auto', extent=extent,
                   cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('min framability \N{MINUS SIGN} Pauli framability')
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title('Min framability \N{MINUS SIGN} Pauli framability (after patching)')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, 'min_minus_pauli_framability.png')
    plt.savefig(fig_path, dpi=150)
    print(f'Figure saved to {fig_path}')
    plt.show()


if __name__ == '__main__':
    main()
