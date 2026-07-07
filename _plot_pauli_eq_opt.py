"""Binary plot: 1 where optimized framability == Pauli framability (diff < 1e-6)."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

out_dir    = Path('results')
n_pts      = 41
gamma_step = 0.2

scan_full = np.load(out_dir / 'scan_full.npy')   # (41, 41, 14)
pauli_fra = scan_full[:, :, 2]
opt_fra   = scan_full[:, :, 3]

tol    = 1e-6
binary = (pauli_fra - opt_fra < tol).astype(float)   # 1 = equal, 0 = opt beat Pauli

gammas = gamma_step * np.arange(n_pts)
half   = gamma_step / 2
extent = [gammas[0] - half, gammas[-1] + half,
          gammas[0] - half, gammas[-1] + half]

fig, ax = plt.subplots(figsize=(6, 5))
cmap = mcolors.ListedColormap(['#d62728', '#2ca02c'])   # red=0, green=1
im = ax.imshow(binary, origin='lower', extent=extent, aspect='auto',
               cmap=cmap, vmin=0, vmax=1, interpolation='nearest')

cbar = fig.colorbar(im, ax=ax, ticks=[0.25, 0.75])
cbar.ax.set_yticklabels(['opt < Pauli\n(optimizer improved)', 'opt = Pauli\n(no improvement)'])

ax.set_xlabel(r"$\gamma'$")
ax.set_ylabel(r"$\gamma$")
ax.set_title("Optimized framability equals Pauli framability\n"
             r"(green: $|\mathcal{F}_\mathrm{opt} - \mathcal{F}_\mathrm{Pauli}| < 10^{-6}$)")

n_equal = int(binary.sum())
n_total = binary.size
ax.text(0.02, 0.02, f'{n_equal}/{n_total} equal',
        transform=ax.transAxes, fontsize=9, color='white',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

fig.tight_layout()
out_png = Path('results_plots') / 'two_qubit_pauli_eq_opt.png'
fig.savefig(out_png, dpi=170)
plt.close(fig)
print(f'[saved] {out_png}')
print(f'{n_equal}/{n_total} points where opt ≈ Pauli (tol={tol})')
