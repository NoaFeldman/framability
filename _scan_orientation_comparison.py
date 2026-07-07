"""Compare Heisenberg vs Schroedinger chi030 in scan-plot orientation."""
import numpy as np
import matplotlib.pyplot as plt

sf    = np.load('results/scan_full.npy')   # (41, 41, 14)
heis  = sf[:, :, 9]    # product_fra_heis  (y=gamma, x=gamma')
schro = sf[:, :, 13]   # product_fra_schro chi030

ext  = [0, 8, 0, 8]
vmin = min(heis[np.isfinite(heis)].min(), schro.min())
vmax = max(heis[np.isfinite(heis)].max(), schro.max())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

kw = dict(origin='lower', aspect='auto', extent=ext,
          vmin=vmin, vmax=vmax, cmap='viridis')

im0 = axes[0].imshow(heis,  **kw)
axes[0].set_title('Heisenberg product-state (col 9)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(schro, **kw)
axes[1].set_title(r'Schrödinger product-state $\chi=30$ (col 13)')
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(schro - heis, origin='lower', aspect='auto',
                     extent=ext, cmap='RdBu_r')
axes[2].set_title('Schrödinger − Heisenberg')
plt.colorbar(im2, ax=axes[2])

for ax in axes:
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r'$\gamma$')

plt.tight_layout()
plt.savefig('results_plots/scan_orientation_comparison.png', dpi=150)
print('Saved results_plots/scan_orientation_comparison.png')
