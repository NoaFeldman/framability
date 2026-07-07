"""Quick comparison plot of Heisenberg vs Schrödinger product-state framability."""
import numpy as np
import matplotlib.pyplot as plt

extra = np.stack([
    np.stack([
        np.load(f'results/point_extra_{ig:04d}_{igp:04d}.npy')
        for igp in range(41)
    ])
    for ig in range(41)
])  # (41,41,7)

heis  = extra[:, :, 4]   # product_fra_heis
schro = extra[:, :, 5]   # product_fra_schro  (== chi030 within 5e-8)

vmin = min(heis[np.isfinite(heis)].min(), schro.min())
vmax = max(heis[np.isfinite(heis)].max(), schro.max())

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
kw = dict(origin='lower', aspect='auto', extent=[0, 8, 0, 8],
          vmin=vmin, vmax=vmax, cmap='viridis')

im0 = axes[0].imshow(heis.T,  **kw)
axes[0].set_title('Heisenberg (product)')
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(schro.T, **kw)
axes[1].set_title(r'Schrödinger (product, $\chi=30$)')
plt.colorbar(im1, ax=axes[1])

diff = schro - heis
im2 = axes[2].imshow(diff.T, origin='lower', aspect='auto',
                     extent=[0, 8, 0, 8], cmap='RdBu_r')
axes[2].set_title("Schrödinger − Heisenberg")
plt.colorbar(im2, ax=axes[2])

for ax in axes:
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\gamma'$")

plt.tight_layout()
plt.savefig('results_plots/heis_vs_schro_comparison.png', dpi=150)
print('Saved results_plots/heis_vs_schro_comparison.png')
