"""
Side-by-side comparison of Heisenberg vs Schrödinger product-state framability.
"""
import numpy as np
import matplotlib.pyplot as plt

N_PTS      = 41
GAMMA_STEP = 0.2

gammas = [GAMMA_STEP * i for i in range(N_PTS)]
extent = [gammas[0], gammas[-1], gammas[0], gammas[-1]]

heis  = np.load('results/product_fra_heisenberg.npy')
schro = np.load('results/product_fra_schroedinger.npy')

vmin = 1.0
vmax = max(np.nanmax(heis), np.nanmax(schro))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, data, title in zip(axes,
                            [heis, schro],
                            ['Heisenberg', 'Schrödinger']):
    im = ax.imshow(data, origin='lower', aspect='auto',
                   extent=extent, vmin=vmin, vmax=vmax)
    ax.contour(data, levels=[1.0 + 1e-6], colors='white',
               linewidths=0.8, extent=extent, origin='lower')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"$\gamma'$")
    ax.set_ylabel(r"$\gamma$")
    ax.set_title(f'Product-state framability ({title})')

fig.suptitle('Product-state framability: Heisenberg vs Schrödinger  (J = 1.0, χ = 6, seed = 42)',
             fontsize=13)
plt.tight_layout()
plt.savefig('results_plots/product_fra_comparison.png', dpi=150)
print('Saved results_plots/product_fra_comparison.png')
