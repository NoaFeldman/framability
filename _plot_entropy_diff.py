import numpy as np
import matplotlib.pyplot as plt

d = np.load('results/scan_full.npy')
diff = d[:, :, 12] - d[:, :, 11]  # steady state - max

gs = 0.1
n = d.shape[0]
extent = [0, gs * (n - 1), 0, gs * (n - 1)]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(diff, origin='lower', aspect='auto', extent=extent)
ax.set_xlabel(r"$\gamma'$")
ax.set_ylabel(r"$\gamma$")
ax.set_title('Steady state bond entropy $-$ Max bond entropy')
fig.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig('results_plots/bond_entropy_diff.png', dpi=150)
print('Saved results_plots/bond_entropy_diff.png')
print(f'min: {diff.min():.6f}  max: {diff.max():.6f}')
