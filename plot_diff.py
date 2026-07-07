"""Plot min_framability - pauli_framability from scan results."""
import numpy as np
import matplotlib.pyplot as plt

n = 20
gs = 0.1
data = np.stack([np.load(f'results/row_{ig:04d}.npy') for ig in range(n)])

diff = data[:, :, 5] - data[:, :, 3]  # min_fra - pauli_fra

gammas = [gs * i for i in range(n)]
extent = [gammas[0], gammas[-1], gammas[0], gammas[-1]]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(diff, origin='lower', aspect='auto', extent=extent)
ax.set_xlabel(r"$\gamma'$")
ax.set_ylabel(r"$\gamma$")
ax.set_title(r"Min framability $-$ Pauli framability")
fig.colorbar(im, ax=ax)
plt.tight_layout()
import os
os.makedirs('results_plots', exist_ok=True)
plt.savefig('results_plots/min_fra_minus_pauli_fra.png', dpi=150)
plt.show()
print(f'Range: [{diff.min():.6f}, {diff.max():.6f}]')
print(f'Points where min_fra > pauli_fra: {np.sum(diff > 1e-9)} / {diff.size}')
print('Saved to results_plots/min_fra_minus_pauli_fra.png')
