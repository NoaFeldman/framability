"""
Plot optimised framability for random two-qubit gates
    U = exp(i*(alpha*XX + beta*YY + gamma*ZZ))
from results_random_kron/.

Only d_ext_single=4 results are used; d=6 results are all 1e6 (LP failed,
needs rerun with use_complex=False).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='results_random_kron')
parser.add_argument('--out',    default='results_plots/random_kron.png')
args = parser.parse_args()

# ── load ─────────────────────────────────────────────────────────────────────
records = []
for f in sorted(Path(args.in_dir).glob('random_kron_4_*.npz')):
    d = np.load(f)
    fra = float(d['framability'])
    if fra > 1e5:
        continue
    records.append(dict(
        fra=fra,
        alpha=float(d['alpha']),
        beta=float(d['beta']),
        gamma=float(d['gamma']),
        label=f.stem.split('_')[-1],
    ))

records.sort(key=lambda r: r['fra'])
fra    = np.array([r['fra']   for r in records])
alphas = np.array([r['alpha'] for r in records])
betas  = np.array([r['beta']  for r in records])
gammas = np.array([r['gamma'] for r in records])
labels = [r['label'] for r in records]
n = len(records)

cmap   = cm.viridis
colors = cmap(np.linspace(0.15, 0.85, n))

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32)

angle_axes = [
    (fig.add_subplot(gs[0, 0]), alphas, r'$\alpha$', 'tab:blue'),
    (fig.add_subplot(gs[0, 1]), betas,  r'$\beta$',  'tab:orange'),
    (fig.add_subplot(gs[1, 0]), gammas, r'$\gamma$', 'tab:green'),
]

for ax, angles, label, c in angle_axes:
    ax.scatter(angles, fra, c=c, s=60, zorder=3)
    for x, y, lbl in zip(angles, fra, labels):
        ax.annotate(lbl, (x, y), textcoords='offset points',
                    xytext=(5, 3), fontsize=7, color='0.4')
    ax.set_xlabel(label + r'  (rad)', fontsize=11)
    ax.set_ylabel(r'framability', fontsize=11)
    ax.grid(alpha=0.3)
    # reference line at 1
    ax.axhline(1.0, color='k', lw=0.8, ls=':')

# bottom-right: sorted bar chart with angle triples annotated
ax_bar = fig.add_subplot(gs[1, 1])
x_pos  = np.arange(n)
bars   = ax_bar.bar(x_pos, fra, color=colors, width=0.6, zorder=3)
ax_bar.axhline(1.0, color='k', lw=0.8, ls=':')

for xi, r in zip(x_pos, records):
    ax_bar.text(xi, r['fra'] + 0.05,
                fr'$({r["alpha"]:.2f},{r["beta"]:.2f},{r["gamma"]:.2f})$',
                ha='center', va='bottom', fontsize=6, rotation=45)

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(labels)
ax_bar.set_xlabel('sample (sorted by framability)', fontsize=11)
ax_bar.set_ylabel(r'framability', fontsize=11)
ax_bar.grid(axis='y', alpha=0.3)

fig.suptitle(
    r'Optimised framability of $e^{i(\alpha XX+\beta YY+\gamma ZZ)}$'
    '\n'
    r'$d_{\rm ext,single}=4$,  10 random angle triples drawn from $[0,\pi/2)^3$',
    fontsize=12,
)

fig.savefig(args.out, dpi=170, bbox_inches='tight')
plt.close(fig)
print(f'Saved {args.out}')
print(f'\nData summary (sorted by framability):')
print(f'  {"sample":>6}  {"alpha":>7}  {"beta":>7}  {"gamma":>7}  {"fra":>9}')
for r in records:
    print(f'  {r["label"]:>6}  {r["alpha"]:7.3f}  {r["beta"]:7.3f}  '
          f'{r["gamma"]:7.3f}  {r["fra"]:9.4f}')
