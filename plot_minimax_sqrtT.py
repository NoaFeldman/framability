"""
Plot per-gate framability vs p for the sqrtT minimax results,
combined with the depol_sweep data (Pauli, extended-Pauli frames).

Panel 1: standalone sqrtT minimax per-gate plot (dashed lines for visibility)
Panel 2: unified plot with depol_sweep frame data + minimax overlay
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load minimax sqrtT results ────────────────────────────────────────────────
MINIMAX_DIR = Path('results_minimax_frame_sqrtT')
MINIMAX_HCT_DIR = Path('results_minimax_frame')
SWEEP_DIR = Path('results_depol_sweep')
D_EXT_SINGLES = [4, 6, 8]
P_MINIMAX = np.array([0.01 * i for i in range(11)])
N_P_MINIMAX = len(P_MINIMAX)

def load_minimax(in_dir, d_ext_singles=D_EXT_SINGLES, n_p=N_P_MINIMAX):
    """Load minimax results. Returns dict[d_ext] -> {gates, per_gate (n_gates, n_p), worst (n_p,)}"""
    results = {}
    for d_ext in d_ext_singles:
        gate_names = None
        worst = np.full(n_p, np.nan)
        per_gate = None
        for pi in range(n_p):
            f = in_dir / f'minimax_{d_ext}_{pi:02d}.npz'
            if not f.exists():
                continue
            data = np.load(f, allow_pickle=True)
            if gate_names is None:
                gate_names = list(data['gates'])
                per_gate = np.full((len(gate_names), n_p), np.nan)
            fra = data['framability']
            worst[pi] = data['worst']
            for gi in range(len(gate_names)):
                per_gate[gi, pi] = fra[gi]
        if gate_names is not None:
            results[d_ext] = dict(gates=gate_names, per_gate=per_gate, worst=worst)
    return results

minimax_sqrtT = load_minimax(MINIMAX_DIR)
minimax_HCT = load_minimax(MINIMAX_HCT_DIR)

# ── Load depol_sweep data ────────────────────────────────────────────────────
sweep = np.load(SWEEP_DIR / 'depol_sweep.npz', allow_pickle=True)
sweep_gates = list(sweep['gates'])
sweep_p = sweep['p_values']
sweep_fra = sweep['framability']  # (n_gates, n_p, n_frames)
sweep_labels = list(sweep['frame_labels'])

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1: Standalone sqrtT minimax per-gate (with dashed lines)
# ──────────────────────────────────────────────────────────────────────────────
GATE_STYLES = {'H': '--', 'CNOT': '-', 'sqrtT': '-.', 'T': ':'}
GATE_COLORS = {'H': 'tab:blue', 'CNOT': 'tab:orange', 'sqrtT': 'tab:green', 'T': 'tab:red'}
GATE_MARKERS = {'H': 's', 'CNOT': 'o', 'sqrtT': '^', 'T': 'D'}

fig1, axes1 = plt.subplots(1, len(D_EXT_SINGLES), figsize=(5 * len(D_EXT_SINGLES), 4.5),
                            sharey=True, sharex=True)

for di, d_ext in enumerate(D_EXT_SINGLES):
    ax = axes1[di]
    if d_ext in minimax_sqrtT:
        res = minimax_sqrtT[d_ext]
        for gi, g in enumerate(res['gates']):
            ax.plot(P_MINIMAX, res['per_gate'][gi],
                    linestyle=GATE_STYLES.get(g, '-'),
                    marker=GATE_MARKERS.get(g, 'o'),
                    color=GATE_COLORS.get(g, f'C{gi}'),
                    label=g, markersize=5, linewidth=1.8)
    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_xlabel(r'depolarisation $p$')
    ax.set_title(fr'$d_{{\rm ext,single}} = {d_ext}$')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

axes1[0].set_ylabel('Framability (per gate, minimax S)')
fig1.suptitle(r'Per-gate framability vs $p$ — minimax over $\{H, \mathrm{CNOT}, \sqrt{T}\}$')
fig1.tight_layout(rect=(0, 0, 1, 0.94))
out1 = Path('results_plots') / 'minimax_sqrtT_per_gate.png'
out1.parent.mkdir(parents=True, exist_ok=True)
fig1.savefig(out1, dpi=170)
plt.close(fig1)
print(f'[saved] {out1}')

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2: Unified — rows = gates, cols = [framability, op bond entropy]
# Overlay: depol_sweep frames + minimax worst-case for both gate sets
# ──────────────────────────────────────────────────────────────────────────────
n_g = len(sweep_gates)
obe = sweep.get('operator_bond_entropy', np.full((n_g, len(sweep_p)), np.nan))

FRAME_COLORS = ['tab:blue', 'tab:orange', 'tab:olive']
FRAME_STYLES = ['-', '--', (0, (5, 2))]
FRAME_MARKERS = ['o', 's', 'P']
FRAME_LW = [2.6, 1.6, 1.6]

# Minimax overlay colors per d_ext_single
D_COLORS = {4: 'tab:green', 6: 'tab:red', 8: 'tab:purple'}
D_MARKERS = {4: '^', 6: 'D', 8: 'v'}

fig2, axes2 = plt.subplots(n_g, 2, figsize=(13, 4.0 * n_g), sharex=True)
if n_g == 1:
    axes2 = axes2[np.newaxis, :]

for ig, gate in enumerate(sweep_gates):
    ax = axes2[ig, 0]

    # Depol sweep frame lines
    for jf, lbl in enumerate(sweep_labels):
        ax.plot(sweep_p, sweep_fra[ig, :, jf] ** 2,
                linestyle=FRAME_STYLES[jf],
                marker=FRAME_MARKERS[jf],
                linewidth=FRAME_LW[jf],
                markersize=6,
                color=FRAME_COLORS[jf], label=lbl)

    # Minimax {H,CNOT,T} overlay (worst-case)
    for d_ext in D_EXT_SINGLES:
        if d_ext in minimax_HCT:
            res = minimax_HCT[d_ext]
            # Trim to sweep_p range
            mask = P_MINIMAX <= sweep_p[-1] + 1e-9
            ax.plot(P_MINIMAX[mask], res['worst'][mask] ** 2,
                    linestyle='--', marker=D_MARKERS[d_ext],
                    color=D_COLORS[d_ext], markersize=5, linewidth=1.5,
                    label=fr'Minimax $\{{H,T,CNOT\}}$ $d={d_ext}$')

    # Minimax {H,CNOT,sqrtT} overlay (worst-case)
    for d_ext in D_EXT_SINGLES:
        if d_ext in minimax_sqrtT:
            res = minimax_sqrtT[d_ext]
            mask = P_MINIMAX <= sweep_p[-1] + 1e-9
            ax.plot(P_MINIMAX[mask], res['worst'][mask] ** 2,
                    linestyle='-.', marker=D_MARKERS[d_ext],
                    color=D_COLORS[d_ext], markersize=5, linewidth=1.5, alpha=0.7,
                    label=fr'Minimax $\{{H,CNOT,\sqrt{{T}}\}}$ $d={d_ext}$')

    ax.axhline(1.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_ylabel(r'Framability$^2$')
    ax.set_title(f'{gate}: framability$^2$')
    ax.grid(alpha=0.3)
    if ig == 0:
        ax.legend(fontsize=6.5, ncol=2)

    # Op bond entropy
    axes2[ig, 1].plot(sweep_p, obe[ig], 'o-', color='tab:brown')
    axes2[ig, 1].set_ylabel('Operator bond entropy')
    axes2[ig, 1].set_title(f'{gate}: op. bond entropy')
    axes2[ig, 1].grid(alpha=0.3)

    for ax_ in axes2[ig]:
        if ig == n_g - 1:
            ax_.set_xlabel(r'depolarisation $p$')

fig2.suptitle('Depolarised gates: framability$^2$ and channel quantities vs. $p$\n'
              '(H, T, $\\sqrt{T}$ lifted to 2 qubits as G$\\otimes$I)')
fig2.tight_layout(rect=(0, 0, 1, 0.96))

out2 = Path('results_plots') / 'depol_sweep_unified.png'
fig2.savefig(out2, dpi=170)
plt.close(fig2)
print(f'[saved] {out2}')

# ── Print S matrix info ──────────────────────────────────────────────────────
for label, d in [('sqrtT', MINIMAX_DIR), ('H_CNOT_T', MINIMAX_HCT_DIR)]:
    f0 = d / 'minimax_6_00.npz'
    if f0.exists():
        data = np.load(f0, allow_pickle=True)
        np.set_printoptions(precision=6, suppress=True)
        print(f'\n{label} — S matrix at p=0, d_ext_single=6:')
        print(data['S'])
        print(f'  worst: {data["worst"]:.6f}, per-gate: {data["framability"]}, gates: {data["gates"]}')
