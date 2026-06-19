"""
gamma_{CH_1} of gate-images over the single-qubit product frame, with
interactive 3D sphere visualisations.

Conventions (compatible with analysis.py / framability.py)
----------------------------------------------------------
- Two-qubit Pauli basis order: [I, X, Y, Z] (x) [I, X, Y, Z], 16 strings,
  flat index = 4*i + j with i the qubit-1 Pauli and j the qubit-2 Pauli.
- A Hermitian operator O is represented by its Pauli-coefficient vector
  c_mu = Tr(P_mu O) / 4, so that O = sum_mu c_mu P_mu.
- A gate is the 16x16 real Pauli-transfer matrix (PTM) acting on those
  coefficient vectors: c' = gate @ c is the image operator's coeff vector.
  (This is the same object as expm(dt*L) elsewhere in the repo.)

Product-frame gauge (optimal single-qubit product frame, F_2 = F_1 (x) F_1):

  gamma_{CH_1}(y) = |y_II|
                  + || y_{sigma (x) I} ||_2          (qubit-1 local block)
                  + || y_{I (x) sigma} ||_2          (qubit-2 local block)
                  + || y_{sigma (x) sigma} ||_*       (nuclear norm of the
                                                       3x3 correlation block)

Single-qubit operators here are the TRACELESS Bloch representation
O = b_x X + b_y Y + b_z Z, so ||O||_inf = |b| and a "Bloch sphere of radius r"
is { r * n : |n| = 1 }. (Change bloch_operator if you want the I/2 component.)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Pauli bookkeeping
# ---------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS_1Q = [_I, _X, _Y, _Z]

# Two-qubit Pauli strings, flat index = 4*i + j
PAULIS_2Q = [np.kron(p1, p2) for p1 in PAULIS_1Q for p2 in PAULIS_1Q]


def pauli_coeffs(op):
    """16-dim real Pauli-coefficient vector of a 4x4 Hermitian operator."""
    op = np.asarray(op, dtype=complex)
    c = np.array([np.trace(P @ op) / 4 for P in PAULIS_2Q])
    if np.max(np.abs(c.imag)) > 1e-9:
        raise ValueError("Operator is not Hermitian (complex Pauli coeffs).")
    return c.real


def bloch_operator(bvec):
    """Single-qubit traceless operator b_x X + b_y Y + b_z Z from a Bloch vector."""
    bvec = np.asarray(bvec, dtype=float)
    return bvec[0] * _X + bvec[1] * _Y + bvec[2] * _Z


def unitary_to_ptm(U):
    """16x16 real PTM of a 4x4 unitary: G_{mu,nu} = Tr(P_mu U P_nu U^dag) / 4."""
    U = np.asarray(U, dtype=complex)
    Udag = U.conj().T
    G = np.empty((16, 16))
    for mu, Pmu in enumerate(PAULIS_2Q):
        for nu, Pnu in enumerate(PAULIS_2Q):
            G[mu, nu] = (np.trace(Pmu @ U @ Pnu @ Udag) / 4).real
    return G


def cnot_ptm():
    """PTM of CNOT (control = qubit 1, target = qubit 2)."""
    U = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=complex)
    return unitary_to_ptm(U)


# ---------------------------------------------------------------------------
# 1. gamma_{CH_1}
# ---------------------------------------------------------------------------
def gamma_CH1(coeffs):
    """Product-frame gauge of an operator given its 16-dim Pauli-coeff vector."""
    c = np.asarray(coeffs, dtype=float).reshape(4, 4)   # c[i, j], i,j in {I,X,Y,Z}
    g_II = abs(c[0, 0])
    loc1 = np.linalg.norm(c[1:, 0])                     # sigma (x) I : column 0
    loc2 = np.linalg.norm(c[0, 1:])                     # I (x) sigma : row 0
    nuc = np.linalg.norm(c[1:, 1:], ord='nuc')          # correlation: nuclear norm
    return float(g_II + loc1 + loc2 + nuc)


def gamma_CH1_image(gate, rho):
    """gamma_{CH_1}(gate @ rho): gauge of the gate-image of operator rho.

    Parameters
    ----------
    gate : (16, 16) real PTM.
    rho  : (4, 4) Hermitian operator OR its 16-dim Pauli-coefficient vector.
    """
    rho = np.asarray(rho)
    if rho.shape == (4, 4):
        c = pauli_coeffs(rho)
    else:
        c = np.asarray(rho, dtype=float).ravel()
    return gamma_CH1(np.asarray(gate, dtype=float) @ c)


# ---------------------------------------------------------------------------
# 2. Sweep rho_2 over the Bloch sphere of radius r (rho_1 fixed)
# ---------------------------------------------------------------------------
def sphere_gamma_grid(gate, rho1, r, theta_step):
    """gamma_{CH_1}(gate @ (rho1 (x) rho2)) for rho2 = r * n over the sphere.

    Parameters
    ----------
    gate : (16, 16) real PTM.
    rho1 : (2, 2) single-qubit operator (qubit 1, held fixed).
    r    : Bloch radius of rho2 (<= 1).
    theta_step : angular step (radians) used for both theta and phi.

    Returns
    -------
    TH, PH : (n_theta, n_phi) meshgrids of polar / azimuthal angle.
    (nx, ny, nz) : unit-sphere direction components on the same grid.
    gamma : (n_theta, n_phi) array of gamma_{CH_1} values.
    """
    thetas = np.arange(0.0, np.pi + 1e-9, theta_step)
    phis = np.arange(0.0, 2 * np.pi + 1e-9, theta_step)
    TH, PH = np.meshgrid(thetas, phis, indexing='ij')
    nx = np.sin(TH) * np.cos(PH)
    ny = np.sin(TH) * np.sin(PH)
    nz = np.cos(TH)
    rho1 = np.asarray(rho1, dtype=complex)

    gamma = np.zeros_like(TH)
    for a in range(TH.shape[0]):
        for b in range(TH.shape[1]):
            rho2 = r * (nx[a, b] * _X + ny[a, b] * _Y + nz[a, b] * _Z)
            op = np.kron(rho1, rho2)
            gamma[a, b] = gamma_CH1_image(gate, op)
    return TH, PH, (nx, ny, nz), gamma


def plot_sphere_gamma(gate, rho1, r, theta_step, html_path=None, title=None):
    """Single interactive 3D colormap of gamma_{CH_1} over the rho2 sphere."""
    import plotly.graph_objects as go
    _, _, (nx, ny, nz), gamma = sphere_gamma_grid(gate, rho1, r, theta_step)
    fig = go.Figure(go.Surface(
        x=r * nx, y=r * ny, z=r * nz,
        surfacecolor=gamma, colorscale='Viridis',
        colorbar=dict(title='gamma_CH1'),
    ))
    fig.update_layout(
        title=title or f'gamma_CH1 over rho2 sphere (r={r:.3f})',
        scene=dict(aspectmode='data', xaxis_title='x', yaxis_title='y',
                   zaxis_title='z'),
    )
    if html_path:
        fig.write_html(html_path)
        print(f'wrote {html_path}')
    return fig, gamma


# ---------------------------------------------------------------------------
# 3. Random panel: 4 random rho_1 x radii [r, 1]  ->  8 interactive plots
# ---------------------------------------------------------------------------
def random_panel(gate, theta_step, seed=0, out_dir='.', gate_name='gate'):
    """Draw 4 random rho_1 in the Bloch ball and one r < 1, then render
    gamma_{CH_1}(gate @ (rho1 (x) rho2)) over the rho2 sphere for each rho_1
    at radii [r, 1].

    Writes 8 standalone interactive HTML files (one per rho1 x radius) plus a
    combined 4x2 overview, all into out_dir.
    """
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 4 random rho_1 uniformly in the Bloch ball (uniform direction, r^(1/3) radius)
    rho1_blochs = []
    for _ in range(4):
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        rad = rng.uniform() ** (1.0 / 3.0)
        rho1_blochs.append(rad * d)

    r = float(rng.uniform(0.0, 1.0))            # one radius r < 1
    radii = [r, 1.0]

    titles = []
    for i, b1 in enumerate(rho1_blochs):
        for rr in radii:
            titles.append(f"rho1#{i} (|b|={np.linalg.norm(b1):.2f}), r={rr:.2f}")

    fig = make_subplots(
        rows=4, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}] for _ in range(4)],
        subplot_titles=titles,
        horizontal_spacing=0.04, vertical_spacing=0.04,
    )

    for i, b1 in enumerate(rho1_blochs):
        rho1 = bloch_operator(b1)
        for k, rr in enumerate(radii):
            _, _, (nx, ny, nz), gamma = sphere_gamma_grid(gate, rho1, rr, theta_step)

            # standalone interactive plot (with its own colorbar)
            ind = go.Figure(go.Surface(
                x=rr * nx, y=rr * ny, z=rr * nz,
                surfacecolor=gamma, colorscale='Viridis',
                colorbar=dict(title='gamma_CH1')))
            ind.update_layout(
                title=f"{gate_name}: gamma_CH1, rho1#{i}, r={rr:.3f}",
                scene=dict(aspectmode='data'))
            ind_path = os.path.join(out_dir, f"{gate_name}_rho{i}_r{rr:.3f}.html")
            ind.write_html(ind_path)
            print(f'wrote {ind_path}  (gamma in [{gamma.min():.3f}, {gamma.max():.3f}])')

            # combined-panel cell (per-cell colour scale, scalebar suppressed)
            fig.add_trace(go.Surface(
                x=rr * nx, y=rr * ny, z=rr * nz,
                surfacecolor=gamma, colorscale='Viridis', showscale=False,
                cmin=float(gamma.min()), cmax=float(gamma.max())),
                row=i + 1, col=k + 1)

    fig.update_layout(
        height=1600, width=1100,
        title=f"{gate_name}: gamma_CH1 over rho2 spheres "
              f"(4 random rho1 x radii [{r:.2f}, 1.0], seed={seed})")
    combined = os.path.join(out_dir, f"{gate_name}_panel.html")
    fig.write_html(combined)
    print(f'wrote {combined}')
    return combined


# ---------------------------------------------------------------------------
# Cluster entry point
# ---------------------------------------------------------------------------
def build_gate(name):
    if name == 'cnot':
        return cnot_ptm()
    if name == 'identity':
        return np.eye(16)
    raise ValueError(f"unknown gate '{name}' (use 'cnot' or 'identity')")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gate', default='cnot', help="cnot | identity")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--theta-step', type=float, default=np.pi / 45,
                    help='angular step in radians (default pi/45 = 4 deg)')
    ap.add_argument('--out-dir', default='gamma_ch1_out')
    args = ap.parse_args()

    gate = build_gate(args.gate)
    random_panel(gate, args.theta_step, seed=args.seed,
                 out_dir=args.out_dir, gate_name=args.gate)


if __name__ == '__main__':
    main()
