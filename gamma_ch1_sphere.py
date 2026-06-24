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

Single-qubit frame operators are identity-padded back onto the operator-norm
unit sphere:  O = (1 - |b|) I + b·sigma,  with eigenvalues {1, 1 - 2|b|} so
||O||_inf = 1 for |b| <= 1.  A sub-unit Bloch vector (|b| = r < 1) thus carries
an extra (1 - r) I; |b| = 1 recovers the traceless b·sigma.  See frame_op_1q.
"""

import numpy as np
from scipy.linalg import expm

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


def operator_bloch(op):
    """Bloch vector (X, Y, Z coefficients) of a single-qubit operator."""
    op = np.asarray(op, dtype=complex)
    return np.array([np.trace(_X @ op) / 2,
                     np.trace(_Y @ op) / 2,
                     np.trace(_Z @ op) / 2]).real


def bloch_expr(b):
    """Human-readable 'a X + b Y + c Z' string for a Bloch vector.

    Uses 4 decimals so the printed coefficients stay consistent with the true
    |b| (2-decimal rounding can make a unit vector look like |b| > 1).
    """
    return " ".join(f"{c:+.4f}{lab}" for c, lab in zip(np.asarray(b, float), "XYZ"))


def frame_op_1q(bvec):
    """Single-qubit frame operator (1 - |b|) I + b·sigma.

    For |b| <= 1 this has operator norm exactly 1 (eigenvalues 1 and 1 - 2|b|):
    a sub-unit Bloch vector is identity-padded back onto the operator-norm unit
    sphere.  |b| = 1 recovers the traceless b·sigma (no identity term).
    """
    b = np.asarray(bvec, dtype=float)
    r = np.linalg.norm(b)
    return (1.0 - r) * _I + b[0] * _X + b[1] * _Y + b[2] * _Z


def op1q_expr(op):
    """Full 'c0 I + bx X + by Y + bz Z' string for a single-qubit operator."""
    op = np.asarray(op, dtype=complex)
    c0 = (np.trace(op) / 2).real
    return f"{c0:+.4f}I " + bloch_expr(operator_bloch(op))


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


def depol_kron_ptm(alpha, beta, gamma):
    """PTM of U = exp(i*(alpha*XX + beta*YY + gamma*ZZ)).

    The entangling gate used in scripts/depol_kron_worker.py (its
    build_gate_superop with the same Tr(P_i U P_j U^dag)/4 convention).
    """
    XX = np.kron(_X, _X)
    YY = np.kron(_Y, _Y)
    ZZ = np.kron(_Z, _Z)
    U = expm(1j * (alpha * XX + beta * YY + gamma * ZZ))
    return unitary_to_ptm(U)


def depol_2q_ptm(p):
    """16x16 two-qubit depolarizing channel PTM (matches _depol_2q in
    sweep_depol_gates_worker).

    Diagonal in the Pauli basis: (1 - 4p) ** weight, where weight is the number
    of non-identity single-qubit factors of the Pauli string (II -> 1,
    weight-1 -> 1-4p, weight-2 -> (1-4p)^2). p = 0 gives the identity.
    """
    diag = np.array(
        [(1.0 - 4 * p) ** ((a != 0) + (b != 0))
         for a in range(4) for b in range(4)],
        dtype=float,
    )
    return np.diag(diag)


def exp_xy_ptm():
    """PTM of U = exp(i * pi * sqrt(2) * (X (x) Y))."""
    return unitary_to_ptm(expm(1j * np.pi * np.sqrt(2.0) * np.kron(_X, _Y)))


def exp_yx_ptm():
    """PTM of U = exp(i * pi * sqrt(2) * (Y (x) X))."""
    return unitary_to_ptm(expm(1j * np.pi * np.sqrt(2.0) * np.kron(_Y, _X)))


def lindbladian_ptm(jump):
    """16x16 superoperator of the dissipator
        D[S](rho) = S rho S^dag - 1/2 {S^dag S, rho},
    as L[i, j] = Tr(P_i D[S](P_j)) / 4 in the [I,X,Y,Z]^2 Pauli basis.
    """
    S = np.asarray(jump, dtype=complex)
    Sd = S.conj().T
    SdS = Sd @ S
    L = np.zeros((16, 16))
    for j, Pj in enumerate(PAULIS_2Q):
        res = S @ Pj @ Sd - 0.5 * (SdS @ Pj + Pj @ SdS)
        for i, Pi in enumerate(PAULIS_2Q):
            L[i, j] = (np.trace(Pi @ res) / 4).real
    return L


def cnot_diss_ptm(dt):
    """CNOT followed by one Euler dissipation step  I + dt * L.

    The Lindbladian L has a single collective jump operator
        S^- = sigma^- (x) I + I (x) sigma^-,   sigma^- = |0><1| = [[0, 1], [0, 0]]
    (decay |1> -> |0>).  Gate order: CNOT first, then (I + dt L).
    """
    sm = np.array([[0, 1], [0, 0]], dtype=complex)        # sigma^- = |0><1|
    S = np.kron(sm, _I) + np.kron(_I, sm)
    L = lindbladian_ptm(S)
    return (np.eye(16) + dt * L) @ cnot_ptm()


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
def sphere_gamma_grid(gate, rho1, r, theta_step, id2=0.0):
    """gamma_{CH_1}(gate @ (rho1 (x) rho2)) over the rho2 sphere.

    rho2 = id2 * I + r * (n . sigma), with |n| = 1 swept over the sphere.

    Parameters
    ----------
    gate : (16, 16) real PTM.
    rho1 : (2, 2) single-qubit operator (qubit 1, held fixed).
    r    : Bloch radius of rho2's traceless part.
    theta_step : angular step (radians) used for both theta and phi.
    id2  : identity coefficient of rho2 (caller ensures |id2| + r < 1).

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
            rho2 = id2 * _I + r * (nx[a, b] * _X + ny[a, b] * _Y + nz[a, b] * _Z)
            op = np.kron(rho1, rho2)
            gamma[a, b] = gamma_CH1_image(gate, op)
    return TH, PH, (nx, ny, nz), gamma


def plot_sphere_gamma(gate, rho1, r, theta_step, html_path=None, title=None):
    """Single interactive 3D colormap of gamma_{CH_1} over the rho2 sphere."""
    import plotly.graph_objects as go
    _, _, (nx, ny, nz), gamma = sphere_gamma_grid(gate, rho1, r, theta_step)
    rho1_str = op1q_expr(rho1)
    fig = go.Figure(go.Surface(
        x=r * nx, y=r * ny, z=r * nz,
        surfacecolor=gamma, colorscale='Viridis',
        colorbar=dict(title='gamma_CH1'),
    ))
    fig.update_layout(
        title=title or f'gamma_CH1 over rho2 sphere (r={r:.3f})<br>'
                       f'rho1 = {rho1_str}',
        scene=dict(aspectmode='data', xaxis_title='x', yaxis_title='y',
                   zaxis_title='z'),
    )
    if html_path:
        fig.write_html(html_path)
        print(f'wrote {html_path}')
    return fig, gamma


# ---------------------------------------------------------------------------
# 3. Panel: grid over rho_1 operators (rows) x rho_2 sphere radii (rs2, cols)
# ---------------------------------------------------------------------------
def gamma_panel(gate, rho1_ops, rs2, id2s, theta_step, out_dir='.', gate_name='gate'):
    """Grid of gamma_{CH_1} spheres: rows = rho_1 operators, cols = rho_2 radii.

    rho1_ops is a list of (2, 2) single-qubit operators; each is one row's fixed
    rho_1. id2s is a per-column list (same length as rs2). Column j renders
    gamma_{CH_1}(gate @ (rho1 (x) rho2)) over
    rho2 = id2s[j] * I + rs2[j] * (n . sigma) swept over the sphere |n| = 1.

    Writes one standalone interactive HTML per cell plus a combined
    len(rho1_ops) x len(rs2) overview with a single shared colorbar.
    """
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    os.makedirs(out_dir, exist_ok=True)

    rho1s = [np.asarray(op, dtype=complex) for op in rho1_ops]
    nrows, ncols = len(rho1s), len(rs2)

    # id2s may be a scalar (one value for all columns) or a per-column list.
    if np.isscalar(id2s):
        id2s = [float(id2s)] * ncols
    else:
        id2s = [float(v) for v in id2s]
    if len(id2s) != ncols:
        raise ValueError(
            f"id2s has length {len(id2s)} but rs2 has {ncols} columns")

    titles = []
    for rho1 in rho1s:
        for j, r2 in enumerate(rs2):
            titles.append(f"rho1 = {op1q_expr(rho1)}"
                          f"<br>rho2: id2={id2s[j]:.3f}, r2={float(r2):.3f}")

    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=[[{'type': 'surface'}] * ncols for _ in range(nrows)],
        subplot_titles=titles,
        horizontal_spacing=0.04, vertical_spacing=0.06,
    )

    gmin, gmax = np.inf, -np.inf
    for i, rho1 in enumerate(rho1s):
        for j, r2 in enumerate(rs2):
            r2 = float(r2)
            _, _, (nx, ny, nz), gamma = sphere_gamma_grid(
                gate, rho1, r2, theta_step, id2=id2s[j])

            # standalone interactive plot (with its own colorbar)
            ind = go.Figure(go.Surface(
                x=r2 * nx, y=r2 * ny, z=r2 * nz,
                surfacecolor=gamma, colorscale='Viridis',
                colorbar=dict(title='gamma_CH1')))
            ind.update_layout(
                title=f"{gate_name}: gamma_CH1 | r2={r2:.3f}<br>"
                      f"rho1 = {op1q_expr(rho1)}",
                scene=dict(aspectmode='data'))
            ind_path = os.path.join(
                out_dir, f"{gate_name}_row{i}_j{j}_r2{r2:.3f}.html")
            ind.write_html(ind_path)
            print(f'wrote {ind_path}  (gamma in [{gamma.min():.3f}, {gamma.max():.3f}])')

            # combined-panel cell: shared colour axis -> a single colorbar
            gmin = min(gmin, float(gamma.min()))
            gmax = max(gmax, float(gamma.max()))
            fig.add_trace(go.Surface(
                x=r2 * nx, y=r2 * ny, z=r2 * nz,
                surfacecolor=gamma, coloraxis='coloraxis'),
                row=i + 1, col=j + 1)

    fig.update_layout(
        height=450 * nrows, width=450 * ncols + 150,
        coloraxis=dict(colorscale='Viridis', cmin=gmin, cmax=gmax,
                       colorbar=dict(title='gamma_CH1')),
        title=f"{gate_name}: gamma_CH1  (rows = rho1 ops, "
              f"cols rho2 radii rs2={[round(float(r), 3) for r in rs2]})")
    combined = os.path.join(out_dir, f"{gate_name}_panel.html")
    fig.write_html(combined)
    print(f'wrote {combined}')
    return combined


# ---------------------------------------------------------------------------
# Cluster entry point
# ---------------------------------------------------------------------------
def build_gate(name, angles=None, p=0.0, dt=0.0):
    if name == 'cnot':
        base = cnot_ptm()
    elif name == 'identity':
        base = np.eye(16)
    elif name == 'depol_kron':
        if angles is None:
            raise ValueError("--gate depol_kron requires --angles ALPHA BETA GAMMA")
        base = depol_kron_ptm(*angles)
    elif name == 'exp_xy':
        base = exp_xy_ptm()
    elif name == 'exp_yx':
        base = exp_yx_ptm()
    elif name == 'cnot_diss':
        base = cnot_diss_ptm(dt)
    else:
        raise ValueError(
            f"unknown gate '{name}' (use 'cnot', 'identity', 'depol_kron', "
            f"'exp_xy', 'exp_yx', or 'cnot_diss')")
    # depol_2q(p) ∘ gate, matching depol_kron_worker.build_channel.
    # p = 0 gives depol_2q(0) = identity, i.e. the bare gate.
    return depol_2q_ptm(p) @ base


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gate', default='cnot',
                    help="cnot | identity | depol_kron | exp_xy | exp_yx | "
                         "cnot_diss")
    ap.add_argument('--angles', type=float, nargs=3, default=None,
                    metavar=('ALPHA', 'BETA', 'GAMMA'),
                    help='for --gate depol_kron: '
                         'U = exp(i*(alpha*XX + beta*YY + gamma*ZZ))')
    ap.add_argument('--dt', type=float, default=0.0,
                    help='for --gate cnot_diss: Euler dissipation step in '
                         'I + dt*L (collective jump S^-)')
    ap.add_argument('--p', type=float, default=0.0,
                    help='depolarizing strength applied to any gate '
                         '(depol_2q(p) ∘ gate); default 0 = bare gate')
    ap.add_argument('--rho1', type=float, nargs=4, default=None,
                    metavar=('CID', 'CX', 'CY', 'CZ'),
                    help='fixed rho1 = CID*I + CX*X + CY*Y + CZ*Z; requires '
                         '|CID| + |(CX,CY,CZ)| < 1. Overrides --rs1 (single row).')
    ap.add_argument('--id2', type=float, default=0.0,
                    help='identity coefficient of the rho2 frame element; '
                         'requires |id2| + r2 < 1 for each r2 in --rs2')
    ap.add_argument('--rs1', type=float, nargs='+', default=[0.5, 1.0],
                    help='rho1 radii (each <= 1), one random rho1 per panel ROW; '
                         'ignored if --rho1 is given.')
    ap.add_argument('--rs2', type=float, nargs='+', default=[0.5, 0.9],
                    help='rho2 Bloch radii (>= 0); |id2| + r2 must be < 1.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--theta-step', type=float, default=np.pi / 45,
                    help='angular step in radians (default pi/45 = 4 deg)')
    ap.add_argument('--out-dir', default='gamma_ch1_out')
    args = ap.parse_args()

    if any(r < 0.0 or r > 1.0 + 1e-9 for r in args.rs1):
        ap.error('--rs1 values must be in [0, 1]')
    if any(r < 0.0 for r in args.rs2):
        ap.error('--rs2 values must be >= 0')
    for r2 in args.rs2:                                  # request 3 constraint
        if abs(args.id2) + r2 >= 1.0:
            ap.error(f'|id2| + r2 must be < 1, violated by id2={args.id2}, '
                     f'r2={r2} (sum {abs(args.id2) + r2:.4f})')
    if args.rho1 is not None:                            # request 2 constraint
        s = abs(args.rho1[0]) + float(np.linalg.norm(args.rho1[1:]))
        if s >= 1.0:
            ap.error('--rho1 requires |CID| + |(CX,CY,CZ)| < 1 '
                     f'(got {s:.4f})')

    gate = build_gate(args.gate, args.angles, args.p, args.dt)

    gate_name = args.gate
    if args.gate == 'depol_kron':
        a, b, g = args.angles
        gate_name = f'depol_kron_a{a:.2f}_b{b:.2f}_g{g:.2f}'
    elif args.gate == 'cnot_diss':
        gate_name = f'cnot_diss_dt{args.dt:.3f}'
    if args.p != 0.0:
        gate_name += f'_p{args.p:.2f}'
    if args.id2 != 0.0:
        gate_name += f'_id2{args.id2:.2f}'
    if args.rho1 is not None:                            # request 1: rho1 in name
        cid, cx, cy, cz = args.rho1
        gate_name += f'_rho1_id{cid:+.2f}_x{cx:+.2f}_y{cy:+.2f}_z{cz:+.2f}'

    # rho_1 rows: a single fixed --rho1, else one random direction per --rs1 radius
    if args.rho1 is not None:
        cid, cx, cy, cz = args.rho1
        rho1_ops = [cid * _I + cx * _X + cy * _Y + cz * _Z]
    else:
        rng = np.random.default_rng(args.seed)
        rho1_ops = []
        for r1 in args.rs1:
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            rho1_ops.append(frame_op_1q(float(r1) * d))

    gamma_panel(gate, rho1_ops, args.rs2, args.id2, args.theta_step,
                out_dir=args.out_dir, gate_name=gate_name)


if __name__ == '__main__':
    main()
