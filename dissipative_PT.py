"""
Dissipative Phase Transition on a 2D qubit lattice.

    H = h Σ_i Z_i + J Σ_{<i,j>} X_i X_j

Jump operators (amplitude damping toward |0>):
    L_i = sqrt(gamma) S^-_i,   S^- = |0><1|

Two-qubit Trotter building block for bond <i,j>:
    L_2(ρ) = -i[J X_i X_j + (h/2)(Z_i⊗I + I⊗Z_j), ρ]
           + (gamma/2) D(S^-_i)(ρ) + (gamma/2) D(S^-_j)(ρ)

where D(L)(ρ) = L ρ L† - ½{L†L, ρ}.

Trotter gate:  U_2 = exp(L_2 * dt),  16×16 real matrix (2-qubit Pauli basis).

Steady state is solved for the full 2×2 (N=4 qubit) open-boundary lattice.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, null_space
from scipy.optimize import minimize
from scipy.sparse import csc_matrix, eye as sp_eye, hstack as sp_hstack, vstack as sp_vstack
from scipy.optimize._linprog_highs import _linprog_highs
from scipy.optimize._linprog_util import _LPProblem, _clean_inputs

# ── single-qubit primitives ───────────────────────────────────────────────────
_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
# S^- = |0><1| = σ+  (lowers excited → ground, drives toward |0>)
S_MINUS = 0.5 * (_SX + 1j * _SY)   # [[0,1],[0,0]]
S_PLUS  = S_MINUS.conj().T           # [[0,0],[1,0]]
_PAULI  = [_I2, _SX, _SY, _SZ]

# ── N-qubit Pauli basis ───────────────────────────────────────────────────────
_P_CACHE: dict[int, np.ndarray] = {}


def _pauli_tensor(n: int) -> np.ndarray:
    """(4^n, 2^n, 2^n) array of all n-qubit Pauli string operators."""
    if n not in _P_CACHE:
        dim, d = 4 ** n, 2 ** n
        P = np.zeros((dim, d, d), dtype=complex)
        for idx in range(dim):
            tmp, s = idx, []
            for _ in range(n):
                s.append(tmp % 4)
                tmp //= 4
            mats = [_PAULI[a] for a in reversed(s)]
            op = mats[0]
            for m in mats[1:]:
                op = np.kron(op, m)
            P[idx] = op
        _P_CACHE[n] = P
    return _P_CACHE[n]


def _site_op(op: np.ndarray, site: int, n: int) -> np.ndarray:
    mats = [_I2] * n
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


# ── 2D lattice geometry ───────────────────────────────────────────────────────
def bonds_2d(Lx: int, Ly: int) -> list[tuple[int, int]]:
    """Nearest-neighbor bonds of an Lx×Ly open-boundary lattice (row-major sites)."""
    bonds = []
    for r in range(Ly):
        for c in range(Lx - 1):
            bonds.append((r * Lx + c, r * Lx + c + 1))
    for r in range(Ly - 1):
        for c in range(Lx):
            bonds.append((r * Lx + c, (r + 1) * Lx + c))
    return bonds


# ── core Lindbladian builder (Pauli basis) ────────────────────────────────────
def _build_lindbladian(H: np.ndarray, jumps: list, n: int) -> np.ndarray:
    """Return (4^n, 4^n) real Lindbladian M[a,b] = Tr(P_a Lind(P_b)) / 2^n.

    jumps: list of (rate, L, L†, L†L).
    """
    d = 2 ** n
    P = _pauli_tensor(n)
    dim = 4 ** n
    M = np.zeros((dim, dim), dtype=float)
    for b in range(dim):
        Pb = P[b]
        res = -1j * (H @ Pb - Pb @ H)
        for rate, L, Ld, LdL in jumps:
            if rate == 0:
                continue
            res += rate * (L @ Pb @ Ld - 0.5 * (LdL @ Pb + Pb @ LdL))
        M[:, b] = np.einsum('aij,ji->a', P, res).real / d
    return M


# ── model Hamiltonians and Lindbladians ───────────────────────────────────────
def hamiltonian_2d(h: float, J: float, Lx: int = 2, Ly: int = 2) -> np.ndarray:
    """H = h Σ_i Z_i + J Σ_{<i,j>} X_i X_j on the Lx×Ly OBC lattice."""
    N = Lx * Ly
    d = 2 ** N
    H = np.zeros((d, d), dtype=complex)
    for i in range(N):
        H += h * _site_op(_SZ, i, N)
    for i, j in bonds_2d(Lx, Ly):
        H += J * (_site_op(_SX, i, N) @ _site_op(_SX, j, N))
    return H


def build_bond_lindbladian(J: float, h: float, gamma: float) -> np.ndarray:
    """16×16 Lindbladian in the 2-qubit Pauli basis for one bond.

    H_2 = J X⊗X + (h/2) Z⊗I + (h/2) I⊗Z
    D   = (gamma/2) D(S^-⊗I) + (gamma/2) D(I⊗S^-)
    """
    H2 = (J       * np.kron(_SX, _SX)
          + (h/2) * np.kron(_SZ, _I2)
          + (h/2) * np.kron(_I2, _SZ))
    jumps = []
    for L in (np.kron(S_MINUS, _I2), np.kron(_I2, S_MINUS)):
        Ld = L.conj().T
        jumps.append((gamma / 2, L, Ld, Ld @ L))
    return _build_lindbladian(H2, jumps, n=2)


def build_full_lindbladian(h: float, J: float, gamma: float,
                            Lx: int = 2, Ly: int = 2) -> np.ndarray:
    """(4^N × 4^N) full Lindbladian for the Lx×Ly OBC lattice."""
    N = Lx * Ly
    H = hamiltonian_2d(h, J, Lx, Ly)
    jumps = []
    for i in range(N):
        L = _site_op(S_MINUS, i, N)
        Ld = L.conj().T
        jumps.append((gamma, L, Ld, Ld @ L))
    return _build_lindbladian(H, jumps, n=N)


def bond_trotter_gate(J: float, h: float, gamma: float, dt: float) -> np.ndarray:
    """16×16 Trotter gate exp(L_2 * dt) in the 2-qubit Pauli basis."""
    return expm(build_bond_lindbladian(J, h, gamma) * dt).real


# ── NESS ──────────────────────────────────────────────────────────────────────
def steady_state_and_decay(L: np.ndarray, N: int) -> tuple:
    """Return (c_ss, decay_rate).

    c_ss is the Pauli-coefficient vector of the NESS (unit-trace), or None if
    the null space is degenerate (e.g. gamma=0 gives no unique NESS).
    """
    d_hilbert = 2 ** N
    ns = null_space(L, rcond=1e-9)
    evals = np.linalg.eigvals(L)
    re_sorted = np.sort(evals.real)[::-1]
    decay = float('nan')
    for r in re_sorted:
        if r < -1e-9:
            decay = float(-r)
            break

    if ns.shape[1] != 1:
        return None, decay
    c = ns[:, 0].real
    if abs(c[0]) < 1e-12:
        return None, decay
    c = c / (c[0] * d_hilbert)
    return c, decay


def pauli_to_rho(c: np.ndarray, N: int) -> np.ndarray:
    P = _pauli_tensor(N)
    rho = np.einsum('a,aij->ij', c.astype(complex), P)
    return (rho + rho.conj().T) / 2.0


# ── state properties ──────────────────────────────────────────────────────────
def vn_entropy(rho: np.ndarray) -> float:
    e = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    e = e[e > 1e-15]
    return float(-np.sum(e * np.log(e)))


def site_magnetization(rho: np.ndarray, N: int) -> np.ndarray:
    """<Z_i> for each site; shape (N,)."""
    return np.array([float(np.trace(_site_op(_SZ, i, N) @ rho).real)
                     for i in range(N)])


def negativity(rho: np.ndarray, d_A: int) -> float:
    """Entanglement negativity for the A|B bipartition with dim(A) = d_A."""
    d = rho.shape[0]
    d_B = d // d_A
    rho_pt = (rho.reshape(d_A, d_B, d_A, d_B)
                 .transpose(0, 3, 2, 1).reshape(d, d))
    e = np.linalg.eigvalsh((rho_pt + rho_pt.conj().T) / 2)
    return float(np.sum(np.abs(e[e < -1e-15])))


def lpdo_bond_entropy(rho: np.ndarray, d_A: int) -> float:
    """LPDO bond entropy for the d_A | (d/d_A) bipartition."""
    from lpdo import purification_sqrt, _bond_entropy
    d = rho.shape[0]
    d_B = d // d_A
    X = purification_sqrt(rho)
    T = X.reshape(d_A, d_B, d_A, d_B).transpose(0, 2, 1, 3)
    sv = np.linalg.svd(T.reshape(d_A * d_A, d_B * d_B), compute_uv=False)
    return float(_bond_entropy(sv))


def max_lpdo_bond_entropy(rho: np.ndarray, N: int) -> float:
    """Max LPDO bond entropy over bipartitions d_A ∈ {2, 2^(N//2)}."""
    d = 2 ** N
    best = float('nan')
    for d_A in sorted({2, d // 2}):
        if 1 < d_A < d:
            try:
                v = lpdo_bond_entropy(rho, d_A)
                best = v if np.isnan(best) else max(best, v)
            except Exception:
                pass
    return best


# ── gate properties ───────────────────────────────────────────────────────────
def pauli_framability(gate: np.ndarray) -> float:
    return float(np.max(np.sum(np.abs(gate), axis=1)))


def channel_stab_purity(gate: np.ndarray) -> float:
    """log2(d^2 * Σ_k G_kk^2 / (d+1)) where d = 2^n_qubits."""
    n = int(round(np.log2(gate.shape[0]) / 2))
    d = 2 ** n
    diag = np.diag(gate).real
    return float(np.log2(d ** 2 * float(np.sum(diag ** 2)) / (d + 1)))


# ── optimised framability (2-qubit bond, D = kron(S,S)) ──────────────────────
_N_BOND = 2
_FIXED_COL = np.array([[1.0], [0.0], [0.0], [0.0]])


def _project_bloch(M: np.ndarray) -> np.ndarray:
    c_I   = np.abs(M[0:1, :])
    bloch = np.linalg.norm(M[1:4, :], axis=0, keepdims=True)
    return M / np.maximum(c_I + bloch, 1.0)


def _kron_power(S: np.ndarray, n: int) -> np.ndarray:
    D = S
    for _ in range(n - 1):
        D = np.kron(D, S)
    return D


def _framability_lp(D: np.ndarray, gate: np.ndarray) -> float:
    D = np.asarray(D, dtype=float)
    nrows, d_ext = D.shape
    Y = gate.real.T @ D
    c_obj = np.concatenate([np.zeros(d_ext), np.ones(d_ext)])
    A_eq  = csc_matrix(np.hstack([D, np.zeros((nrows, d_ext))]))
    Ide   = sp_eye(d_ext, format='csc')
    A_ub  = sp_vstack([sp_hstack([Ide, -Ide]), sp_hstack([-Ide, -Ide])], format='csc')
    b_ub  = np.zeros(2 * d_ext)
    bounds = [(None, None)] * d_ext + [(0.0, None)] * d_ext
    lp = _clean_inputs(_LPProblem(c_obj, A_ub, b_ub, A_eq, np.zeros(nrows), bounds, None))
    best = 0.0
    for j in range(d_ext):
        r = _linprog_highs(lp._replace(b_eq=Y[:, j].copy()), solver=None, presolve=False)
        if r['status'] != 0:
            # Column j of the gate cannot be expanded in this frame: the frame
            # is incomplete for the gate, so its framability is +inf.  Returning
            # 0 here (the old behaviour) silently dropped the column and let the
            # optimiser cheat by collapsing D to a rank-deficient frame.
            return float('inf')
        best = max(best, float(np.sum(np.abs(r['x'][:d_ext]))))
    return best


def _ixyz_init(d_ext_single: int) -> np.ndarray:
    b = 1.0 / np.sqrt(2)
    base = np.array([[0., 0., 0., 0., 0., 0., 0.],
                     [1., 0., 0., b,  b,  b,  0.],
                     [0., 1., 0., b, -b,  0., b ],
                     [0., 0., 1., 0., 0., b,  b ]])
    n_free = d_ext_single - 1
    free = np.zeros((4, n_free))
    k = min(n_free, base.shape[1])
    free[:, :k] = base[:, :k]
    return free.ravel()


def frame_from_params(x: np.ndarray, d_ext_single: int) -> np.ndarray:
    """Reconstruct the single-qubit frame S (4 × d_ext_single) from a flat
    parameter vector: column 0 is the fixed identity (1,0,0,0)ᵀ and the
    remaining d_ext_single-1 columns are the Bloch-projected free parameters
    (matching the encoding used inside optimise_framability)."""
    if x is None:
        return np.full((4, d_ext_single), np.nan)
    n_free = d_ext_single - 1
    free = _project_bloch(np.asarray(x, dtype=float).reshape(4, n_free))
    return np.hstack([_FIXED_COL, free])


def params_from_frame(S: np.ndarray) -> np.ndarray:
    """Flat parameter vector (the free columns) from a frame S — inverse of
    frame_from_params, suitable as a warm-start seed for optimise_framability."""
    return np.asarray(S, dtype=float)[:, 1:].ravel()


def embed_frame_params(x_small: np.ndarray, d_ext_small: int,
                       d_ext_large: int) -> np.ndarray:
    """Embed an optimised frame from a smaller d_ext into a larger one.

    The flat parameter vector encodes the (4 × n_free) free columns of S.
    The extra columns needed for the larger frame are filled by replicating
    ("multiplying") the last free column.  Because kron(S_large, S_large) then
    contains every column of kron(S_small, S_small), the framability over the
    larger frame is guaranteed ≤ the smaller-frame value — a monotone warm
    start for the larger d_ext optimisation.
    """
    nf_s = d_ext_small - 1
    nf_l = d_ext_large - 1
    if nf_l < nf_s:
        raise ValueError('target d_ext must be >= source d_ext')
    block = np.asarray(x_small, dtype=float).reshape(4, nf_s)
    pad   = np.repeat(block[:, -1:], nf_l - nf_s, axis=1)
    return np.hstack([block, pad]).ravel()


def optimise_framability(gate: np.ndarray, d_ext_single: int,
                         n_restarts: int = 5, maxfev: int = 1000,
                         seed: int = 0, extra_init_xs=None,
                         return_x: bool = False):
    """Min framability of the 2-qubit gate over D = kron(S, S) (S: 4×d_ext_single).

    extra_init_xs : optional list of flat parameter vectors (length 4*(d_ext-1))
        used as additional warm-start seeds, appended after the standard
        ixyz + random restarts (e.g. a neighbour's optimum or an embedded
        smaller-d_ext frame).
    return_x : if True, return (f_best, x_best) instead of just f_best, where
        x_best is the flat parameter vector for the optimal S.
    """
    n_free   = d_ext_single - 1
    n_params = 4 * n_free

    def _to_S(p):
        return np.hstack([_FIXED_COL, _project_bloch(p.reshape(4, n_free))])

    _PENALTY = 1e6   # finite stand-in for an infeasible (rank-deficient) frame

    def objective(p):
        D = _kron_power(_to_S(p), _N_BOND)
        f = _framability_lp(D, gate)
        return f if np.isfinite(f) else _PENALTY

    rng   = np.random.default_rng(seed)
    seeds = [_ixyz_init(d_ext_single)]
    seeds += [rng.standard_normal(n_params) * 0.3 for _ in range(max(0, n_restarts - 1))]
    if extra_init_xs:
        seeds += [np.asarray(x, dtype=float).ravel() for x in extra_init_xs]

    best, best_x = float('inf'), None
    for x0 in seeds:
        res = minimize(objective, x0, method='Powell',
                       options={'maxfev': maxfev, 'maxiter': maxfev,
                                'ftol': 1e-6, 'xtol': 1e-6})
        if float(res.fun) < best:
            best, best_x = float(res.fun), np.asarray(res.x, dtype=float).copy()
    return (best, best_x) if return_x else best


# ── sign problem (with local single-qubit rotation search) ───────────────────
def _sq_superop(R: np.ndarray) -> np.ndarray:
    M = np.zeros((4, 4), dtype=complex)
    for i, Pi in enumerate(_PAULI):
        for j, Pj in enumerate(_PAULI):
            M[i, j] = np.trace(Pi @ R @ Pj @ R.conj().T) / 2.0
    return M.real if np.allclose(M.imag, 0, atol=1e-12) else M


def _apply_local_rot(gate: np.ndarray, M: np.ndarray, n: int = _N_BOND) -> np.ndarray:
    d = 4 ** n
    g = gate.reshape((4,) * (2 * n))
    for k in range(n):
        g = np.moveaxis(np.tensordot(M, g, axes=([1], [k])), 0, k)
    for k in range(n, 2 * n):
        g = np.moveaxis(np.tensordot(M, g, axes=([1], [k])), 0, k)
    return g.reshape(d, d)


def sign_problem_results(gate: np.ndarray, n_restarts: int = 10,
                         seed: int = 0) -> tuple[float, float]:
    """Return (sign_init, sign_opt) for the gate."""
    from sign_problem import sign_problem as _sp
    rng = np.random.default_rng(seed)

    def neg_s(p):
        R = expm(1j * np.pi * (p[0] * _SX + p[1] * _SY + p[2] * _SZ))
        M = _sq_superop(R)
        if not np.isrealobj(M) and np.max(np.abs(np.imag(M))) > 1e-10:
            return 0.0
        return -float(_sp(_apply_local_rot(gate, np.real(M))))

    sign_init = float(_sp(gate))
    best = sign_init
    for _ in range(n_restarts):
        x0 = rng.standard_normal(3)
        x0 /= max(np.linalg.norm(x0), 1e-14)
        res = minimize(neg_s, x0, method='BFGS',
                       options={'maxiter': 200, 'gtol': 1e-5})
        best = max(best, float(-res.fun))
    return sign_init, best


# ── combined per-point computation ───────────────────────────────────────────
def compute_point(h: float, J: float, gamma: float, dt: float = 0.05,
                  fra_restarts: int = 5, fra_maxfev_4: int = 1000,
                  fra_maxfev_6: int = 500, sign_restarts: int = 10,
                  seed: int = 0, Lx: int = 2, Ly: int = 2,
                  verbose: bool = False) -> dict:
    """Compute all quantities for one (h, J, gamma) parameter point.

    Returns a dict with keys:
      Gate-level (2-qubit bond Trotter, 16×16):
        pauli_fra, opt_fra_4, opt_fra_6, sign_init, sign_opt, chan_stab
      Full lattice (Lx×Ly NESS, 4^N-dim Lindbladian):
        decay_rate, ss_vn, mean_mag, neg_half, max_lpdo
    """
    N = Lx * Ly
    out: dict = dict(h=h, J=J, gamma=gamma, dt=dt)

    # ── Trotter gate (2-qubit bond) ───────────────────────────────────────────
    gate = bond_trotter_gate(J, h, gamma, dt)

    out['pauli_fra'] = pauli_framability(gate)
    out['chan_stab'] = channel_stab_purity(gate)

    if verbose:
        print(f'  pauli_fra={out["pauli_fra"]:.4f}', flush=True)

    out['opt_fra_4'], x4 = optimise_framability(gate, 4, fra_restarts,
                                                 fra_maxfev_4, seed, return_x=True)
    out['opt_fra_6'], x6 = optimise_framability(gate, 6, fra_restarts,
                                                 fra_maxfev_6, seed + 1, return_x=True)
    out['opt_S_4'] = frame_from_params(x4, 4)
    out['opt_S_6'] = frame_from_params(x6, 6)

    if verbose:
        print(f'  opt_fra: d4={out["opt_fra_4"]:.4f}  d6={out["opt_fra_6"]:.4f}', flush=True)

    out['sign_init'], out['sign_opt'] = sign_problem_results(gate, sign_restarts, seed)

    if verbose:
        print(f'  sign: init={out["sign_init"]:.4f}  opt={out["sign_opt"]:.4f}', flush=True)

    # ── Full Lindbladian + NESS ───────────────────────────────────────────────
    L_full = build_full_lindbladian(h, J, gamma, Lx, Ly)
    c_ss, decay = steady_state_and_decay(L_full, N)
    out['decay_rate'] = decay

    if c_ss is not None:
        rho = pauli_to_rho(c_ss, N)
        out['ss_vn']    = vn_entropy(rho)
        out['mean_mag'] = float(np.mean(site_magnetization(rho, N)))
        d_half          = 2 ** (N // 2)       # 4 for a 4-qubit system
        out['neg_half'] = negativity(rho, d_half)
        try:
            out['max_lpdo'] = max_lpdo_bond_entropy(rho, N)
        except Exception:
            out['max_lpdo'] = float('nan')
    else:
        out['ss_vn'] = out['mean_mag'] = out['neg_half'] = out['max_lpdo'] = float('nan')

    if verbose:
        print(f'  decay={out["decay_rate"]:.4e}  vn={out["ss_vn"]:.4f}'
              f'  neg={out["neg_half"]:.4f}', flush=True)

    return out


# ── parameter grid (shared with worker and collect scripts) ──────────────────
H_LIST     = [round(0.1 * i, 10) for i in range(-10, 10)]   # 20 h values: -1.0 … 0.9
# gamma grid: fine (step 0.1) up to 1.0 — indices 0..9 (0.0…0.9) reuse the
# original scan data — then coarse (step 0.2) from 1.0 up to 5.0.
GAMMA_LIST = ([round(0.1 * i, 10) for i in range(11)]            # 0.0 … 1.0  (11 vals)
              + [round(1.0 + 0.2 * i, 10) for i in range(1, 21)])  # 1.2 … 5.0  (20 vals)
J_DEFAULT  = 1.0
DT_DEFAULT = 0.05
N_H        = len(H_LIST)
N_G        = len(GAMMA_LIST)  # 31
N_TOTAL    = N_H * N_G   # 620


# ── self-test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    J, h, gamma, dt = 1.0, 0.5, 0.5, 0.05
    print(f'Test point:  J={J}  h={h}  gamma={gamma}  dt={dt}')
    print(f'2×2 lattice, bonds: {bonds_2d(2, 2)}')

    t0 = time.perf_counter()
    res = compute_point(h, J, gamma, dt, fra_restarts=3, fra_maxfev_4=300,
                        fra_maxfev_6=150, sign_restarts=5, seed=0, verbose=True)
    elapsed = time.perf_counter() - t0

    print(f'\n{"─"*50}')
    for k, v in res.items():
        if isinstance(v, float):
            print(f'  {k:20s} = {v:.6f}')
    print(f'  {"elapsed":20s} = {elapsed:.1f}s')
