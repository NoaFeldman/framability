r"""
kw_asymmetry.py -- the Kramers-Wannier (KW) asymmetry Delta S_KW(t) of
arXiv:2610.23111 (Yang, Lin & Chen, "Liouvillian Non-Invertible Symmetry"),
Eq. (14), for model10 (the Shibata-Katsura dissipative transverse-field Ising
chain, https://arxiv.org/abs/1904.12505), by exact evolution of the full
density matrix.

Model (J, h, D1, D2 free; trotter_lindbladian_scan's model10 is J = h = 1):

    H = -J sum_b Z_b - h sum_j X_j
    jumps  --jumps local       sqrt(D1) X_j (every site),  sqrt(D2) Z_b (every bond)
           --jumps collective  sqrt(D1) A,  sqrt(D2) B,   A = sum_j X_j,  B = sum_b Z_b
                               (the paper's L_A, L_B with gamma_A = D1, gamma_B = D2)

Z_b is a Z-string on the sites of b: Z_j Z_{j+1} on the periodic ring.  The
geometry (sites, Z-strings, the KW image of every X_j) is a Geometry, so a 2D
lattice drops in as one more Geometry (see TWO_D_NOTE: the 2D choice is open).

KW asymmetry (paper Eq. 14, global mixed-state version of Vescovo-Calabrese-
Ares' entanglement asymmetry):

    rho_hat = D rho D^dag,   rho_sym = (rho + rho_hat)/2,   S2 = -ln Tr rho^2
    Delta S_KW = S2(rho_sym) - S2(rho) = ln[ 4a / (a + b + 2c) ]
    a = Tr rho^2,  b = Tr rho_hat^2,  c = Tr(rho rho_hat)

Both formulas are recorded: the general one, and ln[2a / (a + c)], which is
the same thing once b = a (true for rho in the even sector, see below).

The KW operator D
-----------------
D is built directly from its defining relations instead of from a gate
sequence.  With input in the X basis |x>_X (X_j |x> = x_j |x>, x_j = +-1)
and output in the Z basis |mu> (Z_j |mu> = mu_j |mu>):

    <mu| D |x>_X = c * delta( x_j = prod_{k in dual[j]} mu_k  for all j )

On the ring dual[j] = (j, j+1), i.e. x_j = mu_j mu_{j+1} (spin <-> domain
wall).  Then, as operator identities on the full 2^N space,

    D X_j = Z_j Z_{j+1} D,     D Z_j Z_{j+1} = X_{j+1} D,
    D^dag D = D D^dag = P_+ = (1 + eta)/2,   eta = prod_j X_j,

for c^2 = 1/2 (in general c^2 = 1/|kernel| of the map mu -> x, 2 on the
ring).  So D = U_KW P_+ with U_KW any unitary that agrees with D on the even
sector -- the paper's D = U_KW P_+ (its Ref. [54]) up to the normalisation
and an overall phase, neither of which enters Delta S.  The paper's
normalisation D^2 = (1/2)(1 + eta) T is a different overall factor; ours makes
rho_hat a normalised state whenever rho is supported on the even sector
(Tr rho_hat = Tr P_+ rho), which is what the entropy formula needs.

Computationally D = M H^{(x)N}: the normalised Walsh-Hadamard transform
H^{(x)N} (computational -> X basis), then a 0/1 selection matrix M with one
column per mu.  So rho_hat = c^2 R[xmap, xmap] with R = H^{(x)N} rho H^{(x)N}:
2N in-place butterfly passes plus one gather, never a 2^N x 2^N dense D.

Even sector.  H, every X_j, Z_b jump and A, B commute with eta (strong Z2
symmetry), so a state with eta rho = rho eta = rho stays there; then
b = Tr(D rho P_+ rho D^dag) = a.  All initial states below are eta-even
(GHZ, the all-but-GHZ superposition |S>, and |+>^N are each eta = +1).

Evolution ("option 1": exact, full density matrix)
--------------------------------------------------
For N = 12, rho is 4096 x 4096 (268 MB complex) while the vectorised
Liouvillian acts on 4^12 = 1.7e7-vectors: a sparse superoperator would hold
~(2N + 1) * 1.7e7 ~ 4e8 nonzeros (~ 6-10 GB) and buy nothing.  Instead the
matrix ODE  d rho/dt = L(rho)  is integrated directly with scipy's adaptive
8th-order Dormand-Prince (DOP853, dense output at the save times, rtol 1e-8 /
atol 1e-12 by default) using a matrix-free L:

  * every diagonal piece is ONE precomputed elementwise factor G:
      -i [H_ZZ, rho]          -> -i (e_i - e_j) rho_ij,      e = -J sum_b z_b
      D2 sum_b (Z_b rho Z_b - rho)  (local)
      D2 (B rho B - {B^2, rho}/2)    (collective; B is diagonal)
      -D1 N rho                      (local X jumps: X_j^2 = 1)
  * every X_j is a bit flip, applied as a strided view (no copies):
      X_j rho, rho X_j, X_j rho X_j  -> np.add(out, flipped view, out=...).

One L(rho) costs ~3N passes over rho (local; ~5N collective), i.e. a few
seconds at N = 12 on one core, and memory is ~ 15-20 copies of rho (DOP853's
13 stages plus buffers), ~ 5-6 GB.

Self-check (selfcheck(), run by the worker with --selfcheck before the long
evolution) verifies on a small ring, against dense matrices and against
n_qubit_lindbladian.build_lindbladian_from_terms (the repo's own sparse
Liouvillian, column-stacking vec):
  D X_j = Z_j Z_{j+1} D,  D Z_j Z_{j+1} = X_{j+1} D,  D^dag D = D D^dag = P_+,
  D H(J, h) = H(h, J) D,  the Liouvillian duality
  D L(J, h; D1, D2)[rho] D^dag = L(h, J; D2, D1)[D rho D^dag],
  the matrix-free L against the dense / repo Liouvillians, the fast rho_hat
  against D rho D^dag, and b = a on the even sector.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import reduce
from typing import Callable, Optional, Tuple

import numpy as np

from n_qubit_lindbladian import (_SX, _SZ, _I2, _site_op, ring_edges,
                                 build_lindbladian_from_terms)

KW_ASYM_VERSION = '1.0'
JUMP_KINDS = ('local', 'collective')
INIT_KINDS = ('ghz_s', 'ghz', 'plus')
ALPHA_PAPER = float(np.sqrt(3.0) / 2.0)      # paper: sqrt(3)/2 |GHZ> + 1/2 |S>

TWO_D_NOTE = (
    "2D is not implemented yet: KW is not a self-duality of the 2D "
    "transverse-field Ising model (it maps it to the Z2 gauge theory, Wegner), "
    "so the paper's Delta S_KW is undefined on a 3x4 TFIM.  Pick one of: "
    "(a) the self-dual 2D plaquette (Xu-Moore) Ising model on a 3x4 torus, "
    "(b) the 1D KW map along a snake ordering (a diagnostic, not a duality), "
    "(c) the Wegner-dual gauge-theory version, (d) drop 2D; then add a "
    "Geometry for it in geometry_from_spec.")


# ---------------------------------------------------------------------------
#  Geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Geometry:
    """Sites 0..N-1 (site k = k-th tensor factor = bit N-1-k of an index).

    bonds  the Z-strings Z_b of the Hamiltonian (-J Z_b) and of the D2 jumps
    dual   dual[j] = the Z-string the KW operator maps X_j to (D X_j = Z D)
    shift  optional: D Z_{bonds[b]} = X_{shift[b]} D (checked by selfcheck)
    """
    name: str
    N: int
    bonds: Tuple[Tuple[int, ...], ...]
    dual: Tuple[Tuple[int, ...], ...]
    shift: Optional[Tuple[int, ...]] = None


def ring_geometry(N: int) -> Geometry:
    """Periodic ring: bonds (j, j+1 mod N) from n_qubit_lindbladian.ring_edges,
    D X_j = Z_j Z_{j+1} D and D Z_j Z_{j+1} = X_{j+1} D."""
    e = tuple(tuple(int(s) for s in b) for b in ring_edges(N))
    return Geometry(f'ring{N}', N, e, e, tuple((j + 1) % N for j in range(N)))


def geometry_from_spec(spec: str) -> Geometry:
    """'ring:12' -> the 12-site ring.  'rect:3x4' is reserved for the 2D
    lattice (see TWO_D_NOTE)."""
    kind, _, size = spec.partition(':')
    if kind == 'ring':
        return ring_geometry(int(size))
    if kind == 'rect':
        raise NotImplementedError(TWO_D_NOTE)
    raise ValueError(f'unknown geometry {spec!r} (use ring:<N>)')


def site_signs(N: int) -> np.ndarray:
    """(N, 2^N): z_k(s) = +1 / -1 for bit N-1-k of s equal to 0 / 1."""
    s = np.arange(1 << N)
    return np.array([1.0 - 2.0 * ((s >> (N - 1 - k)) & 1) for k in range(N)])


def string_signs(z: np.ndarray, strings) -> np.ndarray:
    """(len(strings), 2^N): the diagonal of each Z-string."""
    return np.array([np.prod(z[list(b)], axis=0) for b in strings])


# ---------------------------------------------------------------------------
#  Walsh-Hadamard transform (in place, normalised) on the row / column index
# ---------------------------------------------------------------------------
def fwht_rows(a: np.ndarray, N: int) -> None:
    """a <- H^{(x)N} a for C-contiguous a of shape (2^N, m)."""
    for k in range(N):
        v = a.reshape(1 << k, 2, -1)
        u = v[:, 0, :].copy()
        v[:, 0, :] += v[:, 1, :]
        np.subtract(u, v[:, 1, :], out=v[:, 1, :])
    a *= 2.0 ** (-N / 2.0)


def fwht_cols(a: np.ndarray, N: int) -> None:
    """a <- a H^{(x)N} for C-contiguous a of shape (m, 2^N)."""
    m = a.shape[0]
    for k in range(N):
        v = a.reshape(m << k, 2, 1 << (N - k - 1))
        u = v[:, 0, :].copy()
        v[:, 0, :] += v[:, 1, :]
        np.subtract(u, v[:, 1, :], out=v[:, 1, :])
    a *= 2.0 ** (-N / 2.0)


def hadamard_dense(N: int) -> np.ndarray:
    h1 = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    return reduce(np.kron, [h1] * N)


# ---------------------------------------------------------------------------
#  KW operator
# ---------------------------------------------------------------------------
class KWMap:
    """D = M H^{(x)N},  <mu|D|x>_X = c delta(x_j = prod_{k in dual[j]} mu_k),
    c^2 = 1 / |kernel|.  See the module docstring."""

    def __init__(self, geom: Geometry):
        N = geom.N
        d = 1 << N
        z = site_signs(N)
        xbits = (string_signs(z, geom.dual) < 0).astype(np.int64)    # (N, d)
        weights = (1 << (N - 1 - np.arange(N))).astype(np.int64)
        self.xmap = (weights[:, None] * xbits).sum(axis=0)          # mu -> x
        counts = np.bincount(self.xmap, minlength=d)
        hit = counts[counts > 0]
        if not np.all(hit == hit[0]):
            raise ValueError('KW map with a non-uniform kernel')
        self.kernel = int(hit[0])
        self.c2 = 1.0 / self.kernel
        self.N = N

    def dense(self) -> np.ndarray:
        """The 2^N x 2^N matrix of D (small N only)."""
        d = 1 << self.N
        M = np.zeros((d, d))
        M[np.arange(d), self.xmap] = np.sqrt(self.c2)
        return M @ hadamard_dense(self.N)

    def conjugate(self, rho: np.ndarray) -> np.ndarray:
        """D rho D^dag (rho is not modified)."""
        R = np.array(rho, dtype=complex, order='C', copy=True)
        fwht_rows(R, self.N)
        fwht_cols(R, self.N)
        return self.c2 * R[np.ix_(self.xmap, self.xmap)]


# ---------------------------------------------------------------------------
#  Matrix-free Liouvillian
# ---------------------------------------------------------------------------
def _add_flip_row(acc, a, k, N):
    """acc += X_k a."""
    s = (1 << k, 2, -1)
    o = acc.reshape(s)
    np.add(o, a.reshape(s)[:, ::-1, :], out=o)


def _add_flip_col(acc, a, k, N):
    """acc += a X_k."""
    s = ((1 << N) << k, 2, 1 << (N - k - 1))
    o = acc.reshape(s)
    np.add(o, a.reshape(s)[:, ::-1, :], out=o)


def _add_flip_both(acc, a, k, N):
    """acc += X_k a X_k."""
    r = 1 << (N - k - 1)
    s = (1 << k, 2, r << k, 2, r)
    o = acc.reshape(s)
    np.add(o, a.reshape(s)[:, ::-1, :, ::-1, :], out=o)


class Model10Liouvillian:
    """rho -> L(rho) for H = -J sum_b Z_b - h sum_j X_j with local or
    collective jumps (module docstring).  Calls return a fresh array (the ODE
    solver keeps its stages); internal buffers are reused."""

    def __init__(self, geom: Geometry, J: float, h: float, d1: float, d2: float,
                 jumps: str = 'local'):
        if jumps not in JUMP_KINDS:
            raise ValueError(f'jumps must be one of {JUMP_KINDS}')
        N = geom.N
        self.N, self.d = N, 1 << N
        self.h, self.d1, self.d2, self.jumps = float(h), float(d1), float(d2), jumps
        z = site_signs(N)
        zb = string_signs(z, geom.bonds)                       # (n_bonds, d)
        e = -float(J) * zb.sum(axis=0)
        G = (-1j) * (e[:, None] - e[None, :])
        if jumps == 'local':
            if self.d2 != 0.0:
                G += self.d2 * (zb.T @ zb)                     # sum_b z_b(i) z_b(j)
            G -= N * self.d1 + len(geom.bonds) * self.d2
        else:
            if self.d2 != 0.0:
                B = zb.sum(axis=0)
                G += self.d2 * (np.outer(B, B)
                                - 0.5 * (B[:, None] ** 2 + B[None, :] ** 2))
        self.G = np.ascontiguousarray(G)
        n_buf = 0
        if self.h != 0.0 or self.d1 != 0.0:
            n_buf = 3 if jumps == 'local' else 5
        self._buf = [np.empty((self.d, self.d), dtype=complex) for _ in range(n_buf)]

    def _sum_flips(self, buf, a, adder):
        buf.fill(0.0)
        for k in range(self.N):
            adder(buf, a, k, self.N)
        return buf

    def __call__(self, rho: np.ndarray) -> np.ndarray:
        N = self.N
        out = self.G * rho
        if not self._buf:
            return out
        if self.jumps == 'local':
            Xr, Xc, Xrc = self._buf
            if self.h != 0.0:
                self._sum_flips(Xr, rho, _add_flip_row)
                self._sum_flips(Xc, rho, _add_flip_col)
                np.subtract(Xr, Xc, out=Xr)
                Xr *= 1j * self.h                    # -i[-h sum X, rho]
                out += Xr
            if self.d1 != 0.0:
                self._sum_flips(Xrc, rho, _add_flip_both)
                Xrc *= self.d1                       # + D1 sum_j X_j rho X_j
                out += Xrc                           # (-D1 N rho is in G)
            return out
        # collective: A = sum_j X_j
        Ar, rA, ArA, AAr, rAA = self._buf
        self._sum_flips(Ar, rho, _add_flip_row)      # A rho
        self._sum_flips(rA, rho, _add_flip_col)      # rho A
        if self.d1 != 0.0:
            self._sum_flips(ArA, Ar, _add_flip_col)  # A rho A
            self._sum_flips(AAr, Ar, _add_flip_row)  # A^2 rho
            self._sum_flips(rAA, rA, _add_flip_col)  # rho A^2
            AAr += rAA
            AAr *= 0.5
            np.subtract(ArA, AAr, out=ArA)
            ArA *= self.d1
            out += ArA
        if self.h != 0.0:
            np.subtract(Ar, rA, out=Ar)
            Ar *= 1j * self.h
            out += Ar
        return out


# ---------------------------------------------------------------------------
#  Initial states (all eta-even)
# ---------------------------------------------------------------------------
def initial_state(N: int, kind: str = 'ghz_s', alpha: float = ALPHA_PAPER) -> np.ndarray:
    """State vector.  'ghz_s': alpha |GHZ> + sqrt(1 - alpha^2) |S>, |S> the
    equal superposition of every basis state except 0...0 and 1...1 (the
    paper's N = 3 state at alpha = sqrt(3)/2); 'ghz'; 'plus' = |+>^N."""
    d = 1 << N
    if kind == 'plus':
        return np.full(d, d ** -0.5, dtype=complex)
    ghz = np.zeros(d, dtype=complex)
    ghz[[0, d - 1]] = 2 ** -0.5
    if kind == 'ghz':
        return ghz
    if kind == 'ghz_s':
        S = np.full(d, (d - 2) ** -0.5, dtype=complex)
        S[[0, d - 1]] = 0.0
        return alpha * ghz + np.sqrt(1.0 - alpha ** 2) * S
    raise ValueError(f'init must be one of {INIT_KINDS}')


# ---------------------------------------------------------------------------
#  Observables and evolution
# ---------------------------------------------------------------------------
def kw_observables(rho: np.ndarray, kw: KWMap) -> dict:
    rh = kw.conjugate(rho)
    a = float(np.vdot(rho, rho).real)
    b = float(np.vdot(rh, rh).real)
    c = float(np.vdot(rho, rh).real)
    return dict(a=a, b=b, c=c,
                tr=float(np.trace(rho).real), tr_hat=float(np.trace(rh).real),
                dS_gen=float(np.log(4.0 * a / (a + b + 2.0 * c))),
                dS_even=float(np.log(2.0 * a / (a + c))))


OBS_KEYS = ('a', 'b', 'c', 'tr', 'tr_hat', 'dS_gen', 'dS_even')


def evolve_kw_asymmetry(geom: Geometry, J: float, h: float, d1: float, d2: float,
                        jumps: str, rho0: np.ndarray, ts: np.ndarray, *,
                        rtol: float = 1e-8, atol: float = 1e-12,
                        log: Callable = print, log_every: int = 25) -> dict:
    """Integrate d rho/dt = L(rho) from ts[0] = 0 and record OBS_KEYS at every
    ts (dense output of DOP853, so only the current state is held)."""
    from scipy.integrate import DOP853
    ts = np.asarray(ts, float)
    if ts[0] != 0.0 or np.any(np.diff(ts) <= 0):
        raise ValueError('ts must start at 0 and increase')
    L = Model10Liouvillian(geom, J, h, d1, d2, jumps)
    kw = KWMap(geom)
    d = L.d
    rec = {k: np.full(len(ts), np.nan) for k in OBS_KEYS}

    def record(i, y):
        for k, v in kw_observables(y.reshape(d, d), kw).items():
            rec[k][i] = v

    record(0, rho0.ravel())
    if len(ts) == 1:
        return dict(rec, n_steps=0, nfev=0)
    solver = DOP853(lambda t, y: L(y.reshape(d, d)).ravel(), 0.0,
                    np.array(rho0, dtype=complex).ravel(), ts[-1],
                    rtol=rtol, atol=atol)
    i, n_steps, t0 = 1, 0, time.time()
    while i < len(ts):
        msg = solver.step()
        if solver.status == 'failed':
            raise RuntimeError(f'DOP853 failed at t = {solver.t}: {msg}')
        n_steps += 1
        if ts[i] <= solver.t:
            interp = None
            while i < len(ts) and ts[i] <= solver.t:
                if ts[i] == solver.t:
                    y = solver.y
                else:
                    interp = interp or solver.dense_output()
                    y = interp(ts[i])
                record(i, y)
                i += 1
        if n_steps % log_every == 0:
            log(f'  step {n_steps}: t = {solver.t:.4f}, dt = {solver.step_size:.3e}, '
                f'nfev = {solver.nfev}, {time.time() - t0:.0f} s')
    return dict(rec, n_steps=n_steps, nfev=solver.nfev)


# ---------------------------------------------------------------------------
#  Self-check (small ring, dense)
# ---------------------------------------------------------------------------
def _zstring_dense(sites, N):
    return reduce(np.kron, [_SZ if k in sites else _I2 for k in range(N)])


def _dense_lindblad(H, jumps, rho):
    out = -1j * (H @ rho - rho @ H)
    for Lk in jumps:
        LdL = Lk.conj().T @ Lk
        out += Lk @ rho @ Lk.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    return out


def _dense_terms(geom, J, h, d1, d2, jumps):
    N = geom.N
    X = [_site_op(_SX, k, N).toarray() for k in range(N)]
    Zb = [_zstring_dense(b, N) for b in geom.bonds]
    H = -J * sum(Zb) - h * sum(X)
    if jumps == 'local':
        Ls = [np.sqrt(d1) * x for x in X] + [np.sqrt(d2) * zz for zz in Zb]
    else:
        Ls = [np.sqrt(d1) * sum(X), np.sqrt(d2) * sum(Zb)]
    return H, Ls


def selfcheck(N: int = 5, J: float = 1.0, h: float = 0.7, d1: float = 0.4,
              d2: float = 0.9, seed: int = 0, tol: float = 1e-10,
              log: Callable = print) -> dict:
    """Dense checks on the N-site ring (module docstring).  Raises
    AssertionError naming every failed check; returns {check: error}."""
    geom = ring_geometry(N)
    d = 1 << N
    kw = KWMap(geom)
    D = kw.dense()
    X = [_site_op(_SX, k, N).toarray() for k in range(N)]
    Zb = [_zstring_dense(b, N) for b in geom.bonds]
    eta = reduce(np.matmul, X)
    Pp = 0.5 * (np.eye(d) + eta)
    err = {}
    err['D X_j = Z_dual D'] = max(np.abs(D @ X[j] - _zstring_dense(geom.dual[j], N) @ D).max()
                                  for j in range(N))
    err['D Z_b = X_shift D'] = max(np.abs(D @ Zb[b] - X[geom.shift[b]] @ D).max()
                                   for b in range(len(geom.bonds)))
    err['D^dag D = P_+'] = np.abs(D.conj().T @ D - Pp).max()
    err['D D^dag = P_+'] = np.abs(D @ D.conj().T - Pp).max()
    HJh = -J * sum(Zb) - h * sum(X)
    HhJ = -h * sum(Zb) - J * sum(X)
    err['D H(J,h) = H(h,J) D'] = np.abs(D @ HJh - HhJ @ D).max()

    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho_any = A @ A.conj().T
    rho_any /= np.trace(rho_any)
    rho = Pp @ rho_any @ Pp
    rho /= np.trace(rho)
    err['fast D rho D^dag'] = np.abs(kw.conjugate(rho_any) - D @ rho_any @ D.conj().T).max()
    ob = kw_observables(rho, kw)
    err['b = a (even sector)'] = abs(ob['b'] - ob['a'])
    err['Tr rho_hat = 1 (even sector)'] = abs(ob['tr_hat'] - 1.0)
    err['dS_gen = dS_even (even sector)'] = abs(ob['dS_gen'] - ob['dS_even'])

    for jumps in JUMP_KINDS:
        H, Ls = _dense_terms(geom, J, h, d1, d2, jumps)
        ref = _dense_lindblad(H, Ls, rho_any)
        err[f'matrix-free L ({jumps})'] = np.abs(
            Model10Liouvillian(geom, J, h, d1, d2, jumps)(rho_any) - ref).max()
        Hd, Lsd = _dense_terms(geom, h, J, d2, d1, jumps)        # dual point
        lhs = D @ _dense_lindblad(H, Ls, rho) @ D.conj().T
        rhs = _dense_lindblad(Hd, Lsd, D @ rho @ D.conj().T)
        err[f'Liouvillian duality ({jumps})'] = np.abs(lhs - rhs).max()
    # the repo's own Liouvillian (local jumps on the ring edges, column-stacking vec)
    ZZ = np.kron(_SZ, _SZ)
    Lrepo = build_lindbladian_from_terms(-h * _SX, -J * ZZ, [np.sqrt(d1) * _SX],
                                         [np.sqrt(d2) * ZZ], N, list(geom.bonds))
    vec = Lrepo @ rho_any.reshape(-1, order='F')
    err['matrix-free L = build_lindbladian_from_terms'] = np.abs(
        Model10Liouvillian(geom, J, h, d1, d2, 'local')(rho_any)
        - vec.reshape(d, d, order='F')).max()

    bad = {k: v for k, v in err.items() if not v < tol}
    for k, v in err.items():
        log(f'  selfcheck {k:45s} {v:.2e}  {"FAIL" if k in bad else "ok"}')
    if bad:
        raise AssertionError(f'KW selfcheck failed: {bad}')
    return err


# ---------------------------------------------------------------------------
#  The 200-point scan (J = 1)
# ---------------------------------------------------------------------------
DIAG_H = np.round(np.linspace(0.3, 1.7, 15), 10)       # h / J, self-dual h = J at the centre
DIAG_GAMMA = np.round(np.linspace(0.0, 1.8, 10), 10)   # D1 = D2 = gamma
OFF_D1 = np.round(np.linspace(0.0, 1.8, 10), 10)       # h = J, D1 != D2 slice
OFF_D2 = np.array([0.2, 0.6, 1.0, 1.4, 1.8])


def scan_points() -> list:
    """200 points: 'diag' (h, gamma) panel, h-major (150), then the 'off'
    (D1, D2) panel at h = J = 1, D1-major (50).  The off panel's D1 = D2 points
    repeat diag points at h = 1 (a free cross-check)."""
    pts = [dict(panel='diag', h=float(h), d1=float(g), d2=float(g), i=i, j=j)
           for i, h in enumerate(DIAG_H) for j, g in enumerate(DIAG_GAMMA)]
    pts += [dict(panel='off', h=1.0, d1=float(a), d2=float(b), i=i, j=j)
            for i, a in enumerate(OFF_D1) for j, b in enumerate(OFF_D2)]
    return pts


def run_tag(geom_name: str, jumps: str, init: str, alpha: float) -> str:
    tag = f'{geom_name}_{jumps}_{init}'
    if init == 'ghz_s' and abs(alpha - ALPHA_PAPER) > 1e-12:
        tag += f'_a{alpha:.4f}'
    return tag
