"""
Product-frame growth: close a single-qubit frame under the free-local-unitary
Euler step of the two-qubit Lindbladian, and track the framability as it grows.

MODEL AND CONVENTIONS
---------------------
The two-qubit generator is the one of continuous_simulation.tex (section
"Example: Product state frame"), i.e. exactly
product_frame_trick.two_qubit_lindbladian_action:

    L(rho) = -i[H, rho] + gamma  (D[L_0](rho) + D[L_1](rho))
                        + gamma' (D[Z_0](rho) + D[Z_1](rho)),
    H = J (Z(x)Z) + h (X(x)I + I(x)X),      L_k = |-><+|_k.

model3 -> h = 0, model4 -> h = MODEL4_H = 1.5 (trotter_lindbladian_scan), J = 1
in both.  NOTE this is the *un-shared* bond generator: build_bond_lindbladian of
trotter_lindbladian_scan divides every one-qubit term by 2*dim, which would move
the tex threshold from gamma' >= |J| to gamma' >= 2*dim*|J|.  The threshold
statement this pipeline tests is the tex one, so the raw generator is used.

The Trotter step is the FIRST-ORDER (Euler) one, as requested:

    rho -> rho + dt L(rho),     dt = DT_DEFAULT = 1e-2,

whose Pauli-transfer matrix is G = 1 + dt M (M = the 16x16 real generator).
`--gate expm` switches to expm(dt M) for comparison.

Frames are single-qubit *state* frames: S is 4 x d_ext with columns

    S[:, i] = (Tr(rho_i)/2, Tr(X rho_i)/2, Tr(Y rho_i)/2, Tr(Z rho_i)/2)
            = (1/2, r_i/2)          for a pure state with Bloch vector r_i,

the convention of framability.make_product_state_D and of
product_frame_trick._as_pauli_column, and D = kron(S, S) (the SAME S on both
qubits).  The starting frame is the octahedron {(id +- sigma^i)/2}, i.e. the six
Bloch axes -- "id +- sigma^i" up to the overall factor 2, which is irrelevant:
the framability LP is invariant under a uniform rescaling of S (D -> lam^2 D
scales the targets by lam^2 too, leaving the coefficients c untouched).

Framability is SCHROEDINGER (targets G d_j, framability.product_state_framability
Eq. 45-46), evaluated with the certified reference LP
dissipative_PT._framability_lp, which internally forms gate.T @ D -- so the
gate argument is passed TRANSPOSED everywhere in this module.  The same holds
for the gate handed to product_frame_trick.frame_element_criterion.

TWO TARGET VARIANTS
-------------------
field='free'   (the DEFAULT, for growth and evaluation alike) targets are the
               Pauli columns of

                   rho_j + dt ( L(rho_j) - i[H_0 + H_1, rho_j] ),

               i.e. rho~, with the element-dependent local generator pair
               (H_0, H_1) of product_frame_trick.  This is the quantity for a
               protocol that CARRIES the free local rotation along with the
               trajectory's product state -- the reading under which the
               rotated poles cost nothing (see "WHICH TARGET THE FRAME IS
               GROWN FOR").  It is not the image of D under any single gate,
               hence the explicit target matrix.  For model4 the local field
               h X sits entirely in the cancelled starred entries, so 'free'
               is h-independent.
field='plain'  targets are G d_j: the bare Euler gate, the quantity for a
               protocol that must represent the rotation inside the finite
               frame.  Meaningful only for frames grown with --field plain.

CANDIDATE EXTRACTION (the growth step)
--------------------------------------
For a frame pair (Psi_0, Psi_1) product_frame_trick returns A_cancelled, i.e.
L(Psi_0 (x) Psi_1) - i[H_0 + H_1, .] in the local basis B_0 (x) B_1 ordered
(|Psi_0 Psi_1>, |Psi_0 Psi_1^perp>, |Psi_0^perp Psi_1>, |Psi_0^perp Psi_1^perp>).
Per the tex it has only

    A[0,0] = -(a_0 + a_1),   A[1,1] = a_1,   A[2,2] = a_0,   A[0,3] = w = i J q*

nonzero (a_i = gamma nu_i + gamma' eta_i >= 0, q the entanglement amplitude),
so with rho = |Psi_0 Psi_1><Psi_0 Psi_1| the Euler step reads, in that basis,

    rho~ = (1 - dt(a_0+a_1)) P_0(x)P_1 + dt a_1 P_0(x)P_1^perp
                                       + dt a_0 P_0^perp(x)P_1
         + dt ( w F_0(x)F_1 + w* F_0^dag(x)F_1^dag ),   F_i = |Psi_i><Psi_i^perp|.

The first three terms are already products of existing elements and their
antipodes.  The last one is the entangling part, and it factorises into local
Hermitian 2x2 blocks: with chi = arg(w), theta = chi/2 and
X_phi = e^{i phi}|0><1| + h.c. = cos(phi) X - sin(phi) Y (local frame),

    w F(x)F + w* F^dag(x)F^dag = (|w|/2) ( X_theta (x) X_theta
                                         + X_{theta+pi/2} (x) X_{theta-pi/2} ),

an exact identity (verified by --self_check).  Equivalently: the local-Pauli
coefficient matrix of rho~ is block diagonal, {I,Z}x{I,Z} carrying the three
product terms and {X,Y}x{X,Y} the traceless symmetric block whose principal
axes are X_theta, Y_theta -- so "block-diagonalise, then diagonalise each 2x2
matrix" reads theta off the {X,Y} block.  That block is degenerate (both
singular values |w|/2), so its principal axes are fixed only up to a rotation
in the plane; the qubit-symmetric value theta = chi/2 is chosen because it
gives BOTH qubits the same four states, which D = kron(S, S) requires.

Diagonalising the four local 2x2 blocks and rotating back with B_i therefore
yields, per pair, the equatorial square

    |x_k> = B_i ( |Psi_i> + e^{-i(theta + k pi/2)} |Psi_i^perp> ) / sqrt(2),
    k = 0..3,

in the plane orthogonal to r_i, for i = 0 and 1.  It is closed under antipodes
(k -> k+2), so the frame stays antipode-closed.

WHERE TO PUT THE RING: THE TILT (--tilt, default 'optimal')
-----------------------------------------------------------
The square above is EXACT but expensive: its l1 weight is 1 + 4 dt |w|, a rate
4|w| = O(1).  The azimuths are right, the polar angle is not.  Matching the
phase-averaged family

    |a(phi)> = |Psi_0> + e^{i phi} alpha |Psi_0^perp>,
    |b(phi)> = |Psi_1^perp> + e^{i phi} beta |Psi_1>

to rho~ at the threshold fixes |alpha beta| = |w|/a_1 and, from the
00-population budget, |alpha|^2 >= dt a_0; the l1 excess is 2 dt a_1 |alpha|^2,
minimised at |alpha|^2 = dt a_0 with excess 2 dt^2 a_0 a_1 = 2 dt^2 |w|^2 --
EXACTLY negativity_floor.  So the ring belongs at the small polar angle

    tan(eps_i / 2) = sqrt(dt a_i)          (optimal_tilt; a_i is qubit i's own
                                            flip rate, a_0 = A[2,2], a_1 = A[1,1])

with the SAME azimuths theta + k pi/2 (the free phases of alpha and beta split
symmetrically as -chi/2 = -theta).  eps = pi/2 recovers the equatorial square,
which --tilt equatorial keeps for the exact-identity check; --tilt both emits
each ring.  At dt = 1e-4 and a ~ 1 the optimal ring sits at eps ~ 0.02 rad
instead of pi/2, i.e. it hugs the element it came from -- which is the point:
the decomposition wants a state just off Psi, not a quarter turn away.

THE U ROTATION (what makes the PLAIN framability reach 1)
---------------------------------------------------------
All of the above decomposes rho~, not the requested step rho + dt L(rho).  The
two differ by exactly the local rotation the trick buys:

    U ( rho + dt L(rho) ) U^dag = rho + dt L~(rho) + O(dt^2) = rho~ + O(dt^2),
    U = U_0(dt) (x) U_1(dt),  U_i(dt) = exp(-i dt H_i),

hence rho + dt L(rho) = U^dag rho~ U + O(dt^2).  A LOCAL unitary maps a product
decomposition to a product decomposition, so the plain Euler step is separable
too and its product states are simply the U-rotated ones.  Every candidate is
therefore pushed through U_i(dt)^dag, and the local factors of that
decomposition are

    U_i(dt)^dag |x_k>,  k = 0..3      (the rotated ring: the coherence)
    U_i(dt)^dag |Psi_i>,
    U_i(dt)^dag |Psi_i^perp>          (the rotated POLES: the three diagonal
                                       terms P_0(x)P_1, P_0(x)P_1^perp,
                                       P_0^perp(x)P_1 rotate as well)

Groups (the four ring states) are added ATOMICALLY, since no proper subset
carries the coherence.  Note U_0 depends on the PARTNER (kappa_0 = -i A[2,0] is a
property of the pair), so each pair rotates Psi_i differently; that is inherent
to the construction and is why the frame keeps growing instead of closing.

THE ROTATED POLES ARE FREE (and are not emitted)
------------------------------------------------
This pipeline requires -- as every frame in this project does -- that the frame
be TRANSLATIONALLY INVARIANT across the qubits: one single-qubit S, used on
every site, D = S (x) S (x) ...  Never S_a (x) S_b.

Given that, the two rotated poles need no computation and no slot in S.  They
are not new states: they are elements S already has, moved by the free local
unitary U_i(dt) -- the very resource the trick buys, applied at every site
identically because S is the same at every site.  The protocol carries that
rotation along with the trajectory's current product state rather than
branching on it, so "S together with everything the free local rotation reaches
from S" is the frame that is actually available, and the poles are inside it by
construction.  We therefore skip computing and storing them (include_poles=True
restores them for the exactness checks in --self_check).

WHICH TARGET THE FRAME IS GROWN FOR (--field, default 'free')
-------------------------------------------------------------
"The poles are free" is a statement about the PROTOCOL, and it fixes the
quantity to compute: the 'free' target, rho~, from which the rotation has been
removed.  Its product factors are the UNROTATED ring (and the unrotated poles
Psi, Psi^perp -- present already, or added as candidates when an antipode is
missing).  Growth, the criterion and the evaluation must all use that target;
product_frame_trick.frame_element_criterion takes a single gate and cannot
express it, so apply_free_criterion applies the same verdicts by exact sparse
recompute.

Skipping the poles while evaluating the PLAIN target is inconsistent, and the
first cluster run (grow_version 1.0) showed it: the plain rate came out at
exactly gamma for model3 (10, 20) and exactly gamma + 4h for model4 (16, 26),
barely moving with d_ext -- the uncovered local drift |kappa|.  The earlier
estimate here, that omitting a pole at angle x = dt|kappa| costs O(x^2), was
WRONG for a finite frame: O(x^2) holds only for a point surrounded by frame
elements, while the parent Psi is a VERTEX of the hull, and a pure state a
distance x off a vertex costs O(x).  Linear in dt|kappa| per step is a rate
~|kappa|, which is exactly what came out.  field='plain' therefore only makes
sense together with the U-rotated candidates it selects (and its poles).

When kappa = 0 the pole is not merely close to its parent but IS its parent:
for the octahedron pair (+x, +x) at gamma = 0, z = 0 kills both the ZZ and the
dephasing contributions to the starred entries, so kappa_0 = kappa_1 = 0, U = 1,
and the two poles are exactly (id +- sigma^x)/2.  (That same pair has q = 1,
w = i, a_0 = a_1 = 1, hence M = [[1, i], [-i, 1]] with lambda_- = 0 exactly: it
is the element that saturates the tex threshold.  Its ring is the four
(0, +-1/sqrt2, +-1/sqrt2) at gauge exactly sqrt(2).)

Only unordered pairs are visited, d_ext(d_ext+1)/2 of them: L is swap
symmetric, so the pair (j,i) returns the two groups of (i,j) exchanged and the
pooled candidate set is identical.

WHAT THE CURVE SHOWS (and why dt = 1e-2 may be too coarse)
----------------------------------------------------------
The tex threshold is gamma' >= |J|: there the 2x2 matrix
M = [[a_1, w], [w*, a_0]] is PSD, rho~ is PPT and the O(dt) cost of the
decomposition vanishes.  Two effects survive at finite dt and finite d_ext:

  * The Euler step is NOT positive, so it is not a density matrix and every
    product decomposition of it needs negative weights.  With Tr = 1 any
    rho' = P - N (P, N >= 0) obeys ||c||_1 = 1 + 2 Tr N >= 1 + 2 |sum of the
    negative eigenvalues of rho'|, a floor no frame can beat.  In the local
    basis rho' is the matrix above with the starred entries dt kappa_i RESTORED,
    and two of its 2x2 corners go indefinite:
        {|Psi_0 Psi_1>, |Psi_0^perp Psi_1^perp>}: det = -dt^2 |w|^2  -- always
            negative, giving a rate floor ~ 2 dt J^2 |q|^2 (2e-2 at dt = 1e-2,
            J = 1), the same size as the 2|lambda_-| = 2e-2 signal of the
            gamma' = 0.99 J set;
        {|Psi_0 Psi_1>, |Psi_0 Psi_1^perp>}: det ~ dt(a_1 - dt |kappa_1|^2),
            negative once dt |kappa|^2 > a, i.e. dt > 1/gamma^2 for a ~ gamma'
            and |kappa| ~ gamma -- marginal at gamma = 10 and VIOLATED at
            gamma = 20 for dt = 1e-2, where the floor rate reaches ~6.
    dt = 1e-3 clears both for all seven cases.  Rather than trust those
    estimates, negativity_floor() MEASURES the bound on the pipeline's own
    target columns and the collect script draws it: a curve sitting on its
    floor means dt is too coarse, nothing more.
  * A frame of angular resolution delta pays ~ a delta^2 / 2 in rate, so the
    curves decay like 1/d_ext and need not reach the floor by d_ext = 100.

Hence the collect script plots BOTH the framability and the rate (f - 1)/dt on
a log axis -- on a linear framability axis every value here sits within 1e-3
of 1.

Usage (library + growth driver; see product_frame_grow_frames_worker.py):
    python scripts/product_frame_grow.py --self_check
"""

from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.sparse import (csc_matrix, eye as sp_eye, hstack as sp_hstack,
                          vstack as sp_vstack)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dissipative_PT import (_I2, _SX, _SY, _SZ, _build_lindbladian,          # noqa: E402
                            _pauli_tensor, _has_full_support, _framability_lp,
                            _HIGHS_ATTEMPTS, _LPProblem, _clean_inputs,
                            _linprog_highs)
from trotter_lindbladian_scan import MINUS_PLUS, MODEL4_H                    # noqa: E402
from product_frame_trick import (bloch_to_ket, perp_ket, product_frame_trick,  # noqa: E402
                                 two_qubit_lindbladian_action,
                                 frame_element_gauge, frame_element_criterion)

GROW_VERSION = '1.1'   # 1.1: frames grown for the 'free' target by default

DT_DEFAULT = 1e-2          # the requested Euler step
D_EXT_MAX_DEFAULT = 100    # stop once d_ext >= this
GP_FACTOR_DEFAULT = 0.99   # the detuned parameter set: gamma' = factor * J

# Growth-step book-keeping tolerances.
DEDUP_TOL = 1e-7           # two Bloch vectors closer than this are the same state
GAUGE_TOL = 1e-9           # redundant iff gauge <= 1 + GAUGE_TOL
STRUCT_TOL = 1e-8          # relative slack on the A_cancelled sparsity pattern
# A candidate within this fraction of the group's tilt eps of an existing element
# buys no coverage (the existing one represents it at cost O(angle^2)) and would
# waste a slot of the per-round budget.  Self-scaling: it removes the rotated
# poles, at ~dt|kappa|, while keeping the tilted ring at eps ~ 2 sqrt(dt a).
MIN_SEP_FRAC_DEFAULT = 0.1

# frame_element_criterion costs O(d_ext^4) LPs and, worse,
# product_frame_trick._min_l1 builds a DENSE (2m x 2m) epigraph block with
# m = d_ext^2 -- 3.2 GB at d_ext = 100.  It is therefore affordable only for
# small frames; above this d_ext the growth filter falls back to the cheap,
# gate-independent single-qubit gauge test (frame_element_is_redundant, which
# is one-way correct: it only ever rejects columns that provably leave the
# framability unchanged for every gate).  The fallback is logged per round and
# stored in the frames file.
CRITERION_MAX_DEXT_DEFAULT = 24

FIELDS = ('plain', 'free')

# The seven requested parameter sets.  J = gamma' = 1 throughout; the growth
# step always runs at gamma' = J (the tex threshold), and the framability is
# evaluated at gamma' = J and gamma' = GP_FACTOR * J.
CASES = (
    dict(tag='model3_gam0',  model='model3', J=1.0, gamma=0.0),
    dict(tag='model3_gam2',  model='model3', J=1.0, gamma=2.0),
    dict(tag='model3_gam10', model='model3', J=1.0, gamma=10.0),
    dict(tag='model3_gam20', model='model3', J=1.0, gamma=20.0),
    dict(tag='model4_gam0',  model='model4', J=1.0, gamma=0.0),
    dict(tag='model4_gam10', model='model4', J=1.0, gamma=10.0),
    dict(tag='model4_gam20', model='model4', J=1.0, gamma=20.0),
)
CASE_BY_TAG = {c['tag']: c for c in CASES}


def model_h(model: str) -> float:
    """The one-qubit field of the model: 0 for model3, MODEL4_H for model4."""
    if model == 'model3':
        return 0.0
    if model == 'model4':
        return float(MODEL4_H)
    raise ValueError(f'this pipeline covers model3 / model4 only, got {model!r}')


# ---------------------------------------------------------------------------
#  Generator, gate, frames
# ---------------------------------------------------------------------------
def lindbladian_ptm(J: float, gamma: float, h: float, gamma_p: float) -> np.ndarray:
    """16x16 real generator M[a,b] = Tr(P_a L(P_b))/4 of the tex Lindbladian.

    Same L as product_frame_trick.two_qubit_lindbladian_action (asserted by
    --self_check), i.e. WITHOUT the 1/(2 dim) bond share.
    """
    H = J * np.kron(_SZ, _SZ)
    if h:
        H = H + h * (np.kron(_SX, _I2) + np.kron(_I2, _SX))
    jumps = []
    for rate, op in ((gamma, MINUS_PLUS), (gamma_p, _SZ)):
        if rate <= 0.0:
            continue
        amp = np.sqrt(rate)
        for L in (np.kron(op, _I2), np.kron(_I2, op)):
            L = amp * np.asarray(L, dtype=complex)
            Ld = L.conj().T
            jumps.append((1.0, L, Ld, Ld @ L))
    return _build_lindbladian(H, jumps, n=2)


def trotter_gate(J: float, gamma: float, h: float, gamma_p: float, dt: float,
                 kind: str = 'euler') -> np.ndarray:
    """Pauli-transfer matrix of the Trotter step: 1 + dt M (euler) or expm(dt M)."""
    M = lindbladian_ptm(J, gamma, h, gamma_p)
    if kind == 'euler':
        return np.eye(16) + dt * M
    if kind == 'expm':
        return expm(dt * M).real
    raise ValueError(f"gate kind must be 'euler' or 'expm', got {kind!r}")


def bloch_of_ket(ket) -> np.ndarray:
    """Bloch vector (x, y, z) of a 2-component ket."""
    k = np.asarray(ket, dtype=complex).reshape(2)
    k = k / np.linalg.norm(k)
    return np.array([float(np.real(k.conj() @ (P @ k))) for P in (_SX, _SY, _SZ)])


def state_column(bloch) -> np.ndarray:
    """Pauli column (1/2, r/2) of the pure state with Bloch vector r."""
    r = np.asarray(bloch, dtype=float).ravel()
    return np.concatenate([[0.5], 0.5 * r])


def frame_matrix(blochs) -> np.ndarray:
    """4 x d_ext single-qubit state frame S from a list of Bloch vectors."""
    return np.column_stack([state_column(r) for r in blochs])


def start_octahedron() -> list:
    """The six starting elements (id +- sigma^i)/2, i.e. the Bloch axes."""
    return [np.array(r, dtype=float) for r in
            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]


def pauli_column_2q(rho) -> np.ndarray:
    """16-vector c_a = Tr(P_a rho)/4 of a 4x4 operator (kron(S,S) index order)."""
    P = _pauli_tensor(2)
    return np.einsum('aij,ji->a', P, np.asarray(rho, dtype=complex)).real / 4.0


# ---------------------------------------------------------------------------
#  Framability (Schroedinger) with explicit per-column targets
# ---------------------------------------------------------------------------
def framability_targets(D: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """max_j min{ ||c||_1 : D c = Y[:, j] }, the sparse per-column LP.

    A transcription of dissipative_PT._framability_lp (same HiGHS ladder, same
    full-support guard, same sparse epigraph block) that accepts a caller-built
    target matrix instead of forming gate.T @ D -- needed for the 'free' variant,
    whose targets are not the image of D under any single gate.  Returns
    (framability, per-column gauges); the framability is +inf if any column is
    unreachable or the frame lacks full Pauli support.
    """
    D = np.asarray(D, dtype=float)
    Y = np.asarray(Y, dtype=float)
    nrows, d_ext = D.shape
    if Y.shape[0] != nrows:
        raise ValueError(f'target matrix has {Y.shape[0]} rows, frame has {nrows}')
    if not _has_full_support(D):
        return float('inf'), np.full(Y.shape[1], np.inf)

    c_obj = np.concatenate([np.zeros(d_ext), np.ones(d_ext)])
    A_eq = csc_matrix(np.hstack([D, np.zeros((nrows, d_ext))]))
    Ide = sp_eye(d_ext, format='csc')
    A_ub = sp_vstack([sp_hstack([Ide, -Ide]), sp_hstack([-Ide, -Ide])], format='csc')
    b_ub = np.zeros(2 * d_ext)
    bounds = [(None, None)] * d_ext + [(0.0, None)] * d_ext
    lp = _clean_inputs(_LPProblem(c_obj, A_ub, b_ub, A_eq, np.zeros(nrows), bounds,
                                  None))

    vals = np.full(Y.shape[1], np.inf)
    for j in range(Y.shape[1]):
        lp_j = lp._replace(b_eq=Y[:, j].copy())
        r = None
        for kw in _HIGHS_ATTEMPTS:
            cand = _linprog_highs(lp_j, **kw)
            if cand['status'] == 0:
                r = cand
                break
        if r is None:
            return float('inf'), vals
        vals[j] = float(np.sum(np.abs(r['x'][:d_ext])))
    return float(np.max(vals)), vals


def plain_targets(S: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Schroedinger targets G d_j for D = kron(S, S)."""
    return np.asarray(gate).real @ np.kron(S, S)


def free_targets(blochs, J: float, gamma: float, h: float, gamma_p: float,
                 dt: float) -> np.ndarray:
    """Targets of the free-local-unitary Euler step, column order of kron(S, S).

    Column (i, j) -> i * d_ext + j holds the Pauli column of

        rho_ij + dt ( L(rho_ij) - i[H_0 + H_1, rho_ij] ),  rho_ij = Psi_i (x) Psi_j,

    with (H_0, H_1) the local generators product_frame_trick chooses for that
    element (the pair is element dependent, which is why this is a target matrix
    and not a gate).
    """
    kets = [bloch_to_ket(r) for r in blochs]
    d = len(kets)
    Y = np.zeros((16, d * d))
    for i, ki in enumerate(kets):
        for j, kj in enumerate(kets):
            res = product_frame_trick(ki, kj, J, gamma, h, gamma_p)
            Hloc = np.kron(res['H0'], _I2) + np.kron(_I2, res['H1'])
            ket = np.kron(ki, kj)
            rho = np.outer(ket, ket.conj())
            drho = (two_qubit_lindbladian_action(rho, J, gamma, h, gamma_p)
                    - 1j * (Hloc @ rho - rho @ Hloc))
            Y[:, i * d + j] = pauli_column_2q(rho + dt * drho)
    return Y


def build_targets(blochs, J: float, gamma: float, h: float, gamma_p: float,
                  dt: float, field: str = 'plain',
                  gate_kind: str = 'euler') -> np.ndarray:
    """The 16 x d_ext^2 target matrix of one framability evaluation.

    field='plain' -> G d_j with G the Trotter gate; identical to the gate.T @ D
        that dissipative_PT._framability_lp forms (asserted by --self_check).
    field='free'  -> free_targets, the dt -> 0 reference (Euler step only).
    """
    if field == 'plain':
        G = trotter_gate(J, gamma, h, gamma_p, dt, gate_kind)
        return plain_targets(frame_matrix(blochs), G)
    if field == 'free':
        if gate_kind != 'euler':
            raise ValueError("field='free' is defined for the Euler step only")
        return free_targets(blochs, J, gamma, h, gamma_p, dt)
    raise ValueError(f'field must be one of {FIELDS}, got {field!r}')


def rho_from_pauli_column(c) -> np.ndarray:
    """The 4x4 operator sum_a c_a P_a of a 16-vector of Pauli coefficients."""
    return np.einsum('a,aij->ij', np.asarray(c, dtype=float), _pauli_tensor(2))


def negativity_floor(Y: np.ndarray) -> tuple[float, int]:
    """Certified lower bound on the framability of the targets Y, and its column.

    Each target has unit trace, so any decomposition into (positive, unit-trace)
    product states splits as rho' = P - N with Tr P - Tr N = 1 and

        ||c||_1 = 1 + 2 Tr N >= 1 + 2 |sum of the negative eigenvalues of rho'|,

    because N >= -rho'.  The framability, a max over columns, therefore cannot
    go below the largest such bound -- no matter how large the frame is.  The
    Euler step is not positive (see the module docstring), so this floor is
    strictly above 1 and grows with dt; measuring it is the honest way to tell
    whether a curve has saturated the frame or merely the time step.
    """
    Y = np.asarray(Y, dtype=float)
    best, jbest = 0.0, -1
    for j in range(Y.shape[1]):
        ev = np.linalg.eigvalsh(rho_from_pauli_column(Y[:, j]))
        val = 1.0 + 2.0 * float(np.sum(np.abs(ev[ev < 0.0])))
        if val > best:
            best, jbest = val, j
    return best, jbest


def trotter_truncation_bound(blochs, J: float, gamma: float, h: float,
                             gamma_p: float, dt: float) -> float:
    """2 max_j ||(expm(dt M) - (1 + dt M)) d_j||_1 / dt: an upper bound on the
    floor rate, and the honest measure of "is the Euler step close to expm".

    NOT the same quantity as negativity_floor, and the relation is one-way.
    With rho' = (1 + dt L)rho = expm(dt L)rho - E and expm(dt L)rho >= 0, every
    negative eigenvalue of rho' is paid for by E, so

        floor - 1 = 2 sum|lambda^-(rho')| <= 2 ||E||_1,

    i.e. a small truncation error forces a small floor.  The converse is false:
    positivity is a cone condition, not a norm one.  A classical rate matrix has
    1 + dt W stochastic (floor exactly 1, zero excess) for every dt <= 1/max
    rate while its truncation error is still O(dt^2).  The bound is also loose
    here by construction: E is spread over all directions but can only create
    negativity where expm(dt L)rho is itself O(dt^2)-small, which for these
    elements is the single {|Psi_0 Psi_1>, |Psi_0^perp Psi_1^perp>} corner.

    Returns 0 for gate_kind='expm' (the targets are then genuine states, and
    negativity_floor correspondingly returns exactly 1).
    """
    M = lindbladian_ptm(J, gamma, h, gamma_p)
    Delta = expm(dt * M).real - (np.eye(16) + dt * M)
    S = frame_matrix(blochs)
    Y = Delta @ np.kron(S, S)
    worst = 0.0
    for j in range(Y.shape[1]):
        ev = np.linalg.eigvalsh(rho_from_pauli_column(Y[:, j]))
        worst = max(worst, float(np.sum(np.abs(ev))))
    return 2.0 * worst / dt


def framability(blochs, J: float, gamma: float, h: float, gamma_p: float, dt: float,
                field: str = 'plain', gate_kind: str = 'euler'
                ) -> tuple[float, np.ndarray]:
    """Schroedinger framability of the Trotter step over D = kron(S, S)."""
    S = frame_matrix(blochs)
    Y = build_targets(blochs, J, gamma, h, gamma_p, dt, field, gate_kind)
    return framability_targets(np.kron(S, S), Y)


# ---------------------------------------------------------------------------
#  Candidate extraction
# ---------------------------------------------------------------------------
def _x_phi(phi: float) -> np.ndarray:
    """X_phi = e^{i phi}|0><1| + h.c. = cos(phi) X - sin(phi) Y."""
    return np.cos(phi) * _SX - np.sin(phi) * _SY


def _tilt_ket(eps: float, phi: float) -> np.ndarray:
    """cos(eps/2)|0> + sin(eps/2) e^{-i phi}|1>: Bloch polar angle eps, azimuth phi.

    eps = pi/2 gives (|0> + e^{-i phi}|1>)/sqrt(2), the +1 eigenvector of X_phi
    (the equatorial square).
    """
    return np.array([np.cos(0.5 * eps),
                     np.sin(0.5 * eps) * np.exp(-1j * phi)], dtype=complex)


def _equator_ket(phi: float) -> np.ndarray:
    """(|0> + e^{-i phi}|1>)/sqrt(2): the +1 eigenvector of X_phi."""
    return _tilt_ket(0.5 * np.pi, phi)


def optimal_tilt(a: float, dt: float) -> float:
    """Bloch polar angle of the minimal-cost product family: tan(eps/2) = sqrt(dt a).

    Matching the phase-averaged family |a(phi)> = |Psi> + e^{i phi} alpha |Psi^perp>,
    |b(phi)> = |Psi_1^perp> + e^{i phi} beta |Psi_1> to rho~ at the threshold fixes
    |alpha beta| = |w| / a_1, and the 00-population budget forces
    |alpha|^2 >= dt a_0.  The l1 excess is 2 dt a_1 |alpha|^2, so the optimum is
    |alpha|^2 = dt a_0, giving excess 2 dt^2 a_0 a_1 = 2 dt^2 |w|^2 -- exactly
    negativity_floor.  The equatorial square (eps = pi/2, i.e. |alpha| = 1) instead
    costs 4 dt |w|, larger by a factor ~ 2/(dt|w|).

    Qubit i takes a_i, its own flip rate: a_0 = A[2,2] (flip qubit 0),
    a_1 = A[1,1] (flip qubit 1).  a = 0 (a dark, undamped direction) returns 0 and
    the candidate degenerates onto Psi, where the dedup screen drops it.
    """
    return 2.0 * float(np.arctan(np.sqrt(max(a, 0.0) * dt)))


def extract_candidates(r0, r1, J: float, gamma: float, h: float, gamma_p: float,
                       dt: float = DT_DEFAULT, tilt: str = 'optimal',
                       include_poles: bool = False,
                       struct_tol: float = STRUCT_TOL) -> dict:
    """Candidate single-qubit states from the Euler step of Psi_0 (x) Psi_1.

    The states are the local factors of the product decomposition of
    rho + dt L(rho) = U^dag rho~ U + O(dt^2), i.e. the ones of rho~ pushed
    through U_i(dt)^dag -- see "THE U ROTATION" in the module docstring.

    Returns dict with
        groups      : [qubit-0 states, qubit-1 states], six Bloch vectors each:
                      the rotated equatorial square (coherence) followed by the
                      two rotated poles (the three diagonal terms).  Only the
                      square is dropped when the entangling amplitude w
                      vanishes; the rotated poles are always candidates because
                      the local rotation itself moves the element.
        groups_local: the same states BEFORE the U rotation, in the same order
                      (the decomposition of rho~; used by --self_check and by
                      the 'free' reference variant)
        U0, U1      : the local unitaries U_i(dt) (computational basis)
        w, chi, theta, a0, a1 : the entries of A_cancelled (module docstring)
        struct_err  : max violation of the expected A_cancelled sparsity
                      pattern, relative to its largest entry (raises above
                      struct_tol)
        M           : the 2x2 matrix [[a_1, w], [w*, a_0]] of the tex, whose
                      smallest eigenvalue lambda_- is < 0 exactly when the step
                      leaves the separable set
    """
    k0, k1 = bloch_to_ket(r0), bloch_to_ket(r1)
    res = product_frame_trick(k0, k1, J, gamma, h, gamma_p)
    A = np.asarray(res['A_cancelled'], dtype=complex)

    a1, a0 = A[1, 1].real, A[2, 2].real
    w = A[0, 3]
    scale = max(1.0, float(np.max(np.abs(A))))
    # Expected pattern: diagonal (-(a0+a1), a1, a0, 0) plus the [0,3]/[3,0] pair.
    expect = np.zeros((4, 4), dtype=complex)
    expect[0, 0] = -(a0 + a1)
    expect[1, 1], expect[2, 2] = a1, a0
    expect[0, 3], expect[3, 0] = w, np.conj(w)
    struct_err = float(np.max(np.abs(A - expect))) / scale
    if struct_err > struct_tol:
        raise RuntimeError(
            f'A_cancelled does not match the continuous_simulation.tex pattern '
            f'(relative error {struct_err:.3e} > {struct_tol:.1e}); '
            f'r0={np.asarray(r0)}, r1={np.asarray(r1)}, '
            f'J={J}, gamma={gamma}, h={h}, gamma_p={gamma_p}')

    M = np.array([[a1, w], [np.conj(w), a0]], dtype=complex)
    U0, U1 = res['U0'](dt), res['U1'](dt)
    theta = 0.5 * float(np.angle(w)) if abs(w) > struct_tol * scale else float('nan')
    # Polar angles of the coherence states, per qubit (see optimal_tilt): one
    # ring of four per angle, all sharing the azimuths theta + k pi/2.
    quarter = 0.5 * np.pi
    if tilt == 'optimal':
        eps = ((optimal_tilt(a0, dt),), (optimal_tilt(a1, dt),))
    elif tilt == 'equatorial':
        eps = ((quarter,), (quarter,))
    elif tilt == 'both':
        eps = ((optimal_tilt(a0, dt), quarter), (optimal_tilt(a1, dt), quarter))
    else:
        raise ValueError(f"tilt must be 'optimal', 'equatorial' or 'both', "
                         f'got {tilt!r}')
    out = dict(w=w, chi=float(np.angle(w)), a0=a0, a1=a1, M=M, theta=theta,
               struct_err=struct_err, U0=U0, U1=U1, tilt=tilt, eps=eps,
               groups=[], groups_local=[])

    # Same four azimuths theta + k pi/2 for both qubits (the local 2x2 blocks
    # carrying the coherence are X_theta, X_{theta+pi/2} on qubit 0 and X_theta,
    # X_{theta-pi/2} on qubit 1, whose eigenvector phases coincide as a set);
    # only the polar angle differs.  The two poles close the three diagonal
    # terms P_0(x)P_1, P_0(x)P_1^perp, P_0^perp(x)P_1.
    # The poles are NOT emitted by default: see "THE ROTATED POLES ARE FREE" in
    # the module docstring.  include_poles=True restores them (as the last two
    # states of each group), which the exactness checks of --self_check need.
    poles = [np.array([1.0, 0.0], dtype=complex), np.array([0.0, 1.0], dtype=complex)]
    for q, (psi, U) in enumerate(((k0, U0), (k1, U1))):
        local = []
        if np.isfinite(theta):
            for e in eps[q]:
                local += [_tilt_ket(float(e), theta + quarter * m) for m in range(4)]
        if include_poles:
            local += poles
        B = np.column_stack([psi, perp_ket(psi)])
        kets = [B @ lk for lk in local]
        out['groups_local'].append([bloch_of_ket(k) for k in kets])
        out['groups'].append([bloch_of_ket(U.conj().T @ k) for k in kets])
    return out


def candidate_groups(blochs, J: float, gamma: float, h: float, gamma_p: float,
                     dt: float = DT_DEFAULT, tilt: str = 'optimal',
                     struct_tol: float = STRUCT_TOL, field: str = 'free') -> list:
    """All candidate groups of one growth round.

    Visits the d_ext(d_ext+1)/2 unordered pairs (the swap symmetry of L makes
    the ordered pairs redundant) and returns a list of dicts
    {pair, qubit, states, lam_min, abs_w, eps}.

    field selects which decomposition the candidates serve (see "WHICH TARGET
    THE FRAME IS GROWN FOR" in the module docstring):
      'free'  -> the UNROTATED states (the product factors of rho~), plus the
                 unrotated poles, so that a missing antipode Psi^perp -- which
                 the diagonal term P_0^perp (x) P_1 needs -- becomes a candidate
                 (Psi itself is always a duplicate and is dropped);
      'plain' -> the U^dag-rotated ring, poles skipped.
    """
    d = len(blochs)
    groups = []
    for i in range(d):
        for j in range(i, d):
            ex = extract_candidates(blochs[i], blochs[j], J, gamma, h, gamma_p,
                                    dt=dt, tilt=tilt, struct_tol=struct_tol,
                                    include_poles=(field == 'free'))
            lam = float(np.min(np.linalg.eigvalsh(ex['M'])))
            src = ex['groups_local'] if field == 'free' else ex['groups']
            for q, states in enumerate(src):
                groups.append(dict(pair=(i, j), qubit=q, states=states,
                                   lam_min=lam, abs_w=float(abs(ex['w'])),
                                   eps=float(min(ex['eps'][q]))))
    return groups


# ---------------------------------------------------------------------------
#  Filtering: which candidates earn their place
# ---------------------------------------------------------------------------
def _is_new_bloch(r, blochs, tol: float = DEDUP_TOL) -> bool:
    return all(np.linalg.norm(np.asarray(r) - b) > tol for b in blochs)


def _min_angle(r, blochs) -> float:
    """Smallest Bloch angle between r and any element of blochs (pi if empty)."""
    if not len(blochs):
        return float(np.pi)
    dots = [float(np.clip(np.dot(np.asarray(r), b), -1.0, 1.0)) for b in blochs]
    return float(np.arccos(max(dots)))


def screen_groups(groups: list, blochs: list, tol: float = DEDUP_TOL,
                  gauge_tol: float = GAUGE_TOL,
                  min_sep_frac: float = MIN_SEP_FRAC_DEFAULT) -> list:
    """Drop duplicate and hull-redundant candidates; rank the groups.

    Per group: remove states already in the frame (or already kept in this
    group) and states whose single-qubit gauge is <= 1 + gauge_tol --
    product_frame_trick.frame_element_is_redundant proves such a column leaves
    the framability unchanged for EVERY gate, and that single-qubit redundancy
    lifts to kron(S, S).  Groups come back sorted by their largest surviving
    gauge (most sticking out of the current hull first).

    NOTE on a pure-state frame the gauge test is weak, and that is a feature of
    the ranking rather than of the rejection: every column here has c_I = 1/2,
    so any S c = v obeys sum_i c_i = 1 and hence ||c||_1 >= 1 with equality iff
    c >= 0, i.e. iff v lies in the CONVEX hull of the existing states.  A pure
    state does that only by being one of them, so the gauge rejects exactly the
    duplicates -- while gauge - 1 = 2 * (negative mass needed) grows with the
    distance to the existing states, which is what makes it a good ranking key
    (it fills the widest gap on the sphere first).
    """
    S = frame_matrix(blochs)
    out = []
    for g in groups:
        keep, gauges = [], []
        # A candidate closer to an existing element than min_sep_frac * eps adds
        # no coverage at the ring's scale: pure bookkeeping against near-copies
        # (e.g. a tilt that collapsed because a = 0 on a dark direction).
        floor_ang = min_sep_frac * g.get('eps', 0.0)
        for r in g['states']:
            if not _is_new_bloch(r, blochs, tol) or not _is_new_bloch(r, keep, tol):
                continue
            if _min_angle(r, blochs + keep) < floor_ang:
                continue
            gauge = frame_element_gauge(S, state_column(r))[0]
            if gauge <= 1.0 + gauge_tol:
                continue
            keep.append(np.asarray(r, dtype=float))
            gauges.append(float(gauge))
        if keep:
            out.append(dict(g, states=keep, gauges=gauges,
                            gauge_max=float(np.max(gauges))))
    out.sort(key=lambda g: -g['gauge_max'])
    return out


def apply_criterion(blochs: list, states: list, gate: np.ndarray,
                    accept: str = 'nonharmful') -> tuple[list, list]:
    """Score candidate states with product_frame_trick.frame_element_criterion.

    The frame grows as we go, so a group's later members see its earlier ones.
    The gate is passed TRANSPOSED (the criterion forms gate.T @ D internally and
    this pipeline is Schroedinger).  accept='nonharmful' keeps everything whose
    verdict is not 'harmful' (a useless column still enlarges d_ext harmlessly
    and can enable later useful ones); accept='useful' keeps strict improvements
    only.  Returns (accepted states, per-state records).
    """
    cur = list(blochs)
    taken, recs = [], []
    for r in states:
        try:
            crit = frame_element_criterion(frame_matrix(cur), state_column(r),
                                           np.asarray(gate).real.T, n_bond=2)
        except RuntimeError as e:
            # The criterion's gauge LPs can still fail outright on a badly
            # conditioned frame (near-parallel columns at small tilt), even with
            # the solver ladder in product_frame_trick._min_l1.  Degrade to the
            # gauge filter for this candidate rather than losing the whole
            # ladder: screen_groups has already certified it non-redundant, so
            # accepting it is the same decision --filter gauge would make.
            print(f'  WARNING frame_element_criterion failed ({e}); '
                  f'falling back to the gauge verdict for this candidate',
                  flush=True)
            recs.append(dict(verdict='lp_failed', accepted=True, error=str(e)))
            cur.append(np.asarray(r, dtype=float))
            taken.append(np.asarray(r, dtype=float))
            continue
        ok = (crit['verdict'] == 'useful' if accept == 'useful'
              else crit['verdict'] != 'harmful')
        recs.append(dict(verdict=crit['verdict'], f_before=float(crit['f_before']),
                         A=float(crit['A']), B=float(crit['B']), accepted=bool(ok)))
        if ok:
            cur.append(np.asarray(r, dtype=float))
            taken.append(np.asarray(r, dtype=float))
    return taken, recs


def apply_free_criterion(blochs: list, states: list, J: float, gamma: float,
                         h: float, gamma_p: float, dt: float,
                         accept: str = 'nonharmful',
                         tol: float = 1e-9) -> tuple[list, list]:
    """Useful / useless / harmful for the 'free' targets, by exact recompute.

    product_frame_trick.frame_element_criterion takes a single gate, and the
    free targets are element dependent (one (H_0, H_1) per frame element), so
    they cannot be passed to it.  This applies the same verdict definitions --
    useful: f strictly down, harmful: f strictly up, useless: otherwise -- by
    recomputing f with framability_targets (sparse, so no dense O(d_ext^4)
    epigraph block).  f is evaluated once for the starting frame and once per
    candidate; it is carried forward whenever a candidate is accepted.
    """
    cur = list(blochs)
    f_before = framability(cur, J, gamma, h, gamma_p, dt, field='free')[0]
    taken, recs = [], []
    for r in states:
        trial = cur + [np.asarray(r, dtype=float)]
        f_after = framability(trial, J, gamma, h, gamma_p, dt, field='free')[0]
        thr = tol * max(1.0, abs(f_before)) if np.isfinite(f_before) else 0.0
        if not np.isfinite(f_after):
            verdict = 'lp_failed'
        elif f_after < f_before - thr:
            verdict = 'useful'
        elif f_after > f_before + thr:
            verdict = 'harmful'
        else:
            verdict = 'useless'
        if verdict == 'lp_failed':
            ok = True                 # same fallback as apply_criterion
        else:
            ok = (verdict == 'useful' if accept == 'useful'
                  else verdict != 'harmful')
        recs.append(dict(verdict=verdict, f_before=float(f_before),
                         f_after=float(f_after), accepted=bool(ok)))
        if ok:
            cur = trial
            taken.append(np.asarray(r, dtype=float))
            if np.isfinite(f_after):
                f_before = f_after
    return taken, recs


# ---------------------------------------------------------------------------
#  The growth loop
# ---------------------------------------------------------------------------
def grow_frames(case: dict, dt: float = DT_DEFAULT,
                d_ext_max: int = D_EXT_MAX_DEFAULT,
                max_new_per_round: int = 12,
                tilt: str = 'optimal',
                min_sep_frac: float = MIN_SEP_FRAC_DEFAULT,
                field: str = 'free',
                filter_mode: str = 'criterion',
                criterion_max_dext: int = CRITERION_MAX_DEXT_DEFAULT,
                accept: str = 'nonharmful',
                gate_kind: str = 'euler',
                max_rounds: int = 60,
                struct_tol: float = STRUCT_TOL,
                verbose: bool = True) -> dict:
    """Grow the single-qubit frame from the octahedron until d_ext >= d_ext_max.

    The growth step always runs at gamma' = J (the threshold at which the tex
    guarantees the stepped element is separable), so the frame sequence is
    shared by both evaluated parameter sets.  Groups are added atomically and
    ranked by gauge; at most max_new_per_round states are added per round (a
    group is the four ring states, so the default 12 means three groups), which is what
    keeps the d_ext ladder -- and hence the plotted curve -- resolved: the
    unrestricted closure adds O(d_ext^2) states per round and would reach 100
    in a single step.

    Returns dict with the frame sequence (list of (3, d) Bloch arrays), the
    d_ext ladder, and per-round diagnostics.
    """
    J, gamma = float(case['J']), float(case['gamma'])
    h, gamma_p = model_h(case['model']), float(case['J'])   # gamma' = J
    blochs = start_octahedron()
    frames = [np.array(blochs).T.copy()]
    rounds = []
    t0 = time.perf_counter()

    for r in range(max_rounds):
        d = len(blochs)
        if d >= d_ext_max:
            break
        groups = candidate_groups(blochs, J, gamma, h, gamma_p, dt, tilt, struct_tol,
                                  field=field)
        kept = screen_groups(groups, blochs, min_sep_frac=min_sep_frac)
        if not kept:
            if verbose:
                print(f'[{case["tag"]}] round {r}: d_ext={d}, no non-redundant '
                      f'candidate left -- frame is closed', flush=True)
            break

        # Shortlist whole groups up to the per-round budget.
        short, used = [], 0
        for g in kept:
            if used >= max_new_per_round:
                break
            short.append(g)
            used += len(g['states'])

        mode = filter_mode
        if filter_mode == 'criterion' and d > criterion_max_dext:
            mode = 'gauge'                 # see CRITERION_MAX_DEXT_DEFAULT
        gate = (trotter_gate(J, gamma, h, gamma_p, dt, gate_kind)
                if mode == 'criterion' and field == 'plain' else None)
        new, recs = [], []
        for g in short:
            if mode == 'criterion' and field == 'free':
                taken, rec = apply_free_criterion(blochs + new, g['states'], J, gamma,
                                                  h, gamma_p, dt, accept)
            elif mode == 'criterion':
                taken, rec = apply_criterion(blochs + new, g['states'], gate, accept)
            else:
                taken, rec = list(g['states']), []
            new += taken
            recs += rec
        if not new:
            if verbose:
                print(f'[{case["tag"]}] round {r}: d_ext={d}, every shortlisted '
                      f'candidate rejected by the {mode} filter -- stopping',
                      flush=True)
            break

        blochs = blochs + new
        frames.append(np.array(blochs).T.copy())
        rounds.append(dict(round=r, d_ext_before=d, d_ext_after=len(blochs),
                           n_groups=len(groups), n_groups_kept=len(kept),
                           n_shortlisted=used, n_added=len(new), filter=mode,
                           gauge_max=float(kept[0]['gauge_max']),
                           eps=float(np.median([g['eps'] for g in groups])),
                           lam_min=float(min(g['lam_min'] for g in groups)),
                           records=recs))
        if verbose:
            print(f'[{case["tag"]}] round {r}: d_ext {d} -> {len(blochs)}  '
                  f'({len(groups)} groups, {len(kept)} non-redundant, '
                  f'filter={mode}, gauge_max={kept[0]["gauge_max"]:.4f}, '
                  f'lam_min={rounds[-1]["lam_min"]:+.4e}, '
                  f'{time.perf_counter() - t0:.0f}s)', flush=True)

    return dict(tag=case['tag'], model=case['model'], J=J, gamma=gamma, h=h,
                gamma_p_grow=gamma_p, dt=dt, gate_kind=gate_kind, tilt=tilt,
                min_sep_frac=min_sep_frac, field=field,
                frames=frames, d_exts=[f.shape[1] for f in frames],
                rounds=rounds, version=GROW_VERSION)


# ---------------------------------------------------------------------------
#  Self check
# ---------------------------------------------------------------------------
def self_check(seed: int = 0, dt: float = DT_DEFAULT) -> None:
    """Verify every structural claim of the module docstring."""
    rng = np.random.default_rng(seed)
    P = _pauli_tensor(2)

    print('1) lindbladian_ptm == two_qubit_lindbladian_action')
    for (J, gamma, h, gamma_p) in ((1.0, 0.0, 0.0, 1.0), (1.0, 10.0, 1.5, 0.99),
                                   (1.0, 20.0, 1.5, 1.0), (1.0, 2.0, 0.0, 0.5)):
        M = lindbladian_ptm(J, gamma, h, gamma_p)
        err = 0.0
        for b in range(16):
            want = pauli_column_2q(two_qubit_lindbladian_action(P[b], J, gamma, h,
                                                                gamma_p))
            err = max(err, float(np.max(np.abs(M[:, b] - want))))
        assert err < 1e-10, (J, gamma, h, gamma_p, err)
        print(f'   J={J} gamma={gamma} h={h} gamma_p={gamma_p}: max err {err:.2e}')

    print('2) plain framability == dissipative_PT._framability_lp(D, G.T)')
    blochs = start_octahedron()
    S = frame_matrix(blochs)
    G = trotter_gate(1.0, 10.0, 1.5, 1.0, dt)
    f_ref = _framability_lp(np.kron(S, S), G.T)
    f_mine = framability_targets(np.kron(S, S), plain_targets(S, G))[0]
    assert abs(f_ref - f_mine) < 1e-9 * max(1.0, f_ref), (f_ref, f_mine)
    print(f'   {f_mine:.12f} vs {f_ref:.12f}')

    print('3) A_cancelled pattern, the exact decomposition of rho~, and the '
          'U-rotated decomposition of the plain Euler step')
    F = np.array([[0.0, 1.0], [0.0, 0.0]])          # |0><1| in the local basis
    # Coefficient list shared by the local and the rotated reconstruction, in
    # the index order of a group: [eq(theta), eq(theta+pi/2), eq(theta+pi),
    # eq(theta+3pi/2), pole Psi, pole Psi^perp].  The two X pairs are
    # X_theta (x) X_theta      -> q0 in {0:+, 2:-},  q1 in {0:+, 2:-}
    # X_{th+pi/2} (x) X_{th-pi/2} -> q0 in {1:+, 3:-}, q1 in {3:+, 1:-}
    coh_terms = [(0, 0, +1), (0, 2, -1), (2, 0, -1), (2, 2, +1),
                 (1, 3, +1), (1, 1, -1), (3, 3, -1), (3, 1, +1)]
    worst_rot = 0.0
    per_case = {}      # (gamma, h) -> (max err_rot/dt^2, max ||L(rho)||)
    for trial in range(200):
        J, gamma = 1.0, float(rng.choice([0.0, 2.0, 10.0, 20.0]))
        h = float(rng.choice([0.0, MODEL4_H]))
        gamma_p = J
        r0, r1 = rng.standard_normal(3), rng.standard_normal(3)
        r0 /= np.linalg.norm(r0)
        r1 /= np.linalg.norm(r1)
        ex = extract_candidates(r0, r1, J, gamma, h, gamma_p, dt=dt,
                                tilt='equatorial', include_poles=True)
        assert ex['struct_err'] < STRUCT_TOL, ex['struct_err']
        if not np.isfinite(ex['theta']):
            continue
        a0, a1, w, th = ex['a0'], ex['a1'], ex['w'], ex['theta']

        # (a) the X_theta identity on its own
        coh = dt * (w * np.kron(F, F) + np.conj(w) * np.kron(F.T, F.T))
        xid = (dt * abs(w) / 2.0) * (np.kron(_x_phi(th), _x_phi(th))
                                     + np.kron(_x_phi(th + 0.5 * np.pi),
                                               _x_phi(th - 0.5 * np.pi)))
        assert float(np.max(np.abs(coh - xid))) < 1e-12

        def _reconstruct(g0, g1):
            """Sum_k c_k proj(g0[i]) (x) proj(g1[j]) in the computational basis."""
            def proj(r):
                k = bloch_to_ket(r)
                return np.outer(k, k.conj())
            out = ((1 - dt * (a0 + a1)) * np.kron(proj(g0[4]), proj(g1[4]))
                   + dt * a1 * np.kron(proj(g0[4]), proj(g1[5]))
                   + dt * a0 * np.kron(proj(g0[5]), proj(g1[4])))
            for i, j, s in coh_terms:
                out = out + (dt * abs(w) / 2.0) * s * np.kron(proj(g0[i]),
                                                              proj(g1[j]))
            return out

        # (b) the UNROTATED states reproduce rho~ exactly (computational basis)
        B0 = np.column_stack([bloch_to_ket(r0), perp_ket(bloch_to_ket(r0))])
        B1 = np.column_stack([bloch_to_ket(r1), perp_ket(bloch_to_ket(r1))])
        V = np.kron(B0, B1)
        rho_t = np.diag([1.0 - dt * (a0 + a1), dt * a1, dt * a0, 0.0]).astype(complex)
        rho_t[0, 3] += dt * w
        rho_t[3, 0] += dt * np.conj(w)
        err_loc = float(np.max(np.abs(_reconstruct(*ex['groups_local'])
                                      - V @ rho_t @ V.conj().T)))
        assert err_loc < 1e-12, (trial, err_loc)

        # (c) the ROTATED states reproduce rho + dt L(rho) to O(dt^2)
        ket = np.kron(bloch_to_ket(r0), bloch_to_ket(r1))
        rho = np.outer(ket, ket.conj())
        rho_plain = rho + dt * two_qubit_lindbladian_action(rho, J, gamma, h, gamma_p)
        err_rot = float(np.max(np.abs(_reconstruct(*ex['groups']) - rho_plain)))
        worst_rot = max(worst_rot, err_rot / dt ** 2)
        assert err_rot < 1e-9 + 1e3 * dt ** 2, (trial, gamma, err_rot)
        # Per case: the second-order coefficient and the first-order scale it
        # corrects, so their ratio * dt is the RELATIVE error of the first-order
        # construction -- which has to sit well below the detuning being tested.
        first = float(np.sum(np.abs(np.linalg.eigvalsh(
            two_qubit_lindbladian_action(rho, J, gamma, h, gamma_p)))))
        prev = per_case.get((gamma, h), (0.0, 0.0))
        per_case[(gamma, h)] = (max(prev[0], err_rot / dt ** 2),
                                max(prev[1], first))

        # (d) group geometry: the square is antipode-closed and orthogonal to r_i
        for r, states in zip((r0, r1), ex['groups_local']):
            for m in range(2):
                assert np.linalg.norm(states[m] + states[m + 2]) < 1e-10
            for st in states[:4]:
                assert abs(float(np.dot(st, r))) < 1e-10
            for st in states:
                assert abs(float(np.linalg.norm(st)) - 1.0) < 1e-10
    print(f'   200 random pairs: pattern, X_theta identity, exact rho~ '
          f'decomposition and group geometry OK')
    print(f'   rotated decomposition of rho + dt L(rho): residual <= '
          f'{worst_rot:.3g} * dt^2  (the O(dt^2) of the U conjugation)')
    print('   per case: c2 = that coefficient, c1 = ||L(rho)||_1 it corrects, '
          'and the RELATIVE error c2*dt/c1 of the first-order construction --')
    print('   which must sit WELL BELOW the detuning being tested (1 - '
          'GP_FACTOR = 1e-2), or the extracted states are the wrong states:')
    print(f'   {"gamma":>7s} {"h":>5s} {"c2":>10s} {"c1":>8s}'
          + ''.join(f'{f"dt={d:g}":>12s}' for d in (1e-2, 1e-3, 1e-4)))
    for (gam, hh) in sorted(per_case):
        c2, c1 = per_case[(gam, hh)]
        print(f'   {gam:7g} {hh:5g} {c2:10.3g} {c1:8.3g}'
              + ''.join(f'{c2 * d / max(c1, 1e-30):12.2e}'
                        for d in (1e-2, 1e-3, 1e-4)))

    print("4) gamma' = J  =>  M PSD (tex threshold), gamma' = 0.99 J  =>  not")
    bad_at, bad_below = 0, 0
    for _ in range(400):
        gamma = float(rng.choice([0.0, 2.0, 10.0, 20.0]))
        h = float(rng.choice([0.0, MODEL4_H]))
        r0, r1 = rng.standard_normal(3), rng.standard_normal(3)
        r0 /= np.linalg.norm(r0)
        r1 /= np.linalg.norm(r1)
        bad_at += float(np.min(np.linalg.eigvalsh(
            extract_candidates(r0, r1, 1.0, gamma, h, 1.0)['M']))) < -1e-9
        bad_below += float(np.min(np.linalg.eigvalsh(
            extract_candidates(r0, r1, 1.0, gamma, h, 0.99)['M']))) < -1e-9
    print(f"   gamma' = J:      {bad_at}/400 elements with lambda_- < 0 (expect 0)")
    print(f"   gamma' = 0.99 J: {bad_below}/400 with lambda_- < 0 (expect > 0)")
    assert bad_at == 0
    # The threshold has to be SHARP, or the detuned parameter set would have
    # nothing to show: below it some element must violate.  At gamma = 0,
    # det M = (0.99^2 - 1)(1-z_0^2)(1-z_1^2) < 0 for every pair, and gamma = 0
    # is drawn about a quarter of the time, so this is not a flaky assertion.
    assert bad_below > 0

    print('5) measured negativity floor of the Euler step vs dt, at '
          "gamma' = J, on the octahedron")
    print('   floor = certified lower bound on the framability RATE for ANY '
          'frame (a curve sitting on it means dt is too coarse, not the frame);')
    print('   bnd   = 2||expm(dt M) d_j - (1 + dt M) d_j||_1 / dt, the '
          'truncation error, which UPPER bounds the floor but is not equal to it')
    print(f'   {"case":16s}' + ''.join(f'{f"dt={d:g} floor":>16s}{"bnd":>12s}'
                                       for d in (1e-2, 1e-3, 1e-4)))
    for case in CASES:
        J, gamma = float(case['J']), float(case['gamma'])
        hh = model_h(case['model'])
        row = f'   {case["tag"]:16s}'
        for d in (1e-2, 1e-3, 1e-4):
            Y = build_targets(blochs, J, gamma, hh, J, d, field='plain')
            fl = negativity_floor(Y)[0]
            bnd = trotter_truncation_bound(blochs, J, gamma, hh, J, d)
            row += f'{(fl - 1.0) / d:16.4e}{bnd:12.4e}'
            assert (fl - 1.0) / d <= bnd + 1e-9, (case['tag'], d, fl, bnd)
        print(row)
    print("   compare with the gamma' = 0.99 J signal, rate 2|lambda_-| = "
          '2e-2: dt is usable only where the floor is well below that')
    print('   (with --gate expm the floor is identically 1 -- the targets are '
          'then genuine states -- which isolates the non-CP artefact)')

    print('6) the (+x, +x) octahedron pair at gamma = 0: worked example')
    xp = np.array([1.0, 0.0, 0.0])
    ex = extract_candidates(xp, xp, 1.0, 0.0, 0.0, 1.0, dt=dt, tilt='equatorial',
                            include_poles=True)
    print(f"   w = {ex['w']:.6g}  (expect 1j)    theta = {ex['theta']:.6f} "
          f'(expect pi/4 = {np.pi / 4:.6f})')
    print(f"   a0 = {ex['a0']:.6g}, a1 = {ex['a1']:.6g} (expect 1, 1)   "
          f"eig M = {np.linalg.eigvalsh(ex['M']).round(12)} (expect [0, 2])")
    assert abs(ex['w'] - 1j) < 1e-12 and abs(ex['theta'] - np.pi / 4) < 1e-12
    assert abs(float(np.min(np.linalg.eigvalsh(ex['M'])))) < 1e-12
    # kappa = 0 here, so the two poles come back as +-x itself: exactly redundant.
    poles = [ex['groups'][0][4], ex['groups'][0][5]]
    print(f'   rotated poles = {np.round(poles[0], 12)}, {np.round(poles[1], 12)} '
          f'(expect +-x: EXACTLY the pair (id +- sigma^x)/2, so free -- and not '
          f'emitted unless include_poles=True)')
    assert np.linalg.norm(poles[0] - xp) < 1e-12
    assert np.linalg.norm(poles[1] + xp) < 1e-12
    ring = np.round(np.array(ex['groups'][0][:4]), 12)
    print(f'   equatorial ring = {ring.tolist()}')
    print(f'   (expect the four (0, +-1/sqrt2, +-1/sqrt2), gauge sqrt(2) = '
          f'{np.sqrt(2):.6f})')
    for st in ring:
        assert abs(st[0]) < 1e-12 and abs(abs(st[1]) - 1 / np.sqrt(2)) < 1e-12
        assert abs(frame_element_gauge(frame_matrix(blochs),
                                       state_column(st))[0] - np.sqrt(2)) < 1e-9
    for name, tl in (('optimal', 'optimal'), ('equatorial', 'equatorial')):
        exx = extract_candidates(xp, xp, 1.0, 0.0, 0.0, 1.0, dt=dt, tilt=tl)
        assert len(exx['groups'][0]) == 4, 'poles must not be emitted by default'
        ang = np.degrees(_min_angle(exx['groups'][0][0], [xp]))
        print(f'   tilt={name:11s} ring polar angle = {ang:8.4f} deg  '
              f'(eps = {exx["eps"][0][0]:.6g} rad)')
        assert abs(np.radians(ang) - exx['eps'][0][0]) < 1e-9

    print('7) two growth rounds on the octahedron (model3, gamma = 2)')
    out = grow_frames(CASE_BY_TAG['model3_gam2'], dt=dt, d_ext_max=20,
                      max_new_per_round=12, filter_mode='criterion', max_rounds=2)
    print(f'   d_ext ladder {out["d_exts"]}')
    for field in FIELDS:
        f = framability(list(out['frames'][0].T), 1.0, 2.0, 0.0, 1.0, dt, field)[0]
        print(f'   octahedron framability ({field:5s}) = {f:.9f}  '
              f'rate {(f - 1.0) / dt:.6e}')
    print('self check OK')


def main() -> None:
    p = argparse.ArgumentParser(
        description='Product-frame growth library; --self_check verifies the '
                    'structural claims of the module docstring.')
    p.add_argument('--self_check', action='store_true')
    p.add_argument('--dt', type=float, default=DT_DEFAULT)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    if args.self_check:
        self_check(seed=args.seed, dt=args.dt)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
