"""Stabilizer Renyi entropy of a state, and non-cliffordness of a unitary.

Definitions
-----------
Choi state of an n-qubit unitary U (2n qubits, normalized):

    |Phi_U> = 2^{-n/2} sum_{i=0}^{2^n - 1} |i> (x) U|i>

which is exactly `rom_of_gate.choi_state_vector` -- that implementation is
reused here, not duplicated.

Stabilizer Renyi entropy (alpha = 2) of a pure N-qubit state |psi>, with
d = 2^N and P running over all 4^N Pauli strings p_0 (x) ... (x) p_{N-1},
p_i in {I, X, Y, Z}:

    M_2(|psi>) = -log( sum_P 2^{-2N} <psi|P|psi>^4 ) - log(2^N)
               = -log( (1/d) sum_P <psi|P|psi>^4 )

(the two forms are algebraically identical; `self_test` checks it numerically).
The general alpha is available through the Renyi form of the same probability
distribution Xi_P = <psi|P|psi>^2 / d, which is normalized for pure states:

    M_alpha = 1/(1-alpha) * log( sum_P Xi_P^alpha ) - log(d).

Non-cliffordness of a unitary:

    NC(U) = M_2(|Phi_U>).

M_2 vanishes exactly on stabilizer states, so NC(U) = 0 for every Clifford U;
the anchor value for the T gate is NC(T) = log(4/3) (its Choi state is
Clifford-equivalent to the T magic state), which `self_test` verifies.

Logarithm base
--------------
Everything above is written with the natural logarithm, which is what these
functions return by default.  Pass `log_base=2` (or divide by log 2) for the
"bits of magic" convention common in the magic literature; the scan pipeline
stores both.

Computing the 4^N expectation values
------------------------------------
<psi|P|psi> = tr(rho P) = sum_{ij} rho_{ij} P_{ji} factorizes over qubits, so
the whole vector is obtained by applying one 4x4 basis-change matrix to each
qubit axis of rho reshaped to (4,)*N -- O(N 4^N) work instead of the O(8^N) of
a Pauli-by-Pauli loop.  For the 8-qubit Choi state of a 4-qubit gate that is
65536 numbers in a few milliseconds.

Usage:
    python magic_measures.py --self_test
"""

from __future__ import annotations

import numpy as np

# Single-qubit Paulis in the I, X, Y, Z order used for the Pauli-string index.
_I2 = np.eye(2, dtype=np.complex128)
_SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI_1Q = (_I2, _SX, _SY, _SZ)
PAULI_LABELS = 'IXYZ'


# ---------------------------------------------------------------------------
#  Choi state (reused from rom_of_gate)
# ---------------------------------------------------------------------------
_ROM_FUNCS: dict[str, object] = {}


def _rom_of_gate():
    """Lazily import rom_of_gate.{choi_state_vector, check_unitary}.

    The import is deferred because rom_of_gate calls sys.exit() at import time
    when the RoM-handbook clone is missing.  Nothing in this pipeline needs the
    handbook itself, so the failure is turned into a normal exception and is
    only ever raised if a Choi state is actually requested (the collect/plot
    stage, for instance, never touches it).
    """
    if not _ROM_FUNCS:
        try:
            from rom_of_gate import choi_state_vector, check_unitary
        except SystemExit as exc:                     # missing RoM-handbook
            raise RuntimeError(
                'rom_of_gate could not be imported (it exits when the '
                'RoM-handbook clone is absent).  Clone it with\n'
                '    git clone https://github.com/quantum-programming/RoM-handbook.git\n'
                'or set ROM_HANDBOOK_DIR.'
            ) from exc
        _ROM_FUNCS['choi'] = choi_state_vector
        _ROM_FUNCS['check'] = check_unitary
    return _ROM_FUNCS['choi'], _ROM_FUNCS['check']


def choi_state_vector(U: np.ndarray) -> np.ndarray:
    """|Phi_U> = 2^{-n/2} sum_i |i> (x) U|i> as a 4^n-dim state vector.

    Thin re-export of rom_of_gate.choi_state_vector so that this module is the
    single entry point of the magic pipeline.
    """
    choi, _ = _rom_of_gate()
    return choi(np.asarray(U, dtype=np.complex128))


# ---------------------------------------------------------------------------
#  Pauli expectation values of a pure state
# ---------------------------------------------------------------------------
def n_qubits_of_state(psi: np.ndarray) -> int:
    """Number of qubits of a state vector; raises if the dimension is not 2^N."""
    psi = np.asarray(psi)
    if psi.ndim != 1:
        raise ValueError(f'psi must be a 1-D state vector, got shape {psi.shape}')
    d = psi.shape[0]
    N = int(d).bit_length() - 1
    if 2 ** N != d:
        raise ValueError(f'state dimension {d} is not a power of two')
    return N


def _pauli_basis_change() -> np.ndarray:
    """(4, 4) matrix W with W[p, 2i + j] = P_p[j, i].

    Contracting W with the (i, j) index pair of one qubit of rho turns that
    pair into a Pauli label, so applying it to every qubit axis maps rho to the
    vector of tr(rho P).
    """
    return np.stack([P.T.reshape(-1) for P in _PAULI_1Q]).astype(np.complex128)


def pauli_expectation_vector(psi: np.ndarray, *, check: bool = True) -> np.ndarray:
    """All 4^N expectations <psi|P|psi>, indexed by base-4 Pauli strings.

    The index of P = p_0 (x) ... (x) p_{N-1} is sum_k label(p_k) * 4^{N-1-k}
    with label = 0,1,2,3 for I,X,Y,Z, i.e. C-order of the (4,)*N tensor.

    Parameters
    ----------
    psi : (2^N,) complex
        Pure state vector; it is normalized internally.
    check : bool
        Verify sum_P <P>^2 = 2^N (Parseval for a pure state) and that the
        expectations came out real.

    Returns
    -------
    (4^N,) float
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    N = n_qubits_of_state(psi)
    nrm = np.linalg.norm(psi)
    if nrm == 0:
        raise ValueError('psi is the zero vector')
    psi = psi / nrm

    rho = np.outer(psi, psi.conj())                       # rho_{ij}
    # (i_0..i_{N-1}, j_0..j_{N-1}) -> interleaved (i_0,j_0, i_1,j_1, ...)
    perm = [x for k in range(N) for x in (k, N + k)]
    t = np.transpose(rho.reshape([2] * (2 * N)), perm).reshape([4] * N)

    W = _pauli_basis_change()
    for k in range(N):
        t = np.moveaxis(np.tensordot(W, t, axes=([1], [k])), 0, k)

    vals = t.reshape(-1)
    if check:
        imag = float(np.max(np.abs(vals.imag))) if vals.size else 0.0
        if imag > 1e-8:
            raise RuntimeError(f'Pauli expectations are not real (max |Im| = {imag:.2e})')
    out = np.ascontiguousarray(vals.real)
    if check:
        parseval = float(np.sum(out ** 2))
        if abs(parseval - 2 ** N) > 1e-6 * 2 ** N:
            raise RuntimeError(
                f'sum_P <P>^2 = {parseval:.8f} != 2^N = {2 ** N} '
                '(state is not pure/normalized?)')
    return out


# ---------------------------------------------------------------------------
#  Stabilizer Renyi entropy
# ---------------------------------------------------------------------------
def stabilizer_renyi_entropy(psi: np.ndarray, alpha: float = 2.0, *,
                             log_base: float | None = None,
                             check: bool = True) -> float:
    """M_alpha of a pure N-qubit state (default alpha = 2, natural log).

    alpha = 2 is exactly the module-docstring formula

        M_2 = -log( sum_P 2^{-2N} <P>^4 ) - log(2^N).

    log_base=None gives natural log (nats); log_base=2 gives bits.
    """
    exps = pauli_expectation_vector(psi, check=check)
    N = n_qubits_of_state(np.asarray(psi).reshape(-1))
    d = float(2 ** N)

    if alpha == 2.0:
        # Written in the literal form of the definition.
        s = float(np.sum(exps ** 4)) * (2.0 ** (-2 * N))
        if s <= 0.0:
            return float('inf')
        val = -np.log(s) - np.log(d)
    else:
        if alpha == 1.0:
            raise ValueError('alpha = 1 is the (Shannon) limit; not implemented')
        xi = exps ** 2 / d                       # normalized for pure states
        s = float(np.sum(xi ** alpha))
        if s <= 0.0:
            return float('inf')
        val = np.log(s) / (1.0 - alpha) - np.log(d)

    # M is >= 0 up to round-off; clip the -1e-16 that stabilizer states give.
    val = float(max(val, 0.0))
    return val if log_base is None else val / np.log(log_base)


def stabilizer_renyi_entropy_2(psi: np.ndarray, **kw) -> float:
    """Alias for the alpha = 2 stabilizer Renyi entropy."""
    return stabilizer_renyi_entropy(psi, 2.0, **kw)


# ---------------------------------------------------------------------------
#  Non-cliffordness of a unitary
# ---------------------------------------------------------------------------
def noncliffordness(U: np.ndarray, alpha: float = 2.0, *,
                    log_base: float | None = None,
                    check_unitarity: bool = True,
                    tol: float = 1e-9) -> float:
    """M_alpha of the Choi state of the n-qubit unitary U (a 2n-qubit state)."""
    U = np.asarray(U, dtype=np.complex128)
    if check_unitarity:
        _, check = _rom_of_gate()
        check(U, tol=tol)
    psi = choi_state_vector(U)
    return stabilizer_renyi_entropy(psi, alpha, log_base=log_base)


# ---------------------------------------------------------------------------
#  Self-test
# ---------------------------------------------------------------------------
def self_test() -> None:
    rng = np.random.default_rng(0)
    H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.diag([1, 1j]).astype(complex)
    T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)
    CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

    # 1. Pauli expectations: |0>, |+>, and Parseval on random pure states.
    e0 = pauli_expectation_vector(np.array([1, 0], dtype=complex))
    assert np.allclose(e0, [1, 0, 0, 1]), e0            # I, X, Y, Z
    ep = pauli_expectation_vector(np.array([1, 1], dtype=complex) / np.sqrt(2))
    assert np.allclose(ep, [1, 1, 0, 0]), ep
    for N in (1, 2, 3):
        v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
        e = pauli_expectation_vector(v)
        assert abs(np.sum(e ** 2) - 2 ** N) < 1e-9

    # 2. Two-qubit ordering: <Z (x) I> and <I (x) Z> on |0>|+>.
    psi = np.kron(np.array([1, 0], dtype=complex),
                  np.array([1, 1], dtype=complex) / np.sqrt(2))
    e = pauli_expectation_vector(psi)
    assert abs(e[3 * 4 + 0] - 1.0) < 1e-12, 'ZI'        # index 3*4 + 0
    assert abs(e[0 * 4 + 3] - 0.0) < 1e-12, 'IZ'
    assert abs(e[0 * 4 + 1] - 1.0) < 1e-12, 'IX'

    # 3. The two written forms of M_2 agree.
    for N in (1, 2, 3):
        v = rng.normal(size=2 ** N) + 1j * rng.normal(size=2 ** N)
        v = v / np.linalg.norm(v)
        ex = pauli_expectation_vector(v)
        lit = -np.log(np.sum(2.0 ** (-2 * N) * ex ** 4)) - np.log(2.0 ** N)
        cmp = -np.log(np.sum(ex ** 4) / 2 ** N)
        assert abs(lit - cmp) < 1e-12
        assert abs(stabilizer_renyi_entropy(v) - lit) < 1e-12

    # 4. Stabilizer states have M_2 = 0; the T magic state has log(4/3).
    for st in (np.array([1, 0], dtype=complex),
               np.array([1, 1], dtype=complex) / np.sqrt(2),
               np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)):
        assert stabilizer_renyi_entropy(st) < 1e-12
    t_state = T @ (np.array([1, 1], dtype=complex) / np.sqrt(2))
    assert abs(stabilizer_renyi_entropy(t_state) - np.log(4 / 3)) < 1e-12
    assert abs(stabilizer_renyi_entropy(t_state, log_base=2) - np.log2(4 / 3)) < 1e-12

    # 5. Choi states: Cliffords are free, the T gate sits at log(4/3), and the
    #    measure is additive on tensor products (T (x) T -> 2 log(4/3)).
    try:
        for C in (_I2, H1, S, CX, np.kron(H1, S)):
            assert noncliffordness(C) < 1e-9, 'Clifford must be non-cliffordness free'
        assert abs(noncliffordness(T) - np.log(4 / 3)) < 1e-10
        assert abs(noncliffordness(np.kron(T, T)) - 2 * np.log(4 / 3)) < 1e-10
        assert abs(noncliffordness(CX @ np.kron(T, _I2)) - np.log(4 / 3)) < 1e-10
        print('choi/non-cliffordness anchors OK  (NC(T) = log(4/3) = '
              f'{np.log(4 / 3):.6f} nats = {np.log2(4 / 3):.6f} bits)')
    except RuntimeError as exc:                          # no RoM-handbook here
        print(f'[skip] Choi-state anchors: {exc}')

    print('magic_measures self-test passed.')


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--self_test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        ap.print_help()
