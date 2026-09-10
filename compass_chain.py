r"""
compass_chain.py -- the dissipative 1D quantum compass chain in a field.

    H = - Jx sum_i X_{2i-1} X_{2i}  - Jy sum_i Y_{2i} Y_{2i+1}
        - h sum_j (n_x X_j + n_y Y_j)
    jumps  sqrt(gamma) Z_j  on every site (pure dephasing)

Bonds alternate XX (odd bonds, 1-based) and YY (even bonds), so every bulk
qubit sits on exactly one bond of each type.

Two builders:

  bond_lindbladian(case, bond, gamma, h, dephasing_frac)
      16x16 real Pauli-basis generator of ONE bond gate ('xx' or 'yy'), via
      trotter_lindbladian_scan.build_bond_lindbladian with dim = 1: each qubit
      sits on 2 bonds, so the field is shared evenly, h/2 per bond.  The
      dephasing is NOT necessarily shared evenly: this bond carries the
      fraction `dephasing_frac` of each site's rate gamma and the other bond
      type carries 1 - dephasing_frac (dephasing_split.balance_split picks the
      fraction).  dephasing_frac = 1/2 is the even share.

  chain_lindbladian(case, gamma, h, n_qubits=8, topology='chain')
      sparse computational-basis Liouvillian of the full chain at full coupling
      (column-stacking vec convention, n_qubit_lindbladian's superoperator
      builders), for the Liouvillian gap and the oscillation rate.  'chain' is
      open boundary (XX bonds at both ends for even n_qubits); 'ring' closes it
      with a YY bond.

Cases: the six (Jx, Jy) x field-direction combinations of the (gamma, h)
phase-diagram scan, on the grid GAMMA_VALS x H_VALS.  The 'xy' field direction
is the unit vector (x + y)/sqrt(2), so h is always the field magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from dissipative_PT import _SX, _SY, _SZ
from trotter_lindbladian_scan import build_bond_lindbladian, _arange
from n_qubit_lindbladian import (ring_edges, _site_op, _two_site_op,
                                 _superop_commutator, _superop_dissipator)

COMPASS_VERSION = '1.0'

# 1D chain: each qubit sits on 2 * BOND_DIM = 2 bonds, one XX and one YY.
BOND_DIM = 1
BONDS = ('xx', 'yy')

GAMMA_VALS = _arange(0.0, 2.0, 0.2)      # 11 values
H_VALS = _arange(0.0, 2.0, 0.2)          # 11 values

FIELD_DIRECTIONS = {
    'x':  (1.0, 0.0),
    'xy': (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
}


@dataclass(frozen=True)
class CompassCase:
    name: str
    Jx: float
    Jy: float
    field: str               # key of FIELD_DIRECTIONS

    @property
    def field_vector(self):
        return FIELD_DIRECTIONS[self.field]

    @property
    def title(self) -> str:
        fld = r'X_j' if self.field == 'x' else r'(X_j+Y_j)/\sqrt{2}'
        return (rf'$H=-{self.Jx:g}\sum_i X_{{2i-1}}X_{{2i}}'
                rf'-{self.Jy:g}\sum_i Y_{{2i}}Y_{{2i+1}}-h\sum_j {fld}$,'
                rf'  jumps $\sqrt{{\gamma}}\,Z_j$')


def _make_case(Jx: float, Jy: float, field: str):
    name = f'jx{Jx:.1f}_jy{Jy:.1f}_h{field}'
    return name, CompassCase(name, Jx, Jy, field)


CASES: dict[str, CompassCase] = dict(
    _make_case(Jx, Jy, field)
    for field in ('x', 'xy')
    for (Jx, Jy) in ((0.5, 1.0), (1.0, 1.0), (1.0, 0.5)))


def field_op(case: CompassCase) -> np.ndarray:
    """n_x X + n_y Y (2x2)."""
    nx, ny = case.field_vector
    return nx * _SX + ny * _SY


# ---------------------------------------------------------------------------
#  Bond gates (framability rates)
# ---------------------------------------------------------------------------
def bond_terms(case: CompassCase, bond: str, gamma: float, h: float,
               dephasing_frac: float = 0.5):
    """(H1, H2, jumps1, jumps2) of one bond gate in build_bond_lindbladian's
    convention.  build_bond_lindbladian scales one-qubit rates by 1/(2*dim), so
    the jump amplitude is chosen to leave exactly dephasing_frac * gamma per
    site on this bond."""
    if bond == 'xx':
        H2 = -case.Jx * np.kron(_SX, _SX)
    elif bond == 'yy':
        H2 = -case.Jy * np.kron(_SY, _SY)
    else:
        raise ValueError(f"bond must be one of {BONDS}, got {bond!r}")
    if not 0.0 <= dephasing_frac <= 1.0:
        raise ValueError(f'dephasing_frac must lie in [0, 1], got {dephasing_frac}')
    H1 = -h * field_op(case)
    rate = 2.0 * BOND_DIM * dephasing_frac * gamma
    jumps1 = [np.sqrt(rate) * _SZ] if rate > 0.0 else []
    return H1, H2, jumps1, []


def bond_lindbladian(case: CompassCase, bond: str, gamma: float, h: float,
                     dephasing_frac: float = 0.5) -> np.ndarray:
    """16x16 real Pauli-basis generator of the `bond` gate."""
    L = build_bond_lindbladian(*bond_terms(case, bond, gamma, h, dephasing_frac),
                               BOND_DIM)
    return np.asarray(L).real


# ---------------------------------------------------------------------------
#  Full chain (gap / oscillation rate)
# ---------------------------------------------------------------------------
def chain_edges(n_qubits: int, topology: str = 'chain'):
    """Edges in chain order; edge k is XX for even k and YY for odd k."""
    if topology == 'chain':
        return [(i, i + 1) for i in range(n_qubits - 1)]
    if topology == 'ring':
        if n_qubits % 2:
            raise ValueError('a compass ring needs an even number of sites')
        return ring_edges(n_qubits)
    raise ValueError(f"topology must be 'chain' or 'ring', got {topology!r}")


def chain_hamiltonian(case: CompassCase, h: float, n_qubits: int = 8,
                      topology: str = 'chain') -> sp.csr_matrix:
    d = 2 ** n_qubits
    H = sp.csr_matrix((d, d), dtype=complex)
    for k, (i, j) in enumerate(chain_edges(n_qubits, topology)):
        if k % 2 == 0:
            H = H - case.Jx * _two_site_op(_SX, _SX, i, j, n_qubits)
        else:
            H = H - case.Jy * _two_site_op(_SY, _SY, i, j, n_qubits)
    if h != 0.0:
        f = field_op(case)
        for s in range(n_qubits):
            H = H - h * _site_op(f, s, n_qubits)
    return H.tocsr()


def chain_lindbladian(case: CompassCase, gamma: float, h: float,
                      n_qubits: int = 8, topology: str = 'chain') -> sp.csr_matrix:
    """Sparse Liouvillian of the full chain, shape (4^n_qubits, 4^n_qubits)."""
    L = _superop_commutator(chain_hamiltonian(case, h, n_qubits, topology))
    if gamma > 0.0:
        amp = np.sqrt(gamma)
        for s in range(n_qubits):
            L = L + _superop_dissipator(amp * _site_op(_SZ, s, n_qubits))
    return L.tocsr()
