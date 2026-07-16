"""Choi-state Robustness of Magic (RoM) of a unitary gate.

Definition
----------
For an n-qubit unitary U, the (normalized) Choi state is the pure 2n-qubit state

    |Phi_U> = (I (x) U) |Omega>,     |Omega> = 2^{-n/2} sum_i |i>|i>,

and we define  RoM(U) := RoM(|Phi_U><Phi_U|), the state robustness of magic

    RoM(rho) = min ||x||_1   s.t.   A x = b,

where the columns of A are all pure stabilizer states written in the Pauli
basis and b_P = tr(rho P).  The LP machinery (fast stabilizer enumeration via
FWHT + column generation) is that of

    H. Hamaguchi, K. Hamada, N. Yoshioka,
    "Handbook for Quantifying Robustness of Magic",
    Quantum 8, 1461 (2024), arXiv:2311.01362,
    code: https://github.com/quantum-programming/RoM-handbook

Feasibility (the Choi state lives on n_choi = 2n qubits; the handbook method
caps at n_choi = 8):

    gate qubits n | n_choi | method               | requirements
    --------------+--------+----------------------+---------------------------
    1, 2          | 2, 4   | naive LP (exact)     | scipy only, runs anywhere
    3             | 6      | column generation    | compiled C++ helper,
                  |        | (exact, certified)   | Gurobi recommended
    4             | 8      | column generation    | C++ helper + Gurobi, heavy
    >= 5          | >= 10  | OUT OF REACH of this method
                  |          (super-exponentially many stabilizer columns)

Sanity anchors:  RoM(Clifford) = 1 exactly;  RoM(T) = sqrt(2) ~ 1.4142136
(the Choi state of T is Clifford-equivalent to the T magic state).  For any
diagonal U, |Phi_U> is Clifford-equivalent to U|+>^n, so e.g. RoM(CCZ) equals
the RoM of the CCZ magic state.

Environment
-----------
ROM_HANDBOOK_DIR : path to the cloned RoM-handbook repository
                   (default: ./RoM-handbook next to this file).
ROM_TMPDIR       : directory for per-call temporary npz files exchanged with
                   the C++ helper (default: system temp).  Unlike the
                   handbook's own get_topK_botK_Amat, the wrapper below uses
                   unique file names, so concurrent Slurm array tasks are safe.

The C++ helper must be compiled once (needed only for n_choi >= 6):

    g++ exputils/dot/fast_dot_products.cpp -o exputils/dot/fast_dot_products.exe \
        -std=c++17 -lz -O2 -DNDEBUG -mtune=native -march=native -fopenmp

Usage
-----
    # single gate
    python rom_of_gate.py --spec T
    python rom_of_gate.py --spec CCZ --solver gurobi
    python rom_of_gate.py --spec "dot(CX,kron(T,T))"
    python rom_of_gate.py --spec "R(ZZ,pi/4)"
    python rom_of_gate.py --spec npy:my_unitary.npy --label my_gate

    # Slurm array worker: process line <task-id> (1-based, comments skipped)
    python rom_of_gate.py --gates-file rom_gates.txt --task-id $SLURM_ARRAY_TASK_ID

Results are written as JSON files into --out (default: results_rom/), one file
per gate, safe across array tasks.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
from functools import reduce
from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse

# --- RoM-handbook import bootstrap -------------------------------------------
ROM_HANDBOOK_DIR = os.environ.get(
    "ROM_HANDBOOK_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "RoM-handbook"),
)
if not os.path.isdir(os.path.join(ROM_HANDBOOK_DIR, "exputils")):
    sys.exit(
        f"RoM-handbook repository not found at {ROM_HANDBOOK_DIR!r}.\n"
        "Clone it with:\n"
        "    git clone https://github.com/quantum-programming/RoM-handbook.git\n"
        "and/or set the ROM_HANDBOOK_DIR environment variable."
    )
sys.path.insert(0, ROM_HANDBOOK_DIR)

from exputils.actual_Amat import generators_to_Amat, get_actual_Amat  # noqa: E402
from exputils.cover_states import cover_states  # noqa: E402
from exputils.RoM.custom import calculate_RoM_custom  # noqa: E402
from exputils.state.state_in_pauli_basis import state_in_pauli_basis  # noqa: E402

FAST_DOT_EXE = os.environ.get(
    "FAST_DOT_EXE",
    os.path.join(ROM_HANDBOOK_DIR, "exputils", "dot", "fast_dot_products.exe"),
)

# ==============================================================================
# Gate library and spec parser
# ==============================================================================

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}


def _diag(phases) -> np.ndarray:
    return np.diag(np.asarray(phases, dtype=np.complex128))


def _controlled(U: np.ndarray, n_controls: int = 1) -> np.ndarray:
    """Controlled-U with n_controls control qubits (controls first)."""
    d = U.shape[0]
    dim = d * 2**n_controls
    C = np.eye(dim, dtype=np.complex128)
    C[dim - d:, dim - d:] = U
    return C


_T = _diag([1, np.exp(1j * np.pi / 4)])
_S = _diag([1, 1j])
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
)

NAMED_GATES = {
    # single qubit
    "I": _I2,
    "X": _X,
    "Y": _Y,
    "Z": _Z,
    "H": _H,
    "S": _S,
    "SDG": _S.conj().T,
    "T": _T,
    "TDG": _T.conj().T,
    # two qubit
    "CX": _controlled(_X),
    "CNOT": _controlled(_X),
    "CZ": _controlled(_Z),
    "CY": _controlled(_Y),
    "CH": _controlled(_H),
    "CS": _controlled(_S),
    "CT": _controlled(_T),
    "SWAP": _SWAP,
    # three qubit
    "CCX": _controlled(_X, 2),
    "TOFFOLI": _controlled(_X, 2),
    "CCZ": _controlled(_Z, 2),
    "CSWAP": _controlled(_SWAP),
    "FREDKIN": _controlled(_SWAP),
    # four qubit
    "CCCX": _controlled(_X, 3),
    "CCCZ": _controlled(_Z, 3),
}

_EVAL_NS = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "arccos": math.acos,
    "arcsin": math.asin,
    "arctan": math.atan,
}


def _eval_angle(expr: str) -> float:
    """Evaluate a numeric expression such as 'pi/4' or '2*arccos(1/3)'."""
    return float(eval(expr, {"__builtins__": {}}, _EVAL_NS))  # noqa: S307


def _pauli_string_matrix(pstr: str) -> np.ndarray:
    assert re.fullmatch(r"[IXYZ]+", pstr), f"bad Pauli string {pstr!r}"
    return reduce(np.kron, (_PAULI[c] for c in pstr))


def _pauli_rotation(pstr: str, theta: float) -> np.ndarray:
    """exp(-i theta/2 P) for a Pauli string P (P^2 = I => closed form)."""
    P = _pauli_string_matrix(pstr)
    dim = P.shape[0]
    return np.cos(theta / 2) * np.eye(dim, dtype=np.complex128) - 1j * np.sin(
        theta / 2
    ) * P


def _split_top_level(s: str) -> List[str]:
    """Split on commas that are not nested inside parentheses."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            assert depth >= 0, f"unbalanced parentheses in {s!r}"
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    assert depth == 0, f"unbalanced parentheses in {s!r}"
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def parse_gate_spec(spec: str) -> np.ndarray:
    """Parse a gate specification string into a unitary matrix.

    Grammar:
        npy:PATH                 unitary loaded from a .npy file
        NAME                     named gate (see NAMED_GATES; case-insensitive)
        RX(a) | RY(a) | RZ(a)    single-qubit rotations exp(-i a/2 P)
        PHASE(a)                 diag(1, e^{ia})
        CP(a)                    diag(1, 1, 1, e^{ia})
        R(PSTR, a)               exp(-i a/2 P) for a Pauli string, e.g. R(ZZ,pi/4)
        kron(g1, g2, ...)        tensor product
        dot(g1, g2, ...)         matrix product g1 g2 ... (left to right)
    Angles are numeric expressions over pi, sqrt, trig, e.g. 'pi/4'.
    """
    spec = spec.strip()
    if spec.lower().startswith("npy:"):
        path = spec[4:].strip()
        U = np.asarray(np.load(path), dtype=np.complex128)
        assert U.ndim == 2 and U.shape[0] == U.shape[1], f"non-square array in {path}"
        return U

    m = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)\s*\((.*)\)", spec, flags=re.DOTALL)
    if m:
        name, inner = m.group(1).upper(), m.group(2)
        args = _split_top_level(inner)
        if name == "KRON":
            return reduce(np.kron, (parse_gate_spec(a) for a in args))
        if name == "DOT":
            mats = [parse_gate_spec(a) for a in args]
            return reduce(np.matmul, mats)
        if name == "R":
            assert len(args) == 2, "R(PSTR, angle) takes exactly two arguments"
            return _pauli_rotation(args[0].strip().upper(), _eval_angle(args[1]))
        if name in ("RX", "RY", "RZ"):
            assert len(args) == 1
            return _pauli_rotation(name[1], _eval_angle(args[0]))
        if name == "PHASE":
            assert len(args) == 1
            return _diag([1, np.exp(1j * _eval_angle(args[0]))])
        if name == "CP":
            assert len(args) == 1
            return _diag([1, 1, 1, np.exp(1j * _eval_angle(args[0]))])
        raise ValueError(f"unknown parametrized gate {name!r} in spec {spec!r}")

    key = spec.upper()
    if key in NAMED_GATES:
        return NAMED_GATES[key].copy()
    raise ValueError(f"cannot parse gate spec {spec!r}")


def check_unitary(U: np.ndarray, tol: float = 1e-10) -> int:
    """Validate U is unitary on an integer number of qubits; return n_qubits."""
    d = U.shape[0]
    n = d.bit_length() - 1
    assert 2**n == d, f"dimension {d} is not a power of 2"
    dev = np.max(np.abs(U.conj().T @ U - np.eye(d)))
    assert dev < tol, f"matrix is not unitary (max |U^dag U - I| = {dev:.2e})"
    return n


# ==============================================================================
# Choi state
# ==============================================================================


def choi_state_vector(U: np.ndarray) -> np.ndarray:
    """|Phi_U> = (I (x) U)|Omega> as a 4^n-dim state vector.

    With |Omega> = d^{-1/2} sum_i |i>|i>:  <i,j|Phi_U> = U_{ji}/sqrt(d),
    i.e. the flattened U^T divided by sqrt(d).
    """
    d = U.shape[0]
    return np.ascontiguousarray(U.T, dtype=np.complex128).reshape(-1) / np.sqrt(d)


def choi_pauli_vector(U: np.ndarray) -> np.ndarray:
    """Pauli-basis vector b_P = <Phi_U| P |Phi_U> of the Choi state (length 16^n)."""
    psi = choi_state_vector(U)
    return state_in_pauli_basis(psi)


# ==============================================================================
# Collision-safe wrapper around the C++ stabilizer enumerator
# ==============================================================================


def _tmpdir() -> str:
    d = os.environ.get("ROM_TMPDIR") or tempfile.gettempdir()
    os.makedirs(d, exist_ok=True)
    return d


def topk_botk_Amat(
    n_qubit: int,
    vec: np.ndarray,
    K: float,
    is_dual: bool,
    verbose: bool = False,
) -> scipy.sparse.csc_matrix:
    """Same contract as exputils.dot.get_topK_botK_Amat.get_topK_botK_Amat, but
    with unique temporary file names so that concurrent jobs cannot collide
    (the handbook's own version writes fixed-name files into the package dir).
    """
    assert 1 <= n_qubit <= 8 and 0 <= K <= 1
    if not os.path.exists(FAST_DOT_EXE):
        raise FileNotFoundError(
            f"C++ helper not found at {FAST_DOT_EXE}. Compile it first (see module "
            "docstring); it is required for n_choi >= 6."
        )
    tag = f"{os.getpid()}_{uuid.uuid4().hex[:12]}"
    rho_base = os.path.join(_tmpdir(), f"rom_rho_{tag}")
    out_base = os.path.join(_tmpdir(), f"rom_res_{tag}")
    np.savez_compressed(rho_base, rho_vec=np.ascontiguousarray(vec, dtype=np.float64))
    try:
        proc = subprocess.run(
            [FAST_DOT_EXE, str(n_qubit), str(K), str(int(is_dual)), "0",
             rho_base, out_base],
            stderr=subprocess.PIPE,
        )
        if verbose and proc.stderr:
            print(proc.stderr.decode("utf-8", errors="ignore"), end="")
        assert proc.returncode == 0, f"fast_dot_products failed: {proc.returncode=}"
        result = np.load(out_base + ".npz")
        rows = result["Amat_rows"]
        data = result["Amat_data"]
        del result
    finally:
        for f in (rho_base + ".npz", out_base + ".npz"):
            if os.path.exists(f):
                os.remove(f)
    if rows.size == data.size == 1 and rows[0] == -1 and data[0] == 0:
        rows = np.array([])
        data = np.array([])
    indptr = np.arange(len(rows) // (2**n_qubit) + 1) * (2**n_qubit)
    return scipy.sparse.csc_matrix(
        (data, rows, indptr), shape=(4**n_qubit, len(rows) // (2**n_qubit))
    )


# ==============================================================================
# RoM solvers
# ==============================================================================

# top/bottom-K fractions suggested by the handbook tutorial per qubit number
DEFAULT_K = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1e-1, 5: 1e-2, 6: 1e-3, 7: 1e-5, 8: 1e-8}


def pick_solver(solver: str, n_qubit: int) -> str:
    if solver != "auto":
        return solver
    try:
        import gurobipy  # noqa: F401

        return "gurobi"
    except ImportError:
        if n_qubit >= 5:
            warnings.warn(
                "Gurobi not available; falling back to scipy. For n_choi >= 5 "
                "the LPs are large and scipy may be slow or fail."
            )
        return "scipy"


def rom_naive(
    n_qubit: int, rho_vec: np.ndarray, solver: str, verbose: bool
) -> Tuple[float, np.ndarray, scipy.sparse.csc_matrix, dict]:
    """Exact RoM via the full precomputed A matrix (available for n <= 5)."""
    assert n_qubit <= 5, "naive method requires the precomputed Amat (n <= 5)"
    Amat = get_actual_Amat(n_qubit)
    rom, coeff = calculate_RoM_custom(Amat, rho_vec, verbose=verbose, method=solver)
    assert not np.isnan(rom), "naive LP failed"
    return rom, coeff, Amat, {"certified": True, "cert_value": None, "iterations": 1}


def rom_column_generation(
    n_qubit: int,
    rho_vec: np.ndarray,
    K: float,
    solver: str,
    verbose: bool,
    certify: bool = True,
    iter_max: int = 100,
    eps: float = 1e-10,
    discard_current_threshold: float = 0.8,
) -> Tuple[float, np.ndarray, scipy.sparse.csc_matrix, dict]:
    """Exact RoM via column generation (handbook tutorial algorithm, Sec. CG).

    Column generation terminates when no stabilizer column violates the dual
    constraint |y^T a| <= 1; since the C++ search scans the extreme inner
    products over ALL stabilizer columns, termination certifies optimality.
    """
    cover_Amat = generators_to_Amat(n_qubit, cover_states(n_qubit))
    cur_A = topk_botk_Amat(n_qubit, rho_vec, K, is_dual=False, verbose=verbose)
    history = []
    result = None
    for it in range(1, iter_max + 1):
        # the cover columns keep the LP feasible for any rho_vec
        cur_A = scipy.sparse.hstack([cur_A, cover_Amat], format="csc")
        rom, coeff, dual = calculate_RoM_custom(
            cur_A,
            rho_vec,
            verbose=verbose,
            method=solver,
            return_dual=True,
            crossover=False,
        )
        if np.any(np.isnan(np.atleast_1d(rom))):
            raise RuntimeError(
                f"LP failed at CG iteration {it} (n={n_qubit}, K={K}). "
                "Try a larger K or solver='gurobi'."
            )
        got_Amat = topk_botk_Amat(n_qubit, dual, K, is_dual=True, verbose=verbose)
        dual_dots = np.abs(dual.T @ got_Amat)
        violated = np.where(dual_dots > 1 + eps)[0]
        history.append(
            {"iter": it, "rom": float(rom), "n_cols": int(cur_A.shape[1]),
             "n_violations": int(len(violated))}
        )
        if verbose:
            print(f"[CG] iter {it}: RoM={rom:.12f}, "
                  f"cols={cur_A.shape[1]}, violations={len(violated)}")
        if len(violated) == 0:
            result = (rom, coeff, dual, cur_A)
            break
        extra_Amat = got_Amat[:, violated]
        keep = np.logical_or(
            np.abs(coeff) > eps,
            np.abs(dual @ cur_A) >= discard_current_threshold,
        )
        cur_A = scipy.sparse.hstack([cur_A[:, keep], extra_Amat], format="csc")
    if result is None:
        raise RuntimeError(f"column generation did not converge in {iter_max} iters")
    rom, coeff, dual, cur_A = result

    cert_value = None
    if certify:
        # global optimality certificate: max |y^T a| over ALL stabilizer states
        extreme = topk_botk_Amat(n_qubit, dual, 0.0, is_dual=True, verbose=verbose)
        cert_value = float(np.max(np.abs(dual.T @ extreme)))
    info = {
        "certified": bool(cert_value is not None and cert_value <= 1 + 1e-8),
        "cert_value": cert_value,
        "iterations": len(history),
        "history": history,
    }
    return rom, coeff, cur_A, info


def rom_approx(
    n_qubit: int, rho_vec: np.ndarray, K: float, solver: str, verbose: bool
) -> Tuple[float, np.ndarray, scipy.sparse.csc_matrix, dict]:
    """Single LP over the top/bottom-K columns plus the feasibility cover.

    Yields an UPPER BOUND on the RoM (typically very close to exact)."""
    cur_A = scipy.sparse.hstack(
        [
            topk_botk_Amat(n_qubit, rho_vec, K, is_dual=False, verbose=verbose),
            generators_to_Amat(n_qubit, cover_states(n_qubit)),
        ],
        format="csc",
    )
    rom, coeff = calculate_RoM_custom(
        cur_A, rho_vec, verbose=verbose, method=solver, crossover=False
    )
    assert not np.isnan(rom), "approximate LP failed; try a larger K"
    return rom, coeff, cur_A, {"certified": False, "cert_value": None, "iterations": 1}


# ==============================================================================
# Worker
# ==============================================================================


def compute_gate_rom(
    spec: str,
    method: str = "auto",
    solver: str = "auto",
    K: Optional[float] = None,
    verbose: bool = False,
    save_decomp_path: Optional[str] = None,
) -> dict:
    U = parse_gate_spec(spec)
    n_gate = check_unitary(U)
    n_choi = 2 * n_gate
    if n_choi > 8:
        raise ValueError(
            f"gate has {n_gate} qubits -> Choi state on {n_choi} qubits; the "
            "handbook method caps at 8 Choi qubits (4-qubit gates). For larger "
            "gates the Choi-state RoM is out of reach of this approach."
        )

    t0 = time.perf_counter()
    rho_vec = choi_pauli_vector(U)
    t_pauli = time.perf_counter() - t0

    if method == "auto":
        method = "naive" if n_choi <= 5 else "cg"
    solver = pick_solver(solver, n_choi)
    K_used = DEFAULT_K[n_choi] if K is None else K

    t0 = time.perf_counter()
    if method == "naive":
        rom, coeff, Amat, info = rom_naive(n_choi, rho_vec, solver, verbose)
    elif method == "cg":
        rom, coeff, Amat, info = rom_column_generation(
            n_choi, rho_vec, K_used, solver, verbose
        )
    elif method == "approx":
        rom, coeff, Amat, info = rom_approx(n_choi, rho_vec, K_used, solver, verbose)
    else:
        raise ValueError(f"unknown method {method!r}")
    t_lp = time.perf_counter() - t0

    residual = float(np.max(np.abs(Amat @ coeff - rho_vec)))
    assert residual < 1e-6, f"decomposition residual too large: {residual:.2e}"

    if save_decomp_path is not None:
        support = np.abs(coeff) > 1e-12
        A_supp = scipy.sparse.csc_matrix(Amat[:, support])
        np.savez_compressed(
            save_decomp_path,
            coeff=coeff[support],
            A_data=A_supp.data,
            A_indices=A_supp.indices,
            A_indptr=A_supp.indptr,
            A_shape=np.array(A_supp.shape),
        )

    return {
        "spec": spec,
        "n_gate": n_gate,
        "n_choi": n_choi,
        "rom": float(rom),
        "log2_rom": float(np.log2(rom)),
        "method": method,
        "solver": solver,
        "K": K_used,
        "certified": info["certified"],
        "cert_value": info["cert_value"],
        "cg_iterations": info["iterations"],
        "residual_inf": residual,
        "n_decomp_terms": int(np.sum(np.abs(coeff) > 1e-12)),
        "time_pauli_vec_s": t_pauli,
        "time_lp_s": t_lp,
        "history": info.get("history"),
    }


def _read_gates_file(path: str) -> List[str]:
    specs = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                specs.append(line)
    return specs


def _sanitize(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec).strip("_")[:80]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Choi-state Robustness of Magic of a unitary gate "
        "(Hamaguchi-Hamada-Yoshioka machinery, arXiv:2311.01362)."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--spec", help="gate specification (see parse_gate_spec)")
    src.add_argument("--gates-file", help="file with one gate spec per line")
    ap.add_argument("--task-id", type=int, default=None,
                    help="1-based line index into --gates-file (Slurm array id)")
    ap.add_argument("--label", default=None, help="output file label")
    ap.add_argument("--out", default="results_rom", help="output directory")
    ap.add_argument("--method", default="auto",
                    choices=["auto", "naive", "cg", "approx"])
    ap.add_argument("--solver", default="auto",
                    choices=["auto", "gurobi", "scipy"])
    ap.add_argument("--K", type=float, default=None,
                    help="top/bottom-K fraction for column generation "
                    "(default: handbook-recommended per qubit count)")
    ap.add_argument("--save-decomp", action="store_true",
                    help="also save the optimal stabilizer decomposition (npz)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.gates_file is not None:
        specs = _read_gates_file(args.gates_file)
        if args.task_id is not None:
            assert 1 <= args.task_id <= len(specs), (
                f"task-id {args.task_id} out of range 1..{len(specs)}"
            )
            specs = [(args.task_id, specs[args.task_id - 1])]
        else:
            specs = list(enumerate(specs, start=1))
    else:
        specs = [(None, args.spec)]

    os.makedirs(args.out, exist_ok=True)
    for task_id, spec in specs:
        label = args.label if (args.label and len(specs) == 1) else _sanitize(spec)
        prefix = f"{task_id:03d}_" if task_id is not None else ""
        out_json = os.path.join(args.out, f"rom_{prefix}{label}.json")
        decomp_path = (
            os.path.join(args.out, f"rom_{prefix}{label}_decomp.npz")
            if args.save_decomp
            else None
        )
        print(f"=== computing Choi-state RoM of {spec!r} ===")
        result = compute_gate_rom(
            spec,
            method=args.method,
            solver=args.solver,
            K=args.K,
            verbose=args.verbose,
            save_decomp_path=decomp_path,
        )
        result["task_id"] = task_id
        result["label"] = label
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(
            f"RoM({spec}) = {result['rom']:.12f}  "
            f"[method={result['method']}, solver={result['solver']}, "
            f"certified={result['certified']}]  -> {out_json}"
        )


if __name__ == "__main__":
    main()
