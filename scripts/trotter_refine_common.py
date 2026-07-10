"""
Shared helpers for the Trotter-scan framability refinement family
(cross-eval sweep, island polish, floor hunt, quick refine).

Centralises:
  * the per-point result-file patterns (base + every refinement flavour),
  * best-known value/frame lookup across all files of a point,
  * per-neighbour frame gathering (ALL 4-connected neighbours, not just the
    single best one — optimal frames come in distinct symmetry branches and
    the branch can switch across the grid, so every neighbour is a candidate),
  * bond-gate construction for a grid point,
  * frame evaluation: fast batched LP for candidate scanning and the
    reference per-column LP (dissipative_PT._framability_lp) to confirm any
    value that is stored, so stored numbers stay consistent with the rest of
    the pipeline.

Grid convention (matches every trotter scan worker):
    ix in 0 .. N_X-1 ; iy in 0 .. N_Y-1 ; point_id = ix * N_Y + iy
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from trotter_lindbladian_scan import bond_trotter_gate, choose_dt, DIM_DEFAULT
from dissipative_PT import (
    _framability_lp, _kron_power,
    frame_from_params, params_from_frame, embed_frame_params,
)
from optimize_framability import _get_framability_fast

NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# key -> (frame key, d_ext_single)
KEYS = {'opt_fra_4': ('opt_S_4', 4), 'opt_fra_6': ('opt_S_6', 6)}

# Every refinement flavour that may hold an improved (value, frame) pair for a
# point.  New refinement stages MUST be added here so later stages (and the
# final collect) see their frames.
REFINE_PATTERNS = ('pt_refine_r*', 'pt_qrefine_r*', 'pt_xeval_r*',
                   'pt_polish_r*', 'pt_fhunt_r*', 'pt_verdict_r*')


def base_path(out_dir: Path, model, ix: int, iy: int) -> Path:
    return Path(out_dir) / model.name / f'pt_{ix:03d}_{iy:03d}.npz'


def all_paths(out_dir: Path, model, ix: int, iy: int) -> list[Path]:
    """Base scan file + every refinement file for this point."""
    d = Path(out_dir) / model.name
    paths = [base_path(out_dir, model, ix, iy)]
    for pat in REFINE_PATTERNS:
        paths += sorted(d.glob(f'{pat}_{ix:03d}_{iy:03d}.npz'))
    return [p for p in paths if p.exists()]


def best_known(out_dir: Path, model, ix: int, iy: int,
               key: str, s_key: str):
    """Lowest-framability (value, frame S) over base + every refinement file."""
    best_val, best_S = np.inf, None
    for f in all_paths(out_dir, model, ix, iy):
        try:
            d = np.load(f)
        except Exception:
            continue
        if key not in d or s_key not in d:
            continue
        v = float(d[key])
        if np.isfinite(v) and v < best_val:
            S = np.asarray(d[s_key], dtype=float)
            if np.all(np.isfinite(S)):
                best_val, best_S = v, S
    return best_val, best_S


def neighbor_frames(out_dir: Path, model, ix: int, iy: int,
                    key: str, s_key: str) -> list[tuple[float, np.ndarray]]:
    """(value, frame) of the best-known result of EVERY 4-connected neighbour.

    Returns one entry per neighbour that has a finite stored frame, ordered by
    ascending value.  All of them should be used as seeds: the single best
    neighbour may sit on a different symmetry branch of optimal frames, while
    another neighbour's frame transfers cleanly.
    """
    out = []
    for dx, dy in NEIGHBORS:
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < model.N_X and 0 <= ny < model.N_Y:
            v, S = best_known(out_dir, model, nx, ny, key, s_key)
            if S is not None:
                out.append((v, S))
    out.sort(key=lambda t: t[0])
    return out


def build_gate(model, ix: int, iy: int, base_npz) -> np.ndarray:
    """Bond Trotter gate of grid point (ix, iy), using the dt/dim stored in the
    base scan file (falling back to the model rules for legacy files)."""
    p1 = float(model.p1_vals[ix])
    p2 = float(model.p2_vals[iy])
    dim = int(base_npz['dim']) if 'dim' in base_npz else DIM_DEFAULT
    dt = float(base_npz['dt']) if 'dt' in base_npz else None
    H1, H2, jumps1, jumps2 = model.build(p1, p2)
    if dt is None:
        dt = model.dt if model.dt is not None else choose_dt(H1, H2, jumps1, jumps2)
    return bond_trotter_gate(H1, H2, jumps1, jumps2, dim, dt)


def eval_frame_fast(S: np.ndarray, gate: np.ndarray) -> float:
    """Framability of `gate` in the fixed product frame kron(S, S): one batched
    LP.  Any finite value is a rigorous upper bound on the point's minimal
    framability.  Used for cheap candidate scanning."""
    f = _get_framability_fast(_kron_power(np.asarray(S, dtype=float), 2), gate)
    return float(f)


def eval_frame_reference(S: np.ndarray, gate: np.ndarray) -> float:
    """Reference per-column LP evaluation (dissipative_PT._framability_lp) —
    the evaluator every stored opt_fra value came from.  Confirm a candidate
    with this before storing it so values stay consistent."""
    f = _framability_lp(_kron_power(np.asarray(S, dtype=float), 2), gate)
    return float(f)


def embed_S4_to_S6(S4: np.ndarray) -> np.ndarray:
    """Embed a d_ext_single=4 frame into d_ext_single=6 (value can only drop)."""
    return frame_from_params(embed_frame_params(params_from_frame(S4), 4, 6), 6)
