#!/bin/bash
# ============================================================
#  SLURM job-array: item 4 -- Lindbladian gap of a full 8-qubit system, one
#  topology (ring or 2x4 lattice) per submission, over the coarsened model3
#  (gamma, gamma') grid (11x11=121 points at the default --stride=5).
#
#  One grid point per array task (121 <= 200, no chunking needed).  Each task
#  builds the sparse 65536x65536 Liouvillian for its (gamma, gamma') point and
#  finds the gap via shift-invert eigs (n_qubit_lindbladian.lindbladian_gap) --
#  tractable sparse, NOT densified.
#
#  Submit (one per topology):
#    mkdir -p logs results_8q
#    TOPOLOGY=ring    sbatch scripts/eight_qubit_gap.slurm.sh
#    TOPOLOGY=lattice sbatch scripts/eight_qubit_gap.slurm.sh
#
#  Output: results_8q/<topology>/pt_<ig>_<igp>.npz
# ============================================================

#SBATCH --job-name=8q_gap
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-120
#SBATCH --output=logs/8qgap_%x_%A_%a.out
#SBATCH --error=logs/8qgap_%x_%A_%a.err

TOPOLOGY=${TOPOLOGY:?set TOPOLOGY to ring or lattice}
OUT_DIR=${OUT_DIR:-results_8q}
# NOTE: STRIDE must match the #SBATCH --array range above (n_grid=len(p1_vals[::STRIDE]),
# array size = n_grid**2 - 1); STRIDE=5 -> 11x11=121 -> array=0-120 (the default).
# If you override STRIDE, edit --array= to match n_grid**2 - 1 first.
STRIDE=${STRIDE:-5}
K=${K:-12}
WHICH=${WHICH:-LR}
# SIGMA is intentionally EMPTY by default: empty => ARPACK regular mode, which
# touches the operator only through matrix-vector products and so exploits its
# sparsity (65536x65536, 1.1e6 nnz, 22 MB, ~3 ms/matvec).  Setting SIGMA turns
# on shift-invert, which needs a sparse LU of the shifted operator -- that
# fill-in did not factor in 110s at N=8 and blows past the job's memory, which
# is why an earlier run of this array completed only its near-trivial
# (small-gamma, nearly diagonal) points.  Set it only for small N.
SIGMA=${SIGMA:-}
NOISE_FLOOR=${NOISE_FLOOR:-1e-6}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$TOPOLOGY] point ${SLURM_ARRAY_TASK_ID}: starting"

python scripts/eight_qubit_gap_worker.py \
    --topology    "$TOPOLOGY" \
    --task_id     "$SLURM_ARRAY_TASK_ID" \
    --stride      "$STRIDE" \
    --out_dir     "$OUT_DIR" \
    --k           "$K" \
    --which       "$WHICH" \
    ${SIGMA:+"--sigma=$SIGMA"} \
    --noise_floor "$NOISE_FLOOR"

echo "[$TOPOLOGY] point ${SLURM_ARRAY_TASK_ID}: done"
