#!/bin/bash
# ============================================================
#  SLURM job-array, STAGE 2 of the product-frame growth pipeline: the
#  framability of every rung (scripts/product_frame_grow_worker.py).
#
#  ONE WORK UNIT = ONE (case, round, gamma' variant, target variant), i.e. one
#  Schroedinger framability LP over D = kron(S, S).  7 cases x ~13 rounds x
#  2 gamma' x 2 targets ~ 360 independent units, strided over the full 0-199
#  array (1-2 units per task).  The unit list is sorted by d_ext DESCENDING
#  before striding, so each task pairs an expensive rung (d_ext ~ 100, i.e.
#  ~10^4 column LPs) with a cheap one and the array load balances.
#
#    gamma' variants: at = J,  lo = GP_FACTOR * J (0.99 J)
#    target variants: free  = free-local-unitary Euler step (the
#                             continuous_simulation.tex quantity, the one that
#                             can reach 1 at gamma' = J)
#                     plain = bare Euler gate (what frame_element_criterion
#                             scores, and what a finite frame really pays)
#
#  Requires the stage-1 ladders (scripts/product_frame_grow_frames.slurm.sh);
#  cases without one are reported and skipped.
#
#  Submit (whole pipeline, with dependencies):
#    mkdir -p logs results_product_frame_grow
#    JID1=$(sbatch --parsable scripts/product_frame_grow_frames.slurm.sh)
#    JID2=$(sbatch --parsable --dependency=afterok:${JID1} scripts/product_frame_grow.slurm.sh)
#    sbatch --dependency=afterok:${JID2} scripts/product_frame_grow_collect.slurm.sh
#
#  Idempotent per unit: an existing finite framability at this GROW_VERSION is
#  skipped, so resubmitting the array fills exactly the holes.
#
#  Output: results_product_frame_grow/<tag>/fra_r<round>_<gp>_<field>.npz
# ============================================================

#SBATCH --job-name=pfg_fra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/pfgfra_%x_%A_%a.out
#SBATCH --error=logs/pfgfra_%x_%A_%a.err

OUT_DIR=${OUT_DIR:-results_product_frame_grow}
N_CHUNKS=${N_CHUNKS:-200}            # must match the --array size above
GP_FACTOR=${GP_FACTOR:-0.99}         # gamma'/J of the detuned set
FIELDS=${FIELDS:-"free"}             # must match stage 1's FIELD
GPS=${GPS:-"at lo"}
FORCE=${FORCE:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

FORCE_FLAG=""
if [ "$FORCE" != "0" ]; then FORCE_FLAG="--force"; fi

echo "[pfg fra] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/product_frame_grow_worker.py \
    --task_id   "$SLURM_ARRAY_TASK_ID" \
    --n_chunks  "$N_CHUNKS" \
    --out_dir   "$OUT_DIR" \
    --gp_factor "$GP_FACTOR" \
    --fields    $FIELDS \
    --gps       $GPS \
    $FORCE_FLAG

echo "[pfg fra] chunk ${SLURM_ARRAY_TASK_ID}: done"
