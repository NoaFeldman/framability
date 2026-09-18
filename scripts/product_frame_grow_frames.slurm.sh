#!/bin/bash
# ============================================================
#  SLURM job-array, STAGE 1 of the product-frame growth pipeline: build the
#  frame ladders (scripts/product_frame_grow_frames_worker.py).
#
#  ONE TASK = ONE PARAMETER SET.  The rounds of a ladder are strictly
#  sequential (round r+1 draws its candidates from the frame of round r), so
#  the seven requested sets are the only parallel axis here -- array 0-6.  The
#  embarrassingly parallel part is stage 2 (scripts/product_frame_grow.slurm.sh,
#  a 0-199 array over the ~360 (case, round, gamma', target) framability LPs).
#
#    task 0  model3  J = gamma' = 1, gamma = 0
#    task 1  model3  J = gamma' = 1, gamma = 2
#    task 2  model3  J = gamma' = 1, gamma = 10
#    task 3  model3  J = gamma' = 1, gamma = 20
#    task 4  model4  J = gamma' = 1, gamma = 0     (h = 1.5)
#    task 5  model4  J = gamma' = 1, gamma = 10
#    task 6  model4  J = gamma' = 1, gamma = 20
#
#  Growth runs at gamma' = J throughout (the continuous_simulation.tex
#  threshold), so each ladder serves both evaluated parameter sets.
#
#  FILTER=criterion (the default) scores the shortlisted candidates with
#  product_frame_trick.frame_element_criterion up to d_ext =
#  CRITERION_MAX_DEXT and with the cheap gauge test above it (that function's
#  dense epigraph block is O(d_ext^4) in memory: 3.2 GB at d_ext = 100).  Set
#  FILTER=gauge for the fast path, which uses the gauge test throughout.
#
#  Submit (whole pipeline, with dependencies):
#    mkdir -p logs results_product_frame_grow
#    JID1=$(sbatch --parsable scripts/product_frame_grow_frames.slurm.sh)
#    JID2=$(sbatch --parsable --dependency=afterok:${JID1} scripts/product_frame_grow.slurm.sh)
#    sbatch --dependency=afterok:${JID2} scripts/product_frame_grow_collect.slurm.sh
#
#  Idempotent: a ladder grown with the same parameters is kept (FORCE=1 to
#  regrow), so the array can be resubmitted after a time-limit kill.
#
#  Output: results_product_frame_grow/<tag>/frames.npz
# ============================================================

#SBATCH --job-name=pfg_frames
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/pfgframes_%x_%A_%a.out
#SBATCH --error=logs/pfgframes_%x_%A_%a.err

OUT_DIR=${OUT_DIR:-results_product_frame_grow}
DT=${DT:-1e-2}                       # the requested Euler step
D_EXT_MAX=${D_EXT_MAX:-100}          # grow until d_ext >= this
MAX_NEW=${MAX_NEW:-12}               # new elements per round (rings of 4 are atomic)
TILT=${TILT:-optimal}                # optimal | equatorial | both
MIN_SEP_FRAC=${MIN_SEP_FRAC:-0.1}    # drop candidates this close (in units of the ring tilt)
FILTER=${FILTER:-criterion}          # criterion | gauge
CRITERION_MAX_DEXT=${CRITERION_MAX_DEXT:-24}
ACCEPT=${ACCEPT:-nonharmful}         # nonharmful | useful
GATE=${GATE:-euler}                  # euler | expm
MAX_ROUNDS=${MAX_ROUNDS:-60}
FORCE=${FORCE:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

FORCE_FLAG=""
if [ "$FORCE" != "0" ]; then FORCE_FLAG="--force"; fi

echo "[pfg frames] case ${SLURM_ARRAY_TASK_ID}: starting"

python scripts/product_frame_grow_frames_worker.py \
    --task_id            "$SLURM_ARRAY_TASK_ID" \
    --out_dir            "$OUT_DIR" \
    --dt                 "$DT" \
    --d_ext_max          "$D_EXT_MAX" \
    --max_new_per_round  "$MAX_NEW" \
    --tilt               "$TILT" \
    --min_sep_frac       "$MIN_SEP_FRAC" \
    --filter             "$FILTER" \
    --criterion_max_dext "$CRITERION_MAX_DEXT" \
    --accept             "$ACCEPT" \
    --gate               "$GATE" \
    --max_rounds         "$MAX_ROUNDS" \
    $FORCE_FLAG

echo "[pfg frames] case ${SLURM_ARRAY_TASK_ID}: done"
