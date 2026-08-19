#!/bin/bash
# ============================================================
#  SLURM job-array: ONE quick-refine round of opt_fra_4/opt_fra_6 over the
#  full (gamma, gamma') grid of one model, at one fixed DT_BASE (base_idx).
#
#  Item 5: "redo the opt Heisenberg framability with d_ext=6 and d_ext=4 by
#  doing 10 stages of neighbor refining ... quick option".  Rounds must run
#  sequentially (round r reads round r-1's output), so this script does ONE
#  round; scripts/submit_dtbase_line_quick_refine.sh chains --round 1..10 via
#  `sbatch --wait` per (model, base_idx), and runs the up-to-20 independent
#  (model, base_idx) chains in parallel (bounded by MAX_PARALLEL_CHAINS).
#
#  The N_TOTAL grid points (2601 for model3/model4) are split across the array
#  via --n_chunks (default 200, i.e. one 0-199 array -> stays at the 200-job
#  cap regardless of grid size); most points have no base_<idx>.npz yet and
#  are skipped near-instantly, so this is cheap even early in the campaign.
#
#  Submit (one round; usually invoked by submit_dtbase_line_quick_refine.sh,
#  not by hand):
#    mkdir -p logs results_dtbase_line
#    MODEL=model3 BASE_IDX=0 ROUND=1 sbatch scripts/trotter_dtbase_line_quick_refine.slurm.sh
#
#  Output: results_dtbase_line/<tag>/base_<idx>_qrefine_r<round>.npz
#  (boundary points only; interior points write nothing).
# ============================================================

#SBATCH --job-name=trot_qrefine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/qrefine_%x_%A_%a.out
#SBATCH --error=logs/qrefine_%x_%A_%a.err

MODEL=${MODEL:-model3}
BASE_IDX=${BASE_IDX:?set BASE_IDX (0..9) to the DT_BASE index to refine}
ROUND=${ROUND:?set ROUND (1..10) to the quick-refine round}
OUT_DIR=${OUT_DIR:-results_dtbase_line}
N_CHUNKS=${N_CHUNKS:-200}
N_RESTARTS=${N_RESTARTS:-3}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-1000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-500}
FRA_TOL=${FRA_TOL:-1e-6}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL base_idx=$BASE_IDX round=$ROUND] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/trotter_dtbase_line_quick_refine_worker.py \
    --model         "$MODEL" \
    --base_idx      "$BASE_IDX" \
    --round         "$ROUND" \
    --task_id       "$SLURM_ARRAY_TASK_ID" \
    --n_chunks      "$N_CHUNKS" \
    --out_dir       "$OUT_DIR" \
    --n_restarts    "$N_RESTARTS" \
    --fra_maxfev_4  "$FRA_MAXFEV_4" \
    --fra_maxfev_6  "$FRA_MAXFEV_6" \
    --fra_tol       "$FRA_TOL" \
    --seed          "$SEED"

echo "[$MODEL base_idx=$BASE_IDX round=$ROUND] chunk ${SLURM_ARRAY_TASK_ID}: done"
