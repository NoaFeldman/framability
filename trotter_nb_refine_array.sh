#!/bin/bash
# ============================================================
#  SLURM job-array: neighbor-seeded refinement of the
#  Lindbladian Trotter framability scan.
#
#  task_id = d_idx * N_GAMMA * N_GP + ig * N_GP + igp
#    (same layout as lindbladian_trotter_array.sh)
#  Total: 2 * 41 * 21 = 1722 tasks (0..1721)
#
#  Requires trotter_summary.npz and all per-task trotter_<d>_*.npz
#  files to already exist in OUT_DIR.
#
#  Submit (after the base scan is complete):
#    mkdir -p logs
#    sbatch --array=0-1721%100 trotter_nb_refine_array.sh
# ============================================================

#SBATCH --job-name=trotter_nb_refine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/trotter_nb_refine_%A_%a.out
#SBATCH --error=logs/trotter_nb_refine_%A_%a.err

OUT_DIR=${OUT_DIR:-results_trotter}
N_RESTARTS=${N_RESTARTS:-5}
MAXFEV=${MAXFEV:-2000}
MAX_ITER=${MAX_ITER:-500}
J=${J:-1.0}
DT=${DT:-0.01}
METHOD=${METHOD:-Nelder-Mead}
SEED=${SEED:-0}
FORCE_REAL=${FORCE_REAL:-1}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

FORCE_REAL_FLAG=""
if [ "$FORCE_REAL" = "1" ]; then
    FORCE_REAL_FLAG="--force_real"
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (OUT_DIR=${OUT_DIR}, FORCE_REAL=${FORCE_REAL})"

python trotter_nb_refine_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV" \
    --max_iter   "$MAX_ITER" \
    --J          "$J" \
    --dt         "$DT" \
    --method     "$METHOD" \
    $FORCE_REAL_FLAG

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
