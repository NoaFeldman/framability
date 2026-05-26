#!/bin/bash
# ============================================================
#  SLURM job-array: optimize framability of the two-qubit
#  Lindbladian Trotter step exp(L*dt) over a 2-D (gamma, gamma')
#  scan for d_ext_single in {4, 6}.
#
#  gamma  : 0.0 .. 8.0  step 0.2  (N_GAMMA = 41)
#  gamma' : 0.0 .. 4.0  step 0.2  (N_GP    = 21)
#  d_ext  : 4, 6                  (N_D     =  2)
#
#  task_id = d_idx * 41 * 21 + ig * 21 + igp
#  Total: 2 * 41 * 21 = 1722 tasks (0..1721)
#
#  Submit:
#    mkdir -p logs results_trotter
#    sbatch --array=0-1721%100 lindbladian_trotter_array.sh
# ============================================================

#SBATCH --job-name=trotter_fra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/trotter_fra_%A_%a.out
#SBATCH --error=logs/trotter_fra_%A_%a.err

OUT_DIR=${OUT_DIR:-results_trotter}
N_RESTARTS=${N_RESTARTS:-10}
MAXFEV=${MAXFEV:-2000}
MAX_ITER=${MAX_ITER:-500}
J=${J:-1.0}
DT=${DT:-0.01}
METHOD=${METHOD:-Nelder-Mead}
SEED=${SEED:-0}
FORCE_REAL=${FORCE_REAL:-1}   # 1 => force real S (override use_complex auto-select)

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

FORCE_REAL_FLAG=""
if [ "$FORCE_REAL" = "1" ]; then
    FORCE_REAL_FLAG="--force_real"
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (OUT_DIR=${OUT_DIR}, FORCE_REAL=${FORCE_REAL})"

python lindbladian_trotter_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV" \
    --max_iter   "$MAX_ITER" \
    --J          "$J" \
    --dt         "$DT" \
    --method     "$METHOD" \
    --seed       "$SEED" \
    $FORCE_REAL_FLAG

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
