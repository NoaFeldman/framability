#!/bin/bash
# ============================================================
#  SLURM job-array script: framability valley exploration for
#  the 2-qubit Lindbladian.
#
#  For each (gamma, gamma_p) grid point find the optimal frame
#  D (Kronecker S(x)S, d_ext_single configurable) minimising
#  heisenberg_framability(D, expm(dt * L)) and locate
#  valley_param_size=10 points on the edge of the plateau.
#
#  Points (see POINTS in valley_worker.py):
#      task 0 -> (gamma, gamma_p) = (6.0, 0.0)
#      task 1 -> (gamma, gamma_p) = (3.0, 0.8)
#
#  Total tasks: 0..1
#
#  Submit:
#      mkdir -p logs results_valley
#      sbatch --array=0-1 scripts/valley_array.sh
# ============================================================

#SBATCH --job-name=valley
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/valley_%A_%a.out
#SBATCH --error=logs/valley_%A_%a.err

OUT_DIR=${OUT_DIR:-results_valley}
D_EXT_SINGLE=${D_EXT_SINGLE:-6}
N_RESTARTS=${N_RESTARTS:-20}
MAXFEV=${MAXFEV:-2000}
MAX_ITER=${MAX_ITER:-500}
SEED=${SEED:-0}
METHOD=${METHOD:-cobyqa}
VALLEY_PARAM_SIZE=${VALLEY_PARAM_SIZE:-10}
PLATEAU_TOL=${PLATEAU_TOL:-1e-4}
INIT_STEP=${INIT_STEP:-0.1}
J=${J:-1.0}
DT=${DT:-0.01}
LONG_TIME=${LONG_TIME:-0}
LONG_TIME_FACTOR=${LONG_TIME_FACTOR:-100.0}
TAG_SUFFIX=${TAG_SUFFIX:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (OUT_DIR=${OUT_DIR}, d=${D_EXT_SINGLE}, LONG_TIME=${LONG_TIME})"

LONG_TIME_FLAG=""
if [ "$LONG_TIME" = "1" ] || [ "$LONG_TIME" = "true" ]; then
    LONG_TIME_FLAG="--long_time"
fi

python scripts/valley_worker.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --out_dir "$OUT_DIR" \
    --d_ext_single "$D_EXT_SINGLE" \
    --n_restarts "$N_RESTARTS" \
    --maxfev "$MAXFEV" \
    --max_iter "$MAX_ITER" \
    --seed "$SEED" \
    --method "$METHOD" \
    --valley_param_size "$VALLEY_PARAM_SIZE" \
    --plateau_tol "$PLATEAU_TOL" \
    --init_step "$INIT_STEP" \
    --J "$J" \
    --dt "$DT" \
    $LONG_TIME_FLAG \
    --long_time_factor "$LONG_TIME_FACTOR" \
    --tag_suffix "$TAG_SUFFIX" \
    --verbose

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
