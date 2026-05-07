#!/bin/bash
# ============================================================
#  SLURM job array: recompute optimised framability for each
#  grid point with updated frame structure.
#  One task per point: task_id = ig * N_PTS + igp.
#
#  Submit via submit_recompute_fra.sh, or directly:
#    export N_PTS=41 J=1.0 GAMMA_STEP=0.2 OUT_DIR=results_opt
#    sbatch --array=0-1680 recompute_fra_array.sh
# ============================================================

#SBATCH --job-name=recompute_fra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/recompute_fra_%A_%a.out
#SBATCH --error=logs/recompute_fra_%A_%a.err

N_PTS=${N_PTS:-41}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_opt}
N_RESTARTS=${N_RESTARTS:-5}
MAXFEV=${MAXFEV:-1000}

# Always run from the submission directory so relative paths work.
cd "${SLURM_SUBMIT_DIR}"

source ".venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: cwd=$(pwd)  OUT_DIR=${OUT_DIR}  N_PTS=${N_PTS}  J=${J}  step=${GAMMA_STEP}"

python recompute_fra_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV"

echo "Task ${SLURM_ARRAY_TASK_ID}: done  exit=$?"
