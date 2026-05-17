#!/bin/bash
# ============================================================
#  SLURM job-array: d_ext_single=6 (ALL columns free) framability scan.
#  One task per gamma row.
#
#  Submit:
#      mkdir -p logs results_free6
#      sbatch --array=0-40 free_6_scan_array.sh
# ============================================================

#SBATCH --job-name=free6_scan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/free6_scan_%A_%a.out
#SBATCH --error=logs/free6_scan_%A_%a.err

N_PTS=${N_PTS:-41}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_free6}
N_RESTARTS=${N_RESTARTS:-10}
MAXFEV=${MAXFEV:-2000}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (N_PTS=${N_PTS}, d_ext=6 free)"

python free_6_scan_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
