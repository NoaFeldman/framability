#!/bin/bash
# ============================================================
#  SLURM job-array: d_ext_single=4 (free columns) framability scan.
#  One task per gamma row.
#
#  Submit:
#      mkdir -p logs results_d4
#      sbatch --array=0-40 d4_scan_array.sh
# ============================================================

#SBATCH --job-name=d4_scan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/d4_scan_%A_%a.out
#SBATCH --error=logs/d4_scan_%A_%a.err

N_PTS=${N_PTS:-41}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_d4}
N_RESTARTS=${N_RESTARTS:-10}
MAXFEV=${MAXFEV:-2000}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (N_PTS=${N_PTS}, d_ext=4 free)"

python d4_scan_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
