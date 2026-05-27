#!/bin/bash
# ============================================================
#  SLURM single-task: collect free_6 scan results and plot.
#  Run after the array job finishes:
#      sbatch --dependency=afterok:<arrayjobid> scripts/free_6_scan_collect.sh
# ============================================================

#SBATCH --job-name=free6_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/free6_collect_%j.out
#SBATCH --error=logs/free6_collect_%j.err

FREE6_DIR=${FREE6_DIR:-results_free6}
D4_DIR=${D4_DIR:-results_d4}
D6_DIR=${D6_DIR:-results}
N_PTS=${N_PTS:-41}
GAMMA_STEP=${GAMMA_STEP:-0.2}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/free_6_scan_collect.py \
    --free6_dir "$FREE6_DIR" \
    --d4_dir "$D4_DIR" \
    --d6_dir "$D6_DIR" \
    --n_pts "$N_PTS" \
    --gamma_step "$GAMMA_STEP"
