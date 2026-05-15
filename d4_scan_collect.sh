#!/bin/bash
# ============================================================
#  SLURM single-task: collect d4 scan results and plot.
#  Run after the array job finishes:
#      sbatch --dependency=afterok:<arrayjobid> d4_scan_collect.sh
# ============================================================

#SBATCH --job-name=d4_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/d4_collect_%j.out
#SBATCH --error=logs/d4_collect_%j.err

D4_DIR=${D4_DIR:-results_d4}
D6_DIR=${D6_DIR:-results}
N_PTS=${N_PTS:-41}
GAMMA_STEP=${GAMMA_STEP:-0.2}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python d4_scan_collect.py \
    --d4_dir "$D4_DIR" \
    --d6_dir "$D6_DIR" \
    --n_pts "$N_PTS" \
    --gamma_step "$GAMMA_STEP"
