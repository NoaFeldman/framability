#!/bin/bash
# ============================================================
#  SLURM single-task script: collect minimax-frame results and
#  build the summary npz + plot.  Run after the array job has
#  finished, optionally as a dependency:
#
#      sbatch --dependency=afterok:<arrayjobid> minimax_frame_collect.sh
# ============================================================

#SBATCH --job-name=minimax_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/minimax_collect_%j.out
#SBATCH --error=logs/minimax_collect_%j.err

IN_DIR=${IN_DIR:-results_minimax_frame}
OUT_DIR=${OUT_DIR:-results_minimax_frame}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python minimax_frame_collect.py --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
