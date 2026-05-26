#!/bin/bash
# ============================================================
#  SLURM single-task: collect minimax Toffoli results.
#  Run after the array job has finished:
#
#    sbatch --dependency=afterok:<array_job_id> minimax_toffoli_collect.sh
# ============================================================

#SBATCH --job-name=minimax_toffoli_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/minimax_toffoli_collect_%j.out
#SBATCH --error=logs/minimax_toffoli_collect_%j.err

IN_DIR=${IN_DIR:-results_minimax_toffoli}
OUT_DIR=${OUT_DIR:-results_minimax_toffoli}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python minimax_toffoli_collect.py --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
