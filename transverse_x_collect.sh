#!/bin/bash
# ============================================================
#  Collect transverse_x results after the array job finishes.
#
#  Usage:
#      sbatch --dependency=afterok:<ARRAY_JOB_ID> transverse_x_collect.sh
# ============================================================

#SBATCH --job-name=transverse_x_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/transverse_x_collect_%j.out
#SBATCH --error=logs/transverse_x_collect_%j.err

IN_DIR=${IN_DIR:-results_transverse_x}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python transverse_x_collect.py --in_dir "$IN_DIR"
