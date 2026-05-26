#!/bin/bash
# ============================================================
#  SLURM single-task: collect Lindbladian Trotter results.
#  Run after the array job has finished:
#
#    sbatch --dependency=afterok:<array_job_id> lindbladian_trotter_collect.sh
# ============================================================

#SBATCH --job-name=trotter_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/trotter_collect_%j.out
#SBATCH --error=logs/trotter_collect_%j.err

IN_DIR=${IN_DIR:-results_trotter}
OUT_DIR=${OUT_DIR:-results_trotter}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python lindbladian_trotter_collect.py --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
