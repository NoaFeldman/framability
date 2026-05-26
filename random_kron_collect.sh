#!/bin/bash
# ============================================================
#  SLURM single-task: collect random-kron results.
#  Run after the array job has finished:
#
#    sbatch --dependency=afterok:<array_job_id> random_kron_collect.sh
# ============================================================

#SBATCH --job-name=random_kron_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/random_kron_collect_%j.out
#SBATCH --error=logs/random_kron_collect_%j.err

IN_DIR=${IN_DIR:-results_random_kron}
OUT_DIR=${OUT_DIR:-results_random_kron}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python random_kron_collect.py --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
