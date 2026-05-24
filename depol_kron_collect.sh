#!/bin/bash
# ============================================================
#  Collect depol_kron results after the array job finishes.
#
#  Usage (as dependency):
#      sbatch --dependency=afterok:<ARRAY_JOB_ID> depol_kron_collect.sh
# ============================================================

#SBATCH --job-name=depol_kron_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/depol_kron_collect_%j.out
#SBATCH --error=logs/depol_kron_collect_%j.err

IN_DIR=${IN_DIR:-results_depol_kron}
OUT_DIR=${OUT_DIR:-results_depol_kron}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python depol_kron_collect.py --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
