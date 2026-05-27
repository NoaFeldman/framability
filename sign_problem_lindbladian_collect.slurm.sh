#!/bin/bash
# ============================================================
#  Collect optimised sign-problem results for both Lindbladians
#  (no field and transverse field) and produce the 2x2 colormap.
#
#  Usage:
#    sbatch --dependency=afterok:<ARRAY_JOB_ID_1>:<ARRAY_JOB_ID_2> \
#           sign_problem_lindbladian_collect.slurm.sh
# ============================================================

#SBATCH --job-name=sign_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/sign_collect_%j.out
#SBATCH --error=logs/sign_collect_%j.err

NOFIELD_DIR=${NOFIELD_DIR:-results_sign_problem_nofield}
TRANSVERSE_DIR=${TRANSVERSE_DIR:-results_sign_problem_transverse}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python sign_problem_lindbladian_collect.py \
    --nofield_dir    "$NOFIELD_DIR" \
    --transverse_dir "$TRANSVERSE_DIR"
