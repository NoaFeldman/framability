#!/bin/bash
# ============================================================
#  Collect 6-qubit star+plaquette scan results into summary + figure.
#
#  Usage:
#    sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/six_qubit_starplaq_collect.slurm.sh
# ============================================================

#SBATCH --job-name=six_starplaq_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/six_starplaq_collect_%j.out
#SBATCH --error=logs/six_starplaq_collect_%j.err

IN_DIR=${IN_DIR:-results_six_starplaq}
OUT_PNG=${OUT_PNG:-results_plots/six_starplaq.png}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
mkdir -p results_plots

python scripts/six_qubit_starplaq_collect.py \
    --in_dir "$IN_DIR" \
    --out    "$OUT_PNG"
