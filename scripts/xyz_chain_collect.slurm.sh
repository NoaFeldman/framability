#!/bin/bash
# ============================================================
#  Collect XYZ-chain results into a summary npz + line-plot figure.
#
#  Usage:
#    sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/xyz_chain_collect.slurm.sh
# ============================================================

#SBATCH --job-name=xyz_chain_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/xyz_chain_collect_%j.out
#SBATCH --error=logs/xyz_chain_collect_%j.err

IN_DIR=${IN_DIR:-results_xyz_chain}
OUT_PNG=${OUT_PNG:-results_plots/xyz_chain.png}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
mkdir -p results_plots

python scripts/xyz_chain_collect.py --in_dir "$IN_DIR" --out "$OUT_PNG"
