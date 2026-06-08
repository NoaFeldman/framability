#!/bin/bash
# ============================================================
#  Collect dissipative-PT results into summary npz + colormap figure.
#
#  Usage (run after the worker array completes):
#    sbatch --dependency=afterok:<ARRAY_JOB_ID> \
#           scripts/dissipative_PT_collect.slurm.sh
# ============================================================

#SBATCH --job-name=dpt_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dpt_collect_%j.out
#SBATCH --error=logs/dpt_collect_%j.err

IN_DIR=${IN_DIR:-results_dpt}
OUT_PNG=${OUT_PNG:-results_plots/dissipative_PT.png}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
mkdir -p results_plots

python scripts/dissipative_PT_collect.py \
    --in_dir  "$IN_DIR" \
    --out_png "$OUT_PNG" \
    --save_npz

echo "Collect done: $OUT_PNG"
