#!/bin/bash
# ============================================================
#  Merge dissipative-PT refinement results into the base scan
#  and regenerate the colormap figure.
#
#  Usage (after the refine array completes):
#    sbatch --dependency=afterok:<REFINE_ARRAY_JOB_ID> \
#           scripts/dissipative_PT_refine_collect.slurm.sh
# ============================================================

#SBATCH --job-name=dpt_refine_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dpt_refine_collect_%j.out
#SBATCH --error=logs/dpt_refine_collect_%j.err

IN_DIR=${IN_DIR:-results_dpt}
OUT_PNG=${OUT_PNG:-results_plots/dissipative_PT.png}
ROUND=${ROUND:-1}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
mkdir -p results_plots

python scripts/dissipative_PT_refine_collect.py \
    --in_dir  "$IN_DIR" \
    --out_png "$OUT_PNG" \
    --round   "$ROUND"

echo "Refine-collect done: $OUT_PNG"
