#!/bin/bash
# ============================================================
#  SLURM single-task script: collect valley results and build
#  the summary npz + plot.  Run as a dependency on the array:
#
#      sbatch --dependency=afterok:<arrayjobid> scripts/valley_collect.sh
# ============================================================

#SBATCH --job-name=valley_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/valley_collect_%j.out
#SBATCH --error=logs/valley_collect_%j.err

IN_DIR=${IN_DIR:-results_valley}
OUT_DIR=${OUT_DIR:-results_valley}
D_EXT_SINGLE=${D_EXT_SINGLE:-6}
TAG_SUFFIX=${TAG_SUFFIX:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/valley_collect.py \
    --in_dir "$IN_DIR" --out_dir "$OUT_DIR" \
    --d_ext_single "$D_EXT_SINGLE" \
    --tag_suffix "$TAG_SUFFIX"
