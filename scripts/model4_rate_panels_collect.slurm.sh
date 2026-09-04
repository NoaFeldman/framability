#!/bin/bash
# ============================================================
#  SLURM job: collect the model4 rate pipeline and draw the eight-panel figure.
#
#  Run after both arrays have finished (or at any time -- missing points are
#  simply left NaN and reported in the log, so this doubles as a progress
#  check).  STRIDE / MB_STRIDE must match what the two arrays were run with.
#
#  Submit:
#    sbatch scripts/model4_rate_panels_collect.slurm.sh
#    MB_STRIDE=1 sbatch scripts/model4_rate_panels_collect.slurm.sh
#
#  Output: results_model4_rate/model4_rate_panels.npz
#          results_model4_rate/model4_rate_panels.png
# ============================================================

#SBATCH --job-name=m4_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/m4collect_%x_%A.out
#SBATCH --error=logs/m4collect_%x_%A.err

IN_DIR=${IN_DIR:-results_model4_rate}
OUT_DIR=${OUT_DIR:-results_model4_rate}
STRIDE=${STRIDE:-1}          # must match the model4_rate_panels array
MB_STRIDE=${MB_STRIDE:-5}    # must match the model4_manybody array

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/model4_rate_panels_collect.py \
    --in_dir    "$IN_DIR" \
    --out_dir   "$OUT_DIR" \
    --stride    "$STRIDE" \
    --mb_stride "$MB_STRIDE"

echo "[model4 collect] done"
