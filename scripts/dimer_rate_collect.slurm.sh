#!/bin/bash
# ============================================================
#  SLURM job: collect one dimer-frame rate run and draw its panels
#  (scripts/dimer_rate_collect.py).  Submitted by scripts/submit_dimer_rate.sh
#  with --dependency=afterok:<array id>; by hand (any time, missing points
#  are left blank):
#      MODEL=model4 sbatch --job-name=m4_dimer_collect scripts/dimer_rate_collect.slurm.sh
# ============================================================

#SBATCH --job-name=dimer_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dimer_collect_%x_%A.out
#SBATCH --error=logs/dimer_collect_%x_%A.err

MODEL=${MODEL:?set MODEL}
DIM=${DIM:-}
TILING=${TILING:-}
MODE=${MODE:-split}
OUT_DIR=${OUT_DIR:-}
STRIDE=${STRIDE:-1}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/dimer_rate_collect.py \
    --model  "$MODEL" \
    --mode   "$MODE" \
    --stride "$STRIDE" \
    ${DIM:+--dim "$DIM"} \
    ${TILING:+--tiling "$TILING"} \
    ${OUT_DIR:+--in_dir "$OUT_DIR"}

echo "[${MODEL} dimer collect] done"
