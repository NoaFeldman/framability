#!/bin/bash
# ============================================================
#  SLURM job-array: model4 product-state framability RATES, chi = 10 and 40.
#
#  For every (gamma, gamma') point of model4's rate grid (51x51 = 2601 points at
#  STRIDE=1, the grid of results_model4_rate/model4_rate_panels.png) this
#  computes product_state_rate of the two-qubit bond generator on the scan's
#  random product frames (PROD_FRAME_SEED, chi=10 and chi=40).
#
#  The 2601 points are split across a 0-199 array (200 tasks, the job cap),
#  ~13 points per task, strided so each task samples the whole grid.  Per-point
#  npz files are skipped if they already exist, so resubmitting fills holes.
#
#  Submit:
#    mkdir -p logs results_model4_rate
#    sbatch scripts/model4_product_rate.slurm.sh
#    STRIDE=5 sbatch scripts/model4_product_rate.slurm.sh   # quick 11x11 preview
#
#  Output: results_model4_rate/model4_product/pt_<ix>_<iy>.npz
#  Then:   sbatch scripts/model4_rate_panels_collect.slurm.sh
#          (adds the two product panels to model4_rate_panels.png)
#
#  RUNTIME: chi=40 is 1600 per-column LPs with 16 equality rows per point,
#  chi=10 is 100; a few tens of seconds per point.
# ============================================================

#SBATCH --job-name=m4_prod_rate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m4prodrate_%x_%A_%a.out
#SBATCH --error=logs/m4prodrate_%x_%A_%a.err

MODEL=${MODEL:-model4}
OUT_DIR=${OUT_DIR:-results_${MODEL}_rate}
N_CHUNKS=${N_CHUNKS:-200}         # must match the --array size above
STRIDE=${STRIDE:-1}               # 1 = full 51x51 grid (same as the rate panels)

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model4 product rates] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model4_product_rate_worker.py \
    --model    "$MODEL" \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --out_dir  "$OUT_DIR" \
    --stride   "$STRIDE"

echo "[model4 product rates] chunk ${SLURM_ARRAY_TASK_ID}: done"
