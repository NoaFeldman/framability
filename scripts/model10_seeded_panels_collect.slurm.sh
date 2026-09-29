#!/bin/bash
# ============================================================
#  SLURM job: draw the model10 rate figure (scripts/model10_seeded_panels_collect.py)
#
#    row 1 | stabilizer-3 | Pauli | opt Heisenberg d_ext=4 | 8 | 12
#    row 2 | product-state chi=10 | chi=40 | 8q osc rate | 8q gap
#
#  Submitted at the end of scripts/submit_model10_seeded_refine.sh,
#  scripts/submit_model10_d12_prod.sh and scripts/submit_model10_rate.sh; by
#  hand (any time -- points a run has not reached fall back to the smaller
#  d_ext, missing product data leaves those panels empty):
#      sbatch scripts/model10_seeded_panels_collect.slurm.sh
#
#  Output: $OUT_PNG (default results_model4_rate/model10_rate_panels.png)
#          $OUT_DIR/model10_seeded_panels.npz
# ============================================================

#SBATCH --job-name=m10_seed_panels
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/m10seedpanels_%x_%A.out
#SBATCH --error=logs/m10seedpanels_%x_%A.err

OUT_DIR=${OUT_DIR:-results_model10_rate}
OUT_PNG=${OUT_PNG:-results_model4_rate/model10_rate_panels.png}
BASE_IN_DIRS=${BASE_IN_DIRS:-"results_model10_rate results_model4_rate"}
BASE_NPZ=${BASE_NPZ:-"results_model4_rate/model10_rate_panels.npz results_model10_rate/model10_rate_panels.npz"}
PROD_DIRS=${PROD_DIRS:-"results_model10_rate results_model4_rate"}
STRIDE=${STRIDE:-1}               # must match the seeded / d_ext = 12 arrays
MB_STRIDE=${MB_STRIDE:-5}         # must match the original model10 pipeline
PROD_STRIDE=${PROD_STRIDE:-1}     # must match the product-state array

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/model10_seeded_panels_collect.py \
    --out_dir      "$OUT_DIR" \
    --out_png      "$OUT_PNG" \
    --base_in_dirs $BASE_IN_DIRS \
    --base_npz     $BASE_NPZ \
    --prod_dirs    $PROD_DIRS \
    --stride       "$STRIDE" \
    --mb_stride    "$MB_STRIDE" \
    --prod_stride  "$PROD_STRIDE"

echo "[model10 panels] done"
