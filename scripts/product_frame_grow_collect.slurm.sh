#!/bin/bash
# ============================================================
#  SLURM job, STAGE 3 of the product-frame growth pipeline: aggregate and plot
#  (scripts/product_frame_grow_collect.py).
#
#  Writes, per case, the framability and the framability RATE (f-1)/dt against
#  d_ext, plus a seven-panel rate figure and one npz holding every curve.  The
#  rate panel is the readable one: at dt = 1e-2 every framability here sits
#  within ~1e-3 of 1.
#
#  Submit after stage 2 (afterok: only if every task succeeded):
#    JID2=$(sbatch --parsable --dependency=afterok:${JID1} scripts/product_frame_grow.slurm.sh)
#    sbatch --dependency=afterok:${JID2} scripts/product_frame_grow_collect.slurm.sh
#  or on its own at any time (missing rungs are simply left out of the curves):
#    sbatch scripts/product_frame_grow_collect.slurm.sh
#
#  Output: results_product_frame_grow/product_frame_grow.npz
#          results_product_frame_grow/<tag>_grow.png
#          results_product_frame_grow/product_frame_grow_rates.png
# ============================================================

#SBATCH --job-name=pfg_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pfgcollect_%x_%A.out
#SBATCH --error=logs/pfgcollect_%x_%A.err

IN_DIR=${IN_DIR:-results_product_frame_grow}
OUT_DIR=${OUT_DIR:-results_product_frame_grow}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/product_frame_grow_collect.py --in_dir "$IN_DIR" --out_dir "$OUT_DIR"

echo "[pfg collect] done"
