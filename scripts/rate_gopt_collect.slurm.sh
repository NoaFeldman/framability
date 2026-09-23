#!/bin/bash
# ============================================================
#  SLURM job: collect the global rate re-optimisation of one model, draw the
#  separate gopt figure, and replot the model's standard figure with the new
#  values folded in (model4_rate_panels.png / <model>_dtbase_extrap.png).
#
#  Submitted by scripts/submit_rate_gopt.sh with
#      --dependency=afterok:<array job id>
#  so it starts only after every worker task finished successfully.
#  By hand (any time; missing points are left blank):
#      MODEL=model4 sbatch --job-name=m4_gopt_collect scripts/rate_gopt_collect.slurm.sh
# ============================================================

#SBATCH --job-name=gopt_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gopt_collect_%x_%A.out
#SBATCH --error=logs/gopt_collect_%x_%A.err

MODEL=${MODEL:?set MODEL}
IN_DIR=${IN_DIR:-results_${MODEL}_rate}
STRIDE=${STRIDE:-1}
D_EXTS=${D_EXTS:-"4 6 8"}
REPLOT=${REPLOT:-1}                # REPLOT=0: only the separate gopt figure

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/rate_gopt_collect.py \
    --model   "$MODEL" \
    --in_dir  "$IN_DIR" \
    --out_dir "$IN_DIR" \
    --stride  "$STRIDE" \
    --d_exts  $D_EXTS \
    $([ "$REPLOT" = "1" ] && echo --replot)

echo "[${MODEL} gopt collect] done"
