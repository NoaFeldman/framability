#!/bin/bash
# ============================================================
#  SLURM job: re-extrapolate and redraw results_dtbase_line/<model>_dtbase_extrap.png
#  with the six random-frame panels of scripts/dtbase_randframe_worker.py
#  (gamma' > 4.2 is left empty in those panels).
#
#  Runs scripts/trotter_dtbase_line_extrap.py, i.e. the standard figure (all
#  MEASURES + osc-rate / Q / Q_obs panels when their data exist) with the
#  random-frame grids added; they are also stored in <model>_dtbase_extrap.npz.
#
#  Submit after the array (afterok: runs only if every task succeeded):
#    JID=$(sbatch --parsable scripts/dtbase_randframe.slurm.sh)
#    sbatch --dependency=afterok:${JID} scripts/dtbase_randframe_collect.slurm.sh
#  or on its own at any time (missing points are drawn empty):
#    sbatch scripts/dtbase_randframe_collect.slurm.sh
# ============================================================

#SBATCH --job-name=dtb_randframe_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/dtbrandframe_collect_%x_%A.out
#SBATCH --error=logs/dtbrandframe_collect_%x_%A.err

MODEL=${MODEL:-model3}
IN_DIR=${IN_DIR:-results_dtbase_line}
OUT_DIR=${OUT_DIR:-results_dtbase_line}
RF_DIR=${RF_DIR:-results_dtbase_randframe}
STRIDE=${STRIDE:-1}           # must match the randframe array and the dtbase-line sweep

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/dtbase_randframe_collect.py --model "$MODEL" --rf_dir "$RF_DIR" \
    --stride "$STRIDE"

python scripts/trotter_dtbase_line_extrap.py \
    --models  "$MODEL" \
    --in_dir  "$IN_DIR" \
    --out_dir "$OUT_DIR" \
    --rf_dir  "$RF_DIR" \
    --stride  "$STRIDE"

echo "[$MODEL randframe collect] done"
