#!/bin/bash
# ============================================================
#  Stage 3: regenerate results_plots/trotter_model[1-5].png from the restored
#  (and, where needed, recomputed) opt_fra_4/6 values.  One array task per model
#  (5 tasks); each is a light matplotlib job.
#
#    mkdir -p logs
#    sbatch scripts/trotter_replot.slurm.sh
#
#  Or run locally in one line (no cluster needed for the plots):
#    for m in model1 model2 model3 model4 model5; do
#      python scripts/trotter_scan_collect.py --model $m \
#          --in_dir results_trotter_v3 \
#          --out_png results_plots/trotter_${m}.png --save_npz
#    done
# ============================================================

#SBATCH --job-name=trotter_replot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --array=1-5
#SBATCH --output=logs/replot_%x_%A_%a.out
#SBATCH --error=logs/replot_%x_%A_%a.err

IN_DIR=${IN_DIR:-results_trotter_v3}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

MODEL="model${SLURM_ARRAY_TASK_ID}"
echo "[replot] ${MODEL} from ${IN_DIR}"

python scripts/trotter_scan_collect.py \
    --model   "$MODEL" \
    --in_dir  "$IN_DIR" \
    --out_png "results_plots/trotter_${MODEL}.png" \
    --save_npz

echo "[replot] ${MODEL}: done"
