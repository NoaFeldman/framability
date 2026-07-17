#!/bin/bash
# ============================================================
#  Collect job for the alternating-scheme re-optimisation: after the
#  trotter_alt_opt array finishes, replot every model's scan summary figure
#  (results_plots/trotter_model*.png, via the unchanged
#  scripts/trotter_scan_collect.py) and the new-vs-old comparison heatmaps
#  (results_plots/trotter_alt_vs_old_model*.png).
#
#  Submit chained after the array:
#    JID=$(sbatch --parsable scripts/trotter_alt_opt.slurm.sh)
#    sbatch --dependency=afterany:$JID scripts/trotter_alt_opt_collect.slurm.sh
# ============================================================

#SBATCH --job-name=alt_opt_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/alt_%x_%j.out
#SBATCH --error=logs/alt_%x_%j.err

IN_DIR=${IN_DIR:-results_trotter_v3}
MODELS=${MODELS:-model1 model2 model3 model4 model5}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

for MODEL in $MODELS; do
    echo "[collect] $MODEL: scan summary figure"
    python scripts/trotter_scan_collect.py --model "$MODEL" --in_dir "$IN_DIR" \
        --out_png "results_plots/trotter_${MODEL}.png"
    echo "[collect] $MODEL: alternating vs old heatmap"
    python scripts/trotter_alt_vs_old_plot.py --model "$MODEL" --in_dir "$IN_DIR" \
        --out_png "results_plots/trotter_alt_vs_old_${MODEL}.png"
done

echo "[collect] done"
