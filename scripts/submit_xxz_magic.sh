#!/bin/bash
# ============================================================
#  One-shot submission of the whole XXZ magic / framability pipeline:
#  the 200-task data array, then the collect+plot job chained to it with
#  --dependency=afterok, so a single command produces the figures.
#
#    bash scripts/submit_xxz_magic.sh
#
#  Environment overrides are passed straight through, e.g.
#    GROUP=nc bash scripts/submit_xxz_magic.sh          # cheap maps only
#    LOG_BASE=2 bash scripts/submit_xxz_magic.sh        # plot in bits
#    FORCE=1 bash scripts/submit_xxz_magic.sh           # ignore cached npz
#
#  afterok fires only if EVERY array task exits 0.  A task that dies (timeout,
#  node failure) therefore blocks the plots; resubmit the array -- finished
#  points are skipped -- or run the collect on its own, which tolerates missing
#  points and draws them blank:
#    sbatch scripts/xxz_magic_collect.slurm.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results_xxz_magic results_plots

DATA_ID=$(sbatch --parsable scripts/xxz_magic.slurm.sh)
echo "data array   : job $DATA_ID  (200 tasks)"

PLOT_ID=$(sbatch --parsable --dependency=afterok:"$DATA_ID" \
                 --kill-on-invalid-dep=yes scripts/xxz_magic_collect.slurm.sh)
echo "collect+plot : job $PLOT_ID  (afterok:$DATA_ID)"

echo
echo "watch:   squeue -u \$USER"
echo "logs:    tail -f logs/xxz_magic_${DATA_ID}_0.out"
echo "figures: results_plots/xxz_magic_xxz_overview.png"
