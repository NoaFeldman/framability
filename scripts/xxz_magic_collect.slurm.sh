#!/bin/bash
# ============================================================
#  SLURM collect job: aggregation + the colormaps of the XXZ magic scan.
#
#  Submitted by scripts/submit_xxz_magic.sh with --dependency=afterok on the
#  data array, so data generation and plotting run in one submission.  Writes
#
#    results_xxz_magic/xxz_magic_summary_<model>.npz
#    results_plots/xxz_magic_<model>_fra_D1.png        3-stabilizer framability, D = 1
#    results_plots/xxz_magic_<model>_nc_n{n}_t{k}.png  non-cliffordness, PBC ring n,
#                                                       t = dt_min * 10**k
#                                                       (18 panels: n in (4,5,6), k=0..5)
#    results_plots/xxz_magic_<model>_overview_n{n}.png the 6 t-decades of ring size n
#                                                       (one grid per n in (4,5,6))
#
#  Safe to run by hand at any time to see partial progress (missing points are
#  drawn blank):
#    sbatch scripts/xxz_magic_collect.slurm.sh
# ============================================================

#SBATCH --job-name=xxz_magic_col
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/xxz_magic_col_%j.out
#SBATCH --error=logs/xxz_magic_col_%j.err

MODEL=${MODEL:-xxz}
IN_DIR=${IN_DIR:-results_xxz_magic}
PLOT_DIR=${PLOT_DIR:-results_plots}
FIT_N=${FIT_N:-15}         # capped at the 10 rungs of the dt ladder
DEG=${DEG:-1}
LOG_BASE=${LOG_BASE:-e}    # e (nats) | 2 (bits) for the non-cliffordness maps
FRA_COLOR=${FRA_COLOR:-auto}   # auto | linear | log   colour scale
FRA_PLOT=${FRA_PLOT:-auto}     # auto | limit | rate   fra**(1/dt) or its log

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$PLOT_DIR"

echo "[collect] ${IN_DIR}/${MODEL} (fit_n ${FIT_N}, deg ${DEG}, log_base ${LOG_BASE})"

python scripts/xxz_magic_collect.py \
    --model     "$MODEL" \
    --in_dir    "$IN_DIR" \
    --plot_dir  "$PLOT_DIR" \
    --fit_n     "$FIT_N" \
    --deg       "$DEG" \
    --log_base  "$LOG_BASE" \
    --fra_color "$FRA_COLOR" \
    --fra_plot  "$FRA_PLOT"

echo "[collect] done"
