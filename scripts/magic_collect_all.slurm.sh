#!/bin/bash
# ============================================================
#  SLURM collect job: aggregation + the colormaps for ALL FIVE unitary
#  magic-scan models in one job.  Submitted by scripts/submit_unitary_magic.sh
#  with --dependency=afterok on all 5 data-array jobs, so the full pipeline
#  (5 models' data generation, then one combined collect) runs from a single
#  `bash scripts/submit_unitary_magic.sh` call.  Writes, per model:
#
#    results_unitary_magic/unitary_magic_summary_<model>.npz
#    results_plots/unitary_magic_<model>_fra_D1.png        3-stabilizer framability, D = 1
#    results_plots/unitary_magic_<model>_nc_n{n}_t{k}.png  non-cliffordness, PBC ring n,
#                                                           t = dt_min * 10**k
#                                                           (18 panels: n in (4,5,6), k=0..5)
#    results_plots/unitary_magic_<model>_overview_n{n}.png the 6 t-decades of ring size n
#                                                           (one grid per n in (4,5,6))
#
#  Safe to run by hand at any time to see partial progress (missing points are
#  drawn blank):
#    sbatch scripts/magic_collect_all.slurm.sh
# ============================================================

#SBATCH --job-name=unitary_magic_col
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/unitary_magic_col_%j.out
#SBATCH --error=logs/unitary_magic_col_%j.err

IN_DIR=${IN_DIR:-results_unitary_magic}
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

# scan_module:model_key pairs, one per model of unitary_models_for_magic.tex.
MODELS=(
    "model1_ising_tilted_magic_scan:ising_tilted"
    "model2_xy_transverse_magic_scan:xy_transverse"
    "model3_xxz_transverse_magic_scan:xxz_transverse"
    "model4_xyz_magic_scan:xyz"
    "model5_xy_dm_magic_scan:xy_dm"
)

STATUS=0
for ENTRY in "${MODELS[@]}"; do
    SCAN_MODULE="${ENTRY%%:*}"
    MODEL="${ENTRY##*:}"
    echo "[collect] ${SCAN_MODULE}/${MODEL} (fit_n ${FIT_N}, deg ${DEG}, log_base ${LOG_BASE})"
    python scripts/magic_collect.py \
        --scan_module "$SCAN_MODULE" \
        --model       "$MODEL" \
        --in_dir      "$IN_DIR" \
        --plot_dir    "$PLOT_DIR" \
        --fit_n       "$FIT_N" \
        --deg         "$DEG" \
        --log_base    "$LOG_BASE" \
        --fra_color   "$FRA_COLOR" \
        --fra_plot    "$FRA_PLOT" \
        || STATUS=1
done

echo "[collect] done (status=${STATUS})"
exit $STATUS
