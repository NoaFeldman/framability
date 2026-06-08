#!/bin/bash
# ============================================================
#  Master submission for dissipative-PT neighbor-seeded
#  framability refinement.
#
#  1. Submits the refine job array (200 tasks, one per grid point).
#  2. Submits a collect job (afterok) that merges the refined
#     framabilities into the base scan and regenerates the figure.
#
#  Requires the base scan files dpt_<ih>_<ig>.npz to exist in OUT_DIR
#  (produced by dissipative_PT.slurm.sh).  Optionally chain after the
#  scan array with --after_job <SCAN_JOB_ID>.
#
#  Usage:
#    bash scripts/submit_dissipative_PT_refine.sh \
#        [--out_dir results_dpt] [--out_png results_plots/dissipative_PT.png] \
#        [--n_restarts 5] [--fra_maxfev_4 1000] [--fra_maxfev_6 500] \
#        [--max_concurrent 200] [--after_job JOB_ID]
# ============================================================

set -euo pipefail

OUT_DIR=results_dpt
OUT_PNG=results_plots/dissipative_PT.png
N_RESTARTS=5
FRA_MAXFEV_4=1000
FRA_MAXFEV_6=500
MAX_CONCURRENT=200
AFTER_JOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --out_png)        OUT_PNG="$2";        shift 2 ;;
        --n_restarts)     N_RESTARTS="$2";     shift 2 ;;
        --fra_maxfev_4)   FRA_MAXFEV_4="$2";   shift 2 ;;
        --fra_maxfev_6)   FRA_MAXFEV_6="$2";   shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --after_job)      AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUT_DIR}" logs results_plots

if [[ -z "$AFTER_JOB" && ! -f "${OUT_DIR}/dpt_00_00.npz" ]]; then
    echo "ERROR: no base scan files found in ${OUT_DIR} (e.g. dpt_00_00.npz)."
    echo "Run the scan first (dissipative_PT.slurm.sh), or pass --after_job <SCAN_JOB_ID>."
    exit 1
fi

DEPEND_FLAG=""
if [[ -n "$AFTER_JOB" ]]; then
    DEPEND_FLAG="--dependency=afterok:${AFTER_JOB}"
fi

echo "========================================================"
echo "  Dissipative-PT neighbor-seeded framability refinement"
echo "  OUT_DIR=${OUT_DIR}  OUT_PNG=${OUT_PNG}"
echo "  n_restarts=${N_RESTARTS}  maxfev_4=${FRA_MAXFEV_4}  maxfev_6=${FRA_MAXFEV_6}"
echo "  max concurrent=${MAX_CONCURRENT}"
[[ -n "$AFTER_JOB" ]] && echo "  dependency=afterok:${AFTER_JOB}"
echo "========================================================"

ARRAY_JOB_ID=$(
    OUT_DIR="$OUT_DIR" N_RESTARTS="$N_RESTARTS" \
    FRA_MAXFEV_4="$FRA_MAXFEV_4" FRA_MAXFEV_6="$FRA_MAXFEV_6" \
    sbatch --parsable \
           --array="0-199%${MAX_CONCURRENT}" \
           ${DEPEND_FLAG} \
           scripts/dissipative_PT_refine.slurm.sh
)
echo "Submitted refine array job: ${ARRAY_JOB_ID}"

COLLECT_JOB_ID=$(
    IN_DIR="$OUT_DIR" OUT_PNG="$OUT_PNG" \
    sbatch --parsable \
           --dependency="afterok:${ARRAY_JOB_ID}" \
           scripts/dissipative_PT_refine_collect.slurm.sh
)
echo "Submitted refine-collect job: ${COLLECT_JOB_ID}  (afterok:${ARRAY_JOB_ID})"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Figure:   ${OUT_PNG}"
