#!/bin/bash
# ============================================================
#  Run 10 sequential rounds of neighbor-seeded framability
#  refinement for the dissipative-PT scan.
#
#  Each round:
#    1. Job array (200 tasks): refine all 620 grid points,
#       output dpt_refine_r{N:02d}_{ih}_{ig}.npz
#    2. Collect job (afterok): merge improved framabilities
#       back into dpt_{ih}_{ig}.npz and regenerate figure.
#  Round N+1 starts only after round N's collect finishes.
#
#  Usage:
#    bash scripts/submit_dissipative_PT_refine_10x.sh \
#        [--out_dir results_dpt] \
#        [--out_png results_plots/dissipative_PT.png] \
#        [--n_rounds 10] \
#        [--start_round 1] \
#        [--n_restarts 5] \
#        [--fra_maxfev_4 1000] \
#        [--fra_maxfev_6 500] \
#        [--max_concurrent 200] \
#        [--after_job JOB_ID]
#
#  The --after_job flag lets you chain this after a scan array:
#    SCAN=$(sbatch --parsable scripts/dissipative_PT.slurm.sh)
#    bash scripts/submit_dissipative_PT_refine_10x.sh --after_job $SCAN
# ============================================================

set -euo pipefail

OUT_DIR=results_dpt
OUT_PNG=results_plots/dissipative_PT.png
N_ROUNDS=10
START_ROUND=1
N_RESTARTS=5
FRA_MAXFEV_4=1000
FRA_MAXFEV_6=500
MAX_CONCURRENT=200
AFTER_JOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --out_png)        OUT_PNG="$2";        shift 2 ;;
        --n_rounds)       N_ROUNDS="$2";       shift 2 ;;
        --start_round)    START_ROUND="$2";    shift 2 ;;
        --n_restarts)     N_RESTARTS="$2";     shift 2 ;;
        --fra_maxfev_4)   FRA_MAXFEV_4="$2";   shift 2 ;;
        --fra_maxfev_6)   FRA_MAXFEV_6="$2";   shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --after_job)      AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUT_DIR}" logs results_plots

END_ROUND=$(( START_ROUND + N_ROUNDS - 1 ))

echo "========================================================"
echo "  Dissipative-PT 10× neighbor-seeded framability refine"
echo "  OUT_DIR=${OUT_DIR}  OUT_PNG=${OUT_PNG}"
echo "  Rounds: ${START_ROUND} .. ${END_ROUND}"
echo "  n_restarts=${N_RESTARTS}  maxfev_4=${FRA_MAXFEV_4}  maxfev_6=${FRA_MAXFEV_6}"
echo "  max concurrent=${MAX_CONCURRENT}"
[[ -n "$AFTER_JOB" ]] && echo "  first round dependency=afterok:${AFTER_JOB}"
echo "========================================================"

PREV_COLLECT=""
[[ -n "$AFTER_JOB" ]] && PREV_COLLECT="$AFTER_JOB"

for ROUND in $(seq "$START_ROUND" "$END_ROUND"); do
    # ── refine array ──────────────────────────────────────────
    DEPEND_FLAG=""
    [[ -n "$PREV_COLLECT" ]] && DEPEND_FLAG="--dependency=afterok:${PREV_COLLECT}"

    ARRAY_JOB_ID=$(
        OUT_DIR="$OUT_DIR" ROUND="$ROUND" N_RESTARTS="$N_RESTARTS" \
        FRA_MAXFEV_4="$FRA_MAXFEV_4" FRA_MAXFEV_6="$FRA_MAXFEV_6" \
        sbatch --parsable \
               --array="0-199%${MAX_CONCURRENT}" \
               ${DEPEND_FLAG} \
               scripts/dissipative_PT_refine.slurm.sh
    )
    echo "Round ${ROUND}: submitted refine array ${ARRAY_JOB_ID}"

    # ── collect (afterok on refine array) ────────────────────
    COLLECT_JOB_ID=$(
        IN_DIR="$OUT_DIR" OUT_PNG="$OUT_PNG" ROUND="$ROUND" \
        sbatch --parsable \
               --dependency="afterok:${ARRAY_JOB_ID}" \
               scripts/dissipative_PT_refine_collect.slurm.sh
    )
    echo "Round ${ROUND}: submitted collect     ${COLLECT_JOB_ID}  (afterok:${ARRAY_JOB_ID})"

    PREV_COLLECT="$COLLECT_JOB_ID"
done

echo ""
echo "All ${N_ROUNDS} rounds submitted.  Final collect job: ${PREV_COLLECT}"
echo "Monitor:  squeue -u \$USER"
echo "Figure:   ${OUT_PNG}  (updated after each collect)"
