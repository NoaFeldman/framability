#!/bin/bash
# ============================================================
#  One-shot: run the dissipative-PT scan (gamma up to 10),
#  then immediately chain 10 rounds of neighbor-seeded
#  framability refinement.
#
#  Usage (from the repo root on the cluster login node):
#    bash scripts/submit_dpt_scan_and_refine.sh
#
#  Optional flags:
#    --out_dir   results_dpt          output directory
#    --out_png   results_plots/dissipative_PT.png
#    --n_rounds  10                   number of refine rounds
#    --n_restarts 5                   restarts per refine point
#    --fra_maxfev_4 1000
#    --fra_maxfev_6 500
# ============================================================

set -euo pipefail

OUT_DIR=results_dpt
OUT_PNG=results_plots/dissipative_PT.png
N_ROUNDS=10
N_RESTARTS=5
FRA_MAXFEV_4=1000
FRA_MAXFEV_6=500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out_dir)      OUT_DIR="$2";      shift 2 ;;
        --out_png)      OUT_PNG="$2";      shift 2 ;;
        --n_rounds)     N_ROUNDS="$2";     shift 2 ;;
        --n_restarts)   N_RESTARTS="$2";   shift 2 ;;
        --fra_maxfev_4) FRA_MAXFEV_4="$2"; shift 2 ;;
        --fra_maxfev_6) FRA_MAXFEV_6="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "${OUT_DIR}" logs results_plots

echo "========================================================"
echo "  dissipative-PT scan → 10× refine"
echo "  OUT_DIR=${OUT_DIR}  OUT_PNG=${OUT_PNG}"
echo "  N_ROUNDS=${N_ROUNDS}"
echo "========================================================"

# ── 1. Base scan (1120 points, 200-task array, ~5-6 pts/task) ──
SCAN_JOB=$(
    OUT_DIR="$OUT_DIR" \
    sbatch --parsable scripts/dissipative_PT.slurm.sh
)
echo "Submitted scan array:  ${SCAN_JOB}"

# ── 2. Collect after scan ──────────────────────────────────────
COLLECT_JOB=$(
    IN_DIR="$OUT_DIR" OUT_PNG="$OUT_PNG" \
    sbatch --parsable \
           --dependency=afterok:${SCAN_JOB} \
           scripts/dissipative_PT_collect.slurm.sh
)
echo "Submitted scan collect: ${COLLECT_JOB}  (afterok:${SCAN_JOB})"

# ── 3. 10 rounds of neighbor-seeded refinement (chained) ───────
bash scripts/submit_dissipative_PT_refine_10x.sh \
    --out_dir       "$OUT_DIR" \
    --out_png       "$OUT_PNG" \
    --n_rounds      "$N_ROUNDS" \
    --n_restarts    "$N_RESTARTS" \
    --fra_maxfev_4  "$FRA_MAXFEV_4" \
    --fra_maxfev_6  "$FRA_MAXFEV_6" \
    --after_job     "$SCAN_JOB"

echo ""
echo "All jobs submitted."
echo "  scan array:    ${SCAN_JOB}"
echo "  scan collect:  ${COLLECT_JOB}  (figure after scan)"
echo "  refine rounds: 1..${N_ROUNDS}  (chained after scan, figure after each round)"
echo "Monitor:  squeue -u \$USER"
