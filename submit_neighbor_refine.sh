#!/bin/bash
# ============================================================
#  Master submission script for neighbor-seeded min-framability
#  refinement.
#
#  Usage:
#    bash submit_neighbor_refine.sh [--n_pts N] [--J J]
#                                   [--gamma_step S] [--out_dir DIR]
#                                   [--n_restarts K] [--maxfev F]
#                                   [--max_concurrent C]
#                                   [--after_job JOB_ID]
#
#  What it does:
#    1. Submits a SLURM job array (one task per grid point, N²
#       tasks total).  Each task checks if any 4-connected neighbor
#       has lower min_fra; if so, re-optimises seeded with the
#       neighbor's D matrix.
#    2. Submits a collect job (afterok dependency) that updates
#       the row files and regenerates two_qubit_scan.png.
#
#  Requires scan_full.npy to already exist in out_dir.
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
N_PTS=41
J=1.0
GAMMA_STEP=0.2
OUT_DIR=results
N_RESTARTS=5
MAXFEV=1000
MAX_CONCURRENT=50
AFTER_JOB=""

# ── parse arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_pts)          N_PTS="$2";          shift 2 ;;
        --J)              J="$2";              shift 2 ;;
        --gamma_step)     GAMMA_STEP="$2";     shift 2 ;;
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --n_restarts)     N_RESTARTS="$2";     shift 2 ;;
        --maxfev)         MAXFEV="$2";         shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --after_job)      AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ARRAY_END=$(( N_PTS * N_PTS - 1 ))
mkdir -p "${OUT_DIR}" logs

# ── sanity check ─────────────────────────────────────────────
if [[ ! -f "${OUT_DIR}/scan_full.npy" ]]; then
    echo "ERROR: ${OUT_DIR}/scan_full.npy not found."
    echo "Run the main scan first (submit_scan.sh)."
    exit 1
fi

echo "========================================================"
echo "  Neighbor-seeded min-framability refinement"
echo "  N_PTS        = ${N_PTS}  (${N_PTS}x${N_PTS} = $((N_PTS*N_PTS)) tasks)"
echo "  J            = ${J}"
echo "  GAMMA_STEP   = ${GAMMA_STEP}"
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  n_restarts   = ${N_RESTARTS}  maxfev = ${MAXFEV}"
echo "  Max jobs     = ${MAX_CONCURRENT} concurrent"
if [[ -n "$AFTER_JOB" ]]; then
    echo "  Dependency   = afterok:${AFTER_JOB}"
fi
echo "========================================================"

# ── build optional dependency flag ───────────────────────────
DEPEND_FLAG=""
if [[ -n "$AFTER_JOB" ]]; then
    DEPEND_FLAG="--dependency=afterok:${AFTER_JOB}"
fi

# ── submit array job ─────────────────────────────────────────
ARRAY_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    N_RESTARTS="$N_RESTARTS" MAXFEV="$MAXFEV" \
    sbatch --parsable \
           --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
           ${DEPEND_FLAG} \
           neighbor_refine_array.sh
)
echo "Submitted neighbor-refine array job: ${ARRAY_JOB_ID}"
echo "  (tasks 0–${ARRAY_END}, up to ${MAX_CONCURRENT} running at once)"

# ── submit collect job (runs after all array tasks succeed) ──
COLLECT_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    sbatch --parsable \
           --dependency="afterok:${ARRAY_JOB_ID}" \
           neighbor_refine_collect.sh
)
echo "Submitted neighbor-refine collect job: ${COLLECT_JOB_ID}"
echo "  (depends on array job ${ARRAY_JOB_ID} via afterok)"

echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/nb_refine_${ARRAY_JOB_ID}_<task>.out"
echo "  tail -f logs/nb_collect_${COLLECT_JOB_ID}.out"
echo "Results will be updated in: ${OUT_DIR}/"
echo "Figure:                     ${OUT_DIR}/two_qubit_scan.png"
