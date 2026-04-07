#!/bin/bash
# ============================================================
#  Master submission script for the steady-state parameter scan.
#
#  Usage:
#    bash submit_scan.sh [--n_pts N] [--J J] [--gamma_step S]
#                        [--out_dir DIR] [--max_concurrent C]
#
#  What it does:
#    1. Validates / creates required directories.
#    2. Submits a SLURM job array (one task per gamma row).
#    3. Submits the collect job that runs after ALL array tasks
#       finish successfully (afterok dependency).
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
N_PTS=20
J=1.0
GAMMA_STEP=0.1
OUT_DIR=results
MAX_CONCURRENT=50   # max simultaneously running array tasks

# ── parse arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_pts)         N_PTS="$2";         shift 2 ;;
        --J)             J="$2";             shift 2 ;;
        --gamma_step)    GAMMA_STEP="$2";    shift 2 ;;
        --out_dir)       OUT_DIR="$2";       shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ARRAY_END=$(( N_PTS - 1 ))

echo "========================================================"
echo "  Framability steady-state scan"
echo "  N_PTS        = ${N_PTS}  (${N_PTS}x${N_PTS} = $((N_PTS*N_PTS)) grid points)"
echo "  J            = ${J}"
echo "  GAMMA_STEP   = ${GAMMA_STEP}"
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  Max jobs     = ${MAX_CONCURRENT} concurrent"
echo "========================================================"

# ── create directories ────────────────────────────────────────
mkdir -p "${OUT_DIR}" logs

# ── submit array job ─────────────────────────────────────────
ARRAY_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    sbatch --parsable \
           --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
           scan_array.sh
)
echo "Submitted scan array job:  ${ARRAY_JOB_ID}"
echo "  (tasks 0–${ARRAY_END}, up to ${MAX_CONCURRENT} running at once)"

# ── submit collect job (runs after all array tasks succeed) ──
COLLECT_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    sbatch --parsable \
           --dependency="afterok:${ARRAY_JOB_ID}" \
           scan_collect.sh
)
echo "Submitted collect job:     ${COLLECT_JOB_ID}"
echo "  (depends on array job ${ARRAY_JOB_ID} via afterok)"

echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/scan_${ARRAY_JOB_ID}_<task>.out"
echo "Results will appear in: ${OUT_DIR}/"
echo "Final figure:            ${OUT_DIR}/two_qubit_scan.png"
