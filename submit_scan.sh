#!/bin/bash
# ============================================================
#  Master submission script for the steady-state parameter scan.
#
#  Usage:
#    bash submit_scan.sh [--n_pts N] [--J J] [--gamma_step S]
#                        [--out_dir DIR] [--max_concurrent C]
#                        [--rel_tol R] [--abs_tol A]
#                        [--max_passes P] [--n_restarts K] [--maxfev F]
#
#  What it does:
#    1. Validates / creates required directories.
#    2. Submits a SLURM job array (one task per gamma row).
#    3. Submits the collect job that runs after ALL array tasks
#       finish successfully (afterok dependency).
#    4. Submits the refine job that re-optimises outlier data points
#       using neighbor seeding, then re-generates the figure.
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
N_PTS=20
J=1.0
GAMMA_STEP=0.1
OUT_DIR=results
MAX_CONCURRENT=50   # max simultaneously running array tasks
REL_TOL=0.05        # outlier relative-gap threshold for refine step
ABS_TOL=1e-3        # outlier absolute-gap threshold for refine step
MAX_PASSES=10       # max refinement passes
N_RESTARTS=3        # restarts per outlier in refine step
MAXFEV=1000         # max function evaluations per restart in refine step

# ── parse arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_pts)          N_PTS="$2";          shift 2 ;;
        --J)              J="$2";              shift 2 ;;
        --gamma_step)     GAMMA_STEP="$2";     shift 2 ;;
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --rel_tol)        REL_TOL="$2";        shift 2 ;;
        --abs_tol)        ABS_TOL="$2";        shift 2 ;;
        --max_passes)     MAX_PASSES="$2";     shift 2 ;;
        --n_restarts)     N_RESTARTS="$2";     shift 2 ;;
        --maxfev)         MAXFEV="$2";         shift 2 ;;
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
echo "  Refine:  rel_tol=${REL_TOL}  abs_tol=${ABS_TOL}  max_passes=${MAX_PASSES}"
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

# ── submit refine job (runs after collect succeeds) ──────────
REFINE_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    REL_TOL="$REL_TOL" ABS_TOL="$ABS_TOL" MAX_PASSES="$MAX_PASSES" \
    N_RESTARTS="$N_RESTARTS" MAXFEV="$MAXFEV" \
    sbatch --parsable \
           --dependency="afterok:${COLLECT_JOB_ID}" \
           refine_scan.sh
)
echo "Submitted refine job:      ${REFINE_JOB_ID}"
echo "  (depends on collect job ${COLLECT_JOB_ID} via afterok)"

echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/scan_${ARRAY_JOB_ID}_<task>.out"
echo "  tail -f logs/refine_${REFINE_JOB_ID}.out"
echo "Results will appear in: ${OUT_DIR}/"
echo "Final figure:            ${OUT_DIR}/two_qubit_scan.png"
