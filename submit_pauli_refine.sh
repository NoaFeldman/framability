#!/bin/bash
# ============================================================
#  Submit pauli_refine_worker as a SLURM job array.
#
#  Usage:
#    bash submit_pauli_refine.sh [--n_pts N] [--J J]
#                                [--gamma_step S] [--out_dir DIR]
#                                [--n_restarts K] [--maxfev F]
#                                [--max_iter I] [--max_concurrent C]
#                                [--after_job JOB_ID]
#
#  Options:
#    --after_job JOB_ID   submit with afterok:JOB_ID dependency,
#                         e.g. chain after the neighbor-refine job.
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
N_PTS=20
J=1.0
GAMMA_STEP=0.1
OUT_DIR=results
N_RESTARTS=5
MAXFEV=1000
MAX_ITER=200
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
        --max_iter)       MAX_ITER="$2";       shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --after_job)      AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ARRAY_END=$(( N_PTS - 1 ))
mkdir -p "${OUT_DIR}" logs

echo "========================================================"
echo "  Pauli-refine pass (min_fra > pauli_fra data points)"
echo "  N_PTS        = ${N_PTS}"
echo "  J            = ${J}"
echo "  GAMMA_STEP   = ${GAMMA_STEP}"
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  n_restarts   = ${N_RESTARTS}  (+1 Pauli seed always appended)"
echo "  maxfev       = ${MAXFEV}   max_iter = ${MAX_ITER}"
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

# ── submit array ─────────────────────────────────────────────
PAULI_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    N_RESTARTS="$N_RESTARTS" MAXFEV="$MAXFEV" MAX_ITER="$MAX_ITER" \
    sbatch --parsable \
           --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
           ${DEPEND_FLAG} \
           pauli_refine_array.sh
)
echo "Submitted pauli-refine array job: ${PAULI_JOB_ID}"
echo "  (tasks 0–${ARRAY_END}, up to ${MAX_CONCURRENT} running at once)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f logs/pauli_ref_${PAULI_JOB_ID}_<task>.out"
echo "Results will be updated in: ${OUT_DIR}/"
