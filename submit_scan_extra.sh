#!/bin/bash
# ============================================================
#  Master submission script for extra steady-state properties
#  (max LPDO bond dim, magnetization).
#
#  Runs without recomputing the original 8 properties.
#  After the extra data is gathered, the collect job merges
#  everything and regenerates the combined figure.
#
#  Usage:
#    bash submit_scan_extra.sh [--n_pts N] [--J J] [--gamma_step S]
#                              [--out_dir DIR] [--max_concurrent C]
#                              [--N_qubits Q] [--after_job JOB_ID]
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
N_PTS=20
J=1.0
GAMMA_STEP=0.1
OUT_DIR=results
MAX_CONCURRENT=50
N_QUBITS=2
AFTER_JOB=""

# ── parse arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_pts)          N_PTS="$2";          shift 2 ;;
        --J)              J="$2";              shift 2 ;;
        --gamma_step)     GAMMA_STEP="$2";     shift 2 ;;
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --N_qubits)       N_QUBITS="$2";       shift 2 ;;
        --after_job)      AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ARRAY_END=$(( N_PTS - 1 ))

echo "========================================================"
echo "  Extra properties scan (max bond dim + magnetization)"
echo "  N_PTS        = ${N_PTS}"
echo "  J            = ${J}"
echo "  GAMMA_STEP   = ${GAMMA_STEP}"
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  N_QUBITS     = ${N_QUBITS}"
echo "  Max jobs     = ${MAX_CONCURRENT} concurrent"
if [[ -n "$AFTER_JOB" ]]; then
    echo "  After job    = ${AFTER_JOB}"
fi
echo "========================================================"

# ── create directories ────────────────────────────────────────
mkdir -p "${OUT_DIR}" logs

# ── optional dependency flag ─────────────────────────────────
DEP_FLAG=""
if [[ -n "$AFTER_JOB" ]]; then
    DEP_FLAG="--dependency=afterok:${AFTER_JOB}"
fi

# ── submit extra-properties array job ────────────────────────
EXTRA_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    N_QUBITS="$N_QUBITS" \
    sbatch --parsable \
           --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
           ${DEP_FLAG} \
           scan_array_extra.sh
)
echo "Submitted extra array job: ${EXTRA_JOB_ID}"
echo "  (tasks 0–${ARRAY_END}, up to ${MAX_CONCURRENT} running at once)"

# ── submit collect job (merges base + extra, regenerates figure) ──
COLLECT_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    sbatch --parsable \
           --dependency="afterok:${EXTRA_JOB_ID}" \
           scan_collect.sh
)
echo "Submitted collect job:     ${COLLECT_JOB_ID}"
echo "  (depends on extra job ${EXTRA_JOB_ID} via afterok)"

echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
