#!/bin/bash
# ============================================================
#  Master submission: product-state Schrödinger framability
#  chi=20 and chi=40 over the full (gamma, gamma') grid.
#
#  Submits a job array (one task per grid point) followed by a
#  collect job that assembles results and regenerates the figure.
#
#  Usage:
#    bash submit_schro_fra.sh [--n_pts N] [--J J] [--gamma_step S]
#                             [--out_dir DIR] [--max_concurrent C]
#                             [--after_job JOB_ID]
#
#  Options:
#    --n_pts          Grid points per axis (default: 41)
#    --J              Coupling constant    (default: 1.0)
#    --gamma_step     Grid spacing         (default: 0.2)
#    --out_dir        Output directory     (default: results)
#    --max_concurrent Max simultaneous array tasks (default: 100)
#    --after_job      Submit after this SLURM job ID completes
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
N_PTS=41
J=1.0
GAMMA_STEP=0.2
OUT_DIR=results
MAX_CONCURRENT=100
AFTER_JOB=""

# ── parse arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_pts)          N_PTS="$2";          shift 2 ;;
        --J)              J="$2";              shift 2 ;;
        --gamma_step)     GAMMA_STEP="$2";     shift 2 ;;
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --after_job)      AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ARRAY_END=$(( N_PTS * N_PTS - 1 ))

echo "========================================================"
echo "  Schrödinger framability scan  (chi=20, chi=40)"
echo "  N_PTS        = ${N_PTS}"
echo "  J            = ${J}"
echo "  GAMMA_STEP   = ${GAMMA_STEP}"
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  Max jobs     = ${MAX_CONCURRENT} concurrent"
echo "  Array tasks  = 0–${ARRAY_END}  (${N_PTS}×${N_PTS} = $((N_PTS * N_PTS)))"
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

# ── submit array job ─────────────────────────────────────────
ARRAY_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    sbatch --parsable \
           --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
           ${DEP_FLAG} \
           schro_fra_array.sh
)
echo "Submitted array job: ${ARRAY_JOB_ID}"
echo "  (tasks 0–${ARRAY_END}, up to ${MAX_CONCURRENT} running at once)"

# ── submit collect job (assembles grids and regenerates figure) ──
COLLECT_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    sbatch --parsable \
           --dependency="afterok:${ARRAY_JOB_ID}" \
           schro_fra_collect.sh
)
echo "Submitted collect job: ${COLLECT_JOB_ID}"
echo "  (depends on array job ${ARRAY_JOB_ID} via afterok)"

echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "After completion, figure is at: ${OUT_DIR}/two_qubit_scan.png"
