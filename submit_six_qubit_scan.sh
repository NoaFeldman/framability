#!/bin/bash
# ============================================================
#  Master submission script for the 6-qubit (2x3) Lindbladian scan.
#
#  Usage:
#    bash submit_six_qubit_scan.sh [--n_pts N] [--n_pts_g NG] [--n_pts_gp NGP]
#                                  [--J J] [--gamma_step S]
#                                  [--out_dir DIR] [--max_concurrent C]
#                                  [--max_steps M] [--fidelity_threshold F]
#
#  What it does:
#    1. Creates required directories (logs/, OUT_DIR).
#    2. Submits a SLURM job array (one task per (gamma, gamma') point;
#       total tasks = n_pts * n_pts).
#    3. Submits the collect job afterok of the array job.
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
N_PTS=41
J=1.0
GAMMA_STEP=0.2
OUT_DIR=results_six
MAX_CONCURRENT=50
MAX_STEPS=100000
FIDELITY_THRESHOLD=0.9

# ── parse arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_pts)              N_PTS="$2";              shift 2 ;;
        --J)                  J="$2";                  shift 2 ;;
        --gamma_step)         GAMMA_STEP="$2";         shift 2 ;;
        --out_dir)            OUT_DIR="$2";            shift 2 ;;
        --max_concurrent)     MAX_CONCURRENT="$2";     shift 2 ;;
        --max_steps)          MAX_STEPS="$2";          shift 2 ;;
        --fidelity_threshold) FIDELITY_THRESHOLD="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ARRAY_END=$(( N_PTS * N_PTS - 1 ))

echo "========================================================"
echo "  6-qubit (2x3) Lindbladian scan"
echo "  N_PTS              = ${N_PTS}  (grid ${N_PTS}x${N_PTS} = $((N_PTS*N_PTS)) points)"
echo "  J                  = ${J}"
echo "  GAMMA_STEP         = ${GAMMA_STEP}"
echo "  OUT_DIR            = ${OUT_DIR}"
echo "  MAX_CONCURRENT     = ${MAX_CONCURRENT}"
echo "  MAX_STEPS          = ${MAX_STEPS}"
echo "  FIDELITY_THRESHOLD = ${FIDELITY_THRESHOLD}"
echo "========================================================"

mkdir -p "${OUT_DIR}" logs

# ── submit array job ─────────────────────────────────────────
ARRAY_JOB_ID=$(
    N_PTS_G="$N_PTS_G" N_PTS_GP="$N_PTS_GP" \
    J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    MAX_STEPS="$MAX_STEPS" FIDELITY_THRESHOLD="$FIDELITY_THRESHOLD" \
    sbatch --parsable \
           --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
           six_qubit_scan_array.sh
)
echo "Submitted six-qubit array job:  ${ARRAY_JOB_ID}"
echo "  (tasks 0-${ARRAY_END}, up to ${MAX_CONCURRENT} concurrent)"

# ── submit collect job (afterok dependency on the array) ─────
COLLECT_JOB_ID=$(
    N_PTS_G="$N_PTS_G" N_PTS_GP="$N_PTS_GP" \
    J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    sbatch --parsable \
           --dependency="afterok:${ARRAY_JOB_ID}" \
           six_qubit_scan_collect.sh
)
echo "Submitted six-qubit collect job: ${COLLECT_JOB_ID}"
echo "  (runs after array job ${ARRAY_JOB_ID} completes successfully)"

echo
echo "Monitor with:"
echo "  squeue -u \$USER"
echo
echo "Output:"
echo "  ${OUT_DIR}/six_qubit_scan.npy"
echo "  ${OUT_DIR}/six_qubit_scan_bond_vs_fra.png"
