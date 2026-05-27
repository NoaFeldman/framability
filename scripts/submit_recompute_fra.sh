#!/bin/bash
# ============================================================
#  Submit the full recompute-framability pipeline:
#    1. Array job: one task per grid point  (N_PTS^2 tasks total)
#    2. Collect job: assembles scan_opt.npy and regenerates figure
#
#  Usage:
#    bash submit_recompute_fra.sh [--n_pts N] [--J J]
#                                 [--gamma_step S] [--out_dir DIR]
#                                 [--max_concurrent C]
#                                 [--n_restarts K] [--maxfev F]
# ============================================================

set -euo pipefail

N_PTS=41
J=1.0
GAMMA_STEP=0.2
OUT_DIR=results_opt
MAX_CONCURRENT=200
N_RESTARTS=5
MAXFEV=1000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_pts)          N_PTS="$2";          shift 2 ;;
        --J)              J="$2";              shift 2 ;;
        --gamma_step)     GAMMA_STEP="$2";     shift 2 ;;
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --n_restarts)     N_RESTARTS="$2";     shift 2 ;;
        --maxfev)         MAXFEV="$2";         shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ARRAY_END=$(( N_PTS * N_PTS - 1 ))

echo "========================================================"
echo "  Recompute optimised framability"
echo "  N_PTS        = ${N_PTS}  (${N_PTS}x${N_PTS} = $((N_PTS*N_PTS)) tasks)"
echo "  J            = ${J}"
echo "  GAMMA_STEP   = ${GAMMA_STEP}"
echo "  OUT_DIR      = ${OUT_DIR}"
echo "  Max concurrent = ${MAX_CONCURRENT}"
echo "  n_restarts   = ${N_RESTARTS}  maxfev = ${MAXFEV}"
echo "========================================================"

mkdir -p "${OUT_DIR}" logs

# ── submit array job ──────────────────────────────────────────
ARRAY_JOB_ID=$(
    N_PTS="$N_PTS" J="$J" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    N_RESTARTS="$N_RESTARTS" MAXFEV="$MAXFEV" \
    sbatch --parsable \
           --array="0-${ARRAY_END}%${MAX_CONCURRENT}" \
           recompute_fra_array.sh
)
echo "Submitted array job: ${ARRAY_JOB_ID}  (tasks 0–${ARRAY_END})"

# ── submit collect job (after all array tasks succeed) ────────
COLLECT_JOB_ID=$(
    N_PTS="$N_PTS" GAMMA_STEP="$GAMMA_STEP" OUT_DIR="$OUT_DIR" \
    sbatch --parsable \
           --dependency="afterok:${ARRAY_JOB_ID}" \
           --job-name=recompute_fra_collect \
           --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:30:00 \
           --output="logs/recompute_fra_collect_%j.out" \
           --error="logs/recompute_fra_collect_%j.err" \
           --wrap="cd \"\${SLURM_SUBMIT_DIR}\" && \
                   source .venv/bin/activate && \
                   python scripts/recompute_fra_collect.py \
                       --out_dir \"${OUT_DIR}\" \
                       --n_pts ${N_PTS} \
                       --gamma_step ${GAMMA_STEP}"
)
echo "Submitted collect job: ${COLLECT_JOB_ID}  (depends on ${ARRAY_JOB_ID})"

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Results:  ${OUT_DIR}/scan_opt.npy"
echo "Figure:   ${OUT_DIR}/two_qubit_scan_opt_bond_vs_fra.png"
