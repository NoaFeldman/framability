#!/bin/bash
# ============================================================
#  Master submission: product-state framability of depolarised
#  gates (H, T, CNOT) vs depolarising probability p.
#
#  1. Submits a 15-task array (one per gate × p-value).
#  2. Submits a collect/plot job that runs after all tasks finish.
#
#  Usage:
#    bash submit_depol_fra.sh [--out_dir DIR] [--max_concurrent C]
#                             [--after_job JOB_ID]
#
#  Options:
#    --out_dir        Output directory        (default: results_depol)
#    --max_concurrent Max simultaneous tasks  (default: 15)
#    --after_job      Start after this job ID (optional)
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
OUT_DIR=results_depol
MAX_CONCURRENT=15
AFTER_JOB=""

# ── parse arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --after_job)      AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

N_GATES=3
N_P=5
TOTAL=$(( N_GATES * N_P - 1 ))   # 0..14

mkdir -p "$OUT_DIR" logs

echo "========================================================"
echo "  Depolarised gate framability scan"
echo "  Gates        : depol∘H, depol∘T, depol⊗²∘CNOT"
echo "  p values     : 0.05 0.07 0.09 0.11 0.13"
echo "  chi          : 30"
echo "  OUT_DIR      : ${OUT_DIR}"
echo "  Array tasks  : 0–${TOTAL}  (${N_GATES} gates × ${N_P} p-values)"
echo "  Max concurrent: ${MAX_CONCURRENT}"
[[ -n "$AFTER_JOB" ]] && echo "  After job    : ${AFTER_JOB}"
echo "========================================================"

# ── dependency flag for array job ────────────────────────────
ARRAY_DEP=""
[[ -n "$AFTER_JOB" ]] && ARRAY_DEP="--dependency=afterok:${AFTER_JOB}"

# ── submit array ─────────────────────────────────────────────
ARRAY_JID=$(sbatch \
    --array="0-${TOTAL}%${MAX_CONCURRENT}" \
    --export=OUT_DIR="${OUT_DIR}" \
    ${ARRAY_DEP} \
    --parsable \
    depol_fra_array.sh)
echo "Array job submitted: ${ARRAY_JID}"

# ── submit collect/plot job ───────────────────────────────────
COLLECT_JID=$(sbatch \
    --dependency=afterok:"${ARRAY_JID}" \
    --job-name=depol_fra_collect \
    --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:10:00 \
    --output=logs/depol_fra_collect_%j.out \
    --error=logs/depol_fra_collect_%j.err \
    --parsable \
    --wrap="source ${PWD}/.venv/bin/activate && \
            export MPLCONFIGDIR=/tmp/matplotlib-\${SLURM_JOB_ID} && \
            python depol_fra_collect.py --out_dir ${OUT_DIR}")
echo "Collect job submitted: ${COLLECT_JID}"

echo ""
echo "Figure will be saved to ${OUT_DIR}/depol_fra.png once complete."
