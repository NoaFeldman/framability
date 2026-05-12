#!/bin/bash
# ============================================================
#  Master submission: depolarised-gate sweep
#    gates  = CNOT, H, T            (gate_idx 0..2)
#    p      = 0.00, 0.01, ..., 0.07 (p_idx 0..7)
#  Each task computes framability for 5 frame choices plus OTOC,
#  channel stabilizer purity and operator bond entropy.
#
#  1. Submits a 24-task array (3 gates × 8 p values).
#  2. Submits a collect/plot job that runs after all tasks finish.
#
#  Usage:
#    bash submit_sweep_depol_gates.sh
#    bash submit_sweep_depol_gates.sh --out_dir results_depol_sweep
#    bash submit_sweep_depol_gates.sh --out_dir results_depol_sweep --max_concurrent 12
#    bash submit_sweep_depol_gates.sh --out_dir results_depol_sweep --n_restarts 5 --after_job 123456
#
#  Options (all optional):
#    --out_dir        Output directory        (default: results_depol_sweep)
#    --n_restarts     Optimizer restarts      (default: 5)
#    --max_concurrent Max simultaneous tasks  (default: 24)
#    --after_job      Start after this SLURM job ID completes
# ============================================================

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────
OUT_DIR=results_depol_sweep
N_RESTARTS=5
MAX_CONCURRENT=24
AFTER_JOB=""

# ── parse arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out_dir)        OUT_DIR="$2";        shift 2 ;;
        --n_restarts)     N_RESTARTS="$2";     shift 2 ;;
        --max_concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --after_job)      AFTER_JOB="$2";      shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

N_GATES=3
N_P=8
TOTAL=$(( N_GATES * N_P - 1 ))   # 0..23

mkdir -p "$OUT_DIR" logs

echo "========================================================"
echo "  Depolarised-gate sweep (framability + OTOC + stab + obe)"
echo "  Gates        : CNOT, H, T"
echo "  p values     : 0.00 0.01 0.02 0.03 0.04 0.05 0.06 0.07"
echo "  Frames       : Pauli, Extended-Pauli, Opt(d_ext_single=4,6,8)"
echo "  N_RESTARTS   : ${N_RESTARTS}"
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
    --export=OUT_DIR="${OUT_DIR}",N_RESTARTS="${N_RESTARTS}" \
    ${ARRAY_DEP} \
    --parsable \
    sweep_depol_gates_array.sh)
echo "Array job submitted: ${ARRAY_JID}"

# ── submit collect/plot job ───────────────────────────────────
COLLECT_JID=$(sbatch \
    --dependency=afterok:"${ARRAY_JID}" \
    --job-name=depol_sweep_collect \
    --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:10:00 \
    --output=logs/depol_sweep_collect_%j.out \
    --error=logs/depol_sweep_collect_%j.err \
    --parsable \
    --wrap="source ${PWD}/.venv/bin/activate && \
            export MPLCONFIGDIR=/tmp/matplotlib-\${SLURM_JOB_ID} && \
            python sweep_depol_gates_collect.py --in_dir ${OUT_DIR} --out_dir ${OUT_DIR}")
echo "Collect job submitted: ${COLLECT_JID}"

echo ""
echo "Figure will be saved to ${OUT_DIR}/depol_sweep.png once complete."
