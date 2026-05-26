#!/bin/bash
# ============================================================
#  SLURM job-array script: minimax-frame optimisation across
#  gates {H, T, CNOT} under 2-qubit depolarisation.
#
#  Uses SO(3)-rotated random starts for better global coverage.
#
#  task_id = d_idx * N_P + p_idx
#    d_idx 0..2   D_EXT_SINGLES = [4, 6, 8]
#    p_idx 0..10  P_VALUES = 0.00..0.10 step 0.01
#
#  Total tasks: 33  (task_ids 0..32)
#
#  Submit:
#      mkdir -p logs results_minimax_H_CNOT_T
#      sbatch --array=0-32 minimax_frame_array_v2.sh
# ============================================================

#SBATCH --job-name=minimax_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=logs/minimax_v2_%A_%a.out
#SBATCH --error=logs/minimax_v2_%A_%a.err

OUT_DIR=${OUT_DIR:-results_minimax_H_CNOT_T}
N_RESTARTS=${N_RESTARTS:-80}
MAX_ITER=${MAX_ITER:-500}
SEED=${SEED:-0}
METHOD=${METHOD:-SLSQP}
GATE_SET=${GATE_SET:-H_CNOT_T}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (OUT_DIR=${OUT_DIR}, METHOD=${METHOD}, n_restarts=${N_RESTARTS})"

python minimax_frame_worker.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --out_dir "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --max_iter "$MAX_ITER" \
    --seed "$SEED" \
    --method "$METHOD" \
    --gate_set "$GATE_SET"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
