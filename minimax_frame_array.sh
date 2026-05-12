#!/bin/bash
# ============================================================
#  SLURM job-array script: minimax-frame optimisation across
#  gates {H, T, CNOT} under 2-qubit depolarisation.
#
#  For each (d_ext_single, p) find S minimising
#      max_{g in {H, T, CNOT}} framability(kron(S,S), N_p^{x2} . g)
#
#  task_id = d_idx * N_P + p_idx
#    d_idx 0..N_D-1   D_EXT_SINGLES = [4, 6, 8]   (N_D = 3)
#    p_idx 0..N_P-1   P_VALUES      = 0..0.10 step 0.01  (N_P = 11)
#
#  Total tasks: 0..32  (3 * 11 = 33)
#
#  Submit:
#      mkdir -p logs results_minimax_frame
#      sbatch --array=0-32 minimax_frame_array.sh
# ============================================================

#SBATCH --job-name=minimax_frame
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/minimax_frame_%A_%a.out
#SBATCH --error=logs/minimax_frame_%A_%a.err

OUT_DIR=${OUT_DIR:-results_minimax_frame}
N_RESTARTS=${N_RESTARTS:-20}
MAXFEV=${MAXFEV:-2000}
MAX_ITER=${MAX_ITER:-500}
SEED=${SEED:-0}
METHOD=${METHOD:-cobyqa}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (OUT_DIR=${OUT_DIR}, METHOD=${METHOD})"

python minimax_frame_worker.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --out_dir "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev "$MAXFEV" \
    --max_iter "$MAX_ITER" \
    --seed "$SEED" \
    --method "$METHOD"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
