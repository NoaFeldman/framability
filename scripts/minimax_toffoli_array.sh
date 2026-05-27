#!/bin/bash
# ============================================================
#  SLURM job-array: minimax framability for {H, Toffoli} under
#  3-qubit depolarisation N_p ⊗ N_p ⊗ N_p.
#
#  H is lifted to 3 qubits as H ⊗ I ⊗ I.
#  Toffoli (CCX) acts on qubits 0,1,2 (controls 0,1; target 2).
#
#  task_id = d_idx * N_P + p_idx
#    d_idx in 0..1   D_EXT_SINGLES = [4, 6]   (N_D = 2)
#    p_idx in 0..10  P_VALUES = 0.00..0.10     (N_P = 11)
#
#  Total: 2 * 11 = 22 tasks (0..21)
#
#  Note: d_ext_single=8 (d_ext=512 for 3 qubits) is excluded because
#  the resulting LP (~524k variables) is impractical to solve.
#
#  Submit:
#    mkdir -p logs results_minimax_toffoli
#    sbatch --array=0-21 scripts/minimax_toffoli_array.sh
# ============================================================

#SBATCH --job-name=minimax_toffoli
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/minimax_toffoli_%A_%a.out
#SBATCH --error=logs/minimax_toffoli_%A_%a.err

OUT_DIR=${OUT_DIR:-results_minimax_toffoli}
N_RESTARTS=${N_RESTARTS:-20}
MAXFEV=${MAXFEV:-2000}
MAX_ITER=${MAX_ITER:-500}
SEED=${SEED:-0}
METHOD=${METHOD:-SLSQP}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (OUT_DIR=${OUT_DIR}, METHOD=${METHOD})"

python scripts/minimax_toffoli_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV" \
    --max_iter   "$MAX_ITER" \
    --seed       "$SEED" \
    --method     "$METHOD"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
