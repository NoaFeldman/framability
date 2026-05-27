#!/bin/bash
# ============================================================
#  SLURM job-array: scan (gamma, gamma') for the two-qubit
#  Lindbladian with transverse field H = J*ZZ + h*(XI + IX).
#
#  Per-task computes: Pauli framability, optimised framability
#  (d_ext=4 and d_ext=6), and max LPDO bond entropy.
#
#  task_id = ig * N_GP + igp
#    ig  in 0..40   gamma   = 0.2*ig  (up to 8.0)
#    igp in 0..20   gamma_p = 0.2*igp (up to 4.0)
#
#  Total tasks: 861  (task_ids 0..860)
#
#  Submit:
#      mkdir -p logs results_transverse_x
#      sbatch --array=0-860%100 transverse_x_array.sh
# ============================================================

#SBATCH --job-name=transverse_x
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/transverse_x_%A_%a.out
#SBATCH --error=logs/transverse_x_%A_%a.err

OUT_DIR=${OUT_DIR:-results_transverse_x}
N_RESTARTS=${N_RESTARTS:-20}
MAX_ITER=${MAX_ITER:-500}
J=${J:-1.0}
H_FIELD=${H_FIELD:-1.0}
DT=${DT:-0.01}
MAX_LPDO_STEPS=${MAX_LPDO_STEPS:-5000}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (gamma/gamma' grid, J=${J}, h=${H_FIELD})"

python transverse_x_worker.py \
    --task_id        "$SLURM_ARRAY_TASK_ID" \
    --out_dir        "$OUT_DIR" \
    --n_restarts     "$N_RESTARTS" \
    --max_iter       "$MAX_ITER" \
    --J              "$J" \
    --h              "$H_FIELD" \
    --dt             "$DT" \
    --max_lpdo_steps "$MAX_LPDO_STEPS" \
    --seed           "$SEED"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
