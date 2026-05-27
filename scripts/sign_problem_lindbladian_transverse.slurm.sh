#!/bin/bash
# ============================================================
#  SLURM job-array: optimised sign problem of the two-qubit
#  Lindbladian Trotter step exp(L*dt) with transverse field:
#  H = J*ZZ + h*(XI + IX).
#
#  task_id = ig * N_GP + igp
#    ig  in 0..40   gamma   = 0.2*ig  (up to 8.0)
#    igp in 0..20   gamma_p = 0.2*igp (up to 4.0)
#  Total: 861 tasks  (task_ids 0..860)
#
#  Submit:
#    mkdir -p logs results_sign_problem_transverse
#    sbatch --array=0-860%100 sign_problem_lindbladian_transverse.slurm.sh
# ============================================================

#SBATCH --job-name=sign_transverse
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sign_transverse_%A_%a.out
#SBATCH --error=logs/sign_transverse_%A_%a.err

OUT_DIR=${OUT_DIR:-results_sign_problem_transverse}
TAG=${TAG:-transverse}
N_RESTARTS=${N_RESTARTS:-30}
METHOD=${METHOD:-BFGS}
J=${J:-1.0}
H_FIELD=${H_FIELD:-1.0}
DT=${DT:-0.01}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (TAG=${TAG}, h=${H_FIELD})"

python scripts/sign_problem_lindbladian_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --out_dir    "$OUT_DIR" \
    --tag        "$TAG" \
    --J          "$J" \
    --h          "$H_FIELD" \
    --dt         "$DT" \
    --n_restarts "$N_RESTARTS" \
    --method     "$METHOD" \
    --seed       "$SEED"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
