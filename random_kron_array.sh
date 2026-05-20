#!/bin/bash
# ============================================================
#  SLURM job-array: minimize framability of random two-qubit
#  gates  U = exp(i*(alpha*XX + beta*YY + gamma*ZZ))
#  for d_ext_single in {4, 6}, over 10 random (alpha,beta,gamma)
#  triples drawn from [0, pi/2)^3 with master seed 42.
#
#  task_id = d_idx * 10 + sample_idx
#    d_idx      in 0..1  (d_ext_single in [4, 6])
#    sample_idx in 0..9
#
#  Total: 20 tasks (0..19)
#
#  Submit:
#    mkdir -p logs results_random_kron
#    sbatch --array=0-19 random_kron_array.sh
# ============================================================

#SBATCH --job-name=random_kron
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/random_kron_%A_%a.out
#SBATCH --error=logs/random_kron_%A_%a.err

OUT_DIR=${OUT_DIR:-results_random_kron}
N_RESTARTS=${N_RESTARTS:-10}
MAXFEV=${MAXFEV:-2000}
MAX_ITER=${MAX_ITER:-500}
METHOD=${METHOD:-dual_annealing}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (OUT_DIR=${OUT_DIR})"

python random_kron_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV" \
    --max_iter   "$MAX_ITER" \
    --method     "$METHOD" \
    --seed       "$SEED"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
