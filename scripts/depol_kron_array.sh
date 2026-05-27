#!/bin/bash
# ============================================================
#  SLURM job-array: optimise framability of depol_2q(p) ∘ g
#  for g = exp(i*(alpha*XX+beta*YY+gamma*ZZ)),
#  10 random angle triples, p in {0.00..0.09}, d_ext in {4,6,8}.
#
#  task_id = sample_idx * N_D * N_P + d_idx * N_P + p_idx
#    sample_idx  0..9   (N_SAMPLES=10)
#    d_idx       0..2   (D_EXT_SINGLES=[4,6,8])
#    p_idx       0..9   (P_VALUES=0.00..0.09)
#
#  Total tasks: 300  (task_ids 0..299)
#
#  Submit:
#      mkdir -p logs results_depol_kron
#      sbatch --array=0-299%200 scripts/depol_kron_array.sh
# ============================================================

#SBATCH --job-name=depol_kron
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/depol_kron_%A_%a.out
#SBATCH --error=logs/depol_kron_%A_%a.err

OUT_DIR=${OUT_DIR:-results_depol_kron}
N_RESTARTS=${N_RESTARTS:-40}
MAX_ITER=${MAX_ITER:-500}
SEED=${SEED:-0}
METHOD=${METHOD:-SLSQP}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (d=${OUT_DIR}, method=${METHOD}, n_restarts=${N_RESTARTS})"

python scripts/depol_kron_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --max_iter   "$MAX_ITER" \
    --seed       "$SEED" \
    --method     "$METHOD"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
