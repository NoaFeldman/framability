#!/bin/bash
# ============================================================
#  SLURM job-array: 6-qubit star+plaquette sign-problem scan.
#
#  Grid: 20 x 20 over (gamma_s, gamma_p), step 0.2, range [0, 4).
#  N_TOTAL = 400 points split across N_JOBS tasks (default 100, so 4 pts/job).
#
#  Submit (set A: h=1, lambda=1):
#    mkdir -p logs results_sign_six_h1_lam1
#    OUT_DIR=results_sign_six_h1_lam1 H_FIELD=1.0 LAM=1.0 \
#      sbatch --array=0-99 scripts/sign_problem_six_qubit.slurm.sh
#
#  Submit (set B: h=1, lambda=1/sqrt(2)):
#    mkdir -p logs results_sign_six_h1_lamsqrt2
#    LAMVAL=$(python -c "import math; print(1/math.sqrt(2))")
#    OUT_DIR=results_sign_six_h1_lamsqrt2 H_FIELD=1.0 LAM=$LAMVAL \
#      sbatch --array=0-99 scripts/sign_problem_six_qubit.slurm.sh
# ============================================================

#SBATCH --job-name=sign_six
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sign_six_%A_%a.out
#SBATCH --error=logs/sign_six_%A_%a.err

OUT_DIR=${OUT_DIR:-results_sign_six}
H_FIELD=${H_FIELD:-1.0}
LAM=${LAM:-1.0}
DT=${DT:-0.02}
N_RESTARTS=${N_RESTARTS:-10}
N_JOBS=${N_JOBS:-100}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}/${N_JOBS}: starting  (h=${H_FIELD}, lam=${LAM}, dt=${DT}, out_dir=${OUT_DIR})"

python scripts/sign_problem_six_qubit_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_jobs     "$N_JOBS" \
    --out_dir    "$OUT_DIR" \
    --h          "$H_FIELD" \
    --lam        "$LAM" \
    --dt         "$DT" \
    --n_restarts "$N_RESTARTS" \
    --seed       "$SEED"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
