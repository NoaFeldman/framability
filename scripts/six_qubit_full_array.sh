#!/bin/bash
# ============================================================
#  SLURM job-array script: comprehensive 6-qubit (2x3) scan.
#  One task per (gamma, gamma') grid POINT.
#
#  task_id = ig * N_PTS_GP + igp
#
#  Default 51x21 grid -> 1071 tasks (array 0-1070).
# ============================================================

#SBATCH --job-name=six_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/six_full_%A_%a.out
#SBATCH --error=logs/six_full_%A_%a.err

N_PTS_G=${N_PTS_G:-51}
N_PTS_GP=${N_PTS_GP:-21}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_six}
MAX_STEPS=${MAX_STEPS:-100000}
FIDELITY_THRESHOLD=${FIDELITY_THRESHOLD:-0.9}
DT_STABILIZER=${DT_STABILIZER:-0.1}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: six_full (n_g=${N_PTS_G}, n_gp=${N_PTS_GP}, J=${J}, step=${GAMMA_STEP})"

python scripts/six_qubit_full_worker.py \
    --task_id            "$SLURM_ARRAY_TASK_ID" \
    --n_pts_g            "$N_PTS_G" \
    --n_pts_gp           "$N_PTS_GP" \
    --J                  "$J" \
    --gamma_step         "$GAMMA_STEP" \
    --out_dir            "$OUT_DIR" \
    --max_steps          "$MAX_STEPS" \
    --fidelity_threshold "$FIDELITY_THRESHOLD" \
    --dt_stabilizer      "$DT_STABILIZER"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
