#!/bin/bash
# ============================================================
#  SLURM job-array: 6-qubit l1-coherence computation.
#
#  Default grid matches results_six six_full_scan: 51 x 21 = 1071 tasks.
#
#  Submit:
#    sbatch --array=0-1070 scripts/six_qubit_coherence_array.sh
# ============================================================

#SBATCH --job-name=six_coh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/six_coh_%A_%a.out
#SBATCH --error=logs/six_coh_%A_%a.err

set -euo pipefail

N_PTS_G=${N_PTS_G:-51}
N_PTS_GP=${N_PTS_GP:-21}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_six_coh}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: N_PTS_G=${N_PTS_G}  N_PTS_GP=${N_PTS_GP}  J=${J}  step=${GAMMA_STEP}"

python scripts/six_qubit_coherence_worker.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --n_pts_g "$N_PTS_G" \
    --n_pts_gp "$N_PTS_GP" \
    --J "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir "$OUT_DIR"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
