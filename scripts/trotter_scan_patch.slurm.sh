#!/bin/bash
# ============================================================
#  SLURM job-array: in-place upgrade of Trotter-scan results 2.2 -> 2.3.
#
#  model2 / model4 : lpdo_max recomputed from the |+>^N start state
#                    (everything else, incl. optimised framability, preserved).
#  model1 / model3 : version re-stamp only (no recompute).
#  Missing / non-2.2 files are left for the base scan worker.
#
#  Run BEFORE the base scan so existing points are not recomputed:
#    mkdir -p logs results_trotter
#    MODEL=model2 sbatch scripts/trotter_scan_patch.slurm.sh
# ============================================================

#SBATCH --job-name=trot_patch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trotpatch_%x_%A_%a.out
#SBATCH --error=logs/trotpatch_%x_%A_%a.err

MODEL=${MODEL:-model1}
OUT_DIR=${OUT_DIR:-results_trotter}
N_CHUNKS=${N_CHUNKS:-200}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] patch chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/trotter_scan_patch_worker.py \
    --model    "$MODEL" \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --out_dir  "$OUT_DIR"

echo "[$MODEL] patch chunk ${SLURM_ARRAY_TASK_ID}: done"
