#!/bin/bash
# ============================================================
#  SLURM job array: find optimal S matrices for tex file.
#  One task per (gamma, gamma') point (4 points total, indices 0-3).
#
#  Submit:
#    sbatch --array=0-3 tex_S_array.sh
#
#  After completion, collect with:
#    python tex_S_collect.py --out_dir results
# ============================================================

#SBATCH --job-name=tex_S
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/tex_S_%A_%a.out
#SBATCH --error=logs/tex_S_%A_%a.err

OUT_DIR=${OUT_DIR:-results}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

python tex_S_worker.py \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --out_dir "${OUT_DIR}" \
    --n_restarts 5 \
    --maxfev 1000 \
    --max_seeds 500
