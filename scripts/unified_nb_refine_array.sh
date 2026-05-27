#!/bin/bash
# ============================================================
#  SLURM job-array: unified neighbor-seeded refinement.
#  One task per grid point (ig, igp).
#
#  Required env vars: VARIANT, ROUND, OUT_DIR
#  Optional: N_PTS, N_IGP, J, GAMMA_STEP, N_RESTARTS, MAXFEV
#
#  Submit example:
#    VARIANT=free6 ROUND=1 OUT_DIR=results_free6 \
#      sbatch --array=0-860 scripts/unified_nb_refine_array.sh
# ============================================================

#SBATCH --job-name=unb_refine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/unb_refine_%A_%a.out
#SBATCH --error=logs/unb_refine_%A_%a.err

VARIANT=${VARIANT:?Must set VARIANT (d6, d4, free6)}
ROUND=${ROUND:?Must set ROUND (1 or 2)}
OUT_DIR=${OUT_DIR:?Must set OUT_DIR}

N_PTS=${N_PTS:-41}
N_IGP=${N_IGP:-21}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
N_RESTARTS=${N_RESTARTS:-5}
MAXFEV=${MAXFEV:-1000}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: ${VARIANT} round ${ROUND}"

python scripts/unified_nb_refine_worker.py \
    --variant    "$VARIANT" \
    --round      "$ROUND" \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --n_igp      "$N_IGP" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
