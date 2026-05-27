#!/bin/bash
# ============================================================
#  SLURM job array: neighbor-seeded refinement of optimised
#  framabilities in results_opt/.
#  One task per grid point: task_id = ig * N_PTS + igp.
#
#  Submit via submit_opt_refine.sh, or directly:
#    export N_PTS=41 J=1.0 GAMMA_STEP=0.2 OUT_DIR=results_opt
#    sbatch --array=0-1680%200 scripts/opt_refine_array.sh
# ============================================================

#SBATCH --job-name=opt_refine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/opt_refine_%A_%a.out
#SBATCH --error=logs/opt_refine_%A_%a.err

N_PTS=${N_PTS:-41}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_opt}
N_RESTARTS=${N_RESTARTS:-5}
MAXFEV=${MAXFEV:-1000}

cd "${SLURM_SUBMIT_DIR}"
source ".venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: cwd=$(pwd)  OUT_DIR=${OUT_DIR}  N_PTS=${N_PTS}  J=${J}  step=${GAMMA_STEP}"

python scripts/opt_refine_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV"

echo "Task ${SLURM_ARRAY_TASK_ID}: done  exit=$?"
