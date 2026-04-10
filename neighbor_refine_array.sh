#!/bin/bash
# ============================================================
#  SLURM job-array script: one task per grid point.
#  Neighbor-seeded refinement of min framability.
#
#  Each task checks whether any 4-connected neighbor has lower
#  min_fra; if so, re-optimises this point seeded with the
#  neighbor's D matrix.
#
#  Direct submission example:
#    export N_PTS=20 J=1.0 GAMMA_STEP=0.1 OUT_DIR=results
#    sbatch --array=0-399 neighbor_refine_array.sh
# ============================================================

#SBATCH --job-name=nb_refine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/nb_refine_%A_%a.out
#SBATCH --error=logs/nb_refine_%A_%a.err

# ── scan parameters (with defaults) ──────────────────────────
N_PTS=${N_PTS:-20}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.1}
OUT_DIR=${OUT_DIR:-results}
N_RESTARTS=${N_RESTARTS:-5}
MAXFEV=${MAXFEV:-1000}

# ── activate Python environment ───────────────────────────────
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# ── run ──────────────────────────────────────────────────────
echo "Task ${SLURM_ARRAY_TASK_ID}: starting neighbor-refine"
echo "  N_PTS=${N_PTS}  J=${J}  step=${GAMMA_STEP}  out=${OUT_DIR}"
echo "  n_restarts=${N_RESTARTS}  maxfev=${MAXFEV}"

python neighbor_refine_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
