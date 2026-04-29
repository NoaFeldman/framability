#!/bin/bash
# ============================================================
#  SLURM job-array script: neighbor-seeded framability refinement
#  restricted to gamma' < N_IGP * GAMMA_STEP.
#
#  Task-id mapping: ig = task_id // N_IGP, igp = task_id % N_IGP
#  Array range: 0 .. N_PTS*N_IGP-1
#
#  Each task checks whether any 4-connected neighbor (on the FULL
#  N_PTS x N_PTS grid) has lower min_fra; if so, re-optimises this
#  point seeded with the neighbor's D matrix.
#
#  Direct submission example:
#    export N_PTS=41 N_IGP=20 J=1.0 GAMMA_STEP=0.2 OUT_DIR=results
#    sbatch --array=0-819 neighbor_refine_restricted_array.sh
# ============================================================

#SBATCH --job-name=nb_refine_restr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/nb_refine_restr_%A_%a.out
#SBATCH --error=logs/nb_refine_restr_%A_%a.err

# ── scan parameters (with defaults) ──────────────────────────
N_PTS=${N_PTS:-41}
N_IGP=${N_IGP:-20}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results}
N_RESTARTS=${N_RESTARTS:-5}
MAXFEV=${MAXFEV:-1000}

# ── activate Python environment ───────────────────────────────
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# ── run ──────────────────────────────────────────────────────
echo "Task ${SLURM_ARRAY_TASK_ID}: starting restricted neighbor-refine"
echo "  N_PTS=${N_PTS}  N_IGP=${N_IGP}  J=${J}  step=${GAMMA_STEP}  out=${OUT_DIR}"
echo "  n_restarts=${N_RESTARTS}  maxfev=${MAXFEV}"

python neighbor_refine_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --n_igp      "$N_IGP" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
