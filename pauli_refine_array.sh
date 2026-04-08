#!/bin/bash
# ============================================================
#  SLURM job-array script: one task per gamma row.
#  Refines data points where min_fra > pauli_fra by seeding the
#  Kronecker-frame optimisation with the Pauli (cycling-identity)
#  starting point.
#
#  Direct submission example:
#    export N_PTS=20 J=1.0 GAMMA_STEP=0.1 OUT_DIR=results
#    sbatch --array=0-19 pauli_refine_array.sh
#
#  Or via submit_pauli_refine.sh (sets --array bounds automatically
#  and exports all parameters).
# ============================================================

#SBATCH --job-name=pauli_ref
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/pauli_ref_%A_%a.out
#SBATCH --error=logs/pauli_ref_%A_%a.err

# ── scan parameters (with defaults) ──────────────────────────
N_PTS=${N_PTS:-20}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.1}
OUT_DIR=${OUT_DIR:-results}
N_RESTARTS=${N_RESTARTS:-5}
MAXFEV=${MAXFEV:-1000}
MAX_ITER=${MAX_ITER:-200}

# ── activate Python environment ───────────────────────────────
# Option A: virtualenv (default)
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# Option B: conda – comment out A and uncomment:
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate framability

# Option C: environment modules – comment out A and uncomment:
# module load python/3.11

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# ── ensure log directory exists ───────────────────────────────
mkdir -p logs

# ── run ──────────────────────────────────────────────────────
echo "Task ${SLURM_ARRAY_TASK_ID}: starting pauli-refine pass"
echo "  N_PTS=${N_PTS}  J=${J}  step=${GAMMA_STEP}  out=${OUT_DIR}"
echo "  n_restarts=${N_RESTARTS}  maxfev=${MAXFEV}  max_iter=${MAX_ITER}"

python pauli_refine_worker.py \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_pts      "$N_PTS" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV" \
    --max_iter   "$MAX_ITER"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
