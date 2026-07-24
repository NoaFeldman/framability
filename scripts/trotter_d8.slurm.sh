#!/bin/bash
# ============================================================
#  SLURM job-array: backfill the d_ext_single=8 Heisenberg optimised framability
#  (opt_fra_8) into an existing Trotter scan, in place (see
#  scripts/trotter_d8_worker.py).  Every other stored quantity is untouched.
#
#  opt_fra_8 / opt_S_8 are computed with the support-enforcing optimiser
#  (alternating certificate + Polyak polish, per-Pauli support required, ixyz
#  seeded), so the frame spans every Pauli.  The gate is rebuilt from each point's
#  own stored (p1, p2, dim, dt) -- no scan parameter is re-derived.
#
#  One MODEL is scanned over its full grid, split across a 200-task array
#  (N_CHUNKS=200); points already carrying a valid opt_fra_8 at the current stamp
#  are skipped, so the array is safely resubmittable.
#
#  Prerequisite: the model's scan pt files must already exist in $IN_DIR.
#
#  Submit one model:
#    mkdir -p logs
#    MODEL=model7a sbatch scripts/trotter_d8.slurm.sh
#
#  Budgets overridable via env vars, e.g.:
#    MODEL=model7e MAXFEV=10000 N_RESTARTS=20 sbatch scripts/trotter_d8.slurm.sh
# ============================================================

#SBATCH --job-name=trot_d8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/d8_%x_%A_%a.out
#SBATCH --error=logs/d8_%x_%A_%a.err

MODEL=${MODEL:-model7a}
IN_DIR=${IN_DIR:-results_trotter_v3}
N_CHUNKS=${N_CHUNKS:-200}
N_RESTARTS=${N_RESTARTS:-12}
MAXFEV=${MAXFEV:-6000}
POLISH_ITERS=${POLISH_ITERS:-300}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] d8 chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/trotter_d8_worker.py \
    --model        "$MODEL" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --in_dir       "$IN_DIR" \
    --n_restarts   "$N_RESTARTS" \
    --maxfev       "$MAXFEV" \
    --polish_iters "$POLISH_ITERS" \
    --seed         "$SEED"

echo "[$MODEL] d8 chunk ${SLURM_ARRAY_TASK_ID}: done"
