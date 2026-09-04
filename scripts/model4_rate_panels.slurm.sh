#!/bin/bash
# ============================================================
#  SLURM job-array: model4 framability-RATE panels (1-6 of the model4 figure).
#
#  For every (gamma, gamma') point of model4's grid (MODELS['model4'].p1_vals x
#  p2_vals = 51x51 = 2601 points at STRIDE=1) this computes the six dt-free
#  framability rates of the two-qubit bond generator:
#     stabilizer-3, Pauli, optimised Heisenberg d_ext=4/6,
#     optimised Schrodinger d_ext=4/6.
#
#  The 2601 points are split across a 0-199 array (200 tasks, the job cap),
#  ~14 points per task, strided so each task samples the whole grid (a task
#  that dies leaves a uniformly thinned grid, not a missing block).
#  Per-point npz files are skipped if they already exist, so resubmitting the
#  same array simply fills the holes.
#
#  Submit:
#    mkdir -p logs results_model4_rate
#    sbatch scripts/model4_rate_panels.slurm.sh
#    STRIDE=5 sbatch scripts/model4_rate_panels.slurm.sh   # quick 11x11 preview
#
#  Output: results_model4_rate/model4/pt_<ix>_<iy>.npz
#
#  RUNTIME: dominated by stabilizer_3_rate (1080 per-column LPs on a 64-row
#  frame) and by the two Nelder-Mead state-frame optimisations.  Start with one
#  array, check a log, and raise --time or lower the restart/maxfev knobs if
#  tasks are timing out.
# ============================================================

#SBATCH --job-name=m4_rate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m4rate_%x_%A_%a.out
#SBATCH --error=logs/m4rate_%x_%A_%a.err

OUT_DIR=${OUT_DIR:-results_model4_rate}
N_CHUNKS=${N_CHUNKS:-200}         # must match the --array size above
STRIDE=${STRIDE:-1}               # 1 = full 51x51 grid
HEIS_RESTARTS=${HEIS_RESTARTS:-8}
HEIS_MAXFEV=${HEIS_MAXFEV:-3000}
POLISH=${POLISH:-300}
SCHRO_RESTARTS=${SCHRO_RESTARTS:-5}
SCHRO_MAXFEV=${SCHRO_MAXFEV:-800}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model4 rates] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model4_rate_panels_worker.py \
    --task_id        "$SLURM_ARRAY_TASK_ID" \
    --n_chunks       "$N_CHUNKS" \
    --out_dir        "$OUT_DIR" \
    --stride         "$STRIDE" \
    --heis_restarts  "$HEIS_RESTARTS" \
    --heis_maxfev    "$HEIS_MAXFEV" \
    --polish         "$POLISH" \
    --schro_restarts "$SCHRO_RESTARTS" \
    --schro_maxfev   "$SCHRO_MAXFEV" \
    --seed           "$SEED"

echo "[model4 rates] chunk ${SLURM_ARRAY_TASK_ID}: done"
