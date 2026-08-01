#!/bin/bash
# ============================================================
#  Single fixed-size SLURM job-array covering the ENTIRE DT_BASE line sweep
#  grid (every model x every (p1,p2) grid point x all 99 base values) in one
#  sbatch call.  Replaces dtbase_line_submit_all.sh's per-point submissions +
#  client-side squeue throttle: this is exactly 200 tasks, submitted once,
#  tracked with plain `squeue` like any other array job.
#
#  Each of the 200 tasks loops sequentially over its shard of the flattened
#  grid (trotter_dtbase_flat_worker.py), so tasks run much longer than the old
#  per-point-per-base jobs -- that's the deliberate trade for a flat 200-job
#  submission.  Already-current npz files are skipped (see
#  trotter_dtbase_line_worker._is_current), so it is safe to resubmit this
#  script unchanged if a task's time limit is hit partway through its shard,
#  or to lower STRIDE later and resubmit to fill in the finer grid.
#
#  Submit:
#    mkdir -p logs results_dtbase_line
#    MODELS="model3 model4" STRIDE=2 sbatch scripts/trotter_dtbase_flat.slurm.sh
#
#  Override the time limit at submit time (directive below is just a default):
#    MODELS="model3" STRIDE=1 sbatch --time=3-00:00:00 scripts/trotter_dtbase_flat.slurm.sh
#
#  Output goes to results_dtbase_line/<tag>/base_<idx>.npz, same as before.
# ============================================================

#SBATCH --job-name=trot_dtbase_flat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/dtbaseflat_%x_%A_%a.out
#SBATCH --error=logs/dtbaseflat_%x_%A_%a.err

MODELS_ARG=${MODELS:-model3 model4}   # unquoted below on purpose: word-split into --models args
STRIDE=${STRIDE:-1}                   # 1 = full grid, 2 = half resolution, ...
OUT_DIR=${OUT_DIR:-results_dtbase_line}
DIM=${DIM:-2}
FRA_RESTARTS=${FRA_RESTARTS:-5}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-1000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-500}
CH1_RESTARTS=${CH1_RESTARTS:-15}
SEED=${SEED:-0}
N_TASKS=200

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[task ${SLURM_ARRAY_TASK_ID}/${N_TASKS}] models=${MODELS_ARG} stride=${STRIDE}"

python scripts/trotter_dtbase_flat_worker.py \
    --models       ${MODELS_ARG} \
    --stride       "$STRIDE" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_tasks      "$N_TASKS" \
    --out_dir      "$OUT_DIR" \
    --dim          "$DIM" \
    --fra_restarts "$FRA_RESTARTS" \
    --fra_maxfev_4 "$FRA_MAXFEV_4" \
    --fra_maxfev_6 "$FRA_MAXFEV_6" \
    --ch1_restarts "$CH1_RESTARTS" \
    --seed         "$SEED"

echo "[task ${SLURM_ARRAY_TASK_ID}] done"
