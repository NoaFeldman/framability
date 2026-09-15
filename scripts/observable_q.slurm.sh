#!/bin/bash
# ============================================================
#  SLURM job-array: observable quality factor Q_obs of the bond generator
#  (Pauli basis + optimised local basis) over a model's grid.
#
#  Q_obs(P) = sum_{P' != P} |A_{P'P}| / (-A_{PP}),  A = L_bond^T (Pauli basis);
#  see liouvillian_quality.observable_quality.  Bond-only (16x16): ~1 s per
#  point with the basis optimisation, microseconds without (NO_OPT=1).
#
#  The 51x51 = 2601 points are split across a 0-199 array (200 tasks, the job
#  cap), ~13 points per task, strided so every task samples the whole grid.
#  Existing per-point files are skipped, so resubmitting fills holes.
#
#  Submit (one array per model):
#    mkdir -p logs results_observable_q
#    MODEL=model3 sbatch scripts/observable_q.slurm.sh
#    MODEL=model4 sbatch scripts/observable_q.slurm.sh
#    MODEL=model8 sbatch scripts/observable_q.slurm.sh
#
#  Output: results_observable_q/<model>/pt_<ix>_<iy>.npz
# ============================================================

#SBATCH --job-name=obs_q
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/obsq_%x_%A_%a.out
#SBATCH --error=logs/obsq_%x_%A_%a.err

MODEL=${MODEL:-model4}
OUT_DIR=${OUT_DIR:-results_observable_q}
N_CHUNKS=${N_CHUNKS:-200}      # must match the --array size above
STRIDE=${STRIDE:-1}            # 1 = full grid, matching the rate panels
N_RESTARTS=${N_RESTARTS:-8}
MAXFEV=${MAXFEV:-2000}
SEED=${SEED:-0}
NO_OPT=${NO_OPT:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

EXTRA=()
[ "$NO_OPT" = "1" ] && EXTRA+=(--no_opt)

echo "[$MODEL Q_obs  no_opt=$NO_OPT] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/observable_q_worker.py \
    --model      "$MODEL" \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_chunks   "$N_CHUNKS" \
    --out_dir    "$OUT_DIR" \
    --stride     "$STRIDE" \
    --n_restarts "$N_RESTARTS" \
    --maxfev     "$MAXFEV" \
    --seed       "$SEED" \
    "${EXTRA[@]}"

echo "[$MODEL Q_obs] chunk ${SLURM_ARRAY_TASK_ID}: done"
