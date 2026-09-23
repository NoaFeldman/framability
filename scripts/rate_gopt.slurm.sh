#!/bin/bash
# ============================================================
#  SLURM job-array: global re-optimisation of the optimised Heisenberg
#  framability rates (d_ext = 4, 6, 8) over one model's full grid
#  (51 x 51 = 2601 points for model3 / model4), seeded with every stored
#  frame so no value can get worse.  ~13 points per task at the 200 cap.
#
#  Submitted by scripts/submit_rate_gopt.sh together with the dependent
#  collect job; by hand:
#    mkdir -p logs results_model4_rate
#    MODEL=model4 sbatch --job-name=m4_gopt scripts/rate_gopt.slurm.sh
#
#  Output: results_<model>_rate/<model>/pt_<ix>_<iy>_gopt.npz
#
#  Runtime knobs (env): DE_POPSIZE / DE_MAXITER (LP frames, default 6 / 25),
#  BUNDLE_ITERS (60), NM_MAXFEV (400); the d_ext = 4 stage uses the closed
#  form and is cheap at its defaults.  Expect ~10-15 min per point.
# ============================================================

#SBATCH --job-name=gopt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/gopt_%x_%A_%a.out
#SBATCH --error=logs/gopt_%x_%A_%a.err

MODEL=${MODEL:?set MODEL (model3 / model4 / model10 ...)}
OUT_DIR=${OUT_DIR:-results_${MODEL}_rate}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size above
STRIDE=${STRIDE:-1}
D_EXTS=${D_EXTS:-"4 6 8"}
DE_POPSIZE=${DE_POPSIZE:-6}
DE_MAXITER=${DE_MAXITER:-25}
BUNDLE_ITERS=${BUNDLE_ITERS:-60}
NM_MAXFEV=${NM_MAXFEV:-400}
SEED=${SEED:-0}
FORCE=${FORCE:-}                   # FORCE=1 recomputes points that have a gopt file

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[${MODEL} gopt] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/rate_gopt_worker.py \
    --model        "$MODEL" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --out_dir      "$OUT_DIR" \
    --stride       "$STRIDE" \
    --d_exts       $D_EXTS \
    --de_popsize   "$DE_POPSIZE" \
    --de_maxiter   "$DE_MAXITER" \
    --bundle_iters "$BUNDLE_ITERS" \
    --nm_maxfev    "$NM_MAXFEV" \
    --seed         "$SEED" \
    ${FORCE:+--force}

echo "[${MODEL} gopt] chunk ${SLURM_ARRAY_TASK_ID}: done"
