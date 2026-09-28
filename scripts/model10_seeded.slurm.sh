#!/bin/bash
# ============================================================
#  SLURM job-array: seeded Heisenberg framability rates of model10 at
#  d_ext = 4 and 8 (scripts/model10_seeded_worker.py) over the full
#  (Delta1, Delta2) grid, 51 x 51 = 2601 points at STRIDE=1, ~13 points per
#  task at the 200-task cap (strided, so a dead task thins the grid evenly).
#
#  Seeds: closed-form frames B / C / P of the rate-zero analysis, structured
#  affine-polygon families (multistart Nelder-Mead in 5-7 parameters), every
#  frame stored by the earlier model10 pipelines (read-only), then the seeded
#  global search of framability_rate_global.  Points where a seed already sits
#  at mu* = 0 skip the global search.
#
#  Submitted by scripts/submit_model10_seeded.sh together with the dependent
#  collect job; by hand:
#    mkdir -p logs results_model10_rate
#    sbatch scripts/model10_seeded.slurm.sh
#    STRIDE=5 sbatch scripts/model10_seeded.slurm.sh      # 11 x 11 preview
#
#  Output: $OUT_DIR/model10_seeded[_s<stride>]/pt_<ix>_<iy>.npz
#  Finished points are skipped, so resubmitting fills holes (FORCE=1 redoes).
#  Runtime: seconds per point on the mu* = 0 plateau, ~10-15 min per point
#  where d_ext = 8 needs the global search.
# ============================================================

#SBATCH --job-name=m10_seed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m10seed_%x_%A_%a.out
#SBATCH --error=logs/m10seed_%x_%A_%a.err

OUT_DIR=${OUT_DIR:-results_model10_rate}
STORED_DIRS=${STORED_DIRS:-"results_model10_rate results_model4_rate"}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size above
STRIDE=${STRIDE:-1}
D_EXTS=${D_EXTS:-"4 8"}
FAM_RANDOM=${FAM_RANDOM:-12}
FAM_POLISH=${FAM_POLISH:-3}
FAM_MAXFEV=${FAM_MAXFEV:-200}
DE_POPSIZE=${DE_POPSIZE:-6}
DE_MAXITER=${DE_MAXITER:-20}
BUNDLE_ITERS=${BUNDLE_ITERS:-80}
NM_MAXFEV=${NM_MAXFEV:-400}
SEED=${SEED:-0}
SELF_CHECK=${SELF_CHECK:-1}        # chunk 0 logs the closed-form self-check
FORCE=${FORCE:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model10 seeded] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model10_seeded_worker.py \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --out_dir      "$OUT_DIR" \
    --stored_dirs  $STORED_DIRS \
    --stride       "$STRIDE" \
    --d_exts       $D_EXTS \
    --fam_random   "$FAM_RANDOM" \
    --fam_polish   "$FAM_POLISH" \
    --fam_maxfev   "$FAM_MAXFEV" \
    --de_popsize   "$DE_POPSIZE" \
    --de_maxiter   "$DE_MAXITER" \
    --bundle_iters "$BUNDLE_ITERS" \
    --nm_maxfev    "$NM_MAXFEV" \
    --seed         "$SEED" \
    $([ "$SELF_CHECK" = "1" ] && echo --self_check) \
    ${FORCE:+--force}

echo "[model10 seeded] chunk ${SLURM_ARRAY_TASK_ID}: done"
