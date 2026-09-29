#!/bin/bash
# ============================================================
#  SLURM job-array: optimised Heisenberg framability rate of model10 at
#  d_ext = 12 (scripts/model10_d12_worker.py), 2601 points over 200 tasks.
#
#    STAGE=seed               seeds from the d_ext = 8 results + winning
#                             families + global polish (one pass)
#    STAGE=refine ROUND=<r>   one quick neighbour-refine round (sequential)
#
#  Driven by scripts/submit_model10_d12_prod.sh; by hand:
#    STAGE=seed sbatch scripts/model10_d12.slurm.sh
#    STAGE=refine ROUND=1 sbatch --time=08:00:00 scripts/model10_d12.slurm.sh
#
#  Points already at mu* = 0 at d_ext = 8 cost one LP; elsewhere expect
#  ~20-40 min per point (144-column LPs), so ~6 such points per task.
#  Output: $OUT_DIR/model10_seeded_d12/pt_<ix>_<iy>[_qrefine_r<NN>].npz
# ============================================================

#SBATCH --job-name=m10_d12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m10d12_%x_%A_%a.out
#SBATCH --error=logs/m10d12_%x_%A_%a.err

STAGE=${STAGE:?set STAGE (seed | refine)}
ROUND=${ROUND:-0}
OUT_DIR=${OUT_DIR:-results_model10_rate}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size above
STRIDE=${STRIDE:-1}
FAM_RANDOM=${FAM_RANDOM:-4}
FAM_POLISH=${FAM_POLISH:-2}
FAM_MAXFEV=${FAM_MAXFEV:-80}
BUNDLE_ITERS=${BUNDLE_ITERS:-40}
NM_MAXFEV=${NM_MAXFEV:-150}
REFINE_BUNDLE=${REFINE_BUNDLE:-25}
RATE_TOL=${RATE_TOL:-1e-6}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model10 d12] stage ${STAGE} round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model10_d12_worker.py \
    --stage         "$STAGE" \
    --round         "$ROUND" \
    --task_id       "$SLURM_ARRAY_TASK_ID" \
    --n_chunks      "$N_CHUNKS" \
    --out_dir       "$OUT_DIR" \
    --stride        "$STRIDE" \
    --fam_random    "$FAM_RANDOM" \
    --fam_polish    "$FAM_POLISH" \
    --fam_maxfev    "$FAM_MAXFEV" \
    --bundle_iters  "$BUNDLE_ITERS" \
    --nm_maxfev     "$NM_MAXFEV" \
    --refine_bundle "$REFINE_BUNDLE" \
    --rate_tol      "$RATE_TOL" \
    --seed          "$SEED"

echo "[model10 d12] stage ${STAGE} round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}: done"
