#!/bin/bash
# ============================================================
#  SLURM job-array: ONE margin round (scripts/model10_margin_worker.py) of the
#  seeded model10 rates at d_ext = D_EXT (8 or 12), 2601 points / 200 tasks.
#
#  Rounds are sequential (round r reads rounds < r); the chain is driven by
#  scripts/submit_model10_margin.sh.  By hand:
#    D_EXT=8 ROUND=1 sbatch scripts/model10_margin.slurm.sh
#
#  Output: $OUT_DIR/model10_seeded[_d12]/pt_<ix>_<iy>_margin_r<NN>.npz
# ============================================================

#SBATCH --job-name=m10_margin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m10margin_%x_%A_%a.out
#SBATCH --error=logs/m10margin_%x_%A_%a.err

D_EXT=${D_EXT:?set D_EXT (8 | 12)}
ROUND=${ROUND:?set ROUND (1..99)}
OUT_DIR=${OUT_DIR:-results_model10_rate}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size above
STRIDE=${STRIDE:-1}
RADIUS=${RADIUS:-2}
RATE_TOL=${RATE_TOL:-1e-6}
MARGIN_ITERS=${MARGIN_ITERS:-40}
PUSH_ITERS=${PUSH_ITERS:-40}
NO_PUSH=${NO_PUSH:-}
SELF_CHECK=${SELF_CHECK:-1}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model10 margin] d_ext ${D_EXT} round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model10_margin_worker.py \
    --d_ext        "$D_EXT" \
    --round        "$ROUND" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --out_dir      "$OUT_DIR" \
    --stride       "$STRIDE" \
    --radius       "$RADIUS" \
    --tol          "$RATE_TOL" \
    --margin_iters "$MARGIN_ITERS" \
    --push_iters   "$PUSH_ITERS" \
    ${NO_PUSH:+--no_push} \
    $([ "$SELF_CHECK" = "1" ] && [ "$ROUND" = "1" ] && echo --self_check)

echo "[model10 margin] d_ext ${D_EXT} round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}: done"
