#!/bin/bash
# ============================================================
#  SLURM job-array: ONE randomised refine round (scripts/model10_rrefine_worker.py)
#  of the seeded model10 rates at d_ext = D_EXT (default 12), 2601 points over
#  200 tasks.  Every positive point (TARGETS=all) or only the suspect ones
#  (TARGETS=nonmono) gets a basin-hopping search: perturbations of the best
#  nearby frame at several scales, each followed by a long margin/rate polish.
#
#  Driven by scripts/submit_model10_rrefine.sh; by hand:
#    ROUND=1 sbatch scripts/model10_rrefine.slurm.sh
#
#  Per point: at most MAX_SECONDS of hops (default 30 min), so a round is at
#  most ~6 points x 30 min per task at d_ext = 12.
#  Output: $OUT_DIR/model10_seeded_d12/pt_<ix>_<iy>_rrefine_r<NN>.npz (improved only)
# ============================================================

#SBATCH --job-name=m10_rrefine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m10rref_%x_%A_%a.out
#SBATCH --error=logs/m10rref_%x_%A_%a.err

D_EXT=${D_EXT:-12}
ROUND=${ROUND:?set ROUND (1..99)}
OUT_DIR=${OUT_DIR:-results_model10_rate}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size above
STRIDE=${STRIDE:-1}
TARGETS=${TARGETS:-all}            # all | nonmono
POLISH_ITERS=${POLISH_ITERS:-100}
HOPS=${HOPS:-6}
SCALES=${SCALES:-"0.02 0.06 0.15"}
MAX_SECONDS=${MAX_SECONDS:-1800}
RATE_TOL=${RATE_TOL:-1e-6}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model10 rrefine] d_ext ${D_EXT} round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model10_rrefine_worker.py \
    --d_ext        "$D_EXT" \
    --round        "$ROUND" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --out_dir      "$OUT_DIR" \
    --stride       "$STRIDE" \
    --targets      "$TARGETS" \
    --polish_iters "$POLISH_ITERS" \
    --hops         "$HOPS" \
    --scales       $SCALES \
    --max_seconds  "$MAX_SECONDS" \
    --tol          "$RATE_TOL" \
    --seed         "$SEED"

echo "[model10 rrefine] d_ext ${D_EXT} round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}: done"
