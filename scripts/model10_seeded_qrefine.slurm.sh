#!/bin/bash
# ============================================================
#  SLURM job-array: ONE quick neighbour-refine round of the seeded model10
#  Heisenberg rates (d_ext = 4 and 8), scripts/model10_seeded_qrefine_worker.py.
#
#  Rounds must run sequentially (round r reads every earlier round); the chain
#  is driven by scripts/submit_model10_seeded_refine.sh.  By hand:
#    ROUND=1 sbatch scripts/model10_seeded_qrefine.slurm.sh
#
#  2601 points over 200 tasks (~13 each); interior points exit at once, so a
#  round costs only the floor boundary.
#  Output: $OUT_DIR/model10_seeded[_s<stride>]/pt_<ix>_<iy>_qrefine_r<NN>.npz
#          (improved points only)
# ============================================================

#SBATCH --job-name=m10_seed_qref
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m10seedqref_%x_%A_%a.out
#SBATCH --error=logs/m10seedqref_%x_%A_%a.err

ROUND=${ROUND:?set ROUND (1..99)}
OUT_DIR=${OUT_DIR:-results_model10_rate}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size above
STRIDE=${STRIDE:-1}
RATE_TOL=${RATE_TOL:-1e-6}
OPTIMIZER=${OPTIMIZER:-rate}       # rate (as the existing quick refine) | global
N_RESTARTS=${N_RESTARTS:-3}
MAXFEV_4=${MAXFEV_4:-1000}
MAXFEV_8=${MAXFEV_8:-500}
POLISH=${POLISH:-150}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model10 seeded qrefine] round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model10_seeded_qrefine_worker.py \
    --round      "$ROUND" \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_chunks   "$N_CHUNKS" \
    --out_dir    "$OUT_DIR" \
    --stride     "$STRIDE" \
    --rate_tol   "$RATE_TOL" \
    --optimizer  "$OPTIMIZER" \
    --n_restarts "$N_RESTARTS" \
    --maxfev_4   "$MAXFEV_4" \
    --maxfev_8   "$MAXFEV_8" \
    --polish     "$POLISH" \
    --seed       "$SEED"

echo "[model10 seeded qrefine] round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}: done"
