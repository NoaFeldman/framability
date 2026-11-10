#!/bin/bash
# ============================================================
#  SLURM job-array: dimer-frame Heisenberg framability rate over one model's
#  grid (scripts/dimer_rate_worker.py).  A 2-qubit frame S on every dimer;
#  the bond gate is certified INSIDE a dimer (frame S) and ACROSS dimers
#  (frame S (x) S); objective = per-bond average (mu_in + (2dim-1) mu_cut)/(2dim).
#  Each task takes every N_CHUNKS-th grid point (disjoint files per point,
#  written atomically), so tasks never collide.
#
#  Submitted by scripts/submit_dimer_rate.sh (which sets --array and the
#  dependent collect job); by hand:
#    mkdir -p logs
#    MODEL=model4 N_CHUNKS=200 sbatch --array=0-199 --job-name=m4_dimer scripts/dimer_rate.slurm.sh
#
#  Output: results_<model>_dimer/<model>_dimer_<tiling>_<mode>/pt_<ix>_<iy>.npz
#          (results_<model>_dim<d>_dimer/... when DIM differs from the model's)
#
#  Knobs (env): MODEL (required), DIM (model's own), TILING (chain / brick),
#  MODE (split), OUT_DIR, N_CHUNKS (= array size), STRIDE (1), M_EXT (unset:
#  m = 16 only; e.g. 20 adds an LP-polished overcomplete frame, ~10x slower),
#  RW_RANDOM (3), RW_MAXFEV (400), N_POLISH (3), SMOOTH_ITERS (300 per
#  smoothing stage), N_KICKS (2 per kick scale), BUNDLE_ITERS (40),
#  NO_CERTIFY (unset; 1 skips the ~1 min per-column LP confirmation),
#  DIMER_DIRS (earlier dimer run folders to seed from), SEED (0), FORCE.
#  Expect ~2-4 min per point at the defaults (LP confirmation ~1 min,
#  smoothed polish of 7 starts ~1-2 min).
# ============================================================

#SBATCH --job-name=dimer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/dimer_%x_%A_%a.out
#SBATCH --error=logs/dimer_%x_%A_%a.err

MODEL=${MODEL:?set MODEL (model4 / model10 / model15 ...)}
DIM=${DIM:-}
TILING=${TILING:-}
MODE=${MODE:-split}
OUT_DIR=${OUT_DIR:-}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size
STRIDE=${STRIDE:-1}
M_EXT=${M_EXT:-}
RW_RANDOM=${RW_RANDOM:-3}
RW_MAXFEV=${RW_MAXFEV:-400}
N_POLISH=${N_POLISH:-3}
SMOOTH_ITERS=${SMOOTH_ITERS:-300}
N_KICKS=${N_KICKS:-2}
BUNDLE_ITERS=${BUNDLE_ITERS:-40}
NO_CERTIFY=${NO_CERTIFY:-}
DIMER_DIRS=${DIMER_DIRS:-}
SEED=${SEED:-0}
FORCE=${FORCE:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[${MODEL} dimer] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/dimer_rate_worker.py \
    --model        "$MODEL" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --stride       "$STRIDE" \
    --mode         "$MODE" \
    --rw_random    "$RW_RANDOM" \
    --rw_maxfev    "$RW_MAXFEV" \
    --n_polish     "$N_POLISH" \
    --smooth_iters "$SMOOTH_ITERS" \
    --n_kicks      "$N_KICKS" \
    --bundle_iters "$BUNDLE_ITERS" \
    --seed         "$SEED" \
    ${DIM:+--dim "$DIM"} \
    ${TILING:+--tiling "$TILING"} \
    ${OUT_DIR:+--out_dir "$OUT_DIR"} \
    ${M_EXT:+--m_ext "$M_EXT"} \
    ${DIMER_DIRS:+--dimer_dirs $DIMER_DIRS} \
    ${NO_CERTIFY:+--no_certify} \
    ${FORCE:+--force}

echo "[${MODEL} dimer] chunk ${SLURM_ARRAY_TASK_ID}: done"
