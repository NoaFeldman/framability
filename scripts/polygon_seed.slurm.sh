#!/bin/bash
# ============================================================
#  SLURM job-array: polygon-seeded Heisenberg rate scan, model3 / model4,
#  half resolution (gamma, gamma' in 0.4 steps: 26x26 = 676 points per model),
#  d_ext = 4, 6, 8.  Per point and d_ext: optimiser seeded with the
#  projector-polygon frame, the seed frame as a fixed frame, and the
#  identity-free polygon frame as a fixed frame.
#
#  Submit (one array per model):
#    mkdir -p logs results_polygon_seed
#    MODEL=model3 sbatch scripts/polygon_seed.slurm.sh
#    MODEL=model4 sbatch scripts/polygon_seed.slurm.sh
#  Collect:
#    python scripts/polygon_seed_collect.py
#
#  Output: results_polygon_seed/<model>/pt_<ix>_<iy>.npz  (resubmit fills holes)
# ============================================================

#SBATCH --job-name=polyseed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/polyseed_%x_%A_%a.out
#SBATCH --error=logs/polyseed_%x_%A_%a.err

MODEL=${MODEL:-model4}
OUT_DIR=${OUT_DIR:-results_polygon_seed}
N_CHUNKS=${N_CHUNKS:-200}          # must match the --array size above
STRIDE=${STRIDE:-2}
D_EXTS=${D_EXTS:-"4 6 8"}
RESTARTS=${RESTARTS:-4}
MAXFEV=${MAXFEV:-2000}
POLISH=${POLISH:-200}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[polyseed] ${MODEL} chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/polygon_seed_worker.py \
    --model     "$MODEL" \
    --task_id   "$SLURM_ARRAY_TASK_ID" \
    --n_chunks  "$N_CHUNKS" \
    --out_dir   "$OUT_DIR" \
    --stride    "$STRIDE" \
    --d_exts    $D_EXTS \
    --restarts  "$RESTARTS" \
    --maxfev    "$MAXFEV" \
    --polish    "$POLISH" \
    --seed      "$SEED"

echo "[polyseed] ${MODEL} chunk ${SLURM_ARRAY_TASK_ID}: done"
