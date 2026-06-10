#!/bin/bash
# ============================================================
#  SLURM job-array: 2D dissipative phase transition scan.
#
#  Grid: N_H=20 × N_G=31 = 620 points covering:
#    h     in [-1.0, -0.9, …, 0.9]                  (N_H=20)
#    gamma in [0.0,0.1,…,1.0, 1.2,1.4,…,5.0]        (N_G=31)
#    J=1   (fixed)
#
#  620 points are split across a 200-task array (N_CHUNKS=200), each task
#  processing a strided subset (≈3–4 points) and skipping any npz already
#  on disk — so the original 0.0…0.9 gamma data is reused, not recomputed.
#
#  Submit:
#    mkdir -p logs results_dpt
#    sbatch scripts/dissipative_PT.slurm.sh
#
#  Override time step or framability budget via env vars, e.g.:
#    DT=0.1 FRA_MAXFEV_4=2000 sbatch scripts/dissipative_PT.slurm.sh
# ============================================================

#SBATCH --job-name=dpt_scan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/dpt_%A_%a.out
#SBATCH --error=logs/dpt_%A_%a.err

OUT_DIR=${OUT_DIR:-results_dpt}
N_CHUNKS=${N_CHUNKS:-200}
DT=${DT:-0.05}
FRA_RESTARTS=${FRA_RESTARTS:-5}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-1000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-500}
SIGN_RESTARTS=${SIGN_RESTARTS:-10}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/dissipative_PT_worker.py \
    --task_id       "$SLURM_ARRAY_TASK_ID" \
    --n_chunks      "$N_CHUNKS" \
    --out_dir       "$OUT_DIR" \
    --dt            "$DT" \
    --fra_restarts  "$FRA_RESTARTS" \
    --fra_maxfev_4  "$FRA_MAXFEV_4" \
    --fra_maxfev_6  "$FRA_MAXFEV_6" \
    --sign_restarts "$SIGN_RESTARTS" \
    --seed          "$SEED"

echo "Chunk ${SLURM_ARRAY_TASK_ID}: done"
