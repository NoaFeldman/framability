#!/bin/bash
# ============================================================
#  SLURM job-array: floor hunt for Trotter-scan models with NO
#  framable (opt_fra == 1) point yet.
#
#  Requires the RANK phase to have run first (single cheap job):
#    python scripts/trotter_scan_floor_hunt_worker.py \
#        --model model2 --round 1 --rank
#  which selects the N_SELECT lowest-value grid points and the
#  elite seed frames.  This array then heavily re-optimises the
#  selected points (Powell over self/neighbour/elite seeds, dual
#  annealing warm start, Polyak floor polish).
#
#    MODEL=model2 ROUND=1 sbatch scripts/trotter_scan_floor_hunt.slurm.sh
# ============================================================

#SBATCH --job-name=trot_fhunt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trotfhunt_%x_%A_%a.out
#SBATCH --error=logs/trotfhunt_%x_%A_%a.err

MODEL=${MODEL:-model2}
ROUND=${ROUND:-1}
OUT_DIR=${OUT_DIR:-results_trotter_v3}
N_CHUNKS=${N_CHUNKS:-200}
N_SELECT=${N_SELECT:-200}
N_RESTARTS=${N_RESTARTS:-3}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-3000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-2000}
DA_MAXFEV=${DA_MAXFEV:-6000}
POLISH_ITERS=${POLISH_ITERS:-500}
FRA_TOL=${FRA_TOL:-1e-6}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} floor hunt round ${ROUND}: starting"

python scripts/trotter_scan_floor_hunt_worker.py \
    --model        "$MODEL" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --round        "$ROUND" \
    --out_dir      "$OUT_DIR" \
    --n_select     "$N_SELECT" \
    --n_restarts   "$N_RESTARTS" \
    --fra_maxfev_4 "$FRA_MAXFEV_4" \
    --fra_maxfev_6 "$FRA_MAXFEV_6" \
    --da_maxfev    "$DA_MAXFEV" \
    --polish_iters "$POLISH_ITERS" \
    --fra_tol      "$FRA_TOL" \
    --seed         "$SEED"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID} floor hunt round ${ROUND}: done"
