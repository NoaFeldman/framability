#!/bin/bash
# ============================================================
#  SLURM job-array: island / boundary polish of the Trotter-scan
#  optimised framabilities (opt_fra_4 / opt_fra_6).
#
#  Run AFTER a cross-eval sweep (trotter_scan_island_fix.slurm.sh).
#  Only boundary points are re-optimised (opt_fra > 1 with a
#  4-connected neighbour at the floor), with the fixes the islands
#  exposed: ALL 4 neighbours' frames as seeds, larger Powell
#  budgets, and a floor-targeted Polyak subgradient polish.
#
#    MODEL=model1 ROUND=1 sbatch scripts/trotter_scan_island_polish.slurm.sh
# ============================================================

#SBATCH --job-name=trot_ipolish
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trotipol_%x_%A_%a.out
#SBATCH --error=logs/trotipol_%x_%A_%a.err

MODEL=${MODEL:-model1}
ROUND=${ROUND:-1}
OUT_DIR=${OUT_DIR:-results_trotter_v3}
N_CHUNKS=${N_CHUNKS:-200}
N_RESTARTS=${N_RESTARTS:-3}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-3000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-2000}
POLISH_ITERS=${POLISH_ITERS:-300}
FRA_TOL=${FRA_TOL:-1e-6}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} polish round ${ROUND}: starting"

python scripts/trotter_scan_island_polish_worker.py \
    --model        "$MODEL" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --round        "$ROUND" \
    --out_dir      "$OUT_DIR" \
    --n_restarts   "$N_RESTARTS" \
    --fra_maxfev_4 "$FRA_MAXFEV_4" \
    --fra_maxfev_6 "$FRA_MAXFEV_6" \
    --polish_iters "$POLISH_ITERS" \
    --fra_tol      "$FRA_TOL" \
    --seed         "$SEED"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID} polish round ${ROUND}: done"
