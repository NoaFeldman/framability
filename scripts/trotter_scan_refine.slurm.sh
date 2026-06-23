#!/bin/bash
# ============================================================
#  SLURM job-array: neighbour-seeded refinement of the Trotter-scan
#  optimised framabilities (opt_fra_4 / opt_fra_6).
#
#  Run 5 sequential rounds (ROUND=1..5); each round reads every earlier round
#  so the best frames propagate across the grid.  Requires the base scan
#  (trotter_scan.slurm.sh) for this MODEL to have finished.
#
#  Submit round 1 (chained after the scan), then chain 2..5:
#    MODEL=model1 ROUND=1 sbatch --dependency=afterok:<SCAN_JOBID> \
#        scripts/trotter_scan_refine.slurm.sh
#    MODEL=model1 ROUND=2 sbatch --dependency=afterok:<R1_JOBID> \
#        scripts/trotter_scan_refine.slurm.sh
#    ... up to ROUND=5
#  (see submit_trotter_scan.sh for an automated chain.)
# ============================================================

#SBATCH --job-name=trot_refine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trotref_%x_%A_%a.out
#SBATCH --error=logs/trotref_%x_%A_%a.err

MODEL=${MODEL:-model1}
ROUND=${ROUND:-1}
OUT_DIR=${OUT_DIR:-results_trotter}
N_CHUNKS=${N_CHUNKS:-200}
N_RESTARTS=${N_RESTARTS:-5}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-1000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-500}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} round ${ROUND}: starting"

python scripts/trotter_scan_refine_worker.py \
    --model        "$MODEL" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --round        "$ROUND" \
    --out_dir      "$OUT_DIR" \
    --n_restarts   "$N_RESTARTS" \
    --fra_maxfev_4 "$FRA_MAXFEV_4" \
    --fra_maxfev_6 "$FRA_MAXFEV_6" \
    --seed         "$SEED"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID} round ${ROUND}: done"
