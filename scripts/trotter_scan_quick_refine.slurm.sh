#!/bin/bash
# ============================================================
#  SLURM job-array: QUICK neighbour-seeded refinement of the
#  Trotter-scan optimised framabilities (opt_fra_4 / opt_fra_6).
#
#  Only boundary points are re-optimised: opt_fra > 1 with a 4-connected
#  neighbour at the framable floor opt_fra == 1 (plus the cross d4->d6
#  embedding step).  Much cheaper than trotter_scan_refine.slurm.sh, so
#  many consecutive rounds are affordable.
#
#  Submit chained rounds via scripts/submit_trotter_quick_refine.sh, or
#  manually:
#    MODEL=model1 ROUND=1 sbatch scripts/trotter_scan_quick_refine.slurm.sh
#    MODEL=model1 ROUND=2 sbatch --dependency=afterany:<R1_JOBID> \
#        scripts/trotter_scan_quick_refine.slurm.sh
# ============================================================

#SBATCH --job-name=trot_qrefine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trotqref_%x_%A_%a.out
#SBATCH --error=logs/trotqref_%x_%A_%a.err

MODEL=${MODEL:-model1}
ROUND=${ROUND:-1}
OUT_DIR=${OUT_DIR:-results_trotter_v3}
N_CHUNKS=${N_CHUNKS:-200}
N_RESTARTS=${N_RESTARTS:-3}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-1000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-500}
FRA_TOL=${FRA_TOL:-1e-6}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} quick round ${ROUND}: starting"

python scripts/trotter_scan_quick_refine_worker.py \
    --model        "$MODEL" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --round        "$ROUND" \
    --out_dir      "$OUT_DIR" \
    --n_restarts   "$N_RESTARTS" \
    --fra_maxfev_4 "$FRA_MAXFEV_4" \
    --fra_maxfev_6 "$FRA_MAXFEV_6" \
    --fra_tol      "$FRA_TOL" \
    --seed         "$SEED"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID} quick round ${ROUND}: done"
