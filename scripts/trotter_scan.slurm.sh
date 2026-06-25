#!/bin/bash
# ============================================================
#  SLURM job-array: generic two-qubit Trotter Lindbladian scan.
#
#  One MODEL (model1..model4) is scanned over its two varying parameters.
#  Grid sizes (point counts):
#    model1  41 x 41 = 1681      model2  21 x 51 = 1071
#    model3  21 x 11 =  231      model4  31 x 16 =  496
#
#  The grid is split across a 200-task array (N_CHUNKS=200); each task processes
#  a strided subset and skips any npz already current on disk.
#
#  Submit one model (default model1):
#    mkdir -p logs results_trotter
#    MODEL=model1 sbatch scripts/trotter_scan.slurm.sh
#
#  The framability/sign Trotter gate uses DIM (default 2, a 2D lattice); the
#  steady-state quantities always use the full 2x2 lattice.  Override the time
#  step / gate dimension / budgets via env vars, e.g.:
#    MODEL=model4 DT=0.1 DIM=2 FRA_MAXFEV_4=2000 sbatch scripts/trotter_scan.slurm.sh
# ============================================================

#SBATCH --job-name=trot_scan
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trot_%x_%A_%a.out
#SBATCH --error=logs/trot_%x_%A_%a.err

MODEL=${MODEL:-model1}
OUT_DIR=${OUT_DIR:-results_trotter}
N_CHUNKS=${N_CHUNKS:-200}
DIM=${DIM:-2}
DT=${DT:-0.1}
FRA_RESTARTS=${FRA_RESTARTS:-5}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-1000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-500}
SIGN_RESTARTS=${SIGN_RESTARTS:-10}
CH1_RESTARTS=${CH1_RESTARTS:-15}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/trotter_scan_worker.py \
    --model         "$MODEL" \
    --task_id       "$SLURM_ARRAY_TASK_ID" \
    --n_chunks      "$N_CHUNKS" \
    --out_dir       "$OUT_DIR" \
    --dim           "$DIM" \
    --dt            "$DT" \
    --fra_restarts  "$FRA_RESTARTS" \
    --fra_maxfev_4  "$FRA_MAXFEV_4" \
    --fra_maxfev_6  "$FRA_MAXFEV_6" \
    --sign_restarts "$SIGN_RESTARTS" \
    --ch1_restarts  "$CH1_RESTARTS" \
    --seed          "$SEED"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}: done"
