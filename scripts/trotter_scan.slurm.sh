#!/bin/bash
# ============================================================
#  SLURM job-array: generic two-qubit Trotter Lindbladian scan.
#
#  One MODEL (model1..model6, model7a..model7e) is scanned over its two varying
#  parameters.  Grid sizes (point counts):
#    model1  21 x 51 = 1071      model2  21 x  51 = 1071
#    model3  51 x 51 = 2601      model4  51 x  51 = 2601
#    model5  21 x 101 = 2121     model6  51 x  51 = 2601
#    model7a..model7d  20 x 20 = 400   model7e  20 x 11 = 220
#
#  The grid is split across a 200-task array (N_CHUNKS=200); each task processes
#  a strided subset and skips any npz already current on disk.
#
#  Submit one model (default model1):
#    mkdir -p logs results_trotter_v3
#    MODEL=model1 sbatch scripts/trotter_scan.slurm.sh
#
#  Every model uses dim=2 and the per-point adaptive Trotter step (choose_dt)
#  unless DIM / DT are set; the steady-state quantities always use the full 2x2
#  lattice.  Output goes to results_trotter_v3 (the earlier results_trotter data
#  is kept but ignored).  Budgets are overridable via env vars, e.g.:
#    MODEL=model4 FRA_MAXFEV_4=2000 sbatch scripts/trotter_scan.slurm.sh
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
OUT_DIR=${OUT_DIR:-results_trotter_v3}
N_CHUNKS=${N_CHUNKS:-200}
DIM=${DIM:-}      # empty -> the model's own dim (all models: 2)
DT=${DT:-}        # empty -> per-point adaptive choose_dt
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

EXTRA_ARGS=()
[ -n "$DIM" ] && EXTRA_ARGS+=(--dim "$DIM")
[ -n "$DT" ]  && EXTRA_ARGS+=(--dt "$DT")

python scripts/trotter_scan_worker.py \
    --model         "$MODEL" \
    --task_id       "$SLURM_ARRAY_TASK_ID" \
    --n_chunks      "$N_CHUNKS" \
    --out_dir       "$OUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    --fra_restarts  "$FRA_RESTARTS" \
    --fra_maxfev_4  "$FRA_MAXFEV_4" \
    --fra_maxfev_6  "$FRA_MAXFEV_6" \
    --sign_restarts "$SIGN_RESTARTS" \
    --ch1_restarts  "$CH1_RESTARTS" \
    --seed          "$SEED"

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}: done"
