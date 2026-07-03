#!/bin/bash
# ============================================================
#  SLURM job-array: build the model5 (dissipative-PT) Trotter-scan results
#  from the existing results_dpt data.
#
#  Grid 20 x 56 = 1120 points, strided over a 200-task array.  Per point the
#  worker reuses every quantity results_dpt already holds (incl. the best
#  opt_fra over all dpt refine rounds) and computes only the missing ones
#  (lpdo, lpdo_max, mag_x, stab_fra, gamma_ch1).
#
#  Submit:
#    mkdir -p logs results_trotter
#    sbatch scripts/trotter_model5_import.slurm.sh
# ============================================================

#SBATCH --job-name=trot_m5imp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trotm5_%x_%A_%a.out
#SBATCH --error=logs/trotm5_%x_%A_%a.err

OUT_DIR=${OUT_DIR:-results_trotter}
DPT_DIR=${DPT_DIR:-results_dpt}
N_CHUNKS=${N_CHUNKS:-200}
CH1_RESTARTS=${CH1_RESTARTS:-15}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model5] import chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/trotter_model5_import_worker.py \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --out_dir      "$OUT_DIR" \
    --dpt_dir      "$DPT_DIR" \
    --ch1_restarts "$CH1_RESTARTS" \
    --seed         "$SEED"

echo "[model5] import chunk ${SLURM_ARRAY_TASK_ID}: done"
