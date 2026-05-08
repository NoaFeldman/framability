#!/bin/bash
# ============================================================
#  SLURM job-array script: one task per (gamma, gamma') point
#  for the 6-qubit (2x3 lattice) Lindbladian scan.
#
#  Submit via submit_six_qubit_scan.sh, which sets --array
#  upper bound (= n_pts*n_pts - 1) and exports parameters.
#
#  Direct submission example:
#    export N_PTS=41 J=1.0 GAMMA_STEP=0.2 OUT_DIR=results_six
#    sbatch --array=0-1680 six_qubit_scan_array.sh
# ============================================================

#SBATCH --job-name=six_fra
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/six_scan_%A_%a.out
#SBATCH --error=logs/six_scan_%A_%a.err

set -euo pipefail

# ── read scan parameters (with defaults) ─────────────────────
N_PTS=${N_PTS:-41}N_PTS_G=${N_PTS_G:-$N_PTS}
N_PTS_GP=${N_PTS_GP:-$N_PTS}J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_six}
MAX_STEPS=${MAX_STEPS:-100000}
FIDELITY_THRESHOLD=${FIDELITY_THRESHOLD:-0.9}

# ── activate Python environment ───────────────────────────────
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"

# ── silence matplotlib config-dir warning ────────────────────
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# Limit BLAS threads (we use cpus-per-task=1)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Task ${SLURM_ARRAY_TASK_ID}: starting  (N_PTS_G=${N_PTS_G}, N_PTS_GP=${N_PTS_GP}, J=${J}, step=${GAMMA_STEP})"

python six_qubit_scan_worker.py \
    --task_id            "$SLURM_ARRAY_TASK_ID" \
    --n_pts_g            "$N_PTS_G" \
    --n_pts_gp           "$N_PTS_GP" \
    --J                  "$J" \
    --gamma_step         "$GAMMA_STEP" \
    --out_dir            "$OUT_DIR" \
    --max_steps          "$MAX_STEPS" \
    --fidelity_threshold "$FIDELITY_THRESHOLD"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
