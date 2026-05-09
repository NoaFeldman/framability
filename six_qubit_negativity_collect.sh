#!/bin/bash
# ============================================================
#  SLURM collect job: aggregate negativity and plot.
#  
#  Submitted as afterok dependency on the array job.
# ============================================================

#SBATCH --job-name=six_neg_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/six_neg_collect_%j.out
#SBATCH --error=logs/six_neg_collect_%j.err

set -euo pipefail

N_PTS_G=${N_PTS_G:-101}
N_PTS_GP=${N_PTS_GP:-7}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_six_neg}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python six_qubit_negativity_collect.py \
    --n_pts_g "$N_PTS_G" \
    --n_pts_gp "$N_PTS_GP" \
    --J "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir "$OUT_DIR"

echo "Collect done."
