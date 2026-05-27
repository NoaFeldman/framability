p#!/bin/bash
# ============================================================
#  SLURM collect job: aggregate l1-coherence and plot.
#
#  Submit as afterok dependency on the array job:
#    AID=$(sbatch --parsable --array=0-1070 scripts/six_qubit_coherence_array.sh)
#    sbatch --dependency=afterok:$AID scripts/six_qubit_coherence_collect.sh
# ============================================================

#SBATCH --job-name=six_coh_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/six_coh_collect_%j.out
#SBATCH --error=logs/six_coh_collect_%j.err

set -euo pipefail

N_PTS_G=${N_PTS_G:-51}
N_PTS_GP=${N_PTS_GP:-21}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_six_coh}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/six_qubit_coherence_collect.py \
    --n_pts_g "$N_PTS_G" \
    --n_pts_gp "$N_PTS_GP" \
    --J "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir "$OUT_DIR"

echo "Collect done."
