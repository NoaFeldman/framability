#!/bin/bash
# ============================================================
#  SLURM collect job for the 6-qubit (2x3) scan.
#  Aggregates per-point results into  six_qubit_scan.npy
#  and produces  six_qubit_scan_bond_vs_fra.png.
#
#  Submitted automatically by submit_six_qubit_scan.sh as an
#  afterok dependency on the array job.
# ============================================================

#SBATCH --job-name=six_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/six_collect_%j.out
#SBATCH --error=logs/six_collect_%j.err

set -euo pipefail

N_PTS=${N_PTS:-41}
N_PTS_G=${N_PTS_G:-$N_PTS}
N_PTS_GP=${N_PTS_GP:-$N_PTS}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_six}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python scripts/six_qubit_scan_collect.py \
    --n_pts_g    "$N_PTS_G" \
    --n_pts_gp   "$N_PTS_GP" \
    --J          "$J" \
    --gamma_step "$GAMMA_STEP" \
    --out_dir    "$OUT_DIR"

echo "Collect done."
