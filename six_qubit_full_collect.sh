#!/bin/bash
# ============================================================
#  Collect + plot job for the comprehensive 6-qubit scan.
#  Aggregates per-point .npy files into six_full_scan.npy and
#  regenerates six_qubit_scan_full.png.
# ============================================================

#SBATCH --job-name=six_full_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/six_full_collect_%j.out
#SBATCH --error=logs/six_full_collect_%j.err

N_PTS_G=${N_PTS_G:-51}
N_PTS_GP=${N_PTS_GP:-21}
J=${J:-1.0}
GAMMA_STEP=${GAMMA_STEP:-0.2}
OUT_DIR=${OUT_DIR:-results_six}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

python six_qubit_full_collect.py \
    --n_pts_g  "$N_PTS_G" \
    --n_pts_gp "$N_PTS_GP" \
    --in_dir   "$OUT_DIR" \
    --out_dir  "$OUT_DIR"

python plot_six_qubit_scan_full.py \
    --in_dir     "$OUT_DIR" \
    --out_dir    "$OUT_DIR" \
    --n_pts_g    "$N_PTS_G" \
    --n_pts_gp   "$N_PTS_GP" \
    --gamma_step "$GAMMA_STEP" \
    --J          "$J"

echo "Collect + plot done -> ${OUT_DIR}/six_qubit_scan_full.png"
