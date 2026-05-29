#!/bin/bash
# ============================================================
#  Collect 6-qubit sign-problem results into summary + figure.
#
#  Usage:
#    IN_DIR=results_sign_six_h1_lam1 \
#      OUT_PNG=results_plots/sign_six_h1_lam1.png \
#      TITLE="6-qubit star+plaquette  h=1, lambda=1  Trotter sign problem" \
#      sbatch --dependency=afterok:<ARRAY_JOB_ID> \
#             scripts/sign_problem_six_qubit_collect.slurm.sh
# ============================================================

#SBATCH --job-name=sign_six_collect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/sign_six_collect_%j.out
#SBATCH --error=logs/sign_six_collect_%j.err

IN_DIR=${IN_DIR:-results_sign_six}
OUT_PNG=${OUT_PNG:-results_plots/sign_six.png}
TITLE=${TITLE:-}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
mkdir -p results_plots

if [ -n "$TITLE" ]; then
  python scripts/sign_problem_six_qubit_collect.py \
      --in_dir "$IN_DIR" --out "$OUT_PNG" --title "$TITLE"
else
  python scripts/sign_problem_six_qubit_collect.py \
      --in_dir "$IN_DIR" --out "$OUT_PNG"
fi
