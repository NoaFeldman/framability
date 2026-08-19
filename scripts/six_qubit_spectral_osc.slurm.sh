#!/bin/bash
# ============================================================
#  SLURM job-array: item 6 -- spectral-oscillation measures
#  (nonequilibrium_phase_characterizers.spectral_oscillation) of a 6-qubit
#  ring Lindbladian, over the same coarsened model3 (gamma, gamma') grid as
#  scripts/eight_qubit_gap.slurm.sh (11x11=121 points at --stride=5; see
#  scripts/six_qubit_spectral_osc_worker.py's docstring for why N=6 not 8).
#
#  One grid point per array task (121 <= 200, no chunking needed).  Each task
#  builds the dense 64x64 H and jump operators for its (gamma, gamma') point
#  and fully diagonalizes the 4096x4096 Liouvillian.
#
#  Submit:
#    mkdir -p logs results_8q/spectral_osc_ring6
#    sbatch scripts/six_qubit_spectral_osc.slurm.sh
#
#  Output: results_8q/spectral_osc_ring6/pt_<ig>_<igp>.npz
# ============================================================

#SBATCH --job-name=6q_specosc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-120
#SBATCH --output=logs/specosc_%x_%A_%a.out
#SBATCH --error=logs/specosc_%x_%A_%a.err

OUT_DIR=${OUT_DIR:-results_8q/spectral_osc_ring6}
# NOTE: STRIDE must match the #SBATCH --array range above, same rule as
# eight_qubit_gap.slurm.sh (STRIDE=5 -> 11x11=121 -> array=0-120, the default).
STRIDE=${STRIDE:-5}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "point ${SLURM_ARRAY_TASK_ID}: starting"

python scripts/six_qubit_spectral_osc_worker.py \
    --task_id "$SLURM_ARRAY_TASK_ID" \
    --stride  "$STRIDE" \
    --out_dir "$OUT_DIR"

echo "point ${SLURM_ARRAY_TASK_ID}: done"
