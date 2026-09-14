#!/bin/bash
# ============================================================
#  SLURM job-array: Lindbladian quality factor Q_max = max_k |Im/Re lambda_k|
#  (damped modes, exact spectra) over a model's (gamma, gamma') grid.
#
#  Per point: the 16x16 bond generator (seconds) and the full 2x3-lattice
#  Lindbladian (4096x4096 dense eig, the cost driver: ~1-3 min single-threaded).
#  The 51x51 = 2601 points are split across a 0-199 array (200 tasks, the job
#  cap), ~13 points per task, strided so every task samples the whole grid.
#  Existing per-point files are skipped, so resubmitting fills holes.
#
#  Submit (one array per model):
#    mkdir -p logs results_liouvillian_q
#    MODEL=model3 sbatch scripts/liouvillian_q.slurm.sh
#    MODEL=model4 sbatch scripts/liouvillian_q.slurm.sh
#    MODEL=model3 NO_LATTICE=1 sbatch --time=00:30:00 scripts/liouvillian_q.slurm.sh
#
#  Output: results_liouvillian_q/<model>/pt_<ix>_<iy>.npz
# ============================================================

#SBATCH --job-name=liouv_q
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/liouvq_%x_%A_%a.out
#SBATCH --error=logs/liouvq_%x_%A_%a.err

MODEL=${MODEL:-model3}
OUT_DIR=${OUT_DIR:-results_liouvillian_q}
N_CHUNKS=${N_CHUNKS:-200}      # must match the --array size above
STRIDE=${STRIDE:-1}            # 1 = full 51x51, matching the framability panels
LX=${LX:-3}
LY=${LY:-2}
TOL_REL=${TOL_REL:-1e-7}
NO_LATTICE=${NO_LATTICE:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

EXTRA=()
[ "$NO_LATTICE" = "1" ] && EXTRA+=(--no_lattice)

echo "[$MODEL Q  lattice ${LY}x${LX}  no_lattice=$NO_LATTICE] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/liouvillian_q_worker.py \
    --model    "$MODEL" \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --out_dir  "$OUT_DIR" \
    --stride   "$STRIDE" \
    --lx       "$LX" \
    --ly       "$LY" \
    --tol_rel  "$TOL_REL" \
    "${EXTRA[@]}"

echo "[$MODEL Q] chunk ${SLURM_ARRAY_TASK_ID}: done"
