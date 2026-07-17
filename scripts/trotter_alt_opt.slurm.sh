#!/bin/bash
# ============================================================
#  SLURM job-array: re-optimise the Heisenberg framabilities
#  (opt_fra_4 / opt_fra_6) of every stored trotter scan point with the
#  certificate-based 'alternating' scheme, in place in results_trotter_v3.
#
#  All five models are handled by ONE array: the 9465 grid points
#  (model1 1071 + model2 1071 + model3 2601 + model4 2601 + model5 2121)
#  are concatenated and strided across 200 tasks (~48 points each; each
#  point runs the d_ext_single=4 and d_ext_single=6 optimisations).
#  Points already stamped with the current OPT_VERSION are skipped, so the
#  array is safely resubmittable.
#
#  Submit:
#    mkdir -p logs
#    sbatch scripts/trotter_alt_opt.slurm.sh
#
#  Budgets / subsets are overridable via env vars, e.g.:
#    MODELS=model3,model4 MAXFEV=4000 sbatch scripts/trotter_alt_opt.slurm.sh
# ============================================================

#SBATCH --job-name=alt_opt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/alt_%x_%A_%a.out
#SBATCH --error=logs/alt_%x_%A_%a.err

IN_DIR=${IN_DIR:-results_trotter_v3}
MODELS=${MODELS:-model1,model2,model3,model4,model5}
N_CHUNKS=${N_CHUNKS:-200}
N_RESTARTS=${N_RESTARTS:-5}
MAXFEV=${MAXFEV:-2000}
POLISH_ITERS=${POLISH_ITERS:-150}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[alt_opt] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: ${MODELS}"

python scripts/trotter_alt_opt_worker.py \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --in_dir       "$IN_DIR" \
    --models       "$MODELS" \
    --n_restarts   "$N_RESTARTS" \
    --maxfev       "$MAXFEV" \
    --polish_iters "$POLISH_ITERS" \
    --seed         "$SEED"

echo "[alt_opt] chunk ${SLURM_ARRAY_TASK_ID}: done"
