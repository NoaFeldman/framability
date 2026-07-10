#!/bin/bash
# ============================================================
#  SLURM job-array: dephasing-anneal continuation of the optimised
#  framability (Heisenberg, d_ext_single = 6) for the minimal-opt_fra_6
#  point of model1 / model2 / model5 of the Trotter scan.
#
#  Layout: 3 models x N_SEEDS independent chains = 198 tasks (array 0-197).
#      model_idx = task / N_SEEDS      seed = task % N_SEEDS
#  Every chain of one model works on the same (deterministically selected)
#  scan point and the same kappa grid; the seeds differ only in the random
#  jitter around the warm starts.  Each chain writes its own
#      results_deph_anneal/<model>/chain_<ix>_<iy>_seed<seed>.npz
#  so there are no cross-job collisions, and re-submission skips chains
#  already at the current code version.
#
#  Submit:
#      mkdir -p logs results_deph_anneal
#      sbatch scripts/trotter_deph_anneal.slurm.sh
#
#  Collect + plot (after the array finishes):
#      python scripts/trotter_deph_anneal_collect.py \
#          --models model1 model2 model5 \
#          --in_dir results_deph_anneal --out_dir results_plots
#
#  Budgets / grid are overridable via env vars, e.g.:
#      PER_DECADE=8 MAXFEV=600 sbatch scripts/trotter_deph_anneal.slurm.sh
# ============================================================

#SBATCH --job-name=deph_anneal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-197
#SBATCH --output=logs/deph_%x_%A_%a.out
#SBATCH --error=logs/deph_%x_%A_%a.err

MODELS_LIST=(${MODELS_LIST:-model1 model2 model5})
N_SEEDS=${N_SEEDS:-66}
SCAN_DIR=${SCAN_DIR:-results_trotter_v3}
OUT_DIR=${OUT_DIR:-results_deph_anneal}
KAPPA_MIN=${KAPPA_MIN:-1e-3}
KAPPA_MAX=${KAPPA_MAX:-1e4}
KAPPA_EXT_MAX=${KAPPA_EXT_MAX:-1e7}
PER_DECADE=${PER_DECADE:-5}
TOL=${TOL:-1e-5}
MAXFEV=${MAXFEV:-400}
N_JITTER=${N_JITTER:-1}
JITTER=${JITTER:-0.05}

TASK=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$((TASK / N_SEEDS))
SEED=$((TASK % N_SEEDS))

if [ "$MODEL_IDX" -ge "${#MODELS_LIST[@]}" ]; then
    echo "task ${TASK}: model index ${MODEL_IDX} beyond ${#MODELS_LIST[@]} models -- nothing to do"
    exit 0
fi
MODEL=${MODELS_LIST[$MODEL_IDX]}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[deph_anneal] task ${TASK}: model=${MODEL} seed=${SEED}"

python scripts/trotter_deph_anneal_worker.py \
    --model         "$MODEL" \
    --seed          "$SEED" \
    --scan_dir      "$SCAN_DIR" \
    --out_dir       "$OUT_DIR" \
    --kappa_min     "$KAPPA_MIN" \
    --kappa_max     "$KAPPA_MAX" \
    --kappa_ext_max "$KAPPA_EXT_MAX" \
    --per_decade    "$PER_DECADE" \
    --tol           "$TOL" \
    --maxfev        "$MAXFEV" \
    --n_jitter      "$N_JITTER" \
    --jitter        "$JITTER"

echo "[deph_anneal] task ${TASK}: done"
