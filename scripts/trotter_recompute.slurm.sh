#!/bin/bash
# ============================================================
#  Stage 2 of the alternating-scheme cleanup: re-optimise ONLY the manifest
#  points (recompute_manifest.npz, produced locally by trotter_alt_confirm.py)
#  with a correct-objective (confirmation-gated) optimiser, in place in
#  results_trotter_v3.
#
#  The manifest is tiny (~dozens of points), so every array task handles at most
#  one point with a generous budget and finishes in a few minutes -- friendly to
#  a packed short queue.  Points already stamped RECOMPUTE_VERSION are skipped,
#  so the array is safely resubmittable.
#
#  Workflow:
#    # 1. locally, produce the manifest (Stage 1) and copy it to the cluster:
#    #    python scripts/trotter_alt_confirm.py --out_manifest recompute_manifest
#    #    scp recompute_manifest.npz user@host:/path/to/framability/
#    mkdir -p logs
#    sbatch scripts/trotter_recompute.slurm.sh
#    # 2. after it finishes, regenerate the figures (locally or on the cluster):
#    #    for m in model1 model2 model3 model4 model5; do
#    #      python scripts/trotter_scan_collect.py --model $m --in_dir results_trotter_v3
#    #    done
#
#  Budgets / manifest path overridable via env vars, e.g.:
#    MAXFEV=10000 N_RESTARTS=20 sbatch scripts/trotter_recompute.slurm.sh
# ============================================================

#SBATCH --job-name=alt_recompute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/recomp_%x_%A_%a.out
#SBATCH --error=logs/recomp_%x_%A_%a.err

IN_DIR=${IN_DIR:-results_trotter_v3}
MANIFEST=${MANIFEST:-recompute_manifest.npz}
N_CHUNKS=${N_CHUNKS:-200}
N_RESTARTS=${N_RESTARTS:-12}
MAXFEV=${MAXFEV:-6000}
POLISH_ITERS=${POLISH_ITERS:-300}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[recompute] task ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} manifest=${MANIFEST}"

python scripts/trotter_recompute_worker.py \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --in_dir       "$IN_DIR" \
    --manifest     "$MANIFEST" \
    --n_restarts   "$N_RESTARTS" \
    --maxfev       "$MAXFEV" \
    --polish_iters "$POLISH_ITERS" \
    --seed         "$SEED"

echo "[recompute] task ${SLURM_ARRAY_TASK_ID}: done"
