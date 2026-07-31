#!/bin/bash
# ============================================================
#  Stage 1 of undoing the flawed alternating re-run: RESTORE opt_fra_4/6 of
#  every trotter_lindbladian_scan point from the OLD frames by reference-LP
#  re-certification (no optimisation), in place in results_trotter_v3.
#
#  ~10.6k points striped over 200 array tasks -> a few minutes per task.  Each
#  task rewrites its own npz slice and drops a per-chunk manifest fragment
#  restore_manifest_chunk<NNN>.npz (no cross-task collisions), so the array is
#  safely resubmittable and points already stamped are skipped.
#
#  Workflow:
#    mkdir -p logs
#    sbatch scripts/trotter_restore.slurm.sh
#    # after it finishes, merge fragments into the recompute manifest:
#    python scripts/trotter_restore_collect.py --out_manifest recompute_manifest
#    # then Stage 2 (cluster) on just the points no old frame could certify:
#    sbatch scripts/trotter_recompute.slurm.sh
#    # then Stage 3, the figures:
#    sbatch scripts/trotter_replot.slurm.sh
#
#  Env overrides: IN_DIR, OLD_DIR (pass OLD_DIR= to use only in-file frames),
#  MODELS, N_CHUNKS.
# ============================================================

#SBATCH --job-name=trotter_restore
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --array=0-199
#SBATCH --output=logs/restore_%x_%A_%a.out
#SBATCH --error=logs/restore_%x_%A_%a.err

IN_DIR=${IN_DIR:-results_trotter_v3}
OLD_DIR=${OLD_DIR-results_trotter_v3_old}   # single dash: allow OLD_DIR= to blank it
MODELS=${MODELS:-model1,model2,model3,model4,model5}
N_CHUNKS=${N_CHUNKS:-200}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[restore] task ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS} in_dir=${IN_DIR} old_dir='${OLD_DIR}'"

python scripts/trotter_restore_confirm_worker.py \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --in_dir   "$IN_DIR" \
    --old_dir  "$OLD_DIR" \
    --models   "$MODELS"

echo "[restore] task ${SLURM_ARRAY_TASK_ID}: done"
