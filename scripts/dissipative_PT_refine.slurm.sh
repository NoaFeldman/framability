#!/bin/bash
# ============================================================
#  SLURM job-array: neighbor-seeded refinement of the
#  dissipative-PT optimised framabilities.
#
#  200 jobs (task_id 0..199), one per (ih, ig) grid point:
#    h     in [-1.0, -0.9, …, 0.9]   (N_H=20)
#    gamma in [0.0, 0.1, …, 0.9]     (N_G=10)
#
#  Each task seeds this point's d=4 / d=6 framability optimisation
#  with the best 4-connected neighbour's frame, then (if d=4 < d=6)
#  re-optimises d=6 from the embedded d=4 frame.
#
#  Requires the base scan (dissipative_PT.slurm.sh) to have finished:
#  the dpt_<ih>_<ig>.npz files must already be in OUT_DIR.
#
#  Submit (after the scan array completes):
#    sbatch --dependency=afterok:<SCAN_JOB_ID> \
#           scripts/dissipative_PT_refine.slurm.sh
# ============================================================

#SBATCH --job-name=dpt_refine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/dpt_refine_%A_%a.out
#SBATCH --error=logs/dpt_refine_%A_%a.err

OUT_DIR=${OUT_DIR:-results_dpt}
N_RESTARTS=${N_RESTARTS:-5}
FRA_MAXFEV_4=${FRA_MAXFEV_4:-1000}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-500}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}/200: starting dpt refine"

python scripts/dissipative_PT_refine_worker.py \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --out_dir      "$OUT_DIR" \
    --n_restarts   "$N_RESTARTS" \
    --fra_maxfev_4 "$FRA_MAXFEV_4" \
    --fra_maxfev_6 "$FRA_MAXFEV_6" \
    --seed         "$SEED"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
