#!/bin/bash
# ============================================================
#  SLURM job-array: ONE quick neighbour-refine round of the optimised
#  Heisenberg framability RATES (rate_heis_4 / rate_heis_6) over the full
#  model4 (gamma, gamma') grid.
#
#  Rounds must run SEQUENTIALLY -- round r reads the base scan plus every
#  earlier round, so the framable floor propagates outward one ring per round.
#  This script therefore does ONE round; scripts/submit_model4_rate_quick_refine.sh
#  chains ROUND=1..10 via `sbatch --wait`.
#
#  The 2601 grid points are split across a 0-199 array (200 tasks, the job
#  cap) via --n_chunks, ~14 points per task.  Most points are interior and are
#  skipped near-instantly (no file written), so a round is far cheaper than
#  the base scan.
#
#  Submit (one round; usually invoked by the chain script, not by hand):
#    mkdir -p logs results_model4_rate
#    ROUND=1 sbatch scripts/model4_rate_quick_refine.slurm.sh
#
#  Output: results_model4_rate/model4/pt_<ix>_<iy>_qrefine_r<NN>.npz
#          (qualifying points only)
# ============================================================

#SBATCH --job-name=m4_qrefine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m4qrefine_%x_%A_%a.out
#SBATCH --error=logs/m4qrefine_%x_%A_%a.err

ROUND=${ROUND:?set ROUND (1..10) to the quick-refine round}
OUT_DIR=${OUT_DIR:-results_model4_rate}
N_CHUNKS=${N_CHUNKS:-200}       # must match the --array size above
RATE_TOL=${RATE_TOL:-1e-6}
N_RESTARTS=${N_RESTARTS:-3}
MAXFEV_4=${MAXFEV_4:-1000}
MAXFEV_6=${MAXFEV_6:-500}
POLISH=${POLISH:-150}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[m4 rate qrefine] round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model4_rate_quick_refine_worker.py \
    --round      "$ROUND" \
    --task_id    "$SLURM_ARRAY_TASK_ID" \
    --n_chunks   "$N_CHUNKS" \
    --out_dir    "$OUT_DIR" \
    --rate_tol   "$RATE_TOL" \
    --n_restarts "$N_RESTARTS" \
    --maxfev_4   "$MAXFEV_4" \
    --maxfev_6   "$MAXFEV_6" \
    --polish     "$POLISH" \
    --seed       "$SEED"

echo "[m4 rate qrefine] round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}: done"
