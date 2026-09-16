#!/bin/bash
# ============================================================
#  SLURM job-array: ONE full neighbour-refine round of the optimised
#  Heisenberg framability RATES (rate_heis_4 / rate_heis_6, d_ext = 4 and 6)
#  over EVERY point of one rate-panel model's grid.
#
#  Rounds must run SEQUENTIALLY -- round r reads the base scan plus every
#  earlier round -- so this script does ONE round and
#  scripts/submit_model4_rate_nb_refine.sh chains ROUND=1..10 via
#  `sbatch --wait`.
#
#  The 51x51 = 2601 grid points are split across a 0-199 array (200 tasks, the
#  job cap) via --n_chunks, ~13 points per task, strided so every task samples
#  the whole grid.  Every processed point writes its round file, so
#  resubmitting the same ROUND skips finished points and fills the holes.
#
#  Submit (one round; usually invoked by the chain script, not by hand):
#    mkdir -p logs
#    MODEL=model10 ROUND=1 sbatch --job-name=m10_nbrefine scripts/model4_rate_nb_refine.slurm.sh
#      (model10 = the Shibata-Katsura dissipative quantum Ising chain,
#       https://arxiv.org/abs/1904.12505)
#
#  Output: results_<model>_rate/<model>/pt_<ix>_<iy>_nrefine_r<NN>.npz
# ============================================================

#SBATCH --job-name=nbrefine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/nbrefine_%x_%A_%a.out
#SBATCH --error=logs/nbrefine_%x_%A_%a.err

ROUND=${ROUND:?set ROUND (1..10) to the refine round}
MODEL=${MODEL:?set MODEL, e.g. MODEL=model10}
OUT_DIR=${OUT_DIR:-results_${MODEL}_rate}
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

echo "[${MODEL} rate nb-refine] round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/model4_rate_nb_refine_worker.py \
    --model      "$MODEL" \
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

echo "[${MODEL} rate nb-refine] round ${ROUND}, chunk ${SLURM_ARRAY_TASK_ID}: done"
