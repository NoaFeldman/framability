#!/bin/bash
# ============================================================
#  SLURM job-array: compass-chain framability-RATE panels (1-6) for ONE case.
#
#  Every (gamma, h) point of compass_chain's 11x11 grid times the six measures
#  (Pauli, stabilizer-3, opt Heisenberg d_ext=4/6, opt Schrodinger d_ext=4/6)
#  is one unit of work: 726 units, strided over this 0-199 array (~4 units per
#  task, measures interleaved).  Each unit balances the dephasing split between
#  the XX and YY bond gates to REL_TOL (dephasing_split.balance_split), so one
#  unit costs up to MAX_EVALS evaluations of the measure on both gates.
#  Finished units are skipped, so resubmitting the same CASE fills the holes.
#
#  Submit (normally through scripts/submit_compass_all.sh):
#    mkdir -p logs results_compass
#    CASE=jx1.0_jy1.0_hx sbatch scripts/compass_rates.slurm.sh
#
#  Output: results_compass/<CASE>/rates/pt_<ig>_<ih>_<measure>.npz
#
#  RUNTIME: dominated by the optimised frames at d_ext=6, where every split
#  evaluation is two full optimisations (warm-started along the search).  Check
#  a log before trusting the 24 h limit; lower MAX_EVALS or the restart/maxfev
#  knobs if tasks time out (a timed-out task also holds back the afterok
#  collect).
# ============================================================

#SBATCH --job-name=cmp_rate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/cmprate_%x_%A_%a.out
#SBATCH --error=logs/cmprate_%x_%A_%a.err

CASE=${CASE:?set CASE to a key of compass_chain.CASES}
OUT_DIR=${OUT_DIR:-results_compass}
N_CHUNKS=${N_CHUNKS:-200}         # must match the --array size above
REL_TOL=${REL_TOL:-0.1}
ABS_TOL=${ABS_TOL:-1e-6}
MAX_EVALS=${MAX_EVALS:-12}
HEIS_RESTARTS=${HEIS_RESTARTS:-8}
HEIS_MAXFEV=${HEIS_MAXFEV:-3000}
POLISH=${POLISH:-300}
SCHRO_RESTARTS=${SCHRO_RESTARTS:-5}
SCHRO_MAXFEV=${SCHRO_MAXFEV:-800}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[compass rates] ${CASE} chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

python scripts/compass_rates_worker.py \
    --case           "$CASE" \
    --task_id        "$SLURM_ARRAY_TASK_ID" \
    --n_chunks       "$N_CHUNKS" \
    --out_dir        "$OUT_DIR" \
    --rel_tol        "$REL_TOL" \
    --abs_tol        "$ABS_TOL" \
    --max_evals      "$MAX_EVALS" \
    --heis_restarts  "$HEIS_RESTARTS" \
    --heis_maxfev    "$HEIS_MAXFEV" \
    --polish         "$POLISH" \
    --schro_restarts "$SCHRO_RESTARTS" \
    --schro_maxfev   "$SCHRO_MAXFEV" \
    --seed           "$SEED"
status=$?

echo "[compass rates] ${CASE} chunk ${SLURM_ARRAY_TASK_ID}: exit ${status}"
exit "$status"
