#!/bin/bash
# ============================================================
#  SLURM job-array: 6-qubit star+plaquette Lindbladian Trotter scan.
#
#  100 jobs (task_id 0..99), each processing 1 grid point
#  (100 total over a 10x10 grid in (gamma_s, gamma_p) with step 0.4).
#  One point per job keeps the wall-clock ~= per-point time even though
#  the d_ext=6 framability is ~hours/point.
#
#  Requires highspy in the venv (per-column LP warm-start, ~10x speedup):
#    pip install highspy
#
#  Submit:
#    mkdir -p logs results_six_starplaq
#    sbatch --array=0-99 scripts/six_qubit_starplaq.slurm.sh
#
#  To skip the slow d_ext=6 framability:
#    DO_FRA_6=0 sbatch --array=0-99 scripts/six_qubit_starplaq.slurm.sh
# ============================================================

#SBATCH --job-name=six_starplaq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/six_starplaq_%A_%a.out
#SBATCH --error=logs/six_starplaq_%A_%a.err

OUT_DIR=${OUT_DIR:-results_six_starplaq}
H_FIELD=${H_FIELD:-1.0}
LAM=${LAM:-1.0}
DT=${DT:-0.04}
DO_FRA_4=${DO_FRA_4:-1}
DO_FRA_6=${DO_FRA_6:-1}
FRA_RESTARTS=${FRA_RESTARTS:-2}
FRA_MAXFEV=${FRA_MAXFEV:-30}
FRA_RESTARTS_6=${FRA_RESTARTS_6:-1}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-10}
N_JOBS=${N_JOBS:-100}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting (n_jobs=${N_JOBS}, DO_FRA_4=${DO_FRA_4} DO_FRA_6=${DO_FRA_6})"

python scripts/six_qubit_starplaq_worker.py \
    --task_id        "$SLURM_ARRAY_TASK_ID" \
    --n_jobs         "$N_JOBS" \
    --out_dir        "$OUT_DIR" \
    --h              "$H_FIELD" \
    --lam            "$LAM" \
    --dt             "$DT" \
    --do_fra_4       "$DO_FRA_4" \
    --do_fra_6       "$DO_FRA_6" \
    --fra_restarts   "$FRA_RESTARTS" \
    --fra_maxfev     "$FRA_MAXFEV" \
    --fra_restarts_6 "$FRA_RESTARTS_6" \
    --fra_maxfev_6   "$FRA_MAXFEV_6" \
    --seed           "$SEED"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
