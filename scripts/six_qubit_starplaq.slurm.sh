#!/bin/bash
# ============================================================
#  SLURM job-array: 6-qubit star+plaquette Lindbladian Trotter scan.
#
#  10 jobs (task_id 0..9), each processing 40 grid points
#  (400 total over a 20x20 grid in (gamma_s, gamma_p) with step 0.2).
#
#  Submit:
#    mkdir -p logs results_six_starplaq
#    sbatch --array=0-9 scripts/six_qubit_starplaq.slurm.sh
#
#  To enable the (very expensive) optimised framabilities, override:
#    DO_FRA_OPT=1 sbatch --array=0-9 scripts/six_qubit_starplaq.slurm.sh
# ============================================================

#SBATCH --job-name=six_starplaq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/six_starplaq_%A_%a.out
#SBATCH --error=logs/six_starplaq_%A_%a.err

OUT_DIR=${OUT_DIR:-results_six_starplaq}
H_FIELD=${H_FIELD:-1.0}
LAM=${LAM:-1.0}
DT=${DT:-0.02}
DO_FRA_4=${DO_FRA_4:-1}
DO_FRA_6=${DO_FRA_6:-1}
FRA_RESTARTS=${FRA_RESTARTS:-2}
FRA_MAXFEV=${FRA_MAXFEV:-30}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}: starting (n_jobs=10, DO_FRA_4=${DO_FRA_4} DO_FRA_6=${DO_FRA_6})"

python scripts/six_qubit_starplaq_worker.py \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_jobs       10 \
    --out_dir      "$OUT_DIR" \
    --h            "$H_FIELD" \
    --lam          "$LAM" \
    --dt           "$DT" \
    --do_fra_4     "$DO_FRA_4" \
    --do_fra_6     "$DO_FRA_6" \
    --fra_restarts "$FRA_RESTARTS" \
    --fra_maxfev   "$FRA_MAXFEV" \
    --seed         "$SEED"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
