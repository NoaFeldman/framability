#!/bin/bash
# ============================================================
#  SLURM job-array: re-optimise ONLY the optimised-framability quantities of an
#  existing Trotter scan, in place, with a stronger optimiser (see
#  scripts/trotter_reopt_worker.py).  Every other stored quantity is untouched.
#
#  Recomputed keys (never degraded — kept only if strictly better in the right
#  direction): opt_fra_4/6 and deph_heis_fra_4 via the alternating-certificate
#  method + Polyak polish; sch_fra_4/6/8 and deph_schro_fra_4 via dual_annealing;
#  gamma_ch1 via a global dual_annealing maximisation.
#
#  One MODEL is scanned over its full grid; the grid is split across a 200-task
#  array (N_CHUNKS=200), each task processing a strided subset.  Points already
#  stamped REOPT_VERSION are skipped, so the array is safely resubmittable.
#
#  Prerequisite: the model's scan pt files must already exist in $IN_DIR
#  (produced by scripts/trotter_scan.slurm.sh).  The gate is rebuilt from each
#  point's own stored (p1, p2, dim, dt) — no scan parameter is re-derived.
#
#  Submit one model:
#    mkdir -p logs
#    MODEL=model7a sbatch scripts/trotter_reopt.slurm.sh
#
#  Budgets overridable via env vars, e.g.:
#    MODEL=model7e MAXFEV=10000 SCH_MAXFEV=5000 sbatch scripts/trotter_reopt.slurm.sh
#  Restrict to a subset of quantities:
#    MODEL=model7a QUANTITIES="opt_fra_4 opt_fra_6" sbatch scripts/trotter_reopt.slurm.sh
# ============================================================

#SBATCH --job-name=trot_reopt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/reopt_%x_%A_%a.out
#SBATCH --error=logs/reopt_%x_%A_%a.err

MODEL=${MODEL:-model7a}
IN_DIR=${IN_DIR:-results_trotter_v3}
N_CHUNKS=${N_CHUNKS:-200}
N_RESTARTS=${N_RESTARTS:-12}
MAXFEV=${MAXFEV:-6000}
SCH_MAXFEV=${SCH_MAXFEV:-3000}
CH1_MAXFEV=${CH1_MAXFEV:-2000}
POLISH_ITERS=${POLISH_ITERS:-300}
QUANTITIES=${QUANTITIES:-}     # empty -> all optimised framabilities
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[$MODEL] reopt chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

EXTRA_ARGS=()
[ -n "$QUANTITIES" ] && EXTRA_ARGS+=(--quantities $QUANTITIES)

python scripts/trotter_reopt_worker.py \
    --model        "$MODEL" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_chunks     "$N_CHUNKS" \
    --in_dir       "$IN_DIR" \
    --n_restarts   "$N_RESTARTS" \
    --maxfev       "$MAXFEV" \
    --sch_maxfev   "$SCH_MAXFEV" \
    --ch1_maxfev   "$CH1_MAXFEV" \
    --polish_iters "$POLISH_ITERS" \
    "${EXTRA_ARGS[@]}" \
    --seed         "$SEED"

echo "[$MODEL] reopt chunk ${SLURM_ARRAY_TASK_ID}: done"
