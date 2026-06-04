#!/bin/bash
# ============================================================
#  SLURM job-array: boundary-driven anisotropic XYZ chain (N=4) analysis.
#
#  21 jobs (task_id 0..20), one drive rate each
#  (gamma in [0,10] step 0.5, units Jz=1).  Jx=1.5, Jy=0.5, Jz=1.0.
#
#  Requires highspy in the venv (framability LP warm-start):
#    /path/to/.venv/bin/python -m pip install highspy
#
#  Submit:
#    mkdir -p logs results_xyz_chain
#    sbatch --array=0-20 scripts/xyz_chain.slurm.sh
#
#  Skip the slow d_ext=6 framability:
#    DO_FRA_6=0 sbatch --array=0-20 scripts/xyz_chain.slurm.sh
# ============================================================

#SBATCH --job-name=xyz_chain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/xyz_chain_%A_%a.out
#SBATCH --error=logs/xyz_chain_%A_%a.err

OUT_DIR=${OUT_DIR:-results_xyz_chain}
DT=${DT:-0.05}
DO_FRA_4=${DO_FRA_4:-1}
DO_FRA_6=${DO_FRA_6:-1}
FRA_RESTARTS=${FRA_RESTARTS:-2}
FRA_MAXFEV=${FRA_MAXFEV:-30}
FRA_RESTARTS_6=${FRA_RESTARTS_6:-1}
FRA_MAXFEV_6=${FRA_MAXFEV_6:-10}
SIGN_RESTARTS=${SIGN_RESTARTS:-10}
N_JOBS=${N_JOBS:-21}
SEED=${SEED:-0}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID}/${N_JOBS}: starting (DO_FRA_4=${DO_FRA_4} DO_FRA_6=${DO_FRA_6})"

python scripts/xyz_chain_worker.py \
    --task_id        "$SLURM_ARRAY_TASK_ID" \
    --n_jobs         "$N_JOBS" \
    --out_dir        "$OUT_DIR" \
    --dt             "$DT" \
    --do_fra_4       "$DO_FRA_4" \
    --do_fra_6       "$DO_FRA_6" \
    --fra_restarts   "$FRA_RESTARTS" \
    --fra_maxfev     "$FRA_MAXFEV" \
    --fra_restarts_6 "$FRA_RESTARTS_6" \
    --fra_maxfev_6   "$FRA_MAXFEV_6" \
    --sign_restarts  "$SIGN_RESTARTS" \
    --seed           "$SEED"

echo "Task ${SLURM_ARRAY_TASK_ID}: done"
