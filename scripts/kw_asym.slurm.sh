#!/bin/bash
# ============================================================
#  SLURM job-array: KW asymmetry Delta S_KW(t) of arXiv:2610.23111 for model10
#  (scripts/kw_asym_worker.py, library kw_asymmetry.py), exact full-density-
#  matrix evolution.  Task t runs scan point t of kw_asymmetry.scan_points()
#  (200 points: 150 on (h, D1 = D2), 50 on (D1, D2) at h = J).
#
#  Submitted together with its collect by scripts/submit_kw_asym.sh; by hand:
#      GEOMETRY=ring:12 JUMPS=local sbatch scripts/kw_asym.slurm.sh
#
#  Cost per task at N = 12 (1 CPU): one L(rho) ~ 2-4 s (local), ~ 1-2x that
#  (collective); DOP853 needs a few thousand L-evals to J t = 3 -> ~2-5 h
#  local.  Collective jumps are stiffer (A^2 ~ N^2): the submit script gives
#  them 24 h.  Memory ~ 8 GB (13 DOP853 stages + dense output + buffers,
#  268 MB each).
#  Output: $OUT_DIR/<geometry>_<jumps>_<init>/pt_<PPP>.npz (one per point;
#  existing files are skipped, so resubmitting finishes what is missing).
# ============================================================

#SBATCH --job-name=kw_asym
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/kwasym_%x_%A_%a.out
#SBATCH --error=logs/kwasym_%x_%A_%a.err

GEOMETRY=${GEOMETRY:-ring:12}
JUMPS=${JUMPS:-local}
INIT=${INIT:-ghz_s}
T_MAX=${T_MAX:-3.0}
N_T=${N_T:-61}
RTOL=${RTOL:-1e-8}
ATOL=${ATOL:-1e-12}
OUT_DIR=${OUT_DIR:-results_kw_asym}
N_TASKS=${N_TASKS:-200}            # must match the --array size above
SELFCHECK=${SELFCHECK:-1}

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[kw asym] ${GEOMETRY} ${JUMPS} ${INIT}, task ${SLURM_ARRAY_TASK_ID}/${N_TASKS}: starting"

python scripts/kw_asym_worker.py \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_tasks  "$N_TASKS" \
    --geometry "$GEOMETRY" \
    --jumps    "$JUMPS" \
    --init     "$INIT" \
    ${ALPHA:+--alpha "$ALPHA"} \
    --t_max    "$T_MAX" \
    --n_t      "$N_T" \
    --rtol     "$RTOL" \
    --atol     "$ATOL" \
    --out_dir  "$OUT_DIR" \
    $( [ "$SELFCHECK" = "1" ] && echo --selfcheck )
status=$?

echo "[kw asym] task ${SLURM_ARRAY_TASK_ID}: exit ${status}"
exit "$status"
