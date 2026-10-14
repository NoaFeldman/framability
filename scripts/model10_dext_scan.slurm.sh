#!/bin/bash
# ============================================================
#  SLURM job-array: model10 d_ext scan (scripts/model10_dext_scan_worker.py)
#  on the diagonal points DELTAS (Delta_1 = Delta_2) at d_ext = D_EXTS.
#  Task t works on (point, d_ext) combo t % n_combos as replica
#  t // n_combos (3 points x 2 d_ext -> 33-34 replicas per combo).
#    STAGE=search          one independent restart per task (every
#                          COLD_EVERY-th replica cold: no pipeline frames)
#    STAGE=hop ROUND=R     one cross-restart basin-hopping replica per task,
#                          seeded from the search + rounds < R
#
#  Driven by scripts/submit_model10_dext_scan.sh (stages need --wait); by hand:
#      STAGE=search sbatch scripts/model10_dext_scan.slurm.sh
#      STAGE=hop ROUND=1 sbatch scripts/model10_dext_scan.slurm.sh
#
#  Cost per task (1 CPU, LP ~0.05 s at d_ext 6, ~0.5 s at d_ext 10):
#  d_ext 6 ~1-2 h, d_ext 10 ~3-5 h for the search (families ~1 h, DE
#  ~1 h, then SEARCH_HOP_SECONDS of hops); a hop task ~HOP_SECONDS.
#  Output (one file per task, unique per stage / round / combo / replica):
#      $OUT_DIR/model10_dext_scan/m<MM>/pt_<ix>_<iy>/search_r<RRR>.npz
#      $OUT_DIR/model10_dext_scan/m<MM>/pt_<ix>_<iy>/hop<NN>_r<RRR>.npz
# ============================================================

#SBATCH --job-name=m10_dext
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/m10dext_%x_%A_%a.out
#SBATCH --error=logs/m10dext_%x_%A_%a.err

STAGE=${STAGE:?set STAGE (search | hop)}
ROUND=${ROUND:-0}
OUT_DIR=${OUT_DIR:-results_model10_rate}
N_TASKS=${N_TASKS:-200}            # must match the --array size above
DELTAS=${DELTAS:-"0.5 0.6 0.7"}
D_EXTS=${D_EXTS:-"6 10"}
COLD_EVERY=${COLD_EVERY:-3}
FAMILIES=${FAMILIES:-"base split probe"}
SEARCH_HOP_SECONDS=${SEARCH_HOP_SECONDS:-3600}
HOP_SECONDS=${HOP_SECONDS:-5400}
POLISH_ITERS=${POLISH_ITERS:-100}
SCALES=${SCALES:-"0.02 0.06 0.15 0.3"}
HOP_TOP=${HOP_TOP:-5}
IMPROVE_TOL=${IMPROVE_TOL:-1e-5}
SEED=${SEED:-0}

if [ "$STAGE" = "search" ]; then SECS=$SEARCH_HOP_SECONDS; else SECS=$HOP_SECONDS; fi

source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

echo "[model10 dext scan] ${STAGE} round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}/${N_TASKS}: starting"

# shellcheck disable=SC2086
python scripts/model10_dext_scan_worker.py \
    --stage        "$STAGE" \
    --round        "$ROUND" \
    --task_id      "$SLURM_ARRAY_TASK_ID" \
    --n_tasks      "$N_TASKS" \
    --deltas       $DELTAS \
    --d_exts       $D_EXTS \
    --out_dir      "$OUT_DIR" \
    ${DIM:+--dim "$DIM"} \
    --cold_every   "$COLD_EVERY" \
    --families     $FAMILIES \
    ${DE_POPSIZE:+--de_popsize "$DE_POPSIZE"} \
    ${DE_MAXITER:+--de_maxiter "$DE_MAXITER"} \
    --polish_iters "$POLISH_ITERS" \
    --scales       $SCALES \
    --hop_seconds  "$SECS" \
    --hop_top      "$HOP_TOP" \
    --improve_tol  "$IMPROVE_TOL" \
    --seed         "$SEED"

echo "[model10 dext scan] ${STAGE} round ${ROUND}, task ${SLURM_ARRAY_TASK_ID}: done"
