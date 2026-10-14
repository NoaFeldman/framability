#!/bin/bash
# ============================================================
#  Reliable model10 rates at d_ext = D_EXTS (default "6 10") on the diagonal
#  points DELTAS (default "0.5 0.6 0.7"), then the figure
#  results_model10_rate/model10_opt_rate_vs_dext.png (mu* vs d_ext):
#
#    1. search   one 200-task array (scripts/model10_dext_scan.slurm.sh,
#                STAGE=search): independent restarts, 33-34 per
#                (point, d_ext), every COLD_EVERY-th one cold
#    2. hop      200-task arrays, rounds 1, 2, ...: cross-restart basin
#                hopping from the best frames so far.  Stops after PATIENCE
#                rounds in a row in which no replica lowers its
#                (point, d_ext) by more than IMPROVE_TOL, or after MAX_ROUNDS
#    3. collect  scripts/model10_dext_scan_collect.slurm.sh: convergence
#                table (hits of independent restarts, last-round gain),
#                merge with the pipelines' d_ext 4 / 8 / 12, nesting, figure
#
#  At most one 200-task array may be queued at a time, so every stage runs
#  with `sbatch --wait`; start on the login node:
#
#      nohup bash scripts/submit_model10_dext_scan.sh > logs/submit_m10_dext.log 2>&1 &
#
#  More hop rounds later (search kept):
#      SKIP_SEARCH=1 MAX_ROUNDS=2 nohup bash scripts/submit_model10_dext_scan.sh \
#          > logs/submit_m10_dext_more.log 2>&1 &
#  Every d_ext through the same test (13 replicas per combo):
#      D_EXTS="4 6 8 10 12" nohup bash scripts/submit_model10_dext_scan.sh ... &
#
#  Knobs (env; sbatch exports them to the slurm scripts): DELTAS, D_EXTS,
#  OUT_DIR, DIM, COLD_EVERY (3), FAMILIES, SEARCH_HOP_SECONDS (3600),
#  HOP_SECONDS (5400), POLISH_ITERS, SCALES, HOP_TOP, IMPROVE_TOL (1e-5),
#  DE_POPSIZE, DE_MAXITER, SEED, MIN_HITS (3), MIN_COLD_HITS (1), TOL_REL,
#  TOL_ABS, OUT_PNG; driver: MAX_ROUNDS (4), PATIENCE (1), START_ROUND
#  (default one past the highest round on disk), SKIP_SEARCH (0).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export OUT_DIR="${OUT_DIR:-results_model10_rate}"
export DELTAS="${DELTAS:-0.5 0.6 0.7}"
export D_EXTS="${D_EXTS:-6 10}"
export IMPROVE_TOL="${IMPROVE_TOL:-1e-5}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
PATIENCE="${PATIENCE:-1}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"
SCAN="$OUT_DIR/model10_dext_scan"
TAG="[m10 dext scan, d_ext ${D_EXTS}, Delta ${DELTAS}]"
mkdir -p logs "$SCAN"

# ---- 1. independent restarts ------------------------------------------
if [ "$SKIP_SEARCH" != "1" ]; then
    jid=$(STAGE=search sbatch --parsable --wait scripts/model10_dext_scan.slurm.sh) \
        || echo "$TAG warning: some search tasks failed; continuing"
    jid="${jid%%;*}"
    n_files=$(find "$SCAN" -name 'search_r[0-9][0-9][0-9].npz' | wc -l)
    n_err=$(cat logs/m10dext_m10_dext_"${jid}"_*.out 2>/dev/null | grep -c 'ERROR' || true)
    echo "$TAG search (job ${jid}): ${n_files} restart file(s) on disk, ${n_err} error(s)"
fi

# ---- 2. cross-restart basin hopping, until exhausted -------------------
if [ -z "${START_ROUND:-}" ]; then
    last=$(find "$SCAN" -name 'hop[0-9][0-9]_r[0-9][0-9][0-9].npz' 2>/dev/null \
           | sed 's/.*\/hop\([0-9][0-9]\)_r[0-9]*\.npz$/\1/' | sort -n | tail -1 || true)
    START_ROUND=$(( 10#${last:-0} + 1 ))
fi
END_ROUND=$(( START_ROUND + MAX_ROUNDS - 1 ))
echo "$TAG hop rounds ${START_ROUND}..${END_ROUND} (stop after ${PATIENCE} round(s)" \
     "without a gain > ${IMPROVE_TOL})"
idle=0
status="not exhausted (MAX_ROUNDS reached)"
for round in $(seq "$START_ROUND" "$END_ROUND"); do
    jid=$(STAGE=hop ROUND="$round" sbatch --parsable --wait scripts/model10_dext_scan.slurm.sh) \
        || echo "$TAG warning: some round-${round} tasks failed; continuing"
    jid="${jid%%;*}"
    logs="logs/m10dext_m10_dext_${jid}_*.out"
    # shellcheck disable=SC2086
    n_done=$(cat $logs 2>/dev/null | grep -c 'saved ' || true)
    # shellcheck disable=SC2086
    n_imp=$(cat $logs 2>/dev/null | grep -c 'IMPROVED' || true)
    echo "$TAG hop round ${round} (job ${jid}): ${n_done} replica(s) done," \
         "${n_imp} lowered their (point, d_ext) by > ${IMPROVE_TOL}"
    if [ "$n_imp" -eq 0 ]; then
        idle=$(( idle + 1 ))
        if [ "$idle" -ge "$PATIENCE" ]; then
            status="EXHAUSTED: ${idle} hop round(s) in a row improved nothing"
            break
        fi
    else
        idle=0
    fi
done
echo "$TAG ${status}"

# ---- 3. collect + figure ------------------------------------------------
cid=$(sbatch --parsable scripts/model10_dext_scan_collect.slurm.sh)
cid="${cid%%;*}"
echo "$TAG collect job ${cid} submitted -> table in logs/m10dext_io_m10_dext_collect_${cid}.out," \
     "figure ${OUT_PNG:-results_model10_rate/model10_opt_rate_vs_dext.png}"
