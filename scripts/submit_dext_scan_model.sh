#!/bin/bash
# ============================================================
#  d_ext scan of ONE registered model at the grid points P1S x P2S:
#  best certified Heisenberg rate at every d_ext in D_EXTS, with a
#  convergence test -- the generic version of submit_dicke_dext_scan.sh
#  (same slurm script, scripts/dicke_dext_scan.slurm.sh, which is
#  model-agnostic; same worker, scripts/model10_dext_scan_worker.py).
#
#    1. search   one N_TASKS-task array: independent restarts, N_TASKS /
#                n_combos per (point, d_ext), every COLD_EVERY-th one cold.
#                Seeded replicas start from OUT_DIR/<MODEL>_seeded(_d12)
#                (padded / reduced / grown to the target d_ext)
#    2. hop      N_TASKS-task arrays, rounds 1, 2, ...: cross-restart basin
#                hopping; stops after PATIENCE rounds without a gain >
#                IMPROVE_TOL, or after MAX_ROUNDS
#    3. collect  scripts/model10_dext_scan_collect.py -> convergence table
#                (logs/dext_<MODEL>_collect_<jid>.out) and
#                OUT_DIR/<MODEL>_dext_scan/summary.npz
#
#  Every stage runs with `sbatch --wait`; start on the login node, e.g.
#
#      MODEL=model4 OUT_DIR=results_model4_seeded P1S="2.0 4.0" P2S="2.0 3.0" \
#          STORED_DIRS=results_model4_rate \
#          nohup bash scripts/submit_dext_scan_model.sh > logs/submit_dext_model4.log 2>&1 &
#
#  More hop rounds later: SKIP_SEARCH=1 MAX_ROUNDS=2 (same MODEL / P1S / P2S).
#
#  Knobs (env): MODEL (required), OUT_DIR (results_<MODEL>_seeded), P1S, P2S
#  (grid values; required), D_EXTS (4 6 8 10 12 14 16), N_TASKS (largest
#  multiple of the number of combos <= 200), STORED_DIRS (roots holding an
#  original six-rate scan <dir>/<MODEL>; default none), COLD_EVERY (3),
#  FAMILIES, FAM_SECONDS (3600), SEARCH_HOP_SECONDS (3600), HOP_SECONDS
#  (5400), POLISH_ITERS, SCALES, HOP_TOP, IMPROVE_TOL (1e-5), DE_POPSIZE,
#  DE_MAXITER, SEED, DIM; collect: MIN_HITS (3), MIN_COLD_HITS (1), TOL_REL,
#  TOL_ABS; driver: SEARCH_TIME (24:00:00), HOP_TIME (12:00:00), MAX_ROUNDS
#  (4), PATIENCE (1), SKIP_SEARCH (0), WAIT_SEEDED (1), PY (.venv/bin/python).
#
#  WAIT_SEEDED=1: before the search, wait until the model's seeded-rate run
#  (rs_<MODEL>_* jobs, scripts/submit_rate_seeded.sh) has nothing queued, so the
#  seeded replicas start from its final d_ext 8 / 12 frames
#  (scripts/wait_seeded_run.sh; ~5 min if no such run).
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
export MODEL="${MODEL:?set MODEL (e.g. model4)}"
export P1S="${P1S:?set P1S (grid values of the first axis)}"
export P2S="${P2S:?set P2S (grid values of the second axis)}"
export OUT_DIR="${OUT_DIR:-results_${MODEL}_seeded}"
export D_EXTS="${D_EXTS:-4 6 8 10 12 14 16}"
export STORED_DIRS="${STORED_DIRS:-}"
export FAM_SECONDS="${FAM_SECONDS:-3600}"
export IMPROVE_TOL="${IMPROVE_TOL:-1e-5}"
PY="${PY:-.venv/bin/python}"
SEARCH_TIME="${SEARCH_TIME:-24:00:00}"
HOP_TIME="${HOP_TIME:-12:00:00}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
PATIENCE="${PATIENCE:-1}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"

n_combos=$(( $(echo $P1S | wc -w) * $(echo $P2S | wc -w) * $(echo $D_EXTS | wc -w) ))
if [ "$n_combos" -gt 200 ]; then
    echo "ERROR: ${n_combos} (point, d_ext) combos > 200 array tasks (fewer points or d_ext)"
    exit 1
fi
export N_TASKS="${N_TASKS:-$(( 200 / n_combos * n_combos ))}"
if [ "$N_TASKS" -lt "$n_combos" ]; then
    echo "ERROR: N_TASKS=${N_TASKS} < ${n_combos} combos (the worker needs one task per combo)"
    exit 1
fi

# ---- guards: model registered, points on its grid, no duplicate driver ------
"$PY" -c "
import sys; sys.path.insert(0, '.')
from trotter_lindbladian_scan import MODELS
s = MODELS['$MODEL']
for v in '$P1S'.split():
    assert min(abs(float(v) - x) for x in s.p1_vals) < 1e-9, f'{s.p1_name} {v} not on the grid'
for v in '$P2S'.split():
    assert min(abs(float(v) - x) for x in s.p2_vals) < 1e-9, f'{s.p2_name} {v} not on the grid'" \
    || { echo "ERROR: ${MODEL} not registered, or P1S / P2S not on its grid"; exit 1; }
JN="dx_${MODEL}"
if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qx "$JN"; then
    echo "ERROR: ${JN} jobs are still queued or running (stop that driver, then scancel)"
    exit 1
fi
if [ "${WAIT_SEEDED:-1}" = "1" ] && [ "$SKIP_SEARCH" != "1" ]; then
    . scripts/wait_seeded_run.sh
    wait_seeded_run "$MODEL"
fi
[ -d "$OUT_DIR/${MODEL}_seeded" ] \
    || echo "note: $OUT_DIR/${MODEL}_seeded missing -- seeded replicas start without pipeline frames"

SCAN="$OUT_DIR/${MODEL}_dext_scan"
TAG="[${MODEL} dext scan]"
ARR="--array=0-$(( N_TASKS - 1 ))"
mkdir -p "$SCAN"
echo "$TAG ${n_combos} combos (points ${P1S} x ${P2S}, d_ext ${D_EXTS}), ${N_TASKS} tasks," \
     "$(( N_TASKS / n_combos )) replicas per combo; out ${SCAN}"

# ---- 1. independent restarts ------------------------------------------------------
if [ "$SKIP_SEARCH" != "1" ]; then
    jid=$(STAGE=search sbatch --parsable --wait --job-name="$JN" --time="$SEARCH_TIME" \
              "$ARR" scripts/dicke_dext_scan.slurm.sh) \
        || echo "$TAG warning: some search tasks failed; continuing"
    jid="${jid%%;*}"
    n_files=$(find "$SCAN" -name 'search_r[0-9][0-9][0-9].npz' | wc -l)
    n_err=$(cat logs/dxdicke_"${JN}"_"${jid}"_*.out 2>/dev/null | grep -c 'ERROR' || true)
    echo "$TAG search (job ${jid}): ${n_files} restart file(s) on disk, ${n_err} error(s)"
fi

# ---- 2. cross-restart basin hopping, until exhausted -----------------------------
last=$(find "$SCAN" -name 'hop[0-9][0-9]_r[0-9][0-9][0-9].npz' 2>/dev/null \
       | sed 's/.*\/hop\([0-9][0-9]\)_r[0-9]*\.npz$/\1/' | sort -n | tail -1 || true)
start=$(( 10#${last:-0} + 1 ))
end=$(( start + MAX_ROUNDS - 1 ))
[ "$end" -le 99 ] || end=99                          # 2-digit round tags
echo "$TAG hop rounds ${start}..${end} (stop after ${PATIENCE} round(s) without a gain > ${IMPROVE_TOL})"
idle=0
status="not exhausted (MAX_ROUNDS reached)"
for round in $(seq "$start" "$end"); do
    jid=$(STAGE=hop ROUND="$round" sbatch --parsable --wait --job-name="$JN" \
              --time="$HOP_TIME" "$ARR" scripts/dicke_dext_scan.slurm.sh) \
        || echo "$TAG warning: some round-${round} tasks failed; continuing"
    jid="${jid%%;*}"
    logs="logs/dxdicke_${JN}_${jid}_*.out"
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

# ---- 3. collect (waits, so a wrapper can chain the paper figure) -----------------
# shellcheck disable=SC2086
cid=$(sbatch --parsable --wait --job-name="dx_${MODEL}_collect" --time=00:30:00 --mem=4G \
          --output="logs/dext_${MODEL}_collect_%j.out" \
          --wrap "${PY} scripts/model10_dext_scan_collect.py --model ${MODEL} \
                  --out_dir ${OUT_DIR} --stored_dirs ${STORED_DIRS} \
                  --p1 ${P1S} --p2 ${P2S} --d_exts ${D_EXTS} \
                  --min_hits ${MIN_HITS:-3} --min_cold_hits ${MIN_COLD_HITS:-1} \
                  --tol_rel ${TOL_REL:-1e-4} --tol_abs ${TOL_ABS:-1e-5}") \
    || echo "$TAG warning: collect failed"
cid="${cid%%;*}"
echo "$TAG collect (job ${cid}): table in logs/dext_${MODEL}_collect_${cid}.out," \
     "summary ${SCAN}/summary.npz"
