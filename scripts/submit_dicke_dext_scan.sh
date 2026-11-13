#!/bin/bash
# ============================================================
#  Best framability rate against d_ext = 4, 6, ..., 16 of the atom-only open
#  Dicke model at ZERO local dephasing (gamma_phi = 0), at
#  lam / lam_c = 0.1, 0.5, 1.0, 1.5, for every N in NS -- then the figure
#  results_model4_rate/dicke_dext_scan.png (+ _logy.png): x = d_ext,
#  y = best certified rate found (an upper bound on mu*), one line per
#  lam / lam_c, one panel per N.
#
#  Per N (MODEL = dicke_N<N>, the registry entry of
#  trotter_lindbladian_scan; points are grid values of its lam_ratio x
#  gamma_phi grid), the drivers running SIDE BY SIDE:
#    1. search   one N_TASKS-task array (scripts/dicke_dext_scan.slurm.sh,
#                STAGE=search): independent restarts, N_TASKS / 28 per
#                (lam, d_ext) combo, every COLD_EVERY-th one cold.  Seeded
#                replicas start from the seeded pipeline's d_ext 4 / 8 / 12
#                frames in results_dicke_N<N>_seeded (padded, reduced or
#                grown to the target d_ext), its rescaled-Pauli optimum and
#                the structured families
#    2. hop      N_TASKS-task arrays, rounds 1, 2, ...: cross-restart basin
#                hopping from the best frames so far (incl. the scan's best
#                frame of every smaller d_ext, padded, and the other lam's
#                best frames).  Stops after PATIENCE rounds in a row without
#                a gain > IMPROVE_TOL in any combo, or after MAX_ROUNDS
#  then, after every N:
#    3. collect  scripts/dicke_dext_scan_collect.slurm.sh: convergence table,
#                merge with the pipeline's d_ext 4 / 8 / 12, nesting in d_ext,
#                the figure (scripts/dicke_dext_scan_plot.py)
#
#  Every stage runs with `sbatch --wait`; start on the login node:
#
#      nohup bash scripts/submit_dicke_dext_scan.sh > logs/submit_dicke_dext.log 2>&1 &
#      NS="10 100" nohup bash scripts/submit_dicke_dext_scan.sh > logs/submit_dicke_dext_10_100.log 2>&1 &
#  More hop rounds later (search kept):
#      SKIP_SEARCH=1 MAX_ROUNDS=2 nohup bash scripts/submit_dicke_dext_scan.sh \
#          > logs/submit_dicke_dext_more.log 2>&1 &
#  Per-N driver logs: logs/submit_dicke_dext_N<N>.log.
#
#  Knobs (env; sbatch exports them to the slurm scripts): NS (5 10 20 100
#  1000), N_TASKS (196 per N = 7 replicas per combo), P1S (0.1 0.5 1.0 1.5),
#  P2S (0), D_EXTS (4 6 8 10 12 14 16), COLD_EVERY (3), FAMILIES,
#  FAM_SECONDS (3600), SEARCH_HOP_SECONDS (3600), HOP_SECONDS (5400),
#  POLISH_ITERS, SCALES, HOP_TOP, IMPROVE_TOL (1e-5), DE_POPSIZE, DE_MAXITER,
#  SEED, MIN_HITS (3), MIN_COLD_HITS (1), TOL_REL, TOL_ABS, OUT_PNG;
#  driver: SEARCH_TIME (24:00:00), HOP_TIME (12:00:00), MAX_ROUNDS (4),
#  PATIENCE (1), SKIP_SEARCH (0), PY (.venv/bin/python).
#  Re-running skips finished search files and continues the hop rounds from
#  the highest round on disk.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
NS="${NS:-5 10 20 100 1000}"
PY="${PY:-.venv/bin/python}"
export N_TASKS="${N_TASKS:-196}"
export P1S="${P1S:-0.1 0.5 1.0 1.5}"
export P2S="${P2S:-0}"
export D_EXTS="${D_EXTS:-4 6 8 10 12 14 16}"
export IMPROVE_TOL="${IMPROVE_TOL:-1e-5}"
SEARCH_TIME="${SEARCH_TIME:-24:00:00}"
HOP_TIME="${HOP_TIME:-12:00:00}"
MAX_ROUNDS="${MAX_ROUNDS:-4}"
PATIENCE="${PATIENCE:-1}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"

# ---- guards against real errors ---------------------------------------------
n_combos=$(( $(echo $P1S | wc -w) * $(echo $P2S | wc -w) * $(echo $D_EXTS | wc -w) ))
if [ "$N_TASKS" -lt "$n_combos" ]; then
    echo "ERROR: N_TASKS=${N_TASKS} < ${n_combos} (point, d_ext) combos (the worker needs one task per combo)"
    exit 1
fi
queued=$(squeue -u "$USER" -h -o '%j' 2>/dev/null || true)
for N in $NS; do
    "$PY" -c "
import sys; sys.path.insert(0, '.')
from trotter_lindbladian_scan import MODELS
s = MODELS['dicke_N$N']
for v in '$P1S'.split():
    assert min(abs(float(v) - x) for x in s.p1_vals) < 1e-9, f'lam_ratio {v} not on the grid'
for v in '$P2S'.split():
    assert min(abs(float(v) - x) for x in s.p2_vals) < 1e-9, f'gamma_phi {v} not on the grid'" \
        || { echo "ERROR: dicke_N${N} not registered, or P1S / P2S not on its grid"; exit 1; }
    if grep -qx "dx_dicke_N${N}" <<< "$queued"; then
        echo "ERROR: dx_dicke_N${N} jobs are still queued or running (stop that driver, then scancel)"
        exit 1
    fi
    [ -d "results_dicke_N${N}_seeded/dicke_N${N}_seeded" ] \
        || echo "note: results_dicke_N${N}_seeded/dicke_N${N}_seeded missing -- N=${N} starts without pipeline frames"
    if grep -q "^rs_dicke_N${N}_" <<< "$queued"; then
        echo "note: the seeded pipeline of N=${N} is still running -- seeds are partial (hop rounds re-read them)"
    fi
done
n_models=$(echo $NS | wc -w)
echo "===== d_ext scan: N = ${NS}; ${n_combos} combos x $(( N_TASKS / n_combos ))+ replicas," \
     "${N_TASKS} tasks per N, $(( n_models * N_TASKS )) in all ====="

# ---- one driver per N ---------------------------------------------------------
run_one() {       # run_one <N>   (search, then hop rounds until exhausted)
    local N="$1"
    export MODEL="dicke_N${N}"
    export OUT_DIR="results_${MODEL}_seeded"
    local JN="dx_${MODEL}" SCAN="$OUT_DIR/${MODEL}_dext_scan"
    local TAG="[${MODEL} dext scan]" ARR="--array=0-$(( N_TASKS - 1 ))"
    local jid n_files n_err logs n_done n_imp last start end round idle=0
    local status="not exhausted (MAX_ROUNDS reached)"
    mkdir -p "$SCAN"

    if [ "$SKIP_SEARCH" != "1" ]; then
        echo "$TAG search: ${N_TASKS} tasks, d_ext ${D_EXTS}, lam/lam_c ${P1S}, gamma_phi ${P2S}"
        jid=$(STAGE=search sbatch --parsable --wait --job-name="$JN" --time="$SEARCH_TIME" \
                  "$ARR" scripts/dicke_dext_scan.slurm.sh) \
            || echo "$TAG warning: some search tasks failed; continuing"
        jid="${jid%%;*}"
        n_files=$(find "$SCAN" -name 'search_r[0-9][0-9][0-9].npz' | wc -l)
        n_err=$(cat logs/dxdicke_"${JN}"_"${jid}"_*.out 2>/dev/null | grep -c 'ERROR' || true)
        echo "$TAG search (job ${jid}): ${n_files} restart file(s) on disk, ${n_err} error(s)"
    fi

    last=$(find "$SCAN" -name 'hop[0-9][0-9]_r[0-9][0-9][0-9].npz' 2>/dev/null \
           | sed 's/.*\/hop\([0-9][0-9]\)_r[0-9]*\.npz$/\1/' | sort -n | tail -1 || true)
    start=$(( 10#${last:-0} + 1 ))
    end=$(( start + MAX_ROUNDS - 1 ))
    [ "$end" -le 99 ] || end=99                      # 2-digit round tags
    echo "$TAG hop rounds ${start}..${end} (stop after ${PATIENCE} round(s) without a gain > ${IMPROVE_TOL})"
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
             "${n_imp} lowered their (lam, d_ext) by > ${IMPROVE_TOL}"
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
}

pids=()
for N in $NS; do
    echo "===== dicke_N${N}: start $(date); log logs/submit_dicke_dext_N${N}.log ====="
    ( run_one "$N" ) > "logs/submit_dicke_dext_N${N}.log" 2>&1 &
    pids+=("$!")
done
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
[ "$rc" = "0" ] || echo "warning: at least one driver failed; see logs/submit_dicke_dext_N*.log"

# ---- collect + figure -----------------------------------------------------------
cid=$(NS="$NS" sbatch --parsable scripts/dicke_dext_scan_collect.slurm.sh)
cid="${cid%%;*}"
echo "===== collect job ${cid} submitted $(date): table in logs/dxdicke_io_dx_dicke_collect_${cid}.out," \
     "figure ${OUT_PNG:-results_model4_rate/dicke_dext_scan.png} ====="
exit "$rc"
