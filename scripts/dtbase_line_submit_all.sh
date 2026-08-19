#!/bin/bash
# ============================================================
#  Submit a full DT_BASE line sweep for every (gamma, gamma') grid point of
#  model3 then model4 -- exactly the grid trotter_lindbladian_scan scans
#  (p1 = gamma = arange(0,10,0.2), p2 = gamma' = arange(0,10,0.2); 51x51 = 2601
#  points per model).
#
#  For each point this fires the existing per-point job array
#      MODEL=<m> P1=<gamma> P2=<gamma'> sbatch scripts/trotter_dtbase_line.slurm.sh
#  whose 0-9 array sweeps the bottom 10 DT_BASE values (item 1 of the
#  dt-extrapolation pipeline redesign).  The worker is idempotent per-KEY (it
#  loads any existing base_<idx>.npz and computes only the MEASURES entries
#  still missing from it -- see trotter_dtbase_line_worker._missing_keys), so
#  the driver is safe to re-run both to fill grid gaps AND to backfill newly
#  added measures (e.g. sch_fra_6/prod_fra_10) onto already-swept points
#  without recomputing anything already stored.
#
#  200-job cap: before each per-point submission the driver blocks until the
#  user's queued+running task count drops below MAXJOBS, so no more than ~MAXJOBS
#  array tasks are ever in flight (the "map cleanly to 200 jobs" constraint).
#
#  Usage:
#      bash scripts/dtbase_line_submit_all.sh
#      MODELS="model3" MAXJOBS=150 bash scripts/dtbase_line_submit_all.sh
#      MODELS="model3" STRIDE=2 bash scripts/dtbase_line_submit_all.sh   # half res
# ============================================================
set -euo pipefail

MODELS="${MODELS:-model3 model4}"     # models to sweep, in order
MAXJOBS="${MAXJOBS:-200}"             # cap on in-flight tasks (see COUNT_STATES)
POLL="${POLL:-30}"                    # seconds between throttle polls
# States counted toward MAXJOBS.  'pending,running' (default) caps the total in
# the queue -- use it when the cluster limits *submitted* jobs.  'running' caps
# only what is executing and lets a backlog sit pending -- use it when the
# cluster caps concurrent *running* jobs below MAXJOBS (tasks stay PENDING), so
# the driver keeps feeding the queue instead of blocking on pending tasks.
COUNT_STATES="${COUNT_STATES:-pending,running}"
# Tasks added per point submission = the array size of the slurm script (0-98 ->
# 99).  The gate leaves this much headroom so a submission never overshoots
# MAXJOBS.  Auto-read from the slurm file; override if you edit the array range.
SLURM_SCRIPT="scripts/trotter_dtbase_line.slurm.sh"
TASKS_PER_POINT="${TASKS_PER_POINT:-$(
    awk -F'=' '/^#SBATCH --array=/{n=$2; sub(/%.*/,"",n); split(n,a,"-");
               print (a[2] != "" ? a[2]-a[1]+1 : 1); exit}' "$SLURM_SCRIPT")}"
TASKS_PER_POINT="${TASKS_PER_POINT:-99}"
GATE=$(( MAXJOBS - TASKS_PER_POINT ))          # submit only when in-flight <= GATE
[ "$GATE" -lt 0 ] && GATE=0

cd "$(dirname "$0")/.."               # repo root
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p logs results_dtbase_line

# Grid values straight from the model spec (single source of truth) --------
# STRIDE=n takes every n-th value of each axis, i.e. multiplies the grid step
# by n (STRIDE=2 halves resolution).  Values are a subset of the full grid
# (vals[::stride], starting at index 0), so each retained point keeps the exact
# same p1/p2 float as the full-resolution run -> the same point_tag() -> the
# same results_dtbase_line/<tag>/ directory.  Points already computed there are
# skipped by the worker's existing per-base _is_current() check; points dropped
# by the stride are simply never (re)submitted.
STRIDE="${STRIDE:-1}"                 # 1 = full grid, 2 = half resolution, ...
grid_vals() {   # $1 = model, $2 = p1|p2
    python - "$1" "$2" "$STRIDE" <<'PY'
import sys
from trotter_lindbladian_scan import MODELS
m = MODELS[sys.argv[1]]
vals = m.p1_vals if sys.argv[2] == 'p1' else m.p2_vals
stride = int(sys.argv[3])
print(' '.join(f'{float(v):g}' for v in vals[::stride]))
PY
}

# In-flight task count for this user in the counted states (all jobs, so the cap
# is genuinely global across every per-point array we submit).
inflight() { squeue -u "$USER" -h -r -t "$COUNT_STATES" | wc -l; }

# Block until there is room for another full array (in-flight <= GATE), so the
# next submission lands at or below MAXJOBS instead of overshooting by an array.
throttle() {
    local n
    n=$(inflight)
    while [ "$n" -gt "$GATE" ]; do
        echo "  [throttle] ${n} ${COUNT_STATES} tasks; need <= ${GATE} (cap ${MAXJOBS}); waiting ${POLL}s..."
        sleep "$POLL"
        n=$(inflight)
    done
}

n_sub=0
for MODEL in $MODELS; do
    P1S=$(grid_vals "$MODEL" p1)
    P2S=$(grid_vals "$MODEL" p2)
    n_p1=$(wc -w <<< "$P1S"); n_p2=$(wc -w <<< "$P2S")
    echo "[$MODEL] submitting ${n_p1} x ${n_p2} = $(( n_p1 * n_p2 )) point sweeps"\
         "(${TASKS_PER_POINT} base tasks each; gate <= ${GATE}, cap ${MAXJOBS}, counting ${COUNT_STATES})"
    for P1 in $P1S; do
        for P2 in $P2S; do
            throttle
            MODEL="$MODEL" P1="$P1" P2="$P2" sbatch "$SLURM_SCRIPT" >/dev/null
            n_sub=$(( n_sub + 1 ))
            printf '\r  [%s] submitted %d sweeps (last gamma=%s gamma_p=%s)      ' \
                   "$MODEL" "$n_sub" "$P1" "$P2"
        done
    done
    echo
done
echo "[done] submitted $n_sub per-point DT_BASE sweeps across: $MODELS"
