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
#      MODELS="model3" RECOMPUTE="opt_fra_6" bash scripts/dtbase_line_submit_all.sh
#      MODELS="model3" RECOMPUTE="all"       bash scripts/dtbase_line_submit_all.sh
#
#  RECOMPUTE forces the named measures to be recomputed even where a finite
#  up-to-date value is already stored (both the resume scan and the worker
#  honour it).  It is NOT needed for measures whose stored generation is below
#  MEASURE_GEN -- those are detected and upgraded automatically.
#
#  Long-running: submits over hours, so run it detached, e.g.
#      nohup bash -c 'MODELS="model3" bash scripts/dtbase_line_submit_all.sh' \
#            > logs/submit_driver.log 2>&1 &
# ============================================================
# NOTE: deliberately NOT 'set -e'.  An earlier version used 'set -euo pipefail',
# which made a single transient sbatch/squeue failure abort the whole submission
# loop silently (sbatch's output went to /dev/null, so the driver just returned
# to the prompt looking like it had finished) -- that is how a full-grid run
# ended up submitting only ~10% of its points.  Failures are now retried and
# tallied instead of being fatal.
set -uo pipefail

MODELS="${MODELS:-model3 model4}"     # models to sweep, in order
MAXJOBS="${MAXJOBS:-200}"             # cap on in-flight tasks (see COUNT_STATES)
POLL="${POLL:-30}"                    # seconds between throttle polls
# Must match the slurm script's own OUT_DIR default; exported so each sbatch
# writes where the resume scan looks.
OUT_DIR="${OUT_DIR:-results_dtbase_line}"
export OUT_DIR
SBATCH_RETRIES="${SBATCH_RETRIES:-5}"  # per-point sbatch retry budget
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

# Points still needing work, as "<p1> <p2>" lines: any point with a missing
# base_<idx>.npz or a missing/non-finite MEASURES key.  Computed in ONE python
# pass (not one per point) so re-running the driver to fill gaps is fast, and
# so an already-complete grid costs a single scan instead of thousands of
# no-op array submissions.
incomplete_points() {
    OUT_DIR="$OUT_DIR" STRIDE="$STRIDE" RECOMPUTE="${RECOMPUTE:-}" \
        python - "$1" <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from trotter_lindbladian_scan import MODELS
# point_needs_work applies exactly the worker's own rule (absent / non-finite /
# superseded generation / explicitly recomputed), so the driver and the worker
# can never disagree about which points still have work to do.
from trotter_dtbase_line_worker import point_tag, point_needs_work, _recompute_set

model = sys.argv[1]
stride = int(os.environ.get('STRIDE', '1'))
out_dir = Path(os.environ.get('OUT_DIR', 'results_dtbase_line'))
recompute = _recompute_set((os.environ.get('RECOMPUTE') or '').split())
m = MODELS[model]
for p1 in m.p1_vals[::stride]:
    for p2 in m.p2_vals[::stride]:
        p1f, p2f = float(p1), float(p2)
        if point_needs_work(out_dir / point_tag(model, p1f, p2f), recompute):
            print(f'{p1f:g} {p2f:g}')
PY
}

# sbatch with retries: a transient scheduler error must not abort the campaign.
n_failed=0
submit_point() {
    local model=$1 p1=$2 p2=$3 tries=0
    while [ "$tries" -lt "$SBATCH_RETRIES" ]; do
        if MODEL="$model" P1="$p1" P2="$p2" RECOMPUTE="${RECOMPUTE:-}" \
                sbatch "$SLURM_SCRIPT" >/dev/null 2>>logs/submit_errors.log; then
            return 0
        fi
        tries=$(( tries + 1 ))
        echo "" >&2
        echo "  [warn] sbatch failed for $model (p1=$p1 p2=$p2); retry ${tries}/${SBATCH_RETRIES} in ${POLL}s (see logs/submit_errors.log)" >&2
        sleep "$POLL"
    done
    echo "  [ERROR] giving up on $model (p1=$p1 p2=$p2) after ${SBATCH_RETRIES} attempts" >&2
    n_failed=$(( n_failed + 1 ))
    return 1
}

n_sub=0
for MODEL in $MODELS; do
    echo "[$MODEL] scanning ${OUT_DIR} for points still needing work..."
    TODO=$(incomplete_points "$MODEL")
    if [ -z "$TODO" ]; then
        echo "[$MODEL] nothing to do -- every point already has all measures."
        continue
    fi
    n_todo=$(wc -l <<< "$TODO")
    echo "[$MODEL] ${n_todo} point sweeps to submit"\
         "(${TASKS_PER_POINT} base tasks each; gate <= ${GATE}, cap ${MAXJOBS}, counting ${COUNT_STATES})"
    while read -r P1 P2; do
        [ -z "${P1:-}" ] && continue
        throttle
        submit_point "$MODEL" "$P1" "$P2"
        n_sub=$(( n_sub + 1 ))
        printf '\r  [%s] submitted %d/%d sweeps (last gamma=%s gamma_p=%s)      ' \
               "$MODEL" "$n_sub" "$n_todo" "$P1" "$P2"
    done <<< "$TODO"
    echo
done
echo "[done] submitted $n_sub per-point DT_BASE sweeps across: $MODELS"
if [ "$n_failed" -gt 0 ]; then
    echo "[WARNING] ${n_failed} point(s) could not be submitted -- see logs/submit_errors.log."
    echo "          Re-run this driver to retry them (it resumes automatically)."
fi
