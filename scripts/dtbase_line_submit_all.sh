#!/bin/bash
# ============================================================
#  Submit a full DT_BASE line sweep for every (gamma, gamma') grid point of
#  model3 then model4 -- exactly the grid trotter_lindbladian_scan scans
#  (p1 = gamma = arange(0,10,0.2), p2 = gamma' = arange(0,10,0.2); 51x51 = 2601
#  points per model).
#
#  For each point this fires the existing per-point job array
#      MODEL=<m> P1=<gamma> P2=<gamma'> sbatch scripts/trotter_dtbase_line.slurm.sh
#  whose 0-98 array sweeps the 99 DT_BASE values.  The worker is idempotent
#  (skips any base_<idx>.npz already at the current code version), so the driver
#  is safe to re-run to fill gaps.
#
#  200-job cap: before each per-point submission the driver blocks until the
#  user's queued+running task count drops below MAXJOBS, so no more than ~MAXJOBS
#  array tasks are ever in flight (the "map cleanly to 200 jobs" constraint).
#
#  Usage:
#      bash scripts/dtbase_line_submit_all.sh
#      MODELS="model3" MAXJOBS=150 bash scripts/dtbase_line_submit_all.sh
# ============================================================
set -euo pipefail

MODELS="${MODELS:-model3 model4}"     # models to sweep, in order
MAXJOBS="${MAXJOBS:-200}"             # global queued+running task cap
POLL="${POLL:-30}"                    # seconds between throttle polls
SLURM_SCRIPT="scripts/trotter_dtbase_line.slurm.sh"

cd "$(dirname "$0")/.."               # repo root
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p logs results_dtbase_line

# Grid values straight from the model spec (single source of truth) --------
grid_vals() {   # $1 = model, $2 = p1|p2
    python - "$1" "$2" <<'PY'
import sys
from trotter_lindbladian_scan import MODELS
m = MODELS[sys.argv[1]]
vals = m.p1_vals if sys.argv[2] == 'p1' else m.p2_vals
print(' '.join(f'{float(v):g}' for v in vals))
PY
}

# Current queued+running task count for this user (all jobs, so the cap is
# genuinely global across every per-point array we submit).
running_tasks() { squeue -u "$USER" -h -r -t pending,running | wc -l; }

throttle() {
    while [ "$(running_tasks)" -ge "$MAXJOBS" ]; do
        echo "  [throttle] $(running_tasks) tasks in flight >= $MAXJOBS; waiting ${POLL}s..."
        sleep "$POLL"
    done
}

n_sub=0
for MODEL in $MODELS; do
    P1S=$(grid_vals "$MODEL" p1)
    P2S=$(grid_vals "$MODEL" p2)
    n_p1=$(wc -w <<< "$P1S"); n_p2=$(wc -w <<< "$P2S")
    echo "[$MODEL] submitting ${n_p1} x ${n_p2} = $(( n_p1 * n_p2 )) point sweeps (99 base tasks each)"
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
