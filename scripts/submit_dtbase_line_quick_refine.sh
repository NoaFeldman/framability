#!/bin/bash
# ============================================================
#  Item 5: run all 10 quick-refine rounds (opt_fra_4/opt_fra_6, "quick"
#  neighbor-refining -- only points touching a framability==1 neighbor) over
#  every (model, base_idx) DT_BASE-line grid.
#
#  Each (model, base_idx) is an independent 10-round chain: round r must wait
#  for round r-1 to finish (it reads r-1's output), so each chain submits its
#  rounds via `sbatch --wait` sequentially.  Different chains are independent
#  of each other, so up to MAX_PARALLEL_CHAINS chains run at once (each
#  chain's active round is itself a <=200-task array, so with the default
#  MAX_PARALLEL_CHAINS this keeps in-flight tasks <= 200*MAX_PARALLEL_CHAINS;
#  lower MAX_PARALLEL_CHAINS or N_CHUNKS if your cluster caps total in-flight
#  jobs below that).
#
#  Usage:
#      bash scripts/submit_dtbase_line_quick_refine.sh
#      MODELS="model3" MAX_PARALLEL_CHAINS=4 bash scripts/submit_dtbase_line_quick_refine.sh
#      N_ROUNDS=10 N_CHUNKS=100 bash scripts/submit_dtbase_line_quick_refine.sh
# ============================================================
set -euo pipefail

MODELS="${MODELS:-model3 model4}"
N_ROUNDS="${N_ROUNDS:-10}"                 # "10 stages of neighbor refining"
N_BASE="${N_BASE:-10}"                     # bottom-10 DT_BASE values (item 1)
MAX_PARALLEL_CHAINS="${MAX_PARALLEL_CHAINS:-2}"
N_CHUNKS="${N_CHUNKS:-200}"
SLURM_SCRIPT="scripts/trotter_dtbase_line_quick_refine.slurm.sh"

cd "$(dirname "$0")/.."               # repo root
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p logs results_dtbase_line

# One (model, base_idx) chain: 10 sequential `sbatch --wait` array submissions.
run_chain() {
    local model=$1 base_idx=$2
    local round
    for round in $(seq 1 "$N_ROUNDS"); do
        echo "[chain $model/base_idx=$base_idx] round $round/${N_ROUNDS}: submitting..."
        MODEL="$model" BASE_IDX="$base_idx" ROUND="$round" N_CHUNKS="$N_CHUNKS" \
            sbatch --wait "$SLURM_SCRIPT"
        echo "[chain $model/base_idx=$base_idx] round $round/${N_ROUNDS}: done"
    done
}

# Bounded-concurrency job pool: at most MAX_PARALLEL_CHAINS run_chain() calls
# in flight at once (each is a background subshell; `wait -n` reaps the first
# to finish before launching the next).
n_running=0
for MODEL in $MODELS; do
    for ((base_idx = 0; base_idx < N_BASE; base_idx++)); do
        run_chain "$MODEL" "$base_idx" &
        n_running=$((n_running + 1))
        if [ "$n_running" -ge "$MAX_PARALLEL_CHAINS" ]; then
            wait -n
            n_running=$((n_running - 1))
        fi
    done
done
wait
echo "[done] all (model, base_idx) quick-refine chains complete: $MODELS x ${N_BASE} base indices x ${N_ROUNDS} rounds"
