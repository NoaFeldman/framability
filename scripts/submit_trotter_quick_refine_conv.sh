#!/bin/bash
# ============================================================
#  Convergence-driven quick neighbour-refinement driver for the Trotter
#  Lindbladian scan: keep running QUICK refinement rounds (boundary points
#  only, 200-task arrays) until a round changes NOTHING in the optimised
#  framability data, then collect and rebuild the figure.
#
#  Unlike submit_trotter_quick_refine.sh (fixed N_ROUNDS chained up front),
#  this driver submits ONE round plus a small chain-controller job
#  (trotter_quick_refine_chain.slurm.sh).  After each round the controller
#  compares the round's pt_qrefine files against the previous best
#  (trotter_quick_refine_check.py); if any opt_fra_4 / opt_fra_6 improved
#  it submits the next round + the next controller, otherwise it submits
#  the final collect.  MAX_ROUNDS (default 100) is a safety cap only.
#
#  The first round starts AFTER the highest already-existing quick round
#  in <OUT_DIR>/<model>/, so it composes with earlier fixed-round runs.
#  Models with no framable point (opt_fra never reaches 1) are skipped via
#  trotter_check_framable.py; override with FORCE=1.
#
#  Requires the base scan (trotter_scan.slurm.sh) for each MODEL to have
#  finished.
#
#  Usage:
#    bash scripts/submit_trotter_quick_refine_conv.sh                # all 5 models, gated
#    bash scripts/submit_trotter_quick_refine_conv.sh model3 model4  # only these
#    bash scripts/submit_trotter_quick_refine_conv.sh model3 MAX_ROUNDS=150
#    bash scripts/submit_trotter_quick_refine_conv.sh model4 FORCE=1
#
#  Overridable via KEY=VALUE args or the environment:
#    OUT_DIR (results_trotter_v3), MAX_ROUNDS (100),
#    CHECK_TOL (1e-9, "no change" threshold, matches the collect merge TOL),
#    TOL (1e-6, framable gate), FORCE (0), plus anything the quick-refine
#    array reads (N_RESTARTS, FRA_MAXFEV_4/6, FRA_TOL, SEED).
# ============================================================
set -euo pipefail

# Split positional model names from KEY=VALUE overrides.
MODELS=()
for a in "$@"; do
    case "$a" in
        *=*) export "$a" ;;
        *)   MODELS+=("$a") ;;
    esac
done
[ ${#MODELS[@]} -eq 0 ] && MODELS=(model1 model2 model3 model4 model5)

OUT_DIR=${OUT_DIR:-results_trotter_v3}
MAX_ROUNDS=${MAX_ROUNDS:-100}
CHECK_TOL=${CHECK_TOL:-1e-9}
TOL=${TOL:-1e-6}
FORCE=${FORCE:-0}

source "${PWD}/.venv/bin/activate"
mkdir -p logs results_plots

for MODEL in "${MODELS[@]}"; do
    # ── gate: only refine models whose opt framability reaches 1 somewhere ──────
    if [ "$FORCE" != "1" ]; then
        if ! python scripts/trotter_check_framable.py \
                --model "$MODEL" --in_dir "$OUT_DIR" --tol "$TOL"; then
            echo "[$MODEL] no framable point -> skipped"
            continue
        fi
    else
        python scripts/trotter_check_framable.py \
            --model "$MODEL" --in_dir "$OUT_DIR" --tol "$TOL" || true
        echo "[$MODEL] FORCE=1 -> refining regardless"
    fi

    # ── continue after the highest already-existing quick round ─────────────────
    LAST=$(ls "$OUT_DIR/$MODEL"/pt_qrefine_r*_*.npz 2>/dev/null \
           | sed -E 's/.*pt_qrefine_r([0-9]+)_.*/\1/' | sort -n | tail -1 || true)
    LAST=${LAST:-0}
    START=$((10#$LAST + 1))
    [ "$START" -gt 1 ] && echo "[$MODEL] found existing quick rounds up to $((10#$LAST)) -> starting at round $START"
    if [ "$START" -gt "$MAX_ROUNDS" ]; then
        echo "[$MODEL] START=$START exceeds MAX_ROUNDS=$MAX_ROUNDS -> raise MAX_ROUNDS; skipped"
        continue
    fi

    # ── first round + chain controller; the controller keeps the chain going ────
    JID=$(MODEL=$MODEL ROUND=$START OUT_DIR=$OUT_DIR \
          sbatch --parsable scripts/trotter_scan_quick_refine.slurm.sh)
    echo "[$MODEL] quick refine round $START : job $JID"
    CHAIN=$(MODEL=$MODEL ROUND=$START OUT_DIR=$OUT_DIR \
            MAX_ROUNDS=$MAX_ROUNDS CHECK_TOL=$CHECK_TOL \
            sbatch --parsable --dependency=afterany:$JID \
                   scripts/trotter_quick_refine_chain.slurm.sh)
    echo "[$MODEL] chain check after round $START : job $CHAIN  (after $JID)"
    echo "[$MODEL] rounds continue automatically until no change (cap $MAX_ROUNDS);"
    echo "[$MODEL] final collect -> results_plots/trotter_${MODEL}.png"
done
