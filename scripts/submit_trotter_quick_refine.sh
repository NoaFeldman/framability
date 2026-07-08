#!/bin/bash
# ============================================================
#  Quick neighbour-refinement driver for the Trotter Lindbladian scan.
#
#  For each requested model, run N_ROUNDS (default 20) chained QUICK
#  refinement rounds: only boundary points (opt_fra > 1 with a 4-connected
#  neighbour at the framable floor opt_fra == 1) are re-optimised, so each
#  round is cheap and the framable region grows at most one ring per round.
#  Models with no framable point are skipped (gate via
#  trotter_check_framable.py, override with FORCE=1).  Rounds are chained
#  with afterany (a single flaky task must not kill the remaining rounds;
#  rounds are incremental and read every earlier file) and followed by a
#  collect that merges all rounds and rebuilds the figure.
#
#  Requires the base scan (trotter_scan.slurm.sh) for each MODEL to have
#  finished.  Earlier full-refine rounds (pt_refine_r*) are read as well.
#
#  Usage:
#    bash scripts/submit_trotter_quick_refine.sh                  # all 5 models, gated
#    bash scripts/submit_trotter_quick_refine.sh model1 model5    # only these
#    bash scripts/submit_trotter_quick_refine.sh model2 N_ROUNDS=10
#    bash scripts/submit_trotter_quick_refine.sh model3 FORCE=1
#
#  Overridable via KEY=VALUE args or the environment:
#    OUT_DIR (results_trotter_v3), N_ROUNDS (20), TOL (1e-6, framable gate),
#    FORCE (0), plus anything the quick-refine job reads
#    (N_RESTARTS, FRA_MAXFEV_4/6, FRA_TOL, SEED).
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
N_ROUNDS=${N_ROUNDS:-20}
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

    # ── N_ROUNDS chained quick-refinement rounds ────────────────────────────────
    PREV=""
    for r in $(seq 1 "$N_ROUNDS"); do
        if [ -z "$PREV" ]; then
            JID=$(MODEL=$MODEL ROUND=$r OUT_DIR=$OUT_DIR \
                  sbatch --parsable scripts/trotter_scan_quick_refine.slurm.sh)
        else
            JID=$(MODEL=$MODEL ROUND=$r OUT_DIR=$OUT_DIR \
                  sbatch --parsable --dependency=afterany:$PREV \
                         scripts/trotter_scan_quick_refine.slurm.sh)
        fi
        echo "[$MODEL] quick refine round $r : job $JID${PREV:+  (after $PREV)}"
        PREV=$JID
    done

    # ── collect: merge all rounds (full + quick), regenerate the figure ─────────
    COLLECT=$(sbatch --parsable --dependency=afterany:$PREV \
        --job-name=trot_qcollect --ntasks=1 --cpus-per-task=1 --mem=4G \
        --time=00:30:00 --output=logs/trotqcol_%x_%A.out --error=logs/trotqcol_%x_%A.err \
        --wrap="source ${PWD}/.venv/bin/activate; cd ${PWD}; \
                export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; \
                python scripts/trotter_scan_refine_collect.py \
                    --model $MODEL --in_dir $OUT_DIR")
    echo "[$MODEL] collect figure : job $COLLECT  (after $PREV)"
    echo "[$MODEL] -> results_plots/trotter_${MODEL}.png"
done
