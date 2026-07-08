#!/bin/bash
# ============================================================
#  Gated neighbour-refinement driver for the Trotter Lindbladian scan.
#
#  For each requested model, refine the optimised framabilities (opt_fra_4 /
#  opt_fra_6) with N_ROUNDS neighbour-seeded rounds (default 6) -- but ONLY if
#  the model's opt-framability plot already has at least one framable point
#  (min opt framability reaches 1, checked by trotter_check_framable.py).  Models
#  with no framable point are skipped.  Each model's rounds are chained with
#  SLURM afterok dependencies and followed by a collect that rebuilds the figure.
#
#  Requires the base scan (trotter_scan.slurm.sh) for each MODEL to have finished.
#
#  Usage:
#    bash scripts/submit_trotter_refine.sh                    # all 5 models, auto-gated
#    bash scripts/submit_trotter_refine.sh model1 model5      # only these (still gated)
#    bash scripts/submit_trotter_refine.sh model2 N_ROUNDS=6  # KEY=VALUE overrides
#    bash scripts/submit_trotter_refine.sh model3 FORCE=1     # refine even if not framable
#
#  Overridable via KEY=VALUE args or the environment:
#    OUT_DIR (results_trotter_v3), N_ROUNDS (6), TOL (1e-6, framable threshold),
#    FORCE (0), plus anything the refine job reads (N_RESTARTS, FRA_MAXFEV_4/6, SEED).
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
N_ROUNDS=${N_ROUNDS:-6}
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

    # ── N_ROUNDS chained refinement rounds ──────────────────────────────────────
    PREV=""
    for r in $(seq 1 "$N_ROUNDS"); do
        if [ -z "$PREV" ]; then
            JID=$(MODEL=$MODEL ROUND=$r OUT_DIR=$OUT_DIR \
                  sbatch --parsable scripts/trotter_scan_refine.slurm.sh)
        else
            JID=$(MODEL=$MODEL ROUND=$r OUT_DIR=$OUT_DIR \
                  sbatch --parsable --dependency=afterok:$PREV \
                         scripts/trotter_scan_refine.slurm.sh)
        fi
        echo "[$MODEL] refine round $r : job $JID${PREV:+  (after $PREV)}"
        PREV=$JID
    done

    # ── collect: merge all rounds, regenerate the figure ────────────────────────
    COLLECT=$(sbatch --parsable --dependency=afterok:$PREV \
        --job-name=trot_collect --ntasks=1 --cpus-per-task=1 --mem=4G \
        --time=00:30:00 --output=logs/trotcol_%x_%A.out --error=logs/trotcol_%x_%A.err \
        --wrap="source ${PWD}/.venv/bin/activate; cd ${PWD}; \
                export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; \
                python scripts/trotter_scan_refine_collect.py \
                    --model $MODEL --in_dir $OUT_DIR --max_round $N_ROUNDS")
    echo "[$MODEL] collect figure : job $COLLECT  (after $PREV)"
    echo "[$MODEL] -> results_plots/trotter_${MODEL}.png"
done
