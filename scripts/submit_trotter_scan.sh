#!/bin/bash
# ============================================================
#  End-to-end driver for one Trotter Lindbladian model:
#    base scan -> 5 neighbour-refinement rounds -> collect figure,
#  all chained with SLURM afterok dependencies.
#
#  Usage:
#    bash scripts/submit_trotter_scan.sh model1
#    bash scripts/submit_trotter_scan.sh model4 DT=0.1 DIM=1
#
#  Run once per model (model1..model4) to produce the 4 figures.
#  Any KEY=VALUE after the model name is exported to every job (e.g. DT, DIM,
#  FRA_MAXFEV_4, SEED).
# ============================================================
set -euo pipefail

MODEL=${1:-model1}
shift || true
for kv in "$@"; do export "$kv"; done
export MODEL

OUT_DIR=${OUT_DIR:-results_trotter}
N_ROUNDS=${N_ROUNDS:-5}
mkdir -p logs "${OUT_DIR}/${MODEL}" results_plots

# ── base scan ────────────────────────────────────────────────────────────────
JID=$(sbatch --parsable scripts/trotter_scan.slurm.sh)
echo "[$MODEL] base scan      : job $JID"

# ── 5 sequential refinement rounds (each waits for the previous) ──────────────
PREV=$JID
for r in $(seq 1 "$N_ROUNDS"); do
    JID=$(ROUND=$r sbatch --parsable --dependency=afterok:$PREV \
              scripts/trotter_scan_refine.slurm.sh)
    echo "[$MODEL] refine round $r : job $JID  (after $PREV)"
    PREV=$JID
done

# ── collect: merge all rounds, regenerate the figure ─────────────────────────
COLLECT=$(sbatch --parsable --dependency=afterok:$PREV \
    --job-name=trot_collect --ntasks=1 --cpus-per-task=1 --mem=4G \
    --time=00:30:00 --output=logs/trotcol_%x_%A.out --error=logs/trotcol_%x_%A.err \
    --wrap="source ${PWD}/.venv/bin/activate; cd ${PWD}; \
            export MPLCONFIGDIR=/tmp/mpl-\$SLURM_JOB_ID; \
            python scripts/trotter_scan_refine_collect.py \
                --model $MODEL --in_dir $OUT_DIR --max_round $N_ROUNDS")
echo "[$MODEL] collect figure : job $COLLECT  (after $PREV)"
echo "[$MODEL] -> results_plots/trotter_${MODEL}.png"
