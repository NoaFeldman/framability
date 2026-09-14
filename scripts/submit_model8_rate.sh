#!/bin/bash
# ============================================================
#  Full model8 panel pipeline (the model4 rate figure, for model8, plus Q):
#
#    1. rates      scripts/model4_rate_panels.slurm.sh  MODEL=model8  (0-199)
#                  six framability rates, optimisers seeded with X-Y rings
#    2. 8q panels  scripts/model4_manybody.slurm.sh     MODEL=model8  (0-120)
#                  osc rate + gap of the 2x4 lattice Lindbladian
#    3. Q          scripts/liouvillian_q.slurm.sh       MODEL=model8  (0-199)
#                  bond + 2x3-lattice quality factor
#    4. collect    scripts/model4_rate_panels_collect.slurm.sh
#                  -> results_model8_rate/model8_rate_panels.{npz,png}
#
#  The stages run as ONE sequential chain (`sbatch --wait`), like
#  submit_model4_rate_quick_refine.sh, so at most one array (<= 200 tasks) is
#  in the queue at a time.  Run it on the login node inside tmux / nohup:
#
#      nohup bash scripts/submit_model8_rate.sh > logs/submit_model8.log 2>&1 &
#
#  Every worker skips points already on disk, so re-running the script after a
#  failure only fills holes.  SKIP_RATES=1 / SKIP_MB=1 / SKIP_Q=1 skip a stage.
#  Q contours: 1, cot(pi/8) = 2.414 (octagon), cot(pi/12) = 3.732 (12-gon).
# ============================================================
set -euo pipefail

MODEL=model8
OUT_DIR="${OUT_DIR:-results_model8_rate}"
Q_DIR="${Q_DIR:-results_liouvillian_q}"
Q_LEVELS="${Q_LEVELS:-1 2.414 3.732}"

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs "$OUT_DIR" "$Q_DIR"

if [ "${SKIP_RATES:-0}" != "1" ]; then
    echo "[model8] stage 1/4: framability rates"
    MODEL=$MODEL OUT_DIR="$OUT_DIR" \
        sbatch --wait --job-name=m8_rate scripts/model4_rate_panels.slurm.sh
fi

if [ "${SKIP_MB:-0}" != "1" ]; then
    echo "[model8] stage 2/4: 8-qubit many-body panels"
    MODEL=$MODEL OUT_DIR="$OUT_DIR" \
        sbatch --wait --job-name=m8_8q scripts/model4_manybody.slurm.sh
fi

if [ "${SKIP_Q:-0}" != "1" ]; then
    echo "[model8] stage 3/4: Lindbladian quality factor"
    MODEL=$MODEL OUT_DIR="$Q_DIR" \
        sbatch --wait --job-name=m8_q scripts/liouvillian_q.slurm.sh
fi

echo "[model8] stage 4/4: collect + figure"
MODEL=$MODEL IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" Q_DIR="$Q_DIR" Q_LEVELS="$Q_LEVELS" \
    sbatch --wait --job-name=m8_collect scripts/model4_rate_panels_collect.slurm.sh

echo "[model8] done: $OUT_DIR/model8_rate_panels.png"
