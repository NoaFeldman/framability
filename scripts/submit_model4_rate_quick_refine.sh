#!/bin/bash
# ============================================================
#  Run all 10 quick neighbour-refine rounds of the optimised Heisenberg
#  framability RATES (rate_heis_4 / rate_heis_6) over one rate-panel model's
#  grid (MODEL, default model4).
#
#  Round r reads the base scan plus every earlier round, so the rounds form
#  ONE sequential chain: each is submitted with `sbatch --wait` and the next
#  starts only after it finishes.  Each round is itself a <=200-task array, so
#  in-flight tasks stay at the 200-job cap throughout.
#
#  Ten rounds propagate the framable floor up to ten grid rings (2.0 in gamma
#  at model4's 0.2 grid step) outward from wherever it already sits.
#
#  RESUMING.  Round numbers must keep increasing: a point that already wrote
#  pt_..._qrefine_r03.npz is skipped forever at round 3 (the worker's
#  out.exists() guard), and the per-point seed is 100000*round, so re-running
#  the same round numbers would re-do frozen work with identical seeds.  So
#  START_ROUND defaults to one past the highest round already on disk -- just
#  run this script again for 10 MORE rounds and it continues at 11, 12, ...
#  Rounds are zero-padded to two digits, so 99 is the ceiling.
#
#  Usage:
#      bash scripts/submit_model4_rate_quick_refine.sh              # next 10 rounds
#      N_ROUNDS=5 bash scripts/submit_model4_rate_quick_refine.sh   # next 5
#      START_ROUND=1 bash scripts/submit_model4_rate_quick_refine.sh  # force from 1
#      N_RESTARTS=5 MAXFEV_4=2000 bash scripts/submit_model4_rate_quick_refine.sh
#      MODEL=model10 bash scripts/submit_model4_rate_quick_refine.sh
#        (model10 = the Shibata-Katsura dissipative quantum Ising chain,
#         https://arxiv.org/abs/1904.12505; OUT_DIR defaults to results_<model>_rate)
#
#  Then replot (picks the min over base + every round automatically):
#      python scripts/model4_rate_panels_collect.py --model <model>
# ============================================================
set -euo pipefail

MODEL="${MODEL:-model4}"
N_ROUNDS="${N_ROUNDS:-10}"
N_CHUNKS="${N_CHUNKS:-200}"
OUT_DIR="${OUT_DIR:-results_${MODEL}_rate}"
SLURM_SCRIPT="scripts/model4_rate_quick_refine.slurm.sh"
JOB_NAME="m${MODEL#model}_qrefine"    # model4 -> m4_qrefine, the script's default
TAG="[m${MODEL#model} rate qrefine]"

cd "$(dirname "$0")/.."               # repo root
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p logs "$OUT_DIR"

# Resume point: one past the highest round already written.  10# forces base-10
# so that "08"/"09" are not parsed as invalid octal.  `|| true`: a model whose
# point directory does not exist yet starts at round 1 instead of tripping
# `set -e` through the failing find.
if [ -z "${START_ROUND:-}" ]; then
    last=$(find "$OUT_DIR/$MODEL" -name '*_qrefine_r[0-9][0-9].npz' 2>/dev/null \
           | sed 's/.*_qrefine_r\([0-9][0-9]\)\.npz$/\1/' | sort -n | tail -1 || true)
    START_ROUND=$(( 10#${last:-0} + 1 ))
fi
END_ROUND=$(( START_ROUND + N_ROUNDS - 1 ))
echo "$TAG running rounds ${START_ROUND}..${END_ROUND}"

for round in $(seq "$START_ROUND" "$END_ROUND"); do
    echo "$TAG round ${round}/${END_ROUND}: submitting..."
    MODEL="$MODEL" ROUND="$round" N_CHUNKS="$N_CHUNKS" OUT_DIR="$OUT_DIR" \
        sbatch --wait --job-name="$JOB_NAME" "$SLURM_SCRIPT"
    n_new=$(find "$OUT_DIR/$MODEL" -name "*_qrefine_r$(printf '%02d' "$round").npz" \
            2>/dev/null | wc -l || true)
    echo "$TAG round ${round}/${END_ROUND}: done (${n_new} point(s) improved)"
    if [ "$n_new" -eq 0 ]; then
        echo "$TAG round ${round} improved nothing -- converged, stopping early."
        break
    fi
done

echo "[done] quick-refine chain complete; now run:"
echo "    python scripts/model4_rate_panels_collect.py --model $MODEL"
