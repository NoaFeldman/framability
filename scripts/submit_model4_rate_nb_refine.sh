#!/bin/bash
# ============================================================
#  10 FULL neighbour-refine rounds of the optimised Heisenberg framability
#  RATES (rate_heis_4 / rate_heis_6, d_ext = 4 and 6) of one rate-panel model,
#  followed by the collect that redraws results_<model>_rate/<model>_rate_panels.png.
#
#  Every round re-optimises EVERY grid point seeded with its own and all
#  4-connected neighbour frames (scripts/model4_rate_nb_refine_worker.py) --
#  unlike scripts/submit_model4_rate_quick_refine.sh, which only touches the
#  boundary of the mu* = 0 region.  Round r reads the base scan and every
#  earlier round (quick and full), so the rounds form ONE sequential chain of
#  `sbatch --wait` arrays and at most 200 tasks are queued at a time.
#
#  A round whose array has failed tasks (e.g. a timeout) does not stop the
#  chain: rounds are incremental, the log says so, and the next round still
#  reads everything that was written.
#
#  RESUMING.  Every processed point writes its round file, so re-running the
#  script continues the highest round already on disk (finished points are
#  skipped) and then runs the remaining rounds up to N_ROUNDS.  Raise N_ROUNDS
#  (e.g. N_ROUNDS=15) for more rounds later.
#
#  Usage (repo root, on the login node inside tmux / nohup):
#      MODEL=model10 nohup bash scripts/submit_model4_rate_nb_refine.sh > logs/nbrefine_model10.log 2>&1 &
#      MODEL=model10 N_ROUNDS=15 bash scripts/submit_model4_rate_nb_refine.sh    # 5 more rounds
#      MODEL=model10 N_RESTARTS=5 MAXFEV_4=2000 bash scripts/submit_model4_rate_nb_refine.sh
#      MODEL=model10 SKIP_COLLECT=1 bash scripts/submit_model4_rate_nb_refine.sh
#
#  model10 = the Shibata-Katsura dissipative quantum Ising chain,
#            https://arxiv.org/abs/1904.12505
# ============================================================
set -euo pipefail

MODEL="${MODEL:?set MODEL, e.g. MODEL=model10}"
N_ROUNDS="${N_ROUNDS:-10}"
N_CHUNKS="${N_CHUNKS:-200}"
OUT_DIR="${OUT_DIR:-results_${MODEL}_rate}"
SLURM_SCRIPT="scripts/model4_rate_nb_refine.slurm.sh"
SHORT="m${MODEL#model}"                 # model10 -> m10
TAG="[${SHORT} rate nb-refine]"

cd "$(dirname "$0")/.."                 # repo root
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p logs

if [ ! -d "$OUT_DIR/$MODEL" ]; then
    echo "$TAG no base rate scan under $OUT_DIR/$MODEL -- run the rate panels first" >&2
    exit 1
fi

# Resume point: the highest full-refine round on disk (its finished points are
# skipped, its holes filled), else round 1.  10# forces base 10 ("08", "09").
if [ -z "${START_ROUND:-}" ]; then
    last=$(find "$OUT_DIR/$MODEL" -name '*_nrefine_r[0-9][0-9].npz' 2>/dev/null \
           | sed 's/.*_nrefine_r\([0-9][0-9]\)\.npz$/\1/' | sort -n | tail -1 || true)
    START_ROUND=$(( 10#${last:-01} ))
fi
echo "$TAG MODEL=$MODEL OUT_DIR=$OUT_DIR rounds ${START_ROUND}..${N_ROUNDS}"

for round in $(seq "$START_ROUND" "$N_ROUNDS"); do
    echo "$TAG round ${round}/${N_ROUNDS}: submitting..."
    if ! MODEL="$MODEL" ROUND="$round" N_CHUNKS="$N_CHUNKS" OUT_DIR="$OUT_DIR" \
            sbatch --wait --job-name="${SHORT}_nbrefine" "$SLURM_SCRIPT"; then
        echo "$TAG round ${round}: some array tasks failed" \
             "(see logs/nbrefine_${SHORT}_nbrefine_*); continuing"
    fi
    python scripts/model4_rate_nb_refine_worker.py --model "$MODEL" \
        --out_dir "$OUT_DIR" --round "$round" --report || true
done

if [ "${SKIP_COLLECT:-0}" != "1" ]; then
    echo "$TAG collect + figure"
    if ! MODEL="$MODEL" IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" \
            sbatch --wait --job-name="${SHORT}_collect" \
            scripts/model4_rate_panels_collect.slurm.sh; then
        echo "$TAG collect job failed -- see logs/m4collect_${SHORT}_collect_*" >&2
        exit 1
    fi
    echo "$TAG done: $OUT_DIR/${MODEL}_rate_panels.png"
fi
