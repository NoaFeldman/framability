#!/bin/bash
# ============================================================
#  Submit, for each model, the global rate re-optimisation array and its
#  collect/replot job chained with --dependency=afterok:<array id>, so one
#  invocation queues the whole pipeline.
#
#  Usage:
#      bash scripts/submit_rate_gopt.sh                  # model3 and model4
#      MODELS="model4" bash scripts/submit_rate_gopt.sh
#      MODELS="model10" DE_MAXITER=15 bash scripts/submit_rate_gopt.sh
#      FORCE=1 bash scripts/submit_rate_gopt.sh          # recompute existing gopt points
#
#  afterok requires every array task to exit 0.  The worker catches per-point
#  errors and always exits 0, so only a wall-time kill or a crash outside the
#  worker holds the collect back; then run the collect by hand (its slurm
#  script) or resubmit the array (finished points are skipped).
# ============================================================
set -euo pipefail

MODELS="${MODELS:-model3 model4}"
cd "$(dirname "$0")/.."               # repo root
mkdir -p logs

for MODEL in $MODELS; do
    OUT_DIR="${OUT_DIR:-results_${MODEL}_rate}"
    mkdir -p "$OUT_DIR"
    tag="m${MODEL#model}"
    jid=$(MODEL="$MODEL" OUT_DIR="$OUT_DIR" \
          sbatch --parsable --job-name="${tag}_gopt" scripts/rate_gopt.slurm.sh)
    jid="${jid%%;*}"                  # --parsable may print "<id>;<cluster>"
    echo "[$MODEL] worker array ${jid} submitted (0-199)"
    cid=$(MODEL="$MODEL" IN_DIR="$OUT_DIR" \
          sbatch --parsable --dependency=afterok:"$jid" \
                 --job-name="${tag}_gopt_collect" scripts/rate_gopt_collect.slurm.sh)
    echo "[$MODEL] collect job ${cid%%;*} queued with --dependency=afterok:${jid}"
    unset OUT_DIR
done
