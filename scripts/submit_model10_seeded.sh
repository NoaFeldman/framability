#!/bin/bash
# ============================================================
#  One-shot submission of the seeded model10 rate pipeline:
#    1. scripts/model10_seeded.slurm.sh          array 0-199 (data generation)
#    2. scripts/model10_seeded_collect.slurm.sh  --dependency=afterok:<array>
#       (neighbour cross-evaluation + figure with the new panels appended)
#
#  Usage (from anywhere; the script moves to the repo root):
#      bash scripts/submit_model10_seeded.sh
#      STRIDE=5 bash scripts/submit_model10_seeded.sh           # 11 x 11 preview
#      OUT_PNG=results_model10_rate/model10_rate_panels.png bash scripts/submit_model10_seeded.sh
#  Every variable of the two slurm scripts (OUT_DIR, STORED_DIRS, DE_MAXITER,
#  BASE_IN_DIRS, BASE_NPZ, Q_DIR, OBS_DIR, N_PROC, ...) passes through, since
#  sbatch exports the submitting environment.
#
#  afterok needs every array task to exit 0.  The worker catches per-point
#  errors and always exits 0, so only a wall-time kill holds the collect back;
#  then resubmit this script (finished points are skipped) or run the collect
#  by hand:  sbatch scripts/model10_seeded_collect.slurm.sh
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
export OUT_DIR="${OUT_DIR:-results_model10_rate}"
export STRIDE="${STRIDE:-1}"
mkdir -p logs "$OUT_DIR" "$(dirname "${OUT_PNG:-results_model4_rate/x}")"

jid=$(sbatch --parsable scripts/model10_seeded.slurm.sh)
jid="${jid%%;*}"                      # --parsable may print "<id>;<cluster>"
echo "[model10 seeded] worker array ${jid} submitted (0-199, stride ${STRIDE})"

cid=$(sbatch --parsable --dependency=afterok:"$jid" \
             scripts/model10_seeded_collect.slurm.sh)
cid="${cid%%;*}"
echo "[model10 seeded] collect job ${cid} queued with --dependency=afterok:${jid}"
echo "  progress:  tail -f logs/m10seed_m10_seed_${jid}_0.out"
echo "  figure:    ${OUT_PNG:-results_model4_rate/model10_rate_panels.png}"
