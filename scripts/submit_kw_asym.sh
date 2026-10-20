#!/bin/bash
# ============================================================
#  One-shot submission of the KW-asymmetry pipeline (arXiv:2610.23111 Eq. 14,
#  model10, exact full-density-matrix evolution):
#    1. the 200-task array  scripts/kw_asym.slurm.sh          (one scan point each)
#    2. its collect + plot  scripts/kw_asym_collect.slurm.sh, held by
#         --dependency=afterok:<array>
#
#  afterok releases the collect only if EVERY task exited 0.  A task that
#  crashed, failed its selfcheck, ran out of memory or hit the time limit
#  leaves the collect PENDING (DependencyNeverSatisfied): scancel it,
#  resubmit with the same settings (finished points are skipped), or collect
#  what is there by hand:
#      TAG=<tag> sbatch scripts/kw_asym_collect.slurm.sh
#
#  Usage (from the repo root on the cluster):
#      bash scripts/submit_kw_asym.sh                            # ring:12, local jumps
#      JUMPS=collective bash scripts/submit_kw_asym.sh           # the paper's L_A, L_B
#      INIT=plus bash scripts/submit_kw_asym.sh                  # |+>^N start
#  Knobs (env): GEOMETRY (ring:12; 2D pending, see kw_asymmetry.TWO_D_NOTE),
#  JUMPS (local | collective), INIT (ghz_s | ghz | plus), ALPHA, T_MAX (3.0),
#  N_T (61), RTOL (1e-8), ATOL (1e-12), OUT_DIR (results_kw_asym),
#  SELFCHECK (1), WALL (12:00:00 local, 24:00:00 collective), MEM (16G).
# ============================================================

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
set -euo pipefail

export GEOMETRY=${GEOMETRY:-ring:12}
export JUMPS=${JUMPS:-local}
export INIT=${INIT:-ghz_s}
export OUT_DIR=${OUT_DIR:-results_kw_asym}
if [ "$JUMPS" = "collective" ]; then WALL=${WALL:-24:00:00}; else WALL=${WALL:-12:00:00}; fi
MEM=${MEM:-16G}

# The run tag (and a fail-fast check of GEOMETRY / INIT) before submitting anything.
TAG=$(python - <<'PY'
import os
from kw_asymmetry import geometry_from_spec, run_tag, ALPHA_PAPER
g = geometry_from_spec(os.environ['GEOMETRY'])
a = float(os.environ.get('ALPHA') or ALPHA_PAPER)
print(run_tag(g.name, os.environ['JUMPS'], os.environ['INIT'], a))
PY
)
export TAG

mkdir -p logs "$OUT_DIR/$TAG"

aid=$(sbatch --parsable --job-name="kw_${TAG}" --time="$WALL" --mem="$MEM" \
             scripts/kw_asym.slurm.sh)
aid=${aid%%;*}
cid=$(sbatch --parsable --job-name="kwcol_${TAG}" --dependency="afterok:${aid}" \
             scripts/kw_asym_collect.slurm.sh)
cid=${cid%%;*}
echo "[kw asym ${TAG}] array ${aid} (200 tasks, ${WALL}, ${MEM}) | collect ${cid} (afterok:${aid})"
echo "figure: ${OUT_DIR}/${TAG}/kw_asym_${TAG}.png"
