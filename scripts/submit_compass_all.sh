#!/bin/bash
# ============================================================
#  One-shot submission of the compass-chain (gamma, h) phase-diagram pipeline.
#
#  For every case of compass_chain.CASES (default: all six) this submits
#    1. the rate array       scripts/compass_rates.slurm.sh     (0-199, panels 1-6)
#    2. the many-body array  scripts/compass_manybody.slurm.sh  (0-120, panels 7-8)
#    3. the case's collect   scripts/compass_collect.slurm.sh, held by
#         --dependency=afterok:<rates array>:<many-body array>
#  so each case's figure appears as soon as both of its arrays have finished.
#
#  afterok releases a collect only if EVERY task of both arrays exited 0.  The
#  workers catch per-point errors (the point is logged and left missing, the
#  task still exits 0), so what holds a collect back is a task that crashed,
#  ran out of memory or hit its time limit.  That collect then stays PENDING
#  (DependencyNeverSatisfied).  To recover, resubmit the array with the same
#  CASE (finished points are skipped), and once it is done collect by hand:
#      CASE=<case> sbatch scripts/compass_collect.slurm.sh
#
#  Usage (from the repo root on the cluster):
#      bash scripts/submit_compass_all.sh
#      CASES="jx1.0_jy1.0_hx jx1.0_jy1.0_hxy" bash scripts/submit_compass_all.sh
# ============================================================

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate
set -euo pipefail

OUT_DIR=${OUT_DIR:-results_compass}
CASES=${CASES:-$(python -c 'from compass_chain import CASES; print(" ".join(CASES))')}

# Fail before submitting anything if a case name is wrong.
python - $CASES <<'PY'
import sys
from compass_chain import CASES
bad = [c for c in sys.argv[1:] if c not in CASES]
if bad:
    sys.exit(f'unknown case(s) {bad}; choose from {list(CASES)}')
PY

mkdir -p logs "$OUT_DIR"

for c in $CASES; do
    rid=$(CASE="$c" OUT_DIR="$OUT_DIR" \
          sbatch --parsable --job-name="cmp_rate_${c}" scripts/compass_rates.slurm.sh)
    rid=${rid%%;*}
    mid=$(CASE="$c" OUT_DIR="$OUT_DIR" \
          sbatch --parsable --job-name="cmp_8q_${c}" scripts/compass_manybody.slurm.sh)
    mid=${mid%%;*}
    cid=$(CASE="$c" IN_DIR="$OUT_DIR" OUT_DIR="$OUT_DIR" \
          sbatch --parsable --job-name="cmp_col_${c}" \
                 --dependency="afterok:${rid}:${mid}" scripts/compass_collect.slurm.sh)
    cid=${cid%%;*}
    echo "[$c] rates array $rid | many-body array $mid | collect $cid (afterok:$rid:$mid)"
done

echo
echo "figures: ${OUT_DIR}/<case>/compass_<case>_panels.png"
