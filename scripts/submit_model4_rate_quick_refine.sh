#!/bin/bash
# ============================================================
#  Run all 10 quick neighbour-refine rounds of the optimised Heisenberg
#  framability RATES (rate_heis_4 / rate_heis_6) over the model4 grid.
#
#  Round r reads the base scan plus every earlier round, so the rounds form
#  ONE sequential chain: each is submitted with `sbatch --wait` and the next
#  starts only after it finishes.  Each round is itself a <=200-task array, so
#  in-flight tasks stay at the 200-job cap throughout.
#
#  Ten rounds propagate the framable floor up to ten grid rings (2.0 in gamma
#  at the 0.2 grid step) outward from wherever it already sits.
#
#  Usage:
#      bash scripts/submit_model4_rate_quick_refine.sh
#      N_ROUNDS=5 bash scripts/submit_model4_rate_quick_refine.sh
#      N_RESTARTS=5 MAXFEV_4=2000 bash scripts/submit_model4_rate_quick_refine.sh
#
#  Then replot (picks the min over base + every round automatically):
#      python scripts/model4_rate_panels_collect.py
# ============================================================
set -euo pipefail

N_ROUNDS="${N_ROUNDS:-10}"
N_CHUNKS="${N_CHUNKS:-200}"
OUT_DIR="${OUT_DIR:-results_model4_rate}"
SLURM_SCRIPT="scripts/model4_rate_quick_refine.slurm.sh"

cd "$(dirname "$0")/.."               # repo root
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p logs "$OUT_DIR"

for round in $(seq 1 "$N_ROUNDS"); do
    echo "[m4 rate qrefine] round ${round}/${N_ROUNDS}: submitting..."
    ROUND="$round" N_CHUNKS="$N_CHUNKS" OUT_DIR="$OUT_DIR" \
        sbatch --wait "$SLURM_SCRIPT"
    n_new=$(find "$OUT_DIR/model4" -name "*_qrefine_r$(printf '%02d' "$round").npz" \
            2>/dev/null | wc -l)
    echo "[m4 rate qrefine] round ${round}/${N_ROUNDS}: done (${n_new} point(s) improved)"
    if [ "$n_new" -eq 0 ]; then
        echo "[m4 rate qrefine] round ${round} improved nothing -- converged, stopping early."
        break
    fi
done

echo "[done] quick-refine chain complete; now run:"
echo "    python scripts/model4_rate_panels_collect.py"
