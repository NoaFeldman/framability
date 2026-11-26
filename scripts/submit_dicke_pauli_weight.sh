#!/bin/bash
# ============================================================
#  Reweighted-Pauli rate of the open Dicke model (uniform and disordered
#  omega0) against the smallest allowed one-site weight w_min: one worker
#  array (scripts/dicke_pauli_weight.slurm.sh), then the collect / figure job
#  (--dependency=afterany).  The array size is the number of combos (the
#  worker's --count), capped at 200; with more combos each task takes
#  every N_TASKS-th one.  Nothing here checks job counts or the queue.
#
#  Usage (login node, repo root):
#      bash scripts/submit_dicke_pauli_weight.sh                 # the 11 labelled points
#      POINTS="U3 D3" bash scripts/submit_dicke_pauli_weight.sh  # only these
#      GRID=1 bash scripts/submit_dicke_pauli_weight.sh          # + w* maps (2 x 336 pts)
#      FORCE=1 bash scripts/submit_dicke_pauli_weight.sh         # recompute
#  Every knob of dicke_pauli_weight.slurm.sh passes through the environment
#  (FAMILY, N_ATOMS, M_SPECIES, POINTS, GRID, GRID_WIDTHS, LOG10_WMIN_END,
#  N_W, N_RANDOM, N_BISECT, OUT_ROOT, FORCE; FIG_DIR, T_VALUES for the collect).
#
#  Output: results_dicke_pauli_weight/..., results_model4_rate/<family>_N<N>_pauli_weight_*.png
#  Logs:   logs/dicke_pw_<array id>_<task>.out, logs/dicke_pw_collect_<id>.out
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs

export FAMILY="${FAMILY:-dicke_dec}"
export N_ATOMS="${N_ATOMS:-100}"
export M_SPECIES="${M_SPECIES:-4}"
export POINTS="${POINTS:-}"
export GRID="${GRID:-}"
export GRID_WIDTHS="${GRID_WIDTHS:-0 0.5}"
export OUT_ROOT="${OUT_ROOT:-results_dicke_pauli_weight}"

source .venv/bin/activate
if [ "$POINTS" = "none" ]; then
    COUNT_ARGS="--points"
elif [ -n "$POINTS" ]; then
    COUNT_ARGS="--points $POINTS"
else
    COUNT_ARGS=""
fi
n_combos=$(python scripts/dicke_pauli_weight_worker.py --count --family "$FAMILY" \
           --N "$N_ATOMS" --grid_widths $GRID_WIDTHS $COUNT_ARGS ${GRID:+--grid})
if [ "$n_combos" -lt 1 ]; then
    echo "nothing to run (0 combos)"; exit 1
fi
export N_TASKS=$(( n_combos < 200 ? n_combos : 200 ))

aid=$(sbatch --parsable --array=0-$((N_TASKS - 1)) scripts/dicke_pauli_weight.slurm.sh)
aid="${aid%%;*}"
echo "===== worker array ${aid}: ${FAMILY} N=${N_ATOMS} | points=${POINTS:-all} grid=${GRID:-0} | ${n_combos} combos on ${N_TASKS} tasks ====="

cid=$(sbatch --parsable --dependency=afterany:"${aid}" scripts/dicke_pauli_weight_collect.slurm.sh)
echo "===== collect job ${cid%%;*} (after ${aid}) ====="
