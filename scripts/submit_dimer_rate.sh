#!/bin/bash
# ============================================================
#  Submit the dimer-frame rate pipeline for several models side by side:
#  per run one worker array (scripts/dimer_rate.slurm.sh) and its collect /
#  panel job chained with --dependency=afterok:<array id>.
#
#  A 2-qubit frame S on every dimer; the bond gate is certified inside a
#  dimer (frame S) and across dimers (frame S (x) S); objective = per-bond
#  average (mu_in + (2 dim - 1) mu_cut) / (2 dim), compared with the stored
#  product-frame rates.  dim 2 uses the brick dimer tiling (TILING=columnar
#  for stacked dimers).
#
#  RUNS entries are <model> or <model>:<dim> (dim defaults to the model's);
#  default "model4 model10 model15" (model4 square lattice, model10 / model15
#  chains; add model10:2 for the 2D Shibata-Katsura lattice).
#  The array sizes share MAX_JOBS (200) in proportion to the grid sizes,
#  one slot per run kept for its collect job, e.g. for the default runs
#  (2601 + 2601 + 224 points): 94 + 94 + 8 tasks + 3 collects = 199,
#  ~28 points per task, ~1.5-3 min per point -> ~1-1.5 h per task.
#
#  Usage (login node, repo root):
#      bash scripts/submit_dimer_rate.sh
#      STRIDE=5 RUNS="model4 model10" MAX_JOBS=100 bash scripts/submit_dimer_rate.sh
#                                          # 11 x 11 preview of the 51 x 51 grids
#      RUNS="model10 model10:2" bash scripts/submit_dimer_rate.sh
#      RUNS="model15" MODE=joint bash scripts/submit_dimer_rate.sh
#      FORCE=1 bash scripts/submit_dimer_rate.sh     # recompute existing points
#  Every knob of dimer_rate.slurm.sh (M_EXT, RW_MAXFEV, NO_CERTIFY, ...)
#  passes through the environment.
#
#  afterok needs every array task to exit 0; the worker catches per-point
#  errors, so only a wall-time kill holds a collect back -- then resubmit
#  (finished points are skipped) or run the collect slurm script by hand.
# ============================================================
set -euo pipefail

RUNS="${RUNS:-model4 model10 model15}"
MAX_JOBS="${MAX_JOBS:-200}"
export MODE="${MODE:-split}"
export STRIDE="${STRIDE:-1}"

cd "$(dirname "$0")/.."               # repo root
mkdir -p logs
source .venv/bin/activate

# grid size and default dim per model (from MODELS)
declare -A NPTS DIMS
total=0
n_runs=0
for run in $RUNS; do
    model="${run%%:*}"
    read -r n d < <(python - "$model" "$STRIDE" <<'EOF'
import sys
sys.path.insert(0, '.')
from trotter_lindbladian_scan import MODELS
m, s = MODELS[sys.argv[1]], int(sys.argv[2])
print(len(m.p1_vals[::s]) * len(m.p2_vals[::s]), m.dim)
EOF
)
    if [[ "$run" == *:* ]]; then d="${run#*:}"; fi
    NPTS[$run]=$n
    DIMS[$run]=$d
    total=$((total + n))
    n_runs=$((n_runs + 1))
done
budget=$((MAX_JOBS - n_runs))         # one slot per run for its collect job
if (( budget < n_runs )); then
    echo "MAX_JOBS=$MAX_JOBS too small for $n_runs runs" >&2
    exit 1
fi

for run in $RUNS; do
    model="${run%%:*}"
    n=${NPTS[$run]}
    dim=${DIMS[$run]}
    tasks=$(( budget * n / total ))
    if (( tasks < 1 )); then tasks=1; fi
    if (( tasks > n )); then tasks=$n; fi
    tiling=""
    if (( dim == 2 )); then tiling="${TILING:-brick}"; fi
    tag="m${model#model}"
    if [[ "$run" == *:* ]]; then tag="${tag}d${dim}"; fi
    tag="${tag}_dimer"
    if [[ "$MODE" != "split" ]]; then tag="${tag}_${MODE}"; fi
    if (( STRIDE != 1 )); then tag="${tag}_s${STRIDE}"; fi
    # the same run (model, dim, mode, stride -> same output folder) must not
    # be queued twice; other dimer runs may be
    if command -v squeue >/dev/null 2>&1 && \
       squeue -u "$USER" -h -o '%j' | grep -Eqx "${tag}(_collect)?"; then
        echo "[$run] $tag is already queued or running: skipped (scancel it first)" >&2
        continue
    fi

    jid=$(MODEL="$model" DIM="$dim" TILING="$tiling" N_CHUNKS="$tasks" \
          sbatch --parsable --array="0-$((tasks - 1))" --job-name="$tag" \
                 scripts/dimer_rate.slurm.sh)
    jid="${jid%%;*}"                  # --parsable may print "<id>;<cluster>"
    echo "[$run] dim=$dim ${tiling:-chain}/$MODE: worker array $jid " \
         "(0-$((tasks - 1)), $n points, ~$(( (n + tasks - 1) / tasks )) per task)"
    cid=$(MODEL="$model" DIM="$dim" TILING="$tiling" \
          sbatch --parsable --dependency=afterok:"$jid" \
                 --job-name="${tag}_collect" scripts/dimer_rate_collect.slurm.sh)
    echo "[$run] collect job ${cid%%;*} queued with --dependency=afterok:$jid"
done
