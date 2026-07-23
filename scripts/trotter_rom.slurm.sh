#!/bin/bash
# ============================================================
#  SLURM job-array: 4-qubit Trotter RoM sub-pipeline (trotter_rom_4q).
#
#  One MODEL (model1..model6) is scanned over the REDUCED parameter ranges of
#  trotter_rom_4q.ROM_GRIDS -- exact subsets of the main-scan grids, so every
#  point reuses the stored stabilizer-3 framability.  Grid sizes:
#    model1   6 x 31 = 186      model2   6 x 21 = 126
#    model3  21 x 22 = 462      model4  31 x 31 = 961
#    model5  11 x 26 = 286      model6  26 x 26 = 676     (total 2697)
#
#  The grid is split across a 200-task array (N_CHUNKS=200); each task
#  processes a strided subset and skips any npz already current on disk, so
#  resubmission after a timeout simply continues.
#
#  Per point the worker reuses the stabilizer-3 framability from
#  results_trotter_v3 when present (models 1-5) and computes the Choi-state
#  RoM of the 4-qubit 2x2-lattice gate (8-qubit Choi state, column generation
#  -- Gurobi strongly recommended, hours per point are possible).
#
#  Submit one model (default model1):
#    mkdir -p logs results_trotter_rom
#    MODEL=model6 sbatch scripts/trotter_rom.slurm.sh
#
#  ONE-TIME SETUP (run on the LOGIN node -- $HOME and pip installs are not
#  available from the compute nodes, so everything must exist beforehand):
#    git clone https://github.com/quantum-programming/RoM-handbook.git
#    g++ RoM-handbook/exputils/dot/fast_dot_products.cpp \
#        -o RoM-handbook/exputils/dot/fast_dot_products.exe \
#        -std=c++17 -lz -O2 -DNDEBUG -march=x86-64-v2 -fopenmp
#    .venv/bin/pip install numba tqdm gurobipy
#  (no -march=native: a login-node build SIGILLs on older compute nodes)
# ============================================================

#SBATCH --job-name=trot_rom
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trot_rom_%A_%a.out
#SBATCH --error=logs/trot_rom_%A_%a.err

MODEL=${MODEL:-model1}
OUT_DIR=${OUT_DIR:-results_trotter_rom}
SCAN_DIR=${SCAN_DIR:-results_trotter_v3}
N_CHUNKS=${N_CHUNKS:-200}
DIM=${DIM:-}          # empty -> the model's own dim (all models: 2)
DT=${DT:-}            # empty -> per-point adaptive choose_dt
METHOD=${METHOD:-auto}
SOLVER=${SOLVER:-auto}
K=${K:-}              # empty -> handbook default (1e-8 at n_choi=8)

# --- environment ---------------------------------------------------------------
# Repo-local venv, exactly like trotter_scan.slurm.sh.  Do NOT use $HOME (it is
# read-only on the compute nodes) and do NOT pip-install here (system python is
# PEP-668 externally managed): all packages must already be in .venv (see the
# one-time setup above).
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

# Gurobi: strongly recommended for the 8-qubit Choi CG LPs.  The pip gurobipy
# alone carries a size-limited license that CANNOT solve them -- rom_of_gate
# probes the license and falls back to scipy/HiGHS automatically.  For a real
# license either load a cluster module or point at a (free academic) license
# file -- $HOME is read-only on the compute nodes but still readable:
#   export GRB_LICENSE_FILE=$HOME/gurobi.lic
module load gurobi 2>/dev/null || true

export ROM_HANDBOOK_DIR="${ROM_HANDBOOK_DIR:-$PWD/RoM-handbook}"
export ROM_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

# Fail fast on a broken environment (missing deps killed every point of the
# first submission silently, one traceback per point).
if ! python -c "import numpy, scipy, numba, tqdm" 2>/dev/null; then
    echo "ERROR: .venv is missing RoM dependencies. On the login node run:" \
         "  .venv/bin/pip install numba tqdm gurobipy" >&2
    exit 1
fi
if [ ! -x "$ROM_HANDBOOK_DIR/exputils/dot/fast_dot_products.exe" ]; then
    echo "ERROR: fast_dot_products.exe not compiled; it is required for the" \
         "8-qubit Choi column generation. See one-time setup above." >&2
    exit 1
fi

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

EXTRA_ARGS=()
[ -n "$DIM" ] && EXTRA_ARGS+=(--dim "$DIM")
[ -n "$DT" ]  && EXTRA_ARGS+=(--dt "$DT")
[ -n "$K" ]   && EXTRA_ARGS+=(--K "$K")

python scripts/trotter_rom_worker.py \
    --model    "$MODEL" \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --out_dir  "$OUT_DIR" \
    --scan_dir "$SCAN_DIR" \
    --method   "$METHOD" \
    --solver   "$SOLVER" \
    "${EXTRA_ARGS[@]}" \
    --verbose

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}: done"
