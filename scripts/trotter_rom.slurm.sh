#!/bin/bash
# ============================================================
#  SLURM job-array: 4-qubit Trotter RoM sub-pipeline (trotter_rom_4q).
#
#  One MODEL (model1..model6) is scanned over its two varying parameters on
#  the same grid as the main trotter scan.  Grid sizes (point counts):
#    model1  21 x  51 = 1071     model2  21 x  51 = 1071
#    model3  51 x  51 = 2601     model4  51 x  51 = 2601
#    model5  21 x 101 = 2121     model6  51 x  51 = 2601
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
#  ONE-TIME SETUP (shared with rom_gate.slurm.sh):
#    git clone https://github.com/quantum-programming/RoM-handbook.git
#    g++ RoM-handbook/exputils/dot/fast_dot_products.cpp \
#        -o RoM-handbook/exputils/dot/fast_dot_products.exe \
#        -std=c++17 -lz -O2 -DNDEBUG -mtune=native -march=native -fopenmp
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

# --- environment (same stack as rom_gate.slurm.sh) ----------------------------
VENV="$HOME/venvs/framability"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy scipy numba tqdm

# Gurobi: effectively required for the 8-qubit Choi CG LPs.  If the cluster
# provides it as a module this picks it up; otherwise --solver auto falls back
# to scipy (slow and fragile at n_choi=8).
module load gurobi 2>/dev/null || true
python -m pip install --quiet gurobipy 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR}"
export ROM_HANDBOOK_DIR="${ROM_HANDBOOK_DIR:-$PWD/RoM-handbook}"
export ROM_TMPDIR="${SLURM_TMPDIR:-/tmp}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

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
