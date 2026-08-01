#!/bin/bash
# ============================================================
#  SLURM job-array: state-RoM sub-pipeline (trotter_rom_state).
#
#  Per grid point of one MODEL (model1..model6):
#    * stabilizer-3 framability of the two-qubit bond Trotter gate -- reused
#      from results_trotter_v3 when present (models 1-5), computed from the
#      bond gate otherwise (all of model6);
#    * RoM of the 2x2-lattice state after ONE application of the exact lattice
#      propagator expm(L_full dt) to the model's lpdo_max start state.
#
#  Each model runs on its FULL trotter_lindbladian_scan grid:
#    model1  21 x  51 = 1071      model2  21 x  51 = 1071
#    model3  51 x  51 = 2601      model4  51 x  51 = 2601
#    model5  21 x 101 = 2121      model6  51 x  51 = 2601    (total 12066)
#
#  The grid is split across a 200-task array (N_CHUNKS=200); each task processes
#  a strided subset and skips any npz already current on disk, so resubmission
#  after a timeout simply continues.
#
#  The four-qubit state RoM is a single sparse scipy/HiGHS LP over the full
#  precomputed stabilizer matrix (256 x 36720) -- seconds per point, and exact.
#  Unlike the Choi-state gate RoM of scripts/trotter_rom.slurm.sh it needs
#  NO Gurobi, NO numba/tqdm and NO compiled C++ enumerator: numpy + scipy and
#  the handbook's data/Amat/Amat4.npz are the whole dependency set.  model6 is
#  the slow model here, because its 2601 framabilities are not precomputed.
#
#  Submit one model (default model1):
#    mkdir -p logs results_trotter_rom_state
#    MODEL=model6 sbatch scripts/trotter_rom_state.slurm.sh
#
#  Submit all six:
#    bash scripts/submit_trotter_rom_state.sh
#
#  ONE-TIME SETUP (login node): the RoM-handbook clone must exist for its
#  data/Amat/Amat4.npz -- no compilation and no pip install are needed:
#    git clone https://github.com/quantum-programming/RoM-handbook.git
# ============================================================

#SBATCH --job-name=trot_roms
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/trot_roms_%A_%a.out
#SBATCH --error=logs/trot_roms_%A_%a.err

MODEL=${MODEL:-model1}
OUT_DIR=${OUT_DIR:-results_trotter_rom_state}
SCAN_DIR=${SCAN_DIR:-results_trotter_v3}
N_CHUNKS=${N_CHUNKS:-200}
DIM=${DIM:-}          # empty -> the model's own dim (all models: 2)
DT=${DT:-}            # empty -> per-point adaptive choose_dt

# --- environment ---------------------------------------------------------------
# Repo-local venv, exactly like trotter_scan.slurm.sh.  Do NOT use $HOME (it is
# read-only on the compute nodes) and do NOT pip-install here (system python is
# PEP-668 externally managed).
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

export ROM_HANDBOOK_DIR="${ROM_HANDBOOK_DIR:-$PWD/RoM-handbook}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"

if ! python -c "import numpy, scipy" 2>/dev/null; then
    echo "ERROR: .venv is missing numpy/scipy." >&2
    exit 1
fi
if [ ! -f "$ROM_HANDBOOK_DIR/data/Amat/Amat4.npz" ]; then
    echo "ERROR: $ROM_HANDBOOK_DIR/data/Amat/Amat4.npz not found." \
         "Clone the RoM-handbook repo (see one-time setup above)." >&2
    exit 1
fi

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

EXTRA_ARGS=()
[ -n "$DIM" ] && EXTRA_ARGS+=(--dim "$DIM")
[ -n "$DT" ]  && EXTRA_ARGS+=(--dt "$DT")

python scripts/trotter_rom_state_worker.py \
    --model    "$MODEL" \
    --task_id  "$SLURM_ARRAY_TASK_ID" \
    --n_chunks "$N_CHUNKS" \
    --out_dir  "$OUT_DIR" \
    --scan_dir "$SCAN_DIR" \
    "${EXTRA_ARGS[@]}" \
    --verbose

echo "[$MODEL] chunk ${SLURM_ARRAY_TASK_ID}: done"
