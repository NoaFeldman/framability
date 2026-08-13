#!/bin/bash
# ============================================================
#  SLURM job-array: "unitary magic scan" data generation, generic across the
#  5 models of unitary_models_for_magic.tex (model1_ising_tilted_magic_scan.py
#  .. model5_xy_dm_magic_scan.py).  Which model to run is selected entirely by
#  the SCAN_MODULE / MODEL environment variables, so this one script covers
#  all five -- see scripts/submit_unitary_magic.sh, which submits it once per
#  model with those variables set.
#
#  Per point the worker computes all seven maps:
#     fra_D1, fra_D2   dt->0 limit of the stabilizer-3 framability of the
#                      nearest-neighbour Trotter bond gate (D = 1, 2), over the
#                      ladder dt_i = 0.1 i * choose_dt, i = 1..10
#     nc_2a..nc_2e     non-cliffordness (alpha=2 stabilizer Renyi entropy of the
#                      Choi state) of exp(iHt) on the 2x2 lattice (4 qubits ->
#                      8-qubit Choi state), by exact diagonalization
#
#  Every model's grid is 20x20 = 400 points; over a 200-task array that is 2
#  points per task, strided.  Each task skips any npz already complete on
#  disk, so a resubmission after a timeout simply continues.  Cost is
#  dominated by the framability group (2 dims x 10 dt x 1080 stabilizer-frame
#  LPs per point); GROUP=nc runs the cheap magic maps alone in minutes.
#
#  Submit one model directly:
#    mkdir -p logs results_unitary_magic results_plots
#    SCAN_MODULE=model1_ising_tilted_magic_scan MODEL=ising_tilted \
#        sbatch scripts/magic_scan.slurm.sh
#
#  All 5 models' data + the dependent collect-all job, in one go:
#    bash scripts/submit_unitary_magic.sh
#
#  ONE-TIME SETUP (login node): the Choi state comes from rom_of_gate, which
#  needs the RoM-handbook clone present at import time (no compilation, no pip):
#    git clone https://github.com/quantum-programming/RoM-handbook.git
# ============================================================

#SBATCH --job-name=unitary_magic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-199
#SBATCH --output=logs/unitary_magic_%A_%a.out
#SBATCH --error=logs/unitary_magic_%A_%a.err

SCAN_MODULE=${SCAN_MODULE:?set SCAN_MODULE, e.g. model1_ising_tilted_magic_scan}
MODEL=${MODEL:?set MODEL, e.g. ising_tilted}
OUT_DIR=${OUT_DIR:-results_unitary_magic}
GROUP=${GROUP:-both}       # both | fra (expensive) | nc (cheap)
N_CHUNKS=${N_CHUNKS:-200}
FORCE=${FORCE:-}           # non-empty -> recompute even if the npz is current

# --- environment ---------------------------------------------------------------
# Repo-local venv, exactly like trotter_rom_dtbase.slurm.sh.  Do NOT use $HOME
# (read-only on the compute nodes) and do NOT pip-install here (system python is
# PEP-668 externally managed).
source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

export ROM_HANDBOOK_DIR="${ROM_HANDBOOK_DIR:-$PWD/RoM-handbook}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}"
# The worker pins the BLAS/LP thread counts to 1 itself (one cpu-per-task).

if ! python -c "import numpy, scipy" 2>/dev/null; then
    echo "ERROR: .venv is missing numpy/scipy." >&2
    exit 1
fi
if [ ! -d "$ROM_HANDBOOK_DIR/exputils" ]; then
    echo "ERROR: $ROM_HANDBOOK_DIR/exputils not found." \
         "Clone the RoM-handbook repo (see one-time setup above)." >&2
    exit 1
fi

echo "[$SCAN_MODULE/$MODEL/$GROUP] chunk ${SLURM_ARRAY_TASK_ID}/${N_CHUNKS}: starting"

EXTRA_ARGS=()
[ -n "$FORCE" ] && EXTRA_ARGS+=(--force)

python scripts/magic_worker.py \
    --scan_module "$SCAN_MODULE" \
    --model       "$MODEL" \
    --task_id     "$SLURM_ARRAY_TASK_ID" \
    --n_chunks    "$N_CHUNKS" \
    --out_dir     "$OUT_DIR" \
    --group       "$GROUP" \
    "${EXTRA_ARGS[@]}"

echo "[$SCAN_MODULE/$MODEL] chunk ${SLURM_ARRAY_TASK_ID}: done"
